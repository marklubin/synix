"""Tests for the LLM executor — sequential and concurrent execution."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from synix.build.executor import (
    ConcurrentExecutor,
    LLMExecutor,
    LLMRequest,
    LLMResult,
    SequentialExecutor,
    create_executor,
)
from synix.build.llm_client import LLMResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client(responses: list[str] | None = None, delay: float = 0.0):
    """Create a mock LLMClient that returns predetermined responses.

    Args:
        responses: List of response strings. If None, returns "response-{i}".
        delay: Sleep time per call to simulate latency.
    """
    call_count = {"n": 0}
    call_log = []

    def mock_complete(messages, max_tokens=None, temperature=None, artifact_desc="artifact"):
        idx = call_count["n"]
        call_count["n"] += 1
        call_log.append({
            "index": idx,
            "messages": messages,
            "artifact_desc": artifact_desc,
            "thread": threading.current_thread().name,
            "time": time.monotonic(),
        })
        if delay > 0:
            time.sleep(delay)
        text = responses[idx] if responses and idx < len(responses) else f"response-{idx}"
        return LLMResponse(
            content=text,
            model="test-model",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )

    client = MagicMock()
    client.complete = mock_complete
    client._call_log = call_log
    client._call_count = call_count
    return client


def _make_requests(n: int) -> list[LLMRequest]:
    """Create n simple LLM requests."""
    return [
        LLMRequest(
            messages=[{"role": "user", "content": f"request {i}"}],
            artifact_desc=f"artifact-{i}",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# LLMRequest tests
# ---------------------------------------------------------------------------


class TestLLMRequest:
    """Tests for LLMRequest dataclass."""

    def test_defaults(self):
        req = LLMRequest(messages=[{"role": "user", "content": "hello"}])
        assert req.max_tokens is None
        assert req.temperature is None
        assert req.artifact_desc == "artifact"

    def test_custom_values(self):
        req = LLMRequest(
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=256,
            temperature=0.7,
            artifact_desc="test-artifact",
        )
        assert req.max_tokens == 256
        assert req.temperature == 0.7
        assert req.artifact_desc == "test-artifact"


# ---------------------------------------------------------------------------
# LLMResult tests
# ---------------------------------------------------------------------------


class TestLLMResult:
    """Tests for LLMResult dataclass."""

    def test_success(self):
        resp = LLMResponse(content="ok", model="m", input_tokens=1, output_tokens=1, total_tokens=2)
        result = LLMResult(index=0, response=resp)
        assert result.success is True
        assert result.error is None

    def test_error(self):
        result = LLMResult(index=1, error=RuntimeError("fail"))
        assert result.success is False
        assert result.response is None
        assert str(result.error) == "fail"


# ---------------------------------------------------------------------------
# SequentialExecutor tests
# ---------------------------------------------------------------------------


class TestSequentialExecutor:
    """Tests for SequentialExecutor."""

    def test_executes_in_order(self):
        """Requests are executed one by one in order."""
        client = _make_mock_client(["first", "second", "third"])
        executor = SequentialExecutor()
        requests = _make_requests(3)

        results = executor.execute(client, requests)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.index == i
            assert result.success is True
            assert result.response.content == ["first", "second", "third"][i]

    def test_empty_requests(self):
        """Empty request list returns empty result list."""
        client = _make_mock_client()
        executor = SequentialExecutor()

        results = executor.execute(client, [])
        assert results == []

    def test_single_request(self):
        """Single request works correctly."""
        client = _make_mock_client(["only"])
        executor = SequentialExecutor()

        results = executor.execute(client, _make_requests(1))
        assert len(results) == 1
        assert results[0].response.content == "only"

    def test_error_captured_not_raised(self):
        """Errors are captured in results, not raised."""
        client = MagicMock()
        client.complete.side_effect = RuntimeError("boom")
        executor = SequentialExecutor()

        results = executor.execute(client, _make_requests(2))

        assert len(results) == 2
        assert results[0].success is False
        assert "boom" in str(results[0].error)
        assert results[1].success is False

    def test_partial_failure(self):
        """Some requests succeed, some fail — all captured."""
        call_count = {"n": 0}

        def mock_complete(**kwargs):
            idx = call_count["n"]
            call_count["n"] += 1
            if idx == 1:
                raise RuntimeError("request 1 failed")
            return LLMResponse(
                content=f"ok-{idx}", model="m",
                input_tokens=1, output_tokens=1, total_tokens=2,
            )

        client = MagicMock()
        client.complete = mock_complete
        executor = SequentialExecutor()

        results = executor.execute(client, _make_requests(3))

        assert results[0].success is True
        assert results[0].response.content == "ok-0"
        assert results[1].success is False
        assert "request 1 failed" in str(results[1].error)
        assert results[2].success is True
        assert results[2].response.content == "ok-2"

    def test_is_llm_executor_subclass(self):
        """SequentialExecutor is a proper LLMExecutor subclass."""
        assert isinstance(SequentialExecutor(), LLMExecutor)


# ---------------------------------------------------------------------------
# ConcurrentExecutor tests
# ---------------------------------------------------------------------------


class TestConcurrentExecutor:
    """Tests for ConcurrentExecutor."""

    def test_executes_all_requests(self):
        """All requests are executed and results returned."""
        client = _make_mock_client()
        executor = ConcurrentExecutor(max_concurrency=4)
        requests = _make_requests(5)

        results = executor.execute(client, requests)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.index == i
            assert result.success is True
        # All responses are present (order of mock call_count is non-deterministic)
        contents = {r.response.content for r in results}
        assert contents == {f"response-{i}" for i in range(5)}

    def test_results_preserve_order(self):
        """Results are returned in request order regardless of completion order."""
        client = _make_mock_client(delay=0.01)
        executor = ConcurrentExecutor(max_concurrency=5)
        requests = _make_requests(10)

        results = executor.execute(client, requests)

        assert len(results) == 10
        for i, result in enumerate(results):
            assert result.index == i

    def test_respects_concurrency_limit(self):
        """No more than max_concurrency requests run simultaneously."""
        max_concurrent = {"value": 0}
        current_concurrent = {"value": 0}
        lock = threading.Lock()

        def mock_complete(**kwargs):
            with lock:
                current_concurrent["value"] += 1
                if current_concurrent["value"] > max_concurrent["value"]:
                    max_concurrent["value"] = current_concurrent["value"]
            time.sleep(0.05)  # Hold the slot briefly
            with lock:
                current_concurrent["value"] -= 1
            return LLMResponse(
                content="ok", model="m",
                input_tokens=1, output_tokens=1, total_tokens=2,
            )

        client = MagicMock()
        client.complete = mock_complete
        executor = ConcurrentExecutor(max_concurrency=3)
        requests = _make_requests(10)

        results = executor.execute(client, requests)

        assert all(r.success for r in results)
        assert max_concurrent["value"] <= 3, (
            f"Max concurrent was {max_concurrent['value']}, expected <= 3"
        )

    def test_empty_requests(self):
        """Empty request list returns empty result list."""
        client = _make_mock_client()
        executor = ConcurrentExecutor(max_concurrency=4)

        results = executor.execute(client, [])
        assert results == []

    def test_single_request(self):
        """Single request works correctly."""
        client = _make_mock_client(["only"])
        executor = ConcurrentExecutor(max_concurrency=4)

        results = executor.execute(client, _make_requests(1))
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].response.content == "only"

    def test_error_captured_not_raised(self):
        """Errors are captured in results, not raised."""
        client = MagicMock()
        client.complete.side_effect = RuntimeError("boom")
        executor = ConcurrentExecutor(max_concurrency=2)

        results = executor.execute(client, _make_requests(3))

        assert len(results) == 3
        assert all(not r.success for r in results)

    def test_partial_failure(self):
        """Some requests succeed, some fail — all captured."""
        call_count = {"n": 0}
        lock = threading.Lock()

        def mock_complete(**kwargs):
            with lock:
                idx = call_count["n"]
                call_count["n"] += 1
            desc = kwargs.get("artifact_desc", "")
            if "artifact-1" in desc:
                raise RuntimeError("request 1 failed")
            return LLMResponse(
                content=f"ok-{idx}", model="m",
                input_tokens=1, output_tokens=1, total_tokens=2,
            )

        client = MagicMock()
        client.complete = mock_complete
        executor = ConcurrentExecutor(max_concurrency=4)

        results = executor.execute(client, _make_requests(3))

        assert len(results) == 3
        # Request 1 should have failed
        assert results[1].success is False
        assert "request 1 failed" in str(results[1].error)
        # Others should have succeeded
        succeeded = [r for r in results if r.success]
        assert len(succeeded) == 2

    @patch("synix.build.executor.time.sleep")
    def test_retry_on_rate_limit(self, mock_sleep):
        """Retries requests that fail with rate limit errors."""
        call_count = {"n": 0}

        def mock_complete(**kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise RuntimeError("Failed to process artifact after 2 attempts: rate limit")
            return LLMResponse(
                content="success after retry", model="m",
                input_tokens=1, output_tokens=1, total_tokens=2,
            )

        client = MagicMock()
        client.complete = mock_complete
        executor = ConcurrentExecutor(max_concurrency=1, max_retries=3, base_delay=0.01)

        results = executor.execute(client, _make_requests(1))

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].response.content == "success after retry"
        assert call_count["n"] == 3

    @patch("synix.build.executor.time.sleep")
    def test_retry_exhausted(self, mock_sleep):
        """After max_retries, returns error."""
        client = MagicMock()
        client.complete.side_effect = RuntimeError(
            "Failed to process artifact after 2 attempts: 429 rate limit"
        )
        executor = ConcurrentExecutor(max_concurrency=1, max_retries=2, base_delay=0.01)

        results = executor.execute(client, _make_requests(1))

        assert len(results) == 1
        assert results[0].success is False
        assert "rate limit" in str(results[0].error).lower()

    def test_non_rate_limit_error_not_retried(self):
        """Non-rate-limit errors are not retried."""
        call_count = {"n": 0}

        def mock_complete(**kwargs):
            call_count["n"] += 1
            raise RuntimeError("LLM API error: bad request")

        client = MagicMock()
        client.complete = mock_complete
        executor = ConcurrentExecutor(max_concurrency=1, max_retries=3)

        results = executor.execute(client, _make_requests(1))

        assert len(results) == 1
        assert results[0].success is False
        # Should have been called only once (no retry for non-rate-limit errors)
        assert call_count["n"] == 1

    def test_concurrency_faster_than_sequential(self):
        """Concurrent execution is faster than sequential for parallel-safe work."""
        client = _make_mock_client(delay=0.05)
        requests = _make_requests(8)

        # Sequential timing
        seq_executor = SequentialExecutor()
        seq_start = time.monotonic()
        seq_results = seq_executor.execute(client, requests)
        seq_time = time.monotonic() - seq_start

        # Reset client
        client = _make_mock_client(delay=0.05)

        # Concurrent timing
        conc_executor = ConcurrentExecutor(max_concurrency=8)
        conc_start = time.monotonic()
        conc_results = conc_executor.execute(client, requests)
        conc_time = time.monotonic() - conc_start

        # Both should succeed
        assert all(r.success for r in seq_results)
        assert all(r.success for r in conc_results)

        # Concurrent should be meaningfully faster
        # (8 * 0.05 = 0.4s sequential vs ~0.05s concurrent)
        assert conc_time < seq_time * 0.75, (
            f"Concurrent ({conc_time:.3f}s) not meaningfully faster than "
            f"sequential ({seq_time:.3f}s)"
        )

    def test_is_llm_executor_subclass(self):
        """ConcurrentExecutor is a proper LLMExecutor subclass."""
        assert isinstance(ConcurrentExecutor(), LLMExecutor)

    def test_max_concurrency_clamped_to_1(self):
        """max_concurrency < 1 is clamped to 1."""
        executor = ConcurrentExecutor(max_concurrency=0)
        assert executor.max_concurrency == 1

        executor = ConcurrentExecutor(max_concurrency=-5)
        assert executor.max_concurrency == 1


# ---------------------------------------------------------------------------
# create_executor factory tests
# ---------------------------------------------------------------------------


class TestCreateExecutor:
    """Tests for the create_executor factory function."""

    def test_default_is_sequential(self):
        executor = create_executor()
        assert isinstance(executor, SequentialExecutor)

    def test_concurrency_1_is_sequential(self):
        executor = create_executor(concurrency=1)
        assert isinstance(executor, SequentialExecutor)

    def test_concurrency_0_is_sequential(self):
        executor = create_executor(concurrency=0)
        assert isinstance(executor, SequentialExecutor)

    def test_concurrency_gt_1_is_concurrent(self):
        executor = create_executor(concurrency=4)
        assert isinstance(executor, ConcurrentExecutor)
        assert executor.max_concurrency == 4

    def test_kwargs_passed_to_concurrent(self):
        executor = create_executor(concurrency=8, timeout=60.0, max_retries=5)
        assert isinstance(executor, ConcurrentExecutor)
        assert executor.max_concurrency == 8
        assert executor.timeout == 60.0
        assert executor.max_retries == 5


# ---------------------------------------------------------------------------
# CLI flag tests
# ---------------------------------------------------------------------------


class TestCLIConcurrencyFlag:
    """Tests for --concurrency / -j CLI flag on build and run commands."""

    def test_build_has_concurrency_option(self):
        """synix build --help shows --concurrency option."""
        runner = CliRunner()
        from synix.cli import main
        result = runner.invoke(main, ["build", "--help"])
        assert result.exit_code == 0
        assert "--concurrency" in result.output
        assert "-j" in result.output

    def test_run_has_concurrency_option(self):
        """synix run --help shows --concurrency option."""
        runner = CliRunner()
        from synix.cli import main
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--concurrency" in result.output
        assert "-j" in result.output
