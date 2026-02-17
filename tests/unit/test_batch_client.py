"""Unit tests for BatchLLMClient."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from synix.build.batch_client import (
    BatchCollecting,
    BatchInProgress,
    BatchLLMClient,
    BatchRequestFailed,
)
from synix.build.batch_state import BatchState
from synix.build.cassette import compute_cassette_key
from synix.core.config import LLMConfig


@pytest.fixture
def llm_config():
    return LLMConfig(provider="openai", model="gpt-4o-mini", max_tokens=512, temperature=0.3)


@pytest.fixture
def batch_state(tmp_path):
    return BatchState(tmp_path, "test-build")


@pytest.fixture
def client(llm_config, batch_state):
    return BatchLLMClient(llm_config, batch_state, "episodes")


@pytest.fixture
def messages():
    return [{"role": "user", "content": "Summarize this text"}]


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client with all needed methods."""
    mock_client = MagicMock()
    mock_file = MagicMock()
    mock_file.id = "file-abc123"
    mock_client.files.create.return_value = mock_file

    mock_batch = MagicMock()
    mock_batch.id = "batch-xyz789"
    mock_client.batches.create.return_value = mock_batch

    return mock_client


class TestCompleteReturnsCached:
    def test_returns_llm_response_when_result_cached(self, client, batch_state, messages, llm_config):
        key = compute_cassette_key(
            llm_config.provider,
            llm_config.model,
            messages,
            llm_config.max_tokens,
            llm_config.temperature,
        )
        batch_state.store_result(key, "cached output", "gpt-4o-mini", {"input": 100, "output": 50})

        resp = client.complete(messages)
        assert resp.content == "cached output"
        assert resp.model == "gpt-4o-mini"
        assert resp.input_tokens == 100
        assert resp.output_tokens == 50

    def test_returns_from_cassette_responses(self, llm_config, batch_state, messages):
        key = compute_cassette_key(
            llm_config.provider,
            llm_config.model,
            messages,
            llm_config.max_tokens,
            llm_config.temperature,
        )
        cassette = {
            key: {
                "content": "cassette output",
                "model": "gpt-4o-mini",
                "tokens": {"input": 80, "output": 40},
            }
        }
        client = BatchLLMClient(llm_config, batch_state, "episodes", cassette_responses=cassette)
        resp = client.complete(messages)
        assert resp.content == "cassette output"
        # Also stored in batch_state
        assert batch_state.get_result(key) is not None


class TestCompleteRaisesExceptions:
    def test_raises_batch_collecting_for_new_request(self, client, messages):
        with pytest.raises(BatchCollecting):
            client.complete(messages)

    def test_raises_batch_in_progress_when_batch_pending(self, client, batch_state, messages, llm_config):
        key = compute_cassette_key(
            llm_config.provider,
            llm_config.model,
            messages,
            llm_config.max_tokens,
            llm_config.temperature,
        )
        batch_state.record_batch("batch-001", "episodes", [key], status="in_progress")

        with pytest.raises(BatchInProgress) as exc_info:
            client.complete(messages)
        assert exc_info.value.batch_id == "batch-001"
        assert exc_info.value.status == "in_progress"

    def test_raises_batch_request_failed_when_batch_done_but_no_result(self, client, batch_state, messages, llm_config):
        key = compute_cassette_key(
            llm_config.provider,
            llm_config.model,
            messages,
            llm_config.max_tokens,
            llm_config.temperature,
        )
        batch_state.record_batch("batch-001", "episodes", [key], status="completed")
        batch_state.update_batch_status("batch-001", "completed")
        batch_state.store_error(key, "content_filter", "Filtered")

        with pytest.raises(BatchRequestFailed) as exc_info:
            client.complete(messages)
        assert exc_info.value.error["code"] == "content_filter"


class TestRequestKeying:
    def test_same_inputs_produce_same_key(self, client, batch_state, messages):
        with pytest.raises(BatchCollecting):
            client.complete(messages)

        pending = batch_state.get_pending("episodes")
        assert len(pending) == 1

        # Queue again with same inputs — same key overwrites
        with pytest.raises(BatchCollecting):
            client.complete(messages)

        pending = batch_state.get_pending("episodes")
        assert len(pending) == 1

    def test_different_messages_produce_different_keys(self, client, batch_state):
        with pytest.raises(BatchCollecting):
            client.complete([{"role": "user", "content": "message A"}])
        with pytest.raises(BatchCollecting):
            client.complete([{"role": "user", "content": "message B"}])

        pending = batch_state.get_pending("episodes")
        assert len(pending) == 2

    def test_different_temperature_produces_different_key(self, llm_config, batch_state, messages):
        client1 = BatchLLMClient(llm_config, batch_state, "episodes")
        with pytest.raises(BatchCollecting):
            client1.complete(messages, temperature=0.3)

        config2 = LLMConfig(provider="openai", model="gpt-4o-mini", max_tokens=512, temperature=0.9)
        client2 = BatchLLMClient(config2, batch_state, "episodes")
        with pytest.raises(BatchCollecting):
            client2.complete(messages)

        pending = batch_state.get_pending("episodes")
        assert len(pending) == 2

    def test_different_model_produces_different_key(self, batch_state, messages):
        config1 = LLMConfig(provider="openai", model="gpt-4o-mini")
        client1 = BatchLLMClient(config1, batch_state, "episodes")
        with pytest.raises(BatchCollecting):
            client1.complete(messages)

        config2 = LLMConfig(provider="openai", model="gpt-4o")
        client2 = BatchLLMClient(config2, batch_state, "episodes")
        with pytest.raises(BatchCollecting):
            client2.complete(messages)

        pending = batch_state.get_pending("episodes")
        assert len(pending) == 2


class TestSubmitBatch:
    def test_creates_jsonl_and_calls_openai(
        self, client, batch_state, messages, llm_config, mock_openai_client, monkeypatch
    ):
        # Queue a request
        with pytest.raises(BatchCollecting):
            client.complete(messages)

        import openai as openai_mod

        monkeypatch.setattr(openai_mod, "OpenAI", lambda **kw: mock_openai_client)

        batch_id = client.submit_batch("episodes")

        assert batch_id == "batch-xyz789"
        mock_openai_client.files.create.assert_called_once()
        mock_openai_client.batches.create.assert_called_once()

        # Verify JSONL content
        file_call_kwargs = mock_openai_client.files.create.call_args
        file_bytes = file_call_kwargs.kwargs.get("file") or file_call_kwargs[1].get("file")
        jsonl_text = file_bytes.read().decode()
        parsed = json.loads(jsonl_text)
        assert parsed["method"] == "POST"
        assert parsed["url"] == "/v1/chat/completions"
        assert parsed["body"]["model"] == "gpt-4o-mini"

        # Pending should be cleared
        assert batch_state.get_pending("episodes") == {}

    def test_raises_on_no_pending(self, client):
        with pytest.raises(RuntimeError, match="No pending requests"):
            client.submit_batch("episodes")


def _make_jsonl_response(custom_ids: list[str], model: str = "gpt-4o-mini") -> str:
    lines = []
    for cid in custom_ids:
        entry = {
            "custom_id": cid,
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": f"Response for {cid}"}}],
                    "model": model,
                    "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
                },
            },
        }
        lines.append(json.dumps(entry))
    return "\n".join(lines)


class TestCheckAndDownload:
    def test_downloads_completed_batch(self, client, batch_state, monkeypatch):
        batch_state.record_batch("batch-001", "episodes", ["k1", "k2"])

        mock_batch = MagicMock()
        mock_batch.status = "completed"
        mock_batch.output_file_id = "file-out"
        mock_batch.error_file_id = None

        mock_content = MagicMock()
        mock_content.text = _make_jsonl_response(["k1", "k2"])

        mock_client = MagicMock()
        mock_client.batches.retrieve.return_value = mock_batch
        mock_client.files.content.return_value = mock_content

        import openai as openai_mod

        monkeypatch.setattr(openai_mod, "OpenAI", lambda **kw: mock_client)

        done = client.check_and_download("batch-001")

        assert done is True
        assert batch_state.get_result("k1")["content"] == "Response for k1"
        assert batch_state.get_result("k2")["content"] == "Response for k2"

    def test_returns_false_for_in_progress(self, client, batch_state, monkeypatch):
        batch_state.record_batch("batch-001", "episodes", ["k1"])

        mock_batch = MagicMock()
        mock_batch.status = "in_progress"

        mock_client = MagicMock()
        mock_client.batches.retrieve.return_value = mock_batch

        import openai as openai_mod

        monkeypatch.setattr(openai_mod, "OpenAI", lambda **kw: mock_client)

        done = client.check_and_download("batch-001")
        assert done is False

    def test_handles_failed_batch(self, client, batch_state, monkeypatch):
        batch_state.record_batch("batch-001", "episodes", ["k1", "k2"])

        mock_batch = MagicMock()
        mock_batch.status = "failed"

        mock_client = MagicMock()
        mock_client.batches.retrieve.return_value = mock_batch

        import openai as openai_mod

        monkeypatch.setattr(openai_mod, "OpenAI", lambda **kw: mock_client)

        done = client.check_and_download("batch-001")

        assert done is True
        assert batch_state.get_error("k1") is not None
        assert batch_state.get_error("k2") is not None

    def test_handles_expired_batch(self, client, batch_state, monkeypatch):
        batch_state.record_batch("batch-001", "episodes", ["k1"])

        mock_batch = MagicMock()
        mock_batch.status = "expired"

        mock_client = MagicMock()
        mock_client.batches.retrieve.return_value = mock_batch

        import openai as openai_mod

        monkeypatch.setattr(openai_mod, "OpenAI", lambda **kw: mock_client)

        done = client.check_and_download("batch-001")

        assert done is True
        assert batch_state.get_error("k1")["code"] == "expired"

    def test_parses_partial_failures(self, client, batch_state, monkeypatch):
        batch_state.record_batch("batch-001", "episodes", ["k1", "k2"])

        success_line = json.dumps(
            {
                "custom_id": "k1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{"message": {"content": "OK"}}],
                        "model": "gpt-4o-mini",
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    },
                },
            }
        )
        failure_line = json.dumps(
            {
                "custom_id": "k2",
                "response": {"status_code": 400},
                "error": {"code": "content_filter", "message": "Filtered"},
            }
        )

        mock_batch = MagicMock()
        mock_batch.status = "completed"
        mock_batch.output_file_id = "file-out"
        mock_batch.error_file_id = None

        mock_content = MagicMock()
        mock_content.text = f"{success_line}\n{failure_line}"

        mock_client = MagicMock()
        mock_client.batches.retrieve.return_value = mock_batch
        mock_client.files.content.return_value = mock_content

        import openai as openai_mod

        monkeypatch.setattr(openai_mod, "OpenAI", lambda **kw: mock_client)

        done = client.check_and_download("batch-001")

        assert done is True
        assert batch_state.get_result("k1") is not None
        assert batch_state.get_error("k2") is not None


class TestClientConfig:
    def test_exposes_config_attribute(self, client, llm_config):
        """BatchLLMClient.config is required by _logged_complete() at llm_transforms.py:45."""
        assert client.config is llm_config
        assert client.config.provider == "openai"
        assert client.config.model == "gpt-4o-mini"
