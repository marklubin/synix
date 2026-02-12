"""Tests for the OpenAI-compatible mock LLM server."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pytest

from tests.mock_server.server import MockLLMServer, _generate_embedding


@pytest.fixture()
def server():
    """Start mock LLM server, yield it, teardown."""
    srv = MockLLMServer(host="127.0.0.1", port=0)
    srv.start_background()
    yield srv
    srv.shutdown()


def _post(base_url: str, path: str, body: dict, headers: dict | None = None, timeout: float = 5) -> tuple[int, dict]:
    """POST JSON to the server, return (status_code, response_body)."""
    url = base_url.rstrip("/") + path if not path.startswith("http") else path
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(base_url: str, path: str, headers: dict | None = None, timeout: float = 5) -> tuple[int, bytes]:
    """GET from the server, return (status_code, raw_body)."""
    url = base_url.rstrip("/") + path
    req = urllib.request.Request(url, method="GET")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


# ── Chat completions ──────────────────────────────────────────────────────────


class TestChatCompletions:
    def test_basic_response(self, server: MockLLMServer):
        status, resp = _post(
            server.base_url,
            "/chat/completions",
            {
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
            },
        )
        assert status == 200
        assert resp["object"] == "chat.completion"
        assert resp["id"].startswith("chatcmpl-")
        assert len(resp["choices"]) == 1
        assert resp["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(resp["choices"][0]["message"]["content"], str)
        assert resp["choices"][0]["finish_reason"] == "stop"
        assert "usage" in resp
        assert resp["usage"]["total_tokens"] == resp["usage"]["prompt_tokens"] + resp["usage"]["completion_tokens"]

    def test_episode_summary_detection(self, server: MockLLMServer):
        status, resp = _post(
            server.base_url,
            "/chat/completions",
            {
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Write an episode summary of the conversation."}],
            },
        )
        assert status == 200
        content = resp["choices"][0]["message"]["content"]
        assert "summary" in content.lower() or "technical" in content.lower()

    def test_monthly_rollup_detection(self, server: MockLLMServer):
        status, resp = _post(
            server.base_url,
            "/chat/completions",
            {
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Create a monthly rollup overview."}],
            },
        )
        assert status == 200
        content = resp["choices"][0]["message"]["content"]
        assert "month" in content.lower() or "themes" in content.lower()

    def test_core_memory_detection(self, server: MockLLMServer):
        status, resp = _post(
            server.base_url,
            "/chat/completions",
            {
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Create a core memory document."}],
            },
        )
        assert status == 200
        content = resp["choices"][0]["message"]["content"]
        assert "Identity" in content or "Profile" in content

    def test_model_switching_default_vs_gpt4o(self, server: MockLLMServer):
        """Different models should return qualitatively different responses."""
        _, resp_default = _post(
            server.base_url,
            "/chat/completions",
            {
                "model": "claude-sonnet-4-5",
                "messages": [{"role": "user", "content": "Write an episode summary of the conversation."}],
            },
        )
        _, resp_gpt4o = _post(
            server.base_url,
            "/chat/completions",
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Write an episode summary of the conversation."}],
            },
        )
        content_default = resp_default["choices"][0]["message"]["content"]
        content_gpt4o = resp_gpt4o["choices"][0]["message"]["content"]
        # They should be different (gpt-4o uses the structured fixture set)
        assert content_default != content_gpt4o

    def test_deterministic_responses(self, server: MockLLMServer):
        """Same input should produce the same output."""
        body = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Tell me about machine learning."}],
        }
        _, resp1 = _post(server.base_url, "/chat/completions", body)
        _, resp2 = _post(server.base_url, "/chat/completions", body)
        assert resp1["choices"][0]["message"]["content"] == resp2["choices"][0]["message"]["content"]


# ── Embeddings ────────────────────────────────────────────────────────────────


class TestEmbeddings:
    def test_single_text(self, server: MockLLMServer):
        status, resp = _post(
            server.base_url,
            "/embeddings",
            {
                "model": "text-embedding-3-small",
                "input": "Hello world",
            },
        )
        assert status == 200
        assert resp["object"] == "list"
        assert len(resp["data"]) == 1
        assert resp["data"][0]["index"] == 0
        embedding = resp["data"][0]["embedding"]
        assert len(embedding) == 256
        # Check it's normalized (L2 norm ≈ 1.0)
        import math

        norm = math.sqrt(sum(x * x for x in embedding))
        assert abs(norm - 1.0) < 1e-6

    def test_batch_texts(self, server: MockLLMServer):
        status, resp = _post(
            server.base_url,
            "/embeddings",
            {
                "model": "text-embedding-3-small",
                "input": ["Hello world", "Goodbye world"],
            },
        )
        assert status == 200
        assert len(resp["data"]) == 2
        assert resp["data"][0]["index"] == 0
        assert resp["data"][1]["index"] == 1
        # Different texts should produce different embeddings
        assert resp["data"][0]["embedding"] != resp["data"][1]["embedding"]

    def test_deterministic_embeddings(self, server: MockLLMServer):
        """Same input should always produce the same embedding."""
        body = {"model": "text-embedding-3-small", "input": "test input"}
        _, resp1 = _post(server.base_url, "/embeddings", body)
        _, resp2 = _post(server.base_url, "/embeddings", body)
        assert resp1["data"][0]["embedding"] == resp2["data"][0]["embedding"]

    def test_custom_dimensions(self, server: MockLLMServer):
        status, resp = _post(
            server.base_url,
            "/embeddings",
            {
                "model": "text-embedding-3-small",
                "input": "test",
                "dimensions": 128,
            },
        )
        assert status == 200
        assert len(resp["data"][0]["embedding"]) == 128

    def test_unit_function_determinism(self):
        """Test _generate_embedding directly for determinism."""
        emb1 = _generate_embedding("hello", 256)
        emb2 = _generate_embedding("hello", 256)
        assert emb1 == emb2
        # Different input → different output
        emb3 = _generate_embedding("world", 256)
        assert emb1 != emb3


# ── Batch API ─────────────────────────────────────────────────────────────────


class TestBatchAPI:
    def test_create_batch(self, server: MockLLMServer):
        status, resp = _post(
            server.base_url,
            "/batches",
            {
                "input_file_id": "file-abc123",
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
        )
        assert status == 200
        assert resp["id"].startswith("batch_")
        assert resp["status"] == "completed"
        assert resp["output_file_id"].startswith("file-")

    def test_get_batch_status(self, server: MockLLMServer):
        # Create a batch first
        _, create_resp = _post(
            server.base_url,
            "/batches",
            {
                "input_file_id": "file-abc123",
                "endpoint": "/v1/chat/completions",
            },
        )
        batch_id = create_resp["id"]

        # Poll status
        status, get_resp = _get(server.base_url, f"/batches/{batch_id}")
        resp = json.loads(get_resp)
        assert status == 200
        assert resp["id"] == batch_id
        assert resp["status"] == "completed"

    def test_get_batch_results(self, server: MockLLMServer):
        # Create batch
        _, create_resp = _post(
            server.base_url,
            "/batches",
            {
                "input_file_id": "file-abc123",
                "endpoint": "/v1/chat/completions",
            },
        )
        output_file_id = create_resp["output_file_id"]

        # Get results file
        status, raw_body = _get(server.base_url, f"/files/{output_file_id}/content")
        assert status == 200
        # Parse JSONL
        lines = raw_body.decode().strip().split("\n")
        assert len(lines) >= 1
        result = json.loads(lines[0])
        assert "response" in result
        assert result["response"]["status_code"] == 200

    def test_batch_not_found(self, server: MockLLMServer):
        status, _ = _get(server.base_url, "/batches/batch_nonexistent")
        resp = json.loads(_)
        assert status == 404

    def test_full_batch_cycle(self, server: MockLLMServer):
        """Full create → poll → retrieve results cycle."""
        # Create
        _, batch = _post(
            server.base_url,
            "/batches",
            {
                "input_file_id": "file-test",
                "endpoint": "/v1/chat/completions",
            },
        )
        assert batch["status"] == "completed"

        # Poll
        status, raw = _get(server.base_url, f"/batches/{batch['id']}")
        poll_resp = json.loads(raw)
        assert poll_resp["status"] == "completed"
        output_file_id = poll_resp["output_file_id"]

        # Get results
        status, content = _get(server.base_url, f"/files/{output_file_id}/content")
        assert status == 200
        result = json.loads(content.decode().strip().split("\n")[0])
        assert result["response"]["body"]["choices"][0]["message"]["content"]


# ── Error Injection ───────────────────────────────────────────────────────────


class TestErrorInjection:
    def test_rate_limit_429(self, server: MockLLMServer):
        status, resp = _post(
            server.base_url,
            "/chat/completions",
            {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Mock-Error": "429"},
        )
        assert status == 429
        assert resp["error"]["type"] == "rate_limit_error"

    def test_server_error_500(self, server: MockLLMServer):
        status, resp = _post(
            server.base_url,
            "/chat/completions",
            {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Mock-Error": "500"},
        )
        assert status == 500
        assert resp["error"]["type"] == "server_error"

    def test_latency_injection(self, server: MockLLMServer):
        start = time.monotonic()
        status, _ = _post(
            server.base_url,
            "/chat/completions",
            {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Mock-Latency-Ms": "200"},
        )
        elapsed = time.monotonic() - start
        assert status == 200
        assert elapsed >= 0.18  # Allow small timing margin

    def test_timeout_injection(self, server: MockLLMServer):
        """Timeout injection should cause the request to hang (client should timeout)."""
        with pytest.raises((urllib.error.URLError, TimeoutError, OSError)):
            _post(
                server.base_url,
                "/chat/completions",
                {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
                headers={"X-Mock-Error": "timeout"},
                timeout=0.5,  # Client timeout shorter than server's 30s hang
            )

    def test_error_injection_on_get(self, server: MockLLMServer):
        """Error injection should work on GET endpoints too."""
        status, raw = _get(
            server.base_url,
            "/batches/batch_test",
            headers={"X-Mock-Error": "500"},
        )
        resp = json.loads(raw)
        assert status == 500


# ── Health & Edge Cases ───────────────────────────────────────────────────────


class TestHealthAndEdgeCases:
    def test_health_endpoint(self, server: MockLLMServer):
        status, body = _get(server.base_url.replace("/v1", ""), "/health")
        resp = json.loads(body)
        assert status == 200
        assert resp["status"] == "ok"

    def test_unknown_endpoint_post(self, server: MockLLMServer):
        status, resp = _post(server.base_url, "/unknown", {"foo": "bar"})
        assert status == 404

    def test_unknown_endpoint_get(self, server: MockLLMServer):
        status, _ = _get(server.base_url, "/unknown")
        assert status == 404

    def test_invalid_json_body(self, server: MockLLMServer):
        """Server should handle malformed JSON gracefully."""
        url = server.base_url + "/chat/completions"
        req = urllib.request.Request(url, data=b"not json", method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Content-Length", "8")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                assert False, "Should have returned error"
        except urllib.error.HTTPError as e:
            assert e.code == 400

    def test_concurrent_requests(self, server: MockLLMServer):
        """Server should handle concurrent requests (ThreadingMixIn)."""
        import concurrent.futures

        def make_request(i):
            return _post(
                server.base_url,
                "/chat/completions",
                {
                    "model": "claude-sonnet-4-5",
                    "messages": [{"role": "user", "content": f"Request {i}"}],
                },
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(status == 200 for status, _ in results)

    def test_base_url_property(self):
        """Server base_url property should return correct URL."""
        srv = MockLLMServer(host="127.0.0.1", port=0)
        srv.start_background()
        try:
            assert srv.base_url.startswith("http://127.0.0.1:")
            assert srv.base_url.endswith("/v1")
        finally:
            srv.shutdown()
