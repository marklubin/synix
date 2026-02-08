"""Standalone OpenAI-compatible HTTP mock server for testing.

Uses stdlib http.server — no Flask/FastAPI dependency.
Supports chat completions, embeddings, and batch API endpoints.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "mock_responses"

# ── Deterministic response generation ──────────────────────────────────────────

# Keyword-based response templates (matches existing mock_llm patterns)
_KEYWORD_RESPONSES: dict[str, dict[str, str]] = {
    "default": {
        "episode_summary": (
            "This is a summary of the conversation. The user discussed technical topics "
            "including programming concepts and best practices. Key decisions were made "
            "regarding implementation approaches."
        ),
        "monthly_rollup": (
            "In this month, the main themes were technical learning and software development. "
            "The user explored machine learning, Docker, and programming languages. "
            "Their understanding deepened across multiple domains."
        ),
        "topical_rollup": (
            "Regarding this topic, the user has shown consistent interest in technical subjects. "
            "Their understanding has evolved from basic concepts to more advanced applications. "
            "Key decisions were made about implementation approaches."
        ),
        "core_memory": (
            "## Identity\nMark is a software engineer interested in AI and systems.\n\n"
            "## Current Focus\nBuilding agent memory systems.\n\n"
            "## Preferences\nPrefers clean code and well-tested systems."
        ),
        "fallback": "Mock response for the given prompt.",
    },
    # Model-switching: GPT-4o responses are more structured/factual
    "gpt-4o": {
        "episode_summary": (
            "Summary: The conversation covered technical topics. "
            "Topics: programming, software architecture. "
            "Decisions: adopted new testing framework. "
            "Action items: implement CI pipeline."
        ),
        "monthly_rollup": (
            "Monthly Report: Primary focus areas were software engineering and AI research. "
            "Key metrics: 12 conversations, 3 major decisions. "
            "Technical stack changes: migrated to new build system."
        ),
        "topical_rollup": (
            "Topic Analysis: Structured overview of the subject matter. "
            "Frequency: discussed in 8 conversations. "
            "Evolution: moved from exploration to implementation phase. "
            "Related topics: architecture, testing, deployment."
        ),
        "core_memory": (
            "## Profile\n- Role: Software Engineer\n- Focus: AI Systems\n\n"
            "## Key Facts\n- Building agent memory systems\n- Prefers tested code\n\n"
            "## Active Projects\n1. Synix build system\n2. Memory pipeline research"
        ),
        "fallback": "Structured mock response from GPT-4o model.",
    },
}

# Models that map to the gpt-4o fixture set
_GPT4O_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4o-2024-05-13"}


def _get_fixture_set(model: str) -> str:
    """Return the fixture set name based on model."""
    if model in _GPT4O_MODELS:
        return "gpt-4o"
    return "default"


def _detect_prompt_type(content: str) -> str:
    """Detect the type of prompt from its content."""
    lower = content.lower()
    if "episode summary" in lower or "summarizing a conversation" in lower:
        return "episode_summary"
    if "monthly" in lower and ("rollup" in lower or "overview" in lower or "synthesiz" in lower):
        return "monthly_rollup"
    if "topical" in lower or "topic:" in lower or ("topic" in lower and "synthe" in lower):
        return "topical_rollup"
    if "core memory" in lower or "core memory document" in lower:
        return "core_memory"
    return "fallback"


def _content_hash(text: str) -> str:
    """SHA-256 hash of text content."""
    return hashlib.sha256(text.encode()).hexdigest()


def _try_fixture_file(content_hash: str) -> str | None:
    """Try to load a fixture response from disk by content hash."""
    fixture_path = FIXTURES_DIR / f"{content_hash}.json"
    if fixture_path.exists():
        data = json.loads(fixture_path.read_text())
        return data.get("content", data.get("response", ""))
    # Try pattern-match files
    pattern_file = FIXTURES_DIR / "patterns.json"
    if pattern_file.exists():
        patterns = json.loads(pattern_file.read_text())
        for pattern_entry in patterns:
            if re.search(pattern_entry["pattern"], content_hash):
                return pattern_entry["content"]
    return None


def _try_structured_fixtures(content: str, model: str) -> str | None:
    """Try to look up a response from structured fixture JSON files.

    Extracts conversation IDs, month identifiers, or topic names from the
    prompt content and looks them up in the corresponding fixture file.
    """
    fixture_set = _get_fixture_set(model)

    # Episode: look for conversation ID patterns like conv_d01, conv_d02, etc.
    conv_match = re.search(r"conv_d\d+", content)
    if conv_match:
        episodes_file = FIXTURES_DIR / "episodes.json"
        if episodes_file.exists():
            data = json.loads(episodes_file.read_text())
            conv_id = conv_match.group(0)
            if conv_id in data:
                entry = data[conv_id]
                # Check model match
                if entry.get("model", "default") in ("default", fixture_set):
                    return entry["content"]

    # Monthly rollup: look for YYYY-MM pattern
    month_match = re.search(r"(20\d{2}-(?:0[1-9]|1[0-2]))", content)
    if month_match:
        monthly_file = FIXTURES_DIR / "monthly_rollups.json"
        if monthly_file.exists():
            data = json.loads(monthly_file.read_text())
            month_key = month_match.group(1)
            if month_key in data:
                entry = data[month_key]
                if entry.get("model", "default") in ("default", fixture_set):
                    return entry["content"]

    # Topical rollup: look for topic names in the prompt
    topic_keywords = {
        "database-migration": ["database migration", "cockroachdb", "postgresql migration"],
        "career": ["career", "staff engineer", "engineering manager"],
        "rust-learning": ["rust learning", "rust programming", "learning rust"],
        "side-projects": ["side project", "expense tracker", "side-project"],
        "ai-and-agents": ["ai and agents", "artificial intelligence", "agent system"],
    }
    lower = content.lower()
    for topic_key, keywords in topic_keywords.items():
        if any(kw in lower for kw in keywords):
            topical_file = FIXTURES_DIR / "topical_rollups.json"
            if topical_file.exists():
                data = json.loads(topical_file.read_text())
                if topic_key in data:
                    entry = data[topic_key]
                    if entry.get("model", "default") in ("default", fixture_set):
                        return entry["content"]
            break  # Only match first topic

    # Core synthesis
    prompt_type = _detect_prompt_type(content)
    if prompt_type == "core_memory":
        core_file = FIXTURES_DIR / "core_synthesis.json"
        if core_file.exists():
            data = json.loads(core_file.read_text())
            if fixture_set in data:
                return data[fixture_set]["content"]
            if "default" in data:
                return data["default"]["content"]

    return None


def _generate_chat_response(model: str, messages: list[dict]) -> str:
    """Generate a deterministic chat response."""
    # Extract user message content
    user_content = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
            break

    # Try content-hash fixture file first
    ch = _content_hash(user_content)
    fixture_resp = _try_fixture_file(ch)
    if fixture_resp is not None:
        return fixture_resp

    # Try structured fixture JSON files (episodes, monthly, topical, core)
    structured_resp = _try_structured_fixtures(user_content, model)
    if structured_resp is not None:
        return structured_resp

    # Fall back to keyword-based response
    fixture_set = _get_fixture_set(model)
    prompt_type = _detect_prompt_type(user_content)
    responses = _KEYWORD_RESPONSES.get(fixture_set, _KEYWORD_RESPONSES["default"])
    return responses.get(prompt_type, responses["fallback"])


def _generate_embedding(text: str, dimensions: int = 256) -> list[float]:
    """Generate a deterministic embedding vector from text content hash."""
    h = hashlib.sha256(text.encode()).digest()
    # Use hash bytes to seed a deterministic sequence
    # Expand to enough bytes for `dimensions` floats
    expanded = b""
    i = 0
    while len(expanded) < dimensions * 4:
        expanded += hashlib.sha256(h + i.to_bytes(4, "big")).digest()
        i += 1
    # Convert to floats in [-1, 1], then L2-normalize
    raw = []
    for j in range(dimensions):
        # Unpack 4 bytes as unsigned int, map to [-1, 1]
        val = struct.unpack(">I", expanded[j * 4 : j * 4 + 4])[0]
        raw.append((val / 2147483648.0) - 1.0)  # [0, 2) - 1 = [-1, 1)
    # L2-normalize
    norm = math.sqrt(sum(x * x for x in raw))
    if norm > 0:
        raw = [x / norm for x in raw]
    return raw


def _estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token."""
    return max(1, len(text) // 4)


# ── HTTP Handler ───────────────────────────────────────────────────────────────


class MockLLMHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OpenAI-compatible mock endpoints."""

    # Suppress default logging
    def log_message(self, format, *args):
        pass

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length > 0 else b""

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: int, message: str, error_type: str = "server_error"):
        self._send_json(
            {"error": {"message": message, "type": error_type, "code": status}},
            status=status,
        )

    def _handle_mock_headers(self) -> bool:
        """Check for error/latency injection headers. Returns True if request was handled."""
        # Latency injection
        latency_ms = self.headers.get("X-Mock-Latency-Ms")
        if latency_ms:
            try:
                time.sleep(int(latency_ms) / 1000.0)
            except ValueError:
                pass

        # Error injection
        error = self.headers.get("X-Mock-Error")
        if error:
            if error == "429":
                self._send_json(
                    {
                        "error": {
                            "message": "Rate limit exceeded",
                            "type": "rate_limit_error",
                            "code": 429,
                        }
                    },
                    status=429,
                )
                return True
            elif error == "500":
                self._send_json(
                    {
                        "error": {
                            "message": "Internal server error",
                            "type": "server_error",
                            "code": 500,
                        }
                    },
                    status=500,
                )
                return True
            elif error == "timeout":
                # Hang for 30 seconds (tests should use a shorter timeout)
                time.sleep(30)
                return True
        return False

    def do_POST(self):
        if self._handle_mock_headers():
            return

        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
        elif self.path == "/v1/embeddings":
            self._handle_embeddings()
        elif self.path == "/v1/batches":
            self._handle_create_batch()
        else:
            self._send_error_json(404, f"Unknown endpoint: {self.path}", "not_found")

    def do_GET(self):
        if self._handle_mock_headers():
            return

        if self.path.startswith("/v1/batches/"):
            self._handle_get_batch()
        elif self.path.startswith("/v1/files/") and self.path.endswith("/content"):
            self._handle_get_file_content()
        elif self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self._send_error_json(404, f"Unknown endpoint: {self.path}", "not_found")

    # ── Chat Completions ──

    def _handle_chat_completions(self):
        try:
            body = json.loads(self._read_body())
        except (json.JSONDecodeError, ValueError):
            self._send_error_json(400, "Invalid JSON body")
            return

        model = body.get("model", "unknown")
        messages = body.get("messages", [])

        content = _generate_chat_response(model, messages)

        # Estimate tokens
        prompt_text = " ".join(m.get("content", "") for m in messages)
        prompt_tokens = _estimate_tokens(prompt_text)
        completion_tokens = _estimate_tokens(content)

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        self._send_json(response)

    # ── Embeddings ──

    def _handle_embeddings(self):
        try:
            body = json.loads(self._read_body())
        except (json.JSONDecodeError, ValueError):
            self._send_error_json(400, "Invalid JSON body")
            return

        model = body.get("model", "text-embedding-3-small")
        input_data = body.get("input", "")
        dimensions = body.get("dimensions", 256)

        # Normalize to list
        if isinstance(input_data, str):
            texts = [input_data]
        else:
            texts = list(input_data)

        data = []
        total_tokens = 0
        for i, text in enumerate(texts):
            embedding = _generate_embedding(text, dimensions)
            data.append({"object": "embedding", "embedding": embedding, "index": i})
            total_tokens += _estimate_tokens(text)

        response = {
            "object": "list",
            "data": data,
            "model": model,
            "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
        }
        self._send_json(response)

    # ── Batch API ──

    def _handle_create_batch(self):
        try:
            body = json.loads(self._read_body())
        except (json.JSONDecodeError, ValueError):
            self._send_error_json(400, "Invalid JSON body")
            return

        batch_id = f"batch_{uuid.uuid4().hex[:24]}"
        output_file_id = f"file-{uuid.uuid4().hex[:24]}"

        # Store batch data on server instance for later retrieval
        batch_store = self.server.batch_store  # type: ignore[attr-defined]
        batch_store[batch_id] = {
            "id": batch_id,
            "object": "batch",
            "endpoint": body.get("endpoint", "/v1/chat/completions"),
            "input_file_id": body.get("input_file_id", ""),
            "status": "completed",
            "output_file_id": output_file_id,
            "created_at": int(time.time()),
            "completed_at": int(time.time()),
            "request_counts": {"total": 1, "completed": 1, "failed": 0},
        }

        # Generate output for the batch
        # For simplicity, produce one result per batch
        result_line = json.dumps({
            "id": f"req-{uuid.uuid4().hex[:12]}",
            "custom_id": "request-1",
            "response": {
                "status_code": 200,
                "body": {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Batch response content."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            },
        })
        batch_store[output_file_id] = result_line + "\n"

        self._send_json(batch_store[batch_id])

    def _handle_get_batch(self):
        batch_id = self.path.split("/v1/batches/")[1].rstrip("/")
        batch_store = self.server.batch_store  # type: ignore[attr-defined]
        if batch_id in batch_store:
            self._send_json(batch_store[batch_id])
        else:
            self._send_error_json(404, f"Batch not found: {batch_id}", "not_found")

    def _handle_get_file_content(self):
        # Extract file ID: /v1/files/{id}/content
        parts = self.path.split("/")
        # parts: ['', 'v1', 'files', '{id}', 'content']
        file_id = parts[3] if len(parts) >= 5 else ""
        batch_store = self.server.batch_store  # type: ignore[attr-defined]
        if file_id in batch_store:
            content = batch_store[file_id]
            body = content.encode() if isinstance(content, str) else content
            self.send_response(200)
            self.send_header("Content-Type", "application/jsonl")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send_error_json(404, f"File not found: {file_id}", "not_found")


# ── Server ─────────────────────────────────────────────────────────────────────


class MockLLMServer(ThreadingMixIn, HTTPServer):
    """Thread-safe OpenAI-compatible mock LLM server."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        super().__init__((host, port), MockLLMHandler)
        self.batch_store: dict[str, object] = {}

    @property
    def base_url(self) -> str:
        """Return the base URL for OpenAI client configuration."""
        host, port = self.server_address
        return f"http://{host}:{port}/v1"

    def start_background(self) -> threading.Thread:
        """Start the server in a daemon thread. Returns the thread."""
        thread = threading.Thread(target=self.serve_forever, daemon=True)
        thread.start()
        return thread
