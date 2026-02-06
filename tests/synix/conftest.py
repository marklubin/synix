"""Pytest fixtures for Synix tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

if TYPE_CHECKING:
    from synix.config import Settings
    from synix.llm.client import LLMResponse


@dataclass
class MockLLMClient:
    """Deterministic LLM for testing — returns predictable responses."""

    responses: dict[str, str] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)
    default_response: str = "This is a mock summary of the input."

    class config:
        """Mock config."""

        model = "mock-model"

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float = 120.0,
    ) -> "LLMResponse":
        """Return a mock response."""
        from synix.llm.client import LLMResponse

        self.calls.append(prompt)

        # Check for matching response
        content = self.responses.get(prompt, self.default_response)

        # Also check for partial matches
        for key, value in self.responses.items():
            if key in prompt:
                content = value
                break

        return LLMResponse(
            content=content,
            model="mock",
            input_tokens=len(prompt.split()),
            output_tokens=len(content.split()),
        )


@pytest.fixture
def mock_llm() -> MockLLMClient:
    """Provide a mock LLM client."""
    return MockLLMClient()


@pytest.fixture
def test_storage_dir(tmp_path: Path) -> Path:
    """Create a temporary storage directory."""
    storage = tmp_path / ".synix"
    storage.mkdir(parents=True)
    return storage


@pytest.fixture
def test_settings(test_storage_dir: Path) -> "Settings":
    """Create test settings with temporary storage."""
    from synix.config import Settings, reset_settings

    reset_settings()
    settings = Settings(
        storage_dir=test_storage_dir,
        llm_api_key="test-key",
    )
    return settings


@pytest.fixture
def initialized_db(test_settings: "Settings") -> "Settings":
    """Initialize databases and return settings."""
    from synix.config import reset_settings
    from synix.db.engine import init_databases, reset_engines

    reset_settings()
    reset_engines()
    init_databases(test_settings)
    yield test_settings
    reset_engines()
    reset_settings()


@pytest.fixture
def claude_export_file(tmp_path: Path) -> Path:
    """Create a sample Claude export file."""
    data = {
        "conversations": [
            {
                "uuid": "conv-001",
                "title": "Rust ownership discussion",
                "created_at": "2024-03-15T10:00:00Z",
                "chat_messages": [
                    {"sender": "human", "text": "Explain Rust ownership"},
                    {
                        "sender": "assistant",
                        "text": "Ownership is Rust's way of managing memory without garbage collection.",
                    },
                ],
            },
            {
                "uuid": "conv-002",
                "title": "Python async patterns",
                "created_at": "2024-03-20T14:00:00Z",
                "chat_messages": [
                    {"sender": "human", "text": "How does async work in Python?"},
                    {
                        "sender": "assistant",
                        "text": "Python's asyncio uses an event loop to manage concurrent tasks.",
                    },
                ],
            },
            {
                "uuid": "conv-003",
                "title": "API design discussion",
                "created_at": "2024-04-05T09:00:00Z",
                "chat_messages": [
                    {"sender": "human", "text": "Best practices for REST API design?"},
                    {
                        "sender": "assistant",
                        "text": "Use proper HTTP methods, status codes, and versioning.",
                    },
                ],
            },
        ]
    }

    file_path = tmp_path / "claude_export.json"
    file_path.write_text(json.dumps(data))
    return file_path


@pytest.fixture
def chatgpt_export_file(tmp_path: Path) -> Path:
    """Create a sample ChatGPT export file."""
    data = [
        {
            "id": "conv-001",
            "title": "Machine learning basics",
            "create_time": 1710500000,
            "mapping": {
                "msg-root": {
                    "id": "msg-root",
                    "message": None,
                    "parent": None,
                    "children": ["msg-001"],
                },
                "msg-001": {
                    "id": "msg-001",
                    "message": {
                        "id": "msg-001",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["What is machine learning?"]},
                        "create_time": 1710500001,
                    },
                    "parent": "msg-root",
                    "children": ["msg-002"],
                },
                "msg-002": {
                    "id": "msg-002",
                    "message": {
                        "id": "msg-002",
                        "author": {"role": "assistant"},
                        "content": {
                            "content_type": "text",
                            "parts": ["Machine learning is a subset of AI that learns from data."],
                        },
                        "create_time": 1710500002,
                    },
                    "parent": "msg-001",
                    "children": [],
                },
            },
        },
        {
            "id": "conv-002",
            "title": "Database optimization",
            "create_time": 1710600000,
            "mapping": {
                "msg-root": {
                    "id": "msg-root",
                    "message": None,
                    "parent": None,
                    "children": ["msg-001"],
                },
                "msg-001": {
                    "id": "msg-001",
                    "message": {
                        "id": "msg-001",
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["How to optimize SQL queries?"]},
                        "create_time": 1710600001,
                    },
                    "parent": "msg-root",
                    "children": ["msg-002"],
                },
                "msg-002": {
                    "id": "msg-002",
                    "message": {
                        "id": "msg-002",
                        "author": {"role": "assistant"},
                        "content": {
                            "content_type": "text",
                            "parts": ["Use indexes, avoid SELECT *, and analyze query plans."],
                        },
                        "create_time": 1710600002,
                    },
                    "parent": "msg-001",
                    "children": [],
                },
            },
        },
    ]

    file_path = tmp_path / "chatgpt_export.json"
    file_path.write_text(json.dumps(data))
    return file_path


@pytest.fixture
def sample_prompt() -> callable:
    """Provide a simple summarize prompt function."""

    def summarize(record):
        return f"Summarize this conversation:\n\n{record.content}\n\nSummary:"

    return summarize


@pytest.fixture
def sample_aggregate_prompt() -> callable:
    """Provide a simple aggregate prompt function."""

    def monthly_reflect(records, period):
        summaries = "\n".join(f"- {r.content[:100]}..." for r in records)
        return f"Reflect on {period}:\n\n{summaries}\n\nReflection:"

    return monthly_reflect
