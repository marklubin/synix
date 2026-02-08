"""Shared test fixtures for Synix."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from synix import Artifact, Layer, Pipeline, Projection, ProvenanceRecord


FIXTURES_DIR = Path(__file__).parent / "synix" / "fixtures"


@pytest.fixture
def tmp_build_dir(tmp_path):
    """Clean build directory for each test."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    return build_dir


@pytest.fixture
def sample_artifacts():
    """Pre-built artifacts for testing downstream modules."""
    return [
        Artifact(
            artifact_id="t-chatgpt-conv001",
            artifact_type="transcript",
            content="user: What is ML?\n\nassistant: Machine learning is...\n\n",
            metadata={
                "source": "chatgpt",
                "source_conversation_id": "conv001",
                "title": "ML discussion",
                "date": "2024-03-15",
                "message_count": 4,
            },
        ),
        Artifact(
            artifact_id="t-chatgpt-conv002",
            artifact_type="transcript",
            content="user: Explain Docker\n\nassistant: Docker packages apps...\n\n",
            metadata={
                "source": "chatgpt",
                "source_conversation_id": "conv002",
                "title": "Docker discussion",
                "date": "2024-03-16",
                "message_count": 2,
            },
        ),
        Artifact(
            artifact_id="t-claude-conv003",
            artifact_type="transcript",
            content="human: What is Rust ownership?\n\nassistant: Rust's ownership...\n\n",
            metadata={
                "source": "claude",
                "source_conversation_id": "conv003",
                "title": "Rust ownership",
                "date": "2024-03-15",
                "message_count": 4,
            },
        ),
        Artifact(
            artifact_id="ep-conv001",
            artifact_type="episode",
            content="In this conversation, the user discussed machine learning fundamentals...",
            input_hashes=["sha256:abc123"],
            prompt_id="episode_summary_v1",
            metadata={
                "source_conversation_id": "conv001",
                "date": "2024-03-15",
            },
        ),
        Artifact(
            artifact_id="ep-conv002",
            artifact_type="episode",
            content="The user asked about Docker containerization...",
            input_hashes=["sha256:def456"],
            prompt_id="episode_summary_v1",
            metadata={
                "source_conversation_id": "conv002",
                "date": "2024-03-16",
            },
        ),
    ]


class MockResponse:
    """Mock Anthropic API response."""

    def __init__(self, text: str):
        self.content = [MagicMock(text=text)]
        self.model = "claude-sonnet-4-20250514"
        self.usage = MagicMock(input_tokens=100, output_tokens=50)


@pytest.fixture
def mock_llm(monkeypatch):
    """Mock Anthropic API — returns deterministic responses based on prompt."""
    calls = []

    def mock_create(**kwargs):
        calls.append(kwargs)
        messages = kwargs.get("messages", [])
        content = messages[0].get("content", "") if messages else ""

        if "episode summary" in content.lower() or "summarizing a conversation" in content.lower():
            return MockResponse(
                "This is a summary of the conversation. The user discussed technical topics "
                "including programming concepts and best practices."
            )
        elif "monthly" in content.lower() or "monthly overview" in content.lower():
            return MockResponse(
                "In this month, the main themes were technical learning and software development. "
                "The user explored machine learning, Docker, and programming languages."
            )
        elif "topical" in content.lower() or "topic:" in content.lower():
            return MockResponse(
                "Regarding this topic, the user has shown consistent interest in technical subjects. "
                "Their understanding has evolved from basic concepts to more advanced applications."
            )
        elif "core memory" in content.lower() or "core memory document" in content.lower():
            return MockResponse(
                "## Identity\nMark is a software engineer interested in AI and systems.\n\n"
                "## Current Focus\nBuilding agent memory systems.\n\n"
                "## Preferences\nPrefers clean code and well-tested systems."
            )
        return MockResponse("Mock response for unknown prompt type.")

    mock_client = MagicMock()
    mock_client.messages.create = mock_create

    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)
    return calls


@pytest.fixture
def sample_pipeline(tmp_build_dir, tmp_path):
    """A complete test pipeline with all layers and projections."""
    source_dir = tmp_path / "exports"
    source_dir.mkdir()

    pipeline = Pipeline("test-pipeline")
    pipeline.source_dir = str(source_dir)
    pipeline.build_dir = str(tmp_build_dir)
    pipeline.llm_config = {
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.3,
        "max_tokens": 1024,
    }

    pipeline.add_layer(Layer(name="transcripts", level=0, transform="parse"))
    pipeline.add_layer(Layer(
        name="episodes", level=1, depends_on=["transcripts"],
        transform="episode_summary", grouping="by_conversation",
    ))
    pipeline.add_layer(Layer(
        name="monthly", level=2, depends_on=["episodes"],
        transform="monthly_rollup", grouping="by_month",
    ))
    pipeline.add_layer(Layer(
        name="core", level=3, depends_on=["monthly"],
        transform="core_synthesis", grouping="single", context_budget=10000,
    ))

    pipeline.add_projection(Projection(
        name="memory-index",
        projection_type="search_index",
        sources=[
            {"layer": "episodes", "search": ["fulltext"]},
            {"layer": "monthly", "search": ["fulltext"]},
            {"layer": "core", "search": ["fulltext"]},
        ],
    ))
    pipeline.add_projection(Projection(
        name="context-doc",
        projection_type="flat_file",
        sources=[{"layer": "core"}],
        config={"output_path": str(tmp_build_dir / "context.md")},
    ))

    return pipeline


@pytest.fixture
def chatgpt_fixture_path():
    """Path to ChatGPT export fixture."""
    return FIXTURES_DIR / "chatgpt_export.json"


@pytest.fixture
def claude_fixture_path():
    """Path to Claude export fixture."""
    return FIXTURES_DIR / "claude_export.json"


@pytest.fixture
def mock_llm_server():
    """Start mock LLM server in background, yield base_url, teardown."""
    from tests.mock_server.server import MockLLMServer

    server = MockLLMServer(host="127.0.0.1", port=0)
    server.start_background()
    yield server.base_url
    server.shutdown()


@pytest.fixture
def source_dir_with_fixtures(tmp_path):
    """A source directory populated with both ChatGPT and Claude export fixtures."""
    source_dir = tmp_path / "exports"
    source_dir.mkdir()

    # Copy fixtures
    import shutil
    shutil.copy(FIXTURES_DIR / "chatgpt_export.json", source_dir / "chatgpt_export.json")
    shutil.copy(FIXTURES_DIR / "claude_export.json", source_dir / "claude_export.json")

    return source_dir
