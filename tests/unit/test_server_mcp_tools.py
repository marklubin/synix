"""Tests for synix.server.mcp_tools — ingest, search, get_context, list_buckets."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from synix.server.config import BucketConfig, ServerConfig
from synix.server.mcp_tools import _state, get_context, ingest, list_buckets, search


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset global MCP state before each test."""
    old_project = _state["project"]
    old_config = _state["config"]
    _state["project"] = None
    _state["config"] = None
    yield
    _state["project"] = old_project
    _state["config"] = old_config


@pytest.fixture()
def server_config(tmp_path: Path) -> ServerConfig:
    """Create a ServerConfig with a real temp directory."""
    return ServerConfig(
        project_dir=str(tmp_path),
        buckets=[
            BucketConfig(
                name="sessions",
                dir="sources/sessions",
                patterns=["**/*.jsonl"],
                description="Session transcripts",
            ),
            BucketConfig(
                name="documents",
                dir="sources/documents",
                patterns=["**/*.md"],
                description="Notes and specs",
            ),
        ],
    )


def test_ingest_writes_file(tmp_path: Path, server_config: ServerConfig):
    """ingest() creates the file in the correct bucket directory."""
    _state["config"] = server_config

    result = ingest("sessions", "hello world", "test.jsonl")

    expected_path = tmp_path / "sources" / "sessions" / "test.jsonl"
    assert expected_path.exists()
    assert expected_path.read_text() == "hello world"
    assert "test.jsonl" in result
    assert "sessions" in result


def test_ingest_invalid_bucket(server_config: ServerConfig):
    """ingest() raises ValueError for unknown bucket."""
    _state["config"] = server_config

    with pytest.raises(ValueError, match="nonexistent"):
        ingest("nonexistent", "content", "file.txt")


def test_list_buckets_shows_configured(server_config: ServerConfig):
    """list_buckets() returns all configured bucket names and descriptions."""
    _state["config"] = server_config

    result = list_buckets()

    assert "sessions" in result
    assert "documents" in result
    assert "Session transcripts" in result
    assert "Notes and specs" in result


def test_list_buckets_empty():
    """list_buckets() handles no configured buckets."""
    _state["config"] = ServerConfig(project_dir="/tmp/test", buckets=[])

    result = list_buckets()
    assert "No buckets configured" in result


def test_search_with_mock_project():
    """search() calls project.release('local').search() and formats results."""
    mock_result = SimpleNamespace(
        score=0.95,
        layer="episodes",
        content="Test search result content",
    )

    mock_release = MagicMock()
    mock_release.search.return_value = [mock_result]

    mock_project = MagicMock()
    mock_project.release.return_value = mock_release

    _state["project"] = mock_project
    _state["config"] = ServerConfig(project_dir="/tmp/test")

    result = search("test query")

    mock_project.release.assert_called_once_with("local")
    mock_release.search.assert_called_once_with(
        "test query", mode="keyword", limit=10, layers=None, surface="search",
    )
    assert "0.950" in result
    assert "episodes" in result
    assert "Test search result content" in result


def test_search_no_results():
    """search() returns 'No results' when nothing matches."""
    mock_release = MagicMock()
    mock_release.search.return_value = []

    mock_project = MagicMock()
    mock_project.release.return_value = mock_release

    _state["project"] = mock_project
    _state["config"] = ServerConfig(project_dir="/tmp/test")

    result = search("nothing matches")
    assert "No results found" in result


def test_search_with_layers_filter():
    """search() parses comma-separated layers string into a list."""
    mock_release = MagicMock()
    mock_release.search.return_value = []

    mock_project = MagicMock()
    mock_project.release.return_value = mock_release

    _state["project"] = mock_project
    _state["config"] = ServerConfig(project_dir="/tmp/test")

    search("query", layers="episodes, topics")

    mock_release.search.assert_called_once_with(
        "query", mode="keyword", limit=10, layers=["episodes", "topics"], surface="search",
    )


def test_get_context_with_mock_project():
    """get_context() calls flat_file on the local release."""
    mock_release = MagicMock()
    mock_release.flat_file.return_value = "# Context\nSome content here."

    mock_project = MagicMock()
    mock_project.release.return_value = mock_release

    _state["project"] = mock_project
    _state["config"] = ServerConfig(project_dir="/tmp/test")

    result = get_context("context-doc")

    mock_project.release.assert_called_once_with("local")
    mock_release.flat_file.assert_called_once_with("context-doc")
    assert result == "# Context\nSome content here."


def test_search_requires_project():
    """search() raises ValueError when no project is open."""
    _state["project"] = None

    with pytest.raises(ValueError, match="No project open"):
        search("test")


def test_get_context_requires_project():
    """get_context() raises ValueError when no project is open."""
    _state["project"] = None

    with pytest.raises(ValueError, match="No project open"):
        get_context()
