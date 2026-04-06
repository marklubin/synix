"""Tests for synix.server.mcp_tools — ingest, search, get_context, list_buckets."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from synix.server import mcp_tools
from synix.server.mcp_tools import get_context, ingest, list_buckets, search
from synix.workspace import BucketConfig, Workspace, WorkspaceConfig


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset global MCP workspace before each test."""
    old_workspace = mcp_tools._workspace
    mcp_tools._workspace = None
    yield
    mcp_tools._workspace = old_workspace


def _make_workspace(project, tmp_path: Path, buckets: list[BucketConfig] | None = None) -> Workspace:
    """Helper: create a Workspace for tests with the given project and buckets."""
    if buckets is None:
        buckets = []
    config = WorkspaceConfig(name="test", buckets=buckets)
    ws = Workspace(project, config)
    # Ensure bucket_dir resolves relative to tmp_path (the mock project root)
    ws._project.project_root = tmp_path
    return ws


@pytest.fixture()
def ws_config(tmp_path: Path) -> WorkspaceConfig:
    """Create a WorkspaceConfig with a real temp directory."""
    return WorkspaceConfig(
        name="test",
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


def test_ingest_writes_file(tmp_path: Path, ws_config: WorkspaceConfig):
    """ingest() creates the file in the correct bucket directory."""
    mock_project = MagicMock()
    mock_project.project_root = tmp_path
    mcp_tools._workspace = Workspace(mock_project, ws_config)

    result = ingest("sessions", "hello world", "test.jsonl")

    expected_path = tmp_path / "sources" / "sessions" / "test.jsonl"
    assert expected_path.exists()
    assert expected_path.read_text() == "hello world"
    assert "test.jsonl" in result
    assert "sessions" in result


def test_ingest_invalid_bucket(tmp_path: Path, ws_config: WorkspaceConfig):
    """ingest() raises ValueError for unknown bucket."""
    mock_project = MagicMock()
    mock_project.project_root = tmp_path
    mcp_tools._workspace = Workspace(mock_project, ws_config)

    with pytest.raises(ValueError, match="nonexistent"):
        ingest("nonexistent", "content", "file.txt")


def test_list_buckets_shows_configured(tmp_path: Path, ws_config: WorkspaceConfig):
    """list_buckets() returns all configured bucket names and descriptions."""
    mock_project = MagicMock()
    mock_project.project_root = tmp_path
    mcp_tools._workspace = Workspace(mock_project, ws_config)

    result = list_buckets()

    assert "sessions" in result
    assert "documents" in result
    assert "Session transcripts" in result
    assert "Notes and specs" in result


def test_list_buckets_empty(tmp_path: Path):
    """list_buckets() handles no configured buckets."""
    mock_project = MagicMock()
    mock_project.project_root = tmp_path
    mcp_tools._workspace = Workspace(mock_project, WorkspaceConfig(name="test", buckets=[]))

    result = list_buckets()
    assert "No buckets configured" in result


def test_search_with_mock_project(tmp_path: Path):
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
    mock_project.project_root = tmp_path

    mcp_tools._workspace = Workspace(mock_project, WorkspaceConfig(name="test"))

    result = search("test query")

    mock_project.release.assert_called_once_with("local")
    mock_release.search.assert_called_once_with(
        "test query",
        mode="keyword",
        limit=10,
        layers=None,
        surface="search",
    )
    assert "0.950" in result
    assert "episodes" in result
    assert "Test search result content" in result


def test_search_no_results(tmp_path: Path):
    """search() returns 'No results' when nothing matches."""
    mock_release = MagicMock()
    mock_release.search.return_value = []

    mock_project = MagicMock()
    mock_project.release.return_value = mock_release
    mock_project.project_root = tmp_path

    mcp_tools._workspace = Workspace(mock_project, WorkspaceConfig(name="test"))

    result = search("nothing matches")
    assert "No results found" in result


def test_search_with_layers_filter(tmp_path: Path):
    """search() parses comma-separated layers string into a list."""
    mock_release = MagicMock()
    mock_release.search.return_value = []

    mock_project = MagicMock()
    mock_project.release.return_value = mock_release
    mock_project.project_root = tmp_path

    mcp_tools._workspace = Workspace(mock_project, WorkspaceConfig(name="test"))

    search("query", layers="episodes, topics")

    mock_release.search.assert_called_once_with(
        "query",
        mode="keyword",
        limit=10,
        layers=["episodes", "topics"],
        surface="search",
    )


def test_get_context_with_mock_project(tmp_path: Path):
    """get_context() calls flat_file on the local release."""
    mock_release = MagicMock()
    mock_release.flat_file.return_value = "# Context\nSome content here."

    mock_project = MagicMock()
    mock_project.release.return_value = mock_release
    mock_project.project_root = tmp_path

    mcp_tools._workspace = Workspace(mock_project, WorkspaceConfig(name="test"))

    result = get_context("context-doc")

    mock_project.release.assert_called_once_with("local")
    mock_release.flat_file.assert_called_once_with("context-doc")
    assert result == "# Context\nSome content here."


def test_search_requires_project():
    """search() raises ValueError when no project is open."""
    mcp_tools._workspace = None

    with pytest.raises(ValueError, match="No project open"):
        search("test")


def test_get_context_requires_project():
    """get_context() raises ValueError when no project is open."""
    mcp_tools._workspace = None

    with pytest.raises(ValueError, match="No project open"):
        get_context()
