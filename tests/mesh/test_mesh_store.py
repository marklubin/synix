"""Tests for synix.mesh.store — session storage with SHA256 dedup."""

from __future__ import annotations

import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from synix.mesh.store import SessionStore

pytestmark = pytest.mark.mesh


@pytest.fixture
def store(tmp_path):
    """Create a SessionStore in a temp directory."""
    return SessionStore(
        db_path=tmp_path / "sessions.db",
        sessions_dir=tmp_path / "sessions",
    )


@pytest.fixture
def sample_content():
    """Sample JSONL content bytes."""
    return b'{"role": "user", "content": "hello"}\n{"role": "assistant", "content": "hi"}\n'


class TestSubmit:
    def test_submit_new_returns_true(self, store, sample_content):
        result = store.submit("sess-001", "project-a", sample_content, "node-1")
        assert result is True

    def test_submit_stores_compressed_file(self, store, sample_content, tmp_path):
        store.submit("sess-001", "project-a", sample_content, "node-1")
        gz_path = tmp_path / "sessions" / "project-a" / "sess-001.jsonl.gz"
        assert gz_path.exists()
        decompressed = gzip.decompress(gz_path.read_bytes())
        assert decompressed == sample_content

    def test_submit_duplicate_returns_false(self, store, sample_content):
        store.submit("sess-001", "project-a", sample_content, "node-1")
        result = store.submit("sess-002", "project-a", sample_content, "node-1")
        assert result is False

    def test_submit_different_content_returns_true(self, store, sample_content):
        store.submit("sess-001", "project-a", sample_content, "node-1")
        other_content = b'{"role": "user", "content": "different"}\n'
        result = store.submit("sess-002", "project-a", other_content, "node-1")
        assert result is True


class TestGetUnprocessed:
    def test_returns_empty_when_no_sessions(self, store):
        assert store.get_unprocessed() == []

    def test_returns_pending_sessions(self, store, sample_content):
        store.submit("sess-001", "project-a", sample_content, "node-1")
        unprocessed = store.get_unprocessed()
        assert len(unprocessed) == 1
        assert unprocessed[0]["session_id"] == "sess-001"
        assert unprocessed[0]["project_dir"] == "project-a"
        assert unprocessed[0]["submitted_by"] == "node-1"

    def test_does_not_return_processed_sessions(self, store, sample_content):
        store.submit("sess-001", "project-a", sample_content, "node-1")
        store.mark_processed(["sess-001"])
        assert store.get_unprocessed() == []


class TestMarkProcessed:
    def test_mark_processed_transitions(self, store, sample_content):
        store.submit("sess-001", "project-a", sample_content, "node-1")
        other_content = b"other content"
        store.submit("sess-002", "project-a", other_content, "node-2")
        store.mark_processed(["sess-001"])

        unprocessed = store.get_unprocessed()
        assert len(unprocessed) == 1
        assert unprocessed[0]["session_id"] == "sess-002"

    def test_mark_empty_list_no_error(self, store):
        store.mark_processed([])  # should not raise


class TestGetSessionContent:
    def test_returns_decompressed_content(self, store, sample_content):
        store.submit("sess-001", "project-a", sample_content, "node-1")
        result = store.get_session_content("sess-001")
        assert result == sample_content

    def test_returns_none_for_missing(self, store):
        assert store.get_session_content("nonexistent") is None

    def test_returns_none_when_file_missing(self, store, sample_content, tmp_path):
        store.submit("sess-001", "project-a", sample_content, "node-1")
        # Delete the file but keep the DB row
        gz_path = tmp_path / "sessions" / "project-a" / "sess-001.jsonl.gz"
        gz_path.unlink()
        assert store.get_session_content("sess-001") is None


class TestCount:
    def test_empty_store(self, store):
        assert store.count() == {"total": 0, "processed": 0, "pending": 0}

    def test_with_sessions(self, store, sample_content):
        store.submit("sess-001", "project-a", sample_content, "node-1")
        other = b"other content"
        store.submit("sess-002", "project-a", other, "node-1")
        assert store.count() == {"total": 2, "processed": 0, "pending": 2}

    def test_with_processed(self, store, sample_content):
        store.submit("sess-001", "project-a", sample_content, "node-1")
        other = b"other content"
        store.submit("sess-002", "project-a", other, "node-1")
        store.mark_processed(["sess-001"])
        assert store.count() == {"total": 2, "processed": 1, "pending": 1}


class TestConcurrentAccess:
    def test_concurrent_submits(self, tmp_path):
        """Multiple threads submitting different sessions concurrently."""
        store = SessionStore(
            db_path=tmp_path / "sessions.db",
            sessions_dir=tmp_path / "sessions",
        )

        def submit_session(i: int) -> bool:
            # Each thread gets its own store instance to avoid sharing connections
            thread_store = SessionStore(
                db_path=tmp_path / "sessions.db",
                sessions_dir=tmp_path / "sessions",
            )
            content = f"session content {i}".encode()
            return thread_store.submit(f"sess-{i:03d}", "project-a", content, f"node-{i}")

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(submit_session, i) for i in range(10)]
            results = [f.result() for f in as_completed(futures)]

        # All should succeed (all have unique content)
        assert all(results)
        assert store.count()["total"] == 10
