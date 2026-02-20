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


class TestSubmitSubsession:
    def test_submit_subsession_seq(self, store, tmp_path):
        content_a = b'{"role": "user", "content": "part 1"}\n'
        content_b = b'{"role": "user", "content": "part 2"}\n'

        assert store.submit("sess-001", "proj", content_a, "node-1", subsession_seq=0)
        assert store.submit("sess-001", "proj", content_b, "node-1", subsession_seq=1)

        # Both stored as separate files
        assert (tmp_path / "sessions" / "proj" / "sess-001.jsonl.gz").exists()
        assert (tmp_path / "sessions" / "proj" / "sess-001_sub0001.jsonl.gz").exists()

    def test_subsession_file_naming(self, store, tmp_path):
        store.submit("s1", "p", b"content-0", "n", subsession_seq=0)
        store.submit("s1", "p", b"content-5", "n", subsession_seq=5)

        assert (tmp_path / "sessions" / "p" / "s1.jsonl.gz").exists()
        assert (tmp_path / "sessions" / "p" / "s1_sub0005.jsonl.gz").exists()

    def test_subsession_dedup_by_sha256(self, store):
        content = b"same content"
        assert store.submit("s1", "p", content, "n", subsession_seq=0)
        assert store.submit("s1", "p", content, "n", subsession_seq=1) is False

    def test_get_unprocessed_includes_subsession_seq(self, store):
        store.submit("s1", "p", b"content-0", "n", subsession_seq=0)
        store.submit("s1", "p", b"content-1", "n", subsession_seq=1)

        unprocessed = store.get_unprocessed()
        assert len(unprocessed) == 2
        seqs = {r["subsession_seq"] for r in unprocessed}
        assert seqs == {0, 1}

    def test_mark_processed_3tuple(self, store):
        store.submit("s1", "p", b"content-0", "n", subsession_seq=0)
        store.submit("s1", "p", b"content-1", "n", subsession_seq=1)

        # Mark only seq=0
        store.mark_processed([("s1", "p", 0)])

        unprocessed = store.get_unprocessed()
        assert len(unprocessed) == 1
        assert unprocessed[0]["subsession_seq"] == 1

    def test_get_session_content_by_subsession(self, store):
        store.submit("s1", "p", b"content-zero", "n", subsession_seq=0)
        store.submit("s1", "p", b"content-one", "n", subsession_seq=1)

        assert store.get_session_content("s1", project_dir="p", subsession_seq=0) == b"content-zero"
        assert store.get_session_content("s1", project_dir="p", subsession_seq=1) == b"content-one"

    def test_get_session_file_path_subsession(self, store):
        store.submit("s1", "p", b"c0", "n", subsession_seq=0)
        store.submit("s1", "p", b"c3", "n", subsession_seq=3)

        p0 = store.get_session_file_path("s1", "p", subsession_seq=0)
        assert p0 is not None
        assert p0.name == "s1.jsonl.gz"

        p3 = store.get_session_file_path("s1", "p", subsession_seq=3)
        assert p3 is not None
        assert p3.name == "s1_sub0003.jsonl.gz"

    def test_list_all_sessions_includes_subsession_seq(self, store):
        store.submit("s1", "p", b"c0", "n", subsession_seq=0)
        store.submit("s1", "p", b"c1", "n", subsession_seq=1)

        all_sessions = store.list_all_sessions()
        assert len(all_sessions) == 2
        seqs = {s["subsession_seq"] for s in all_sessions}
        assert seqs == {0, 1}


class TestSchemaMigration:
    def test_v1_to_v2_migration(self, tmp_path):
        """Create a v1 schema DB, then open with v2 SessionStore — migration should work."""
        import sqlite3

        db_path = tmp_path / "sessions.db"
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        # Create v1 schema
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE sessions (
                session_id TEXT NOT NULL,
                project_dir TEXT NOT NULL,
                submitted_by TEXT NOT NULL,
                jsonl_sha256 TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                processed INTEGER DEFAULT 0,
                PRIMARY KEY (session_id, project_dir)
            )
        """)
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
            ("s1", "p1", "node-1", "abc123", "2024-01-01T00:00:00", 0),
        )
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
            ("s2", "p1", "node-2", "def456", "2024-01-02T00:00:00", 1),
        )
        conn.commit()
        conn.close()

        # Write dummy files for the existing sessions
        (sessions_dir / "p1").mkdir()
        (sessions_dir / "p1" / "s1.jsonl.gz").write_bytes(gzip.compress(b"content1"))
        (sessions_dir / "p1" / "s2.jsonl.gz").write_bytes(gzip.compress(b"content2"))

        # Open with v2 store — should migrate
        store = SessionStore(db_path=db_path, sessions_dir=sessions_dir)

        # Verify data preserved
        unprocessed = store.get_unprocessed()
        assert len(unprocessed) == 1
        assert unprocessed[0]["session_id"] == "s1"
        assert unprocessed[0]["subsession_seq"] == 0

        # Verify we can now submit with subsession_seq
        assert store.submit("s1", "p1", b"subsession content", "node-1", subsession_seq=1)
        assert store.count()["total"] == 3

        store.close()


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
