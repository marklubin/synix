"""Unit tests for the SQLite document queue."""

from __future__ import annotations

import time

from synix.server.queue import DocumentQueue


class TestEnqueue:
    def test_enqueue_returns_doc_id(self, tmp_path):
        q = DocumentQueue(tmp_path / "test.db")
        doc_id = q.enqueue("notes", "a.txt", "hash1", "/tmp/a.txt")
        assert isinstance(doc_id, str)
        assert len(doc_id) == 32  # uuid4().hex

    def test_enqueue_dedup_same_hash(self, tmp_path):
        q = DocumentQueue(tmp_path / "test.db")
        id1 = q.enqueue("notes", "a.txt", "hash1", "/tmp/a.txt")
        id2 = q.enqueue("notes", "b.txt", "hash1", "/tmp/b.txt")
        assert id1 == id2

    def test_enqueue_different_hash(self, tmp_path):
        q = DocumentQueue(tmp_path / "test.db")
        id1 = q.enqueue("notes", "a.txt", "hash1", "/tmp/a.txt")
        id2 = q.enqueue("notes", "b.txt", "hash2", "/tmp/b.txt")
        assert id1 != id2


class TestPendingCount:
    def test_pending_count(self, tmp_path):
        q = DocumentQueue(tmp_path / "test.db")
        assert q.pending_count() == 0
        q.enqueue("notes", "a.txt", "h1", "/tmp/a.txt")
        assert q.pending_count() == 1
        q.enqueue("notes", "b.txt", "h2", "/tmp/b.txt")
        assert q.pending_count() == 2


class TestLastEnqueueTime:
    def test_last_enqueue_time(self, tmp_path):
        q = DocumentQueue(tmp_path / "test.db")
        assert q.last_enqueue_time() is None
        q.enqueue("notes", "a.txt", "h1", "/tmp/a.txt")
        ts = q.last_enqueue_time()
        assert isinstance(ts, float)
        # Should be a recent timestamp (within last 10 seconds)
        assert abs(ts - time.time()) < 10


class TestClaimPendingBatch:
    def test_claim_pending_batch(self, tmp_path):
        q = DocumentQueue(tmp_path / "test.db")
        id1 = q.enqueue("notes", "a.txt", "h1", "/tmp/a.txt")
        id2 = q.enqueue("notes", "b.txt", "h2", "/tmp/b.txt")

        claimed = q.claim_pending_batch("run-1")
        assert set(claimed) == {id1, id2}
        assert q.pending_count() == 0

        # Build run was created
        row = q._conn.execute("SELECT * FROM build_runs WHERE run_id = 'run-1'").fetchone()
        assert row is not None
        assert row["status"] == "running"
        assert row["documents_count"] == 2

    def test_claim_empty_queue(self, tmp_path):
        q = DocumentQueue(tmp_path / "test.db")
        claimed = q.claim_pending_batch("run-1")
        assert claimed == []


class TestMarkBuilt:
    def test_mark_built(self, tmp_path):
        q = DocumentQueue(tmp_path / "test.db")
        doc_id = q.enqueue("notes", "a.txt", "h1", "/tmp/a.txt")
        q.claim_pending_batch("run-1")
        q.mark_built("run-1", built_count=1, cached_count=0)

        status = q.document_status(doc_id)
        assert status["status"] == "built"
        assert status["built_at"] is not None

        run_row = q._conn.execute("SELECT * FROM build_runs WHERE run_id = 'run-1'").fetchone()
        assert run_row["status"] == "completed"
        assert run_row["built_count"] == 1
        assert run_row["cached_count"] == 0


class TestMarkReleased:
    def test_mark_released(self, tmp_path):
        q = DocumentQueue(tmp_path / "test.db")
        doc_id = q.enqueue("notes", "a.txt", "h1", "/tmp/a.txt")
        q.claim_pending_batch("run-1")
        q.mark_built("run-1")
        q.mark_released("run-1")

        status = q.document_status(doc_id)
        assert status["status"] == "released"
        assert status["released_at"] is not None


class TestMarkFailed:
    def test_mark_failed_requeues(self, tmp_path):
        q = DocumentQueue(tmp_path / "test.db")
        doc_id = q.enqueue("notes", "a.txt", "h1", "/tmp/a.txt")
        q.claim_pending_batch("run-1")
        q.mark_failed("run-1", "LLM timeout")

        status = q.document_status(doc_id)
        assert status["status"] == "pending"
        assert status["error_message"] == "LLM timeout"
        assert status["build_run_id"] is None
        assert status["processing_started_at"] is None
        assert q.pending_count() == 1

        run_row = q._conn.execute("SELECT * FROM build_runs WHERE run_id = 'run-1'").fetchone()
        assert run_row["status"] == "failed"
        assert run_row["error_message"] == "LLM timeout"


class TestDocumentStatus:
    def test_document_status(self, tmp_path):
        q = DocumentQueue(tmp_path / "test.db")
        doc_id = q.enqueue("notes", "a.txt", "h1", "/tmp/a.txt", client_id="cli-1")

        status = q.document_status(doc_id)
        assert status is not None
        assert status["doc_id"] == doc_id
        assert status["bucket"] == "notes"
        assert status["filename"] == "a.txt"
        assert status["content_hash"] == "h1"
        assert status["file_path"] == "/tmp/a.txt"
        assert status["client_id"] == "cli-1"
        assert status["status"] == "pending"
        assert status["created_at"] is not None

    def test_document_status_not_found(self, tmp_path):
        q = DocumentQueue(tmp_path / "test.db")
        assert q.document_status("nonexistent") is None

    def test_document_status_queue_position(self, tmp_path):
        q = DocumentQueue(tmp_path / "test.db")
        id1 = q.enqueue("notes", "a.txt", "h1", "/tmp/a.txt")
        id2 = q.enqueue("notes", "b.txt", "h2", "/tmp/b.txt")
        id3 = q.enqueue("notes", "c.txt", "h3", "/tmp/c.txt")

        # Positions are 1-based, ordered by created_at
        s1 = q.document_status(id1)
        s3 = q.document_status(id3)
        assert s1["queue_position"] is not None
        assert s3["queue_position"] is not None
        assert s1["queue_position"] <= s3["queue_position"]

        # After claiming, position should be None
        q.claim_pending_batch("run-1")
        s1_after = q.document_status(id1)
        assert s1_after["queue_position"] is None


class TestRecentHistory:
    def test_recent_history(self, tmp_path):
        q = DocumentQueue(tmp_path / "test.db")
        q.enqueue("notes", "a.txt", "h1", "/tmp/a.txt")
        q.enqueue("notes", "b.txt", "h2", "/tmp/b.txt")
        q.enqueue("notes", "c.txt", "h3", "/tmp/c.txt")

        history = q.recent_history(limit=2)
        assert len(history) == 2
        # Most recent first
        assert history[0]["filename"] == "c.txt"
        assert history[1]["filename"] == "b.txt"


class TestQueueStats:
    def test_queue_stats(self, tmp_path):
        q = DocumentQueue(tmp_path / "test.db")
        q.enqueue("notes", "a.txt", "h1", "/tmp/a.txt")
        q.enqueue("notes", "b.txt", "h2", "/tmp/b.txt")
        q.enqueue("notes", "c.txt", "h3", "/tmp/c.txt")

        q.claim_pending_batch("run-1")
        q.mark_built("run-1", built_count=3)
        q.mark_released("run-1")

        q.enqueue("notes", "d.txt", "h4", "/tmp/d.txt")

        stats = q.queue_stats()
        assert stats["pending_count"] == 1
        assert stats["processing_count"] == 0
        assert stats["built_count"] == 0
        assert stats["released_count"] == 3
        assert stats["total_processed"] == 3
        assert stats["failed_count"] == 0


class TestFullLifecycle:
    def test_full_lifecycle(self, tmp_path):
        q = DocumentQueue(tmp_path / "test.db")

        # Enqueue
        doc_id = q.enqueue("notes", "a.txt", "h1", "/tmp/a.txt")
        assert q.pending_count() == 1

        # Claim
        claimed = q.claim_pending_batch("run-1")
        assert claimed == [doc_id]
        assert q.pending_count() == 0
        assert q.document_status(doc_id)["status"] == "processing"

        # Built
        q.mark_built("run-1", built_count=1, cached_count=0)
        assert q.document_status(doc_id)["status"] == "built"

        # Released
        q.mark_released("run-1")
        assert q.document_status(doc_id)["status"] == "released"
        assert q.document_status(doc_id)["released_at"] is not None


class TestClientId:
    def test_client_id_recorded(self, tmp_path):
        q = DocumentQueue(tmp_path / "test.db")
        doc_id = q.enqueue("notes", "a.txt", "h1", "/tmp/a.txt", client_id="agent-007")
        status = q.document_status(doc_id)
        assert status["client_id"] == "agent-007"


class TestDedupAfterReleased:
    def test_dedup_after_released(self, tmp_path):
        q = DocumentQueue(tmp_path / "test.db")

        # Full lifecycle to released
        original_id = q.enqueue("notes", "a.txt", "h1", "/tmp/a.txt")
        q.claim_pending_batch("run-1")
        q.mark_built("run-1")
        q.mark_released("run-1")

        # Re-enqueue same content_hash
        dedup_id = q.enqueue("notes", "a-copy.txt", "h1", "/tmp/a-copy.txt")
        assert dedup_id == original_id
        # Status should still be released (not re-queued)
        assert q.document_status(original_id)["status"] == "released"
