"""End-to-end tests for the document queue integration.

Tests the full lifecycle: ingest → queue → status tracking.
Build execution is mocked since it requires a full pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synix.server.queue import DocumentQueue


@pytest.fixture
def queue(tmp_path: Path) -> DocumentQueue:
    return DocumentQueue(tmp_path / "queue.db")


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Minimal project directory with .synix/."""
    synix_dir = tmp_path / "project" / ".synix"
    synix_dir.mkdir(parents=True)
    sources = tmp_path / "project" / "sources" / "documents"
    sources.mkdir(parents=True)
    return tmp_path / "project"


class TestIngestToQueueLifecycle:
    """Test the ingest → queue → build → release lifecycle."""

    def test_enqueue_and_check_status(self, queue: DocumentQueue) -> None:
        doc_id = queue.enqueue("docs", "test.md", "abc123hash", "/tmp/test.md", client_id="test-client")
        status = queue.document_status(doc_id)

        assert status is not None
        assert status["status"] == "pending"
        assert status["bucket"] == "docs"
        assert status["filename"] == "test.md"
        assert status["client_id"] == "test-client"

    def test_full_lifecycle_pending_to_released(self, queue: DocumentQueue) -> None:
        # Ingest
        doc_id = queue.enqueue("docs", "file.md", "hash1", "/tmp/file.md")
        assert queue.pending_count() == 1

        # Claim for build
        run_id = "test-run-001"
        claimed = queue.claim_pending_batch(run_id)
        assert len(claimed) == 1
        assert doc_id in claimed
        assert queue.pending_count() == 0

        status = queue.document_status(doc_id)
        assert status["status"] == "processing"

        # Build completes
        queue.mark_built(run_id, built_count=3, cached_count=10)
        status = queue.document_status(doc_id)
        assert status["status"] == "built"

        # Release completes
        queue.mark_released(run_id)
        status = queue.document_status(doc_id)
        assert status["status"] == "released"
        assert status["released_at"] is not None

    def test_failed_build_requeues_documents(self, queue: DocumentQueue) -> None:
        doc_id = queue.enqueue("docs", "file.md", "hash2", "/tmp/file.md")
        run_id = "fail-run"
        queue.claim_pending_batch(run_id)

        # Build fails
        queue.mark_failed(run_id, "LLM timeout")
        status = queue.document_status(doc_id)
        assert status["status"] == "pending"  # requeued
        assert status["error_message"] == "LLM timeout"

        # Can be claimed again
        run_id2 = "retry-run"
        claimed = queue.claim_pending_batch(run_id2)
        assert len(claimed) == 1

    def test_dedup_prevents_reprocessing(self, queue: DocumentQueue) -> None:
        doc_id1 = queue.enqueue("docs", "file.md", "samehash", "/tmp/file.md")
        doc_id2 = queue.enqueue("docs", "file.md", "samehash", "/tmp/file.md")
        assert doc_id1 == doc_id2
        assert queue.pending_count() == 1

    def test_batch_of_multiple_documents(self, queue: DocumentQueue) -> None:
        ids = []
        for i in range(5):
            doc_id = queue.enqueue("docs", f"file{i}.md", f"hash{i}", f"/tmp/file{i}.md")
            ids.append(doc_id)

        assert queue.pending_count() == 5

        run_id = "batch-run"
        claimed = queue.claim_pending_batch(run_id)
        assert len(claimed) == 5
        assert set(claimed) == set(ids)

        queue.mark_built(run_id, built_count=5, cached_count=0)
        queue.mark_released(run_id)

        for doc_id in ids:
            status = queue.document_status(doc_id)
            assert status["status"] == "released"

    def test_queue_stats_after_lifecycle(self, queue: DocumentQueue) -> None:
        # Enqueue and process some
        for i in range(3):
            queue.enqueue("docs", f"done{i}.md", f"donehash{i}", f"/tmp/done{i}.md")
        run_id = "stats-run"
        queue.claim_pending_batch(run_id)
        queue.mark_built(run_id, built_count=3, cached_count=0)
        queue.mark_released(run_id)

        # Enqueue more (pending)
        queue.enqueue("docs", "pending.md", "pendinghash", "/tmp/pending.md")

        stats = queue.queue_stats()
        assert stats["pending_count"] == 1
        assert stats["released_count"] == 3

    def test_recent_history_ordered(self, queue: DocumentQueue) -> None:
        for i in range(5):
            queue.enqueue("docs", f"file{i}.md", f"hist{i}", f"/tmp/file{i}.md")

        recent = queue.recent_history(limit=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0]["filename"] == "file4.md"

    def test_client_id_tracking(self, queue: DocumentQueue) -> None:
        doc_id = queue.enqueue(
            "sessions", "session.jsonl.gz", "sesshash",
            "/tmp/session.jsonl.gz", client_id="Claude@Salinas",
        )
        status = queue.document_status(doc_id)
        assert status["client_id"] == "Claude@Salinas"
