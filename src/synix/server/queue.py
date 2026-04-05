"""SQLite-backed document queue for the synix knowledge server.

Tracks documents through the ingest → build → release lifecycle with
content-hash deduplication and WAL-mode concurrency.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(UTC).isoformat()


class DocumentQueue:
    """Persistent document queue backed by SQLite.

    Two tables:
    - document_queue: tracks individual documents through pending → processing → built → released.
    - build_runs: tracks batch build executions.

    Thread-safe (check_same_thread=False) with WAL journal mode for
    concurrent readers during builds.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._create_tables()
        logger.info("DocumentQueue initialized at %s", db_path)

    def _create_tables(self) -> None:
        """Create tables if they don't exist."""
        with self._conn:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS document_queue (
                    doc_id TEXT PRIMARY KEY,
                    bucket TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    client_id TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    processing_started_at TEXT,
                    built_at TEXT,
                    released_at TEXT,
                    build_run_id TEXT,
                    UNIQUE(content_hash)
                )
            """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS build_runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL DEFAULT 'pending',
                    started_at TEXT,
                    completed_at TEXT,
                    documents_count INTEGER DEFAULT 0,
                    built_count INTEGER DEFAULT 0,
                    cached_count INTEGER DEFAULT 0,
                    error_message TEXT
                )
            """)

    def enqueue(
        self,
        bucket: str,
        filename: str,
        content_hash: str,
        file_path: str,
        client_id: str | None = None,
    ) -> str:
        """Add a document to the queue, deduplicating by content_hash.

        If a document with the same content_hash already exists (any status),
        returns the existing doc_id without inserting a duplicate.

        Returns the doc_id (new or existing).
        """
        with self._lock:
            doc_id = uuid.uuid4().hex
            now = _utcnow()
            with self._conn:
                self._conn.execute(
                    """INSERT OR IGNORE INTO document_queue
                       (doc_id, bucket, filename, content_hash, file_path, client_id, status, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
                    (doc_id, bucket, filename, content_hash, file_path, client_id, now),
                )
                row = self._conn.execute(
                    "SELECT doc_id FROM document_queue WHERE content_hash = ?",
                    (content_hash,),
                ).fetchone()
            actual_id = row["doc_id"]
            if actual_id == doc_id:
                logger.info(
                    "Enqueued doc %s: bucket=%s filename=%s hash=%s",
                    doc_id,
                    bucket,
                    filename,
                    content_hash[:12],
                )
            else:
                logger.info(
                    "Dedup hit for hash %s — returning existing doc %s",
                    content_hash[:12],
                    actual_id,
                )
            return actual_id

    def pending_count(self) -> int:
        """Return the number of documents with status='pending'."""
        row = self._conn.execute("SELECT COUNT(*) AS cnt FROM document_queue WHERE status = 'pending'").fetchone()
        return row["cnt"]

    def last_enqueue_time(self) -> float | None:
        """Return the timestamp of the most recently enqueued pending document.

        Returns None if there are no pending documents.
        """
        row = self._conn.execute(
            "SELECT MAX(created_at) AS max_ts FROM document_queue WHERE status = 'pending'"
        ).fetchone()
        if row["max_ts"] is None:
            return None
        return datetime.fromisoformat(row["max_ts"]).timestamp()

    def claim_pending_batch(self, run_id: str) -> list[str]:
        """Atomically claim all pending documents for a build run.

        Creates a build_runs record and transitions all pending documents to
        'processing' with the given run_id.

        Returns the list of claimed doc_ids.
        """
        with self._lock:
            now = _utcnow()
            with self._conn:
                cur = self._conn.execute("SELECT doc_id FROM document_queue WHERE status = 'pending'")
                doc_ids = [r["doc_id"] for r in cur.fetchall()]

                if not doc_ids:
                    logger.info("claim_pending_batch(%s): no pending documents", run_id)
                    return []

                self._conn.execute(
                    """INSERT INTO build_runs (run_id, status, started_at, documents_count)
                       VALUES (?, 'running', ?, ?)""",
                    (run_id, now, len(doc_ids)),
                )
                self._conn.execute(
                    """UPDATE document_queue
                       SET status = 'processing',
                           build_run_id = ?,
                           processing_started_at = ?
                       WHERE status = 'pending'""",
                    (run_id, now),
                )

            logger.info(
                "claim_pending_batch(%s): claimed %d documents",
                run_id,
                len(doc_ids),
            )
            return doc_ids

    def mark_built(self, run_id: str, built_count: int = 0, cached_count: int = 0) -> None:
        """Transition processing documents to built, complete the build run."""
        with self._lock:
            now = _utcnow()
            with self._conn:
                self._conn.execute(
                    """UPDATE document_queue
                       SET status = 'built', built_at = ?
                       WHERE build_run_id = ? AND status = 'processing'""",
                    (now, run_id),
                )
                self._conn.execute(
                    """UPDATE build_runs
                       SET status = 'completed', completed_at = ?,
                           built_count = ?, cached_count = ?
                       WHERE run_id = ?""",
                    (now, built_count, cached_count, run_id),
                )
            logger.info(
                "mark_built(%s): built=%d cached=%d",
                run_id,
                built_count,
                cached_count,
            )

    def mark_released(self, run_id: str) -> None:
        """Transition built documents to released."""
        with self._lock:
            now = _utcnow()
            with self._conn:
                self._conn.execute(
                    """UPDATE document_queue
                       SET status = 'released', released_at = ?
                       WHERE build_run_id = ? AND status = 'built'""",
                    (now, run_id),
                )
            logger.info("mark_released(%s)", run_id)

    def mark_failed(self, run_id: str, error: str) -> None:
        """Requeue documents back to pending on build or release failure.

        Handles docs in either 'processing' or 'built' state (release can
        fail after a successful build). Clears the build_run_id so documents
        can be claimed in a future run.
        """
        with self._lock:
            now = _utcnow()
            with self._conn:
                self._conn.execute(
                    """UPDATE document_queue
                       SET status = 'pending',
                           error_message = ?,
                           build_run_id = NULL,
                           processing_started_at = NULL,
                           built_at = NULL
                       WHERE build_run_id = ? AND status IN ('processing', 'built')""",
                    (error, run_id),
                )
                self._conn.execute(
                    """UPDATE build_runs
                       SET status = 'failed', completed_at = ?, error_message = ?
                       WHERE run_id = ?""",
                    (now, error, run_id),
                )
            logger.info("mark_failed(%s): %s", run_id, error)

    def document_status(self, doc_id: str) -> dict | None:
        """Return full status dict for a document, or None if not found.

        Includes a ``queue_position`` field: the 1-based position among pending
        documents ordered by created_at, or None if the document is not pending.
        """
        row = self._conn.execute("SELECT * FROM document_queue WHERE doc_id = ?", (doc_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)

        if result["status"] == "pending":
            pos_row = self._conn.execute(
                """SELECT COUNT(*) AS pos FROM document_queue
                   WHERE status = 'pending' AND created_at <= ?""",
                (result["created_at"],),
            ).fetchone()
            result["queue_position"] = pos_row["pos"]
        else:
            result["queue_position"] = None
        return result

    def recent_history(self, limit: int = 50) -> list[dict]:
        """Return the most recent documents ordered by created_at descending."""
        rows = self._conn.execute(
            "SELECT * FROM document_queue ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def queue_stats(self) -> dict:
        """Return aggregate queue statistics."""
        counts = self._conn.execute(
            """SELECT
                   SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending_count,
                   SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) AS processing_count,
                   SUM(CASE WHEN status = 'built' THEN 1 ELSE 0 END) AS built_count,
                   SUM(CASE WHEN status = 'released' THEN 1 ELSE 0 END) AS released_count,
                   SUM(CASE WHEN status = 'pending' AND error_message IS NOT NULL THEN 1 ELSE 0 END) AS failed_count
               FROM document_queue"""
        ).fetchone()

        total_processed = counts["released_count"] or 0

        avg_row = self._conn.execute(
            """SELECT AVG(
                   (julianday(completed_at) - julianday(started_at)) * 86400
               ) AS avg_secs
               FROM build_runs
               WHERE completed_at IS NOT NULL"""
        ).fetchone()
        avg_build_time = avg_row["avg_secs"]

        return {
            "pending_count": counts["pending_count"] or 0,
            "processing_count": counts["processing_count"] or 0,
            "built_count": counts["built_count"] or 0,
            "released_count": counts["released_count"] or 0,
            "failed_count": counts["failed_count"] or 0,
            "total_processed": total_processed,
            "avg_build_time_seconds": round(avg_build_time, 2) if avg_build_time is not None else None,
        }

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
