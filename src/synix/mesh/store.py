"""Session storage — SQLite + compressed files with SHA256 dedup."""

from __future__ import annotations

import gzip
import hashlib
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class SessionStore:
    """Stores submitted session files with SHA256 dedup.

    Uses SQLite (WAL mode) for metadata and gzip-compressed files on disk.
    """

    def __init__(self, db_path: Path, sessions_dir: Path):
        self.db_path = db_path
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                project_dir TEXT NOT NULL,
                submitted_by TEXT NOT NULL,
                jsonl_sha256 TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                processed INTEGER DEFAULT 0
            )
        """)
        conn.commit()

    def submit(
        self,
        session_id: str,
        project_dir: str,
        content: bytes,
        submitted_by: str,
    ) -> bool:
        """Submit a session file. Returns True if new, False if duplicate.

        Content is gzip-compressed and stored as
        sessions/{project_dir}/{session_id}.jsonl.gz.
        Deduplication is based on SHA256 of the raw content.
        """
        content_hash = hashlib.sha256(content).hexdigest()

        conn = self._get_conn()

        # Check for duplicate by sha256
        row = conn.execute("SELECT 1 FROM sessions WHERE jsonl_sha256 = ?", (content_hash,)).fetchone()
        if row is not None:
            logger.debug(
                "Duplicate session content (sha256=%s), skipping %s",
                content_hash[:12],
                session_id,
            )
            return False

        # Compress and write to disk
        proj_dir = self.sessions_dir / project_dir
        proj_dir.mkdir(parents=True, exist_ok=True)
        gz_path = proj_dir / f"{session_id}.jsonl.gz"
        gz_path.write_bytes(gzip.compress(content))

        # Insert metadata
        now = datetime.now(UTC).isoformat()
        conn.execute(
            """INSERT INTO sessions
               (session_id, project_dir, submitted_by, jsonl_sha256, submitted_at, processed)
               VALUES (?, ?, ?, ?, ?, 0)""",
            (session_id, project_dir, submitted_by, content_hash, now),
        )
        conn.commit()

        logger.info("Submitted session %s (sha256=%s)", session_id, content_hash[:12])
        return True

    def get_unprocessed(self) -> list[dict]:
        """Return list of unprocessed session metadata dicts."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT session_id, project_dir, submitted_by, jsonl_sha256, submitted_at "
            "FROM sessions WHERE processed = 0 ORDER BY submitted_at"
        ).fetchall()
        return [dict(row) for row in rows]

    def mark_processed(self, session_ids: list[str]) -> None:
        """Mark sessions as processed after a build."""
        if not session_ids:
            return
        conn = self._get_conn()
        placeholders = ",".join("?" for _ in session_ids)
        conn.execute(
            f"UPDATE sessions SET processed = 1 WHERE session_id IN ({placeholders})",
            session_ids,
        )
        conn.commit()
        logger.info("Marked %d sessions as processed", len(session_ids))

    def get_session_content(self, session_id: str) -> bytes | None:
        """Read and decompress a session file. Returns None if not found."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT project_dir FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None

        gz_path = self.sessions_dir / row["project_dir"] / f"{session_id}.jsonl.gz"
        if not gz_path.exists():
            logger.warning(
                "Session %s metadata exists but file missing: %s",
                session_id,
                gz_path,
            )
            return None

        return gzip.decompress(gz_path.read_bytes())

    def count(self) -> dict:
        """Return {"total": N, "processed": M, "pending": P}."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) as total, SUM(CASE WHEN processed = 1 THEN 1 ELSE 0 END) as processed FROM sessions"
        ).fetchone()
        total = row["total"]
        processed = row["processed"] or 0
        return {"total": total, "processed": processed, "pending": total - processed}

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
