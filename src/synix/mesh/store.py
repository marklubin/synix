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
                session_id TEXT NOT NULL,
                project_dir TEXT NOT NULL,
                submitted_by TEXT NOT NULL,
                jsonl_sha256 TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                processed INTEGER DEFAULT 0,
                PRIMARY KEY (session_id, project_dir)
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

    def mark_processed(self, session_keys: list[str] | list[tuple[str, str]]) -> None:
        """Mark sessions as processed after a build.

        Accepts either:
        - list of session_id strings (marks all matching, backward compat)
        - list of (session_id, project_dir) tuples (composite key)
        """
        if not session_keys:
            return
        conn = self._get_conn()
        if session_keys and isinstance(session_keys[0], tuple):
            for sid, pdir in session_keys:
                conn.execute(
                    "UPDATE sessions SET processed = 1 WHERE session_id = ? AND project_dir = ?",
                    (sid, pdir),
                )
        else:
            placeholders = ",".join("?" for _ in session_keys)
            conn.execute(
                f"UPDATE sessions SET processed = 1 WHERE session_id IN ({placeholders})",
                list(session_keys),
            )
        conn.commit()
        logger.info("Marked %d sessions as processed", len(session_keys))

    def get_session_content(self, session_id: str, project_dir: str | None = None) -> bytes | None:
        """Read and decompress a session file. Returns None if not found.

        If project_dir is given, looks up by composite key (session_id, project_dir).
        Otherwise falls back to session_id only (raises if ambiguous).
        """
        conn = self._get_conn()
        if project_dir is not None:
            row = conn.execute(
                "SELECT project_dir FROM sessions WHERE session_id = ? AND project_dir = ?",
                (session_id, project_dir),
            ).fetchone()
        else:
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

    def list_all_sessions(self) -> list[dict]:
        """Return all sessions with metadata for sync manifest.

        Returns list of dicts with session_id, project_dir, jsonl_sha256.
        """
        conn = self._get_conn()
        rows = conn.execute("SELECT session_id, project_dir, jsonl_sha256 FROM sessions ORDER BY session_id").fetchall()
        return [dict(row) for row in rows]

    def get_session_file_path(self, session_id: str, project_dir: str) -> Path | None:
        """Return the on-disk path to a compressed session file, or None if missing."""
        gz_path = self.sessions_dir / project_dir / f"{session_id}.jsonl.gz"
        return gz_path if gz_path.exists() else None

    @classmethod
    def bootstrap_from_archive(
        cls,
        db_path: Path,
        sessions_dir: Path,
        archive_dir: Path,
    ) -> int:
        """Import sessions from an archive directory into a fresh store.

        Walks archive_dir for .jsonl.gz files, decompresses, and submits each.
        Uses SHA-256 dedup so duplicate content is skipped automatically.
        All imported sessions start as processed=0 so the first build reprocesses.

        Args:
            db_path: Path for the new SQLite database
            sessions_dir: Directory for session files
            archive_dir: Archive directory to import from

        Returns:
            Number of sessions imported (excluding duplicates)
        """
        store = cls(db_path=db_path, sessions_dir=sessions_dir)
        imported = 0

        if not archive_dir.exists():
            logger.warning("Archive directory does not exist: %s", archive_dir)
            return 0

        for gz_file in sorted(archive_dir.rglob("*.jsonl.gz")):
            try:
                content = gzip.decompress(gz_file.read_bytes())
            except Exception:
                logger.warning("Failed to decompress archive file %s, skipping", gz_file, exc_info=True)
                continue

            # Derive session_id and project_dir from path
            rel = gz_file.relative_to(archive_dir)
            session_id = gz_file.stem.replace(".jsonl", "")  # strip .jsonl from .jsonl.gz
            project_dir = str(rel.parent) if rel.parent != Path(".") else "default"

            is_new = store.submit(session_id, project_dir, content, "bootstrap")
            if is_new:
                imported += 1

        logger.info("Bootstrapped %d sessions from archive %s", imported, archive_dir)
        store.close()
        return imported

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
