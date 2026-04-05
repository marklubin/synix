"""Versioned prompt template storage backed by SQLite.

Standalone subsystem colocated in .synix/prompts.db. Independent of
synix core — does NOT modify Transform.load_prompt() or any core models.
The server manages it; the viewer exposes a UI for it.

Every edit creates a new version (append-only). Content-hash dedup
prevents version inflation when content hasn't changed.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS prompts (
    key TEXT NOT NULL,
    version INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (key, version)
);
CREATE INDEX IF NOT EXISTS idx_prompts_key ON prompts(key);
"""


class PromptStore:
    """Versioned prompt template storage.

    Each prompt is identified by a string key. Every call to ``put()``
    that changes the content creates a new version. Reads default to the
    latest version.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(_SCHEMA)
        logger.debug("PromptStore opened at %s", self._db_path)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, key: str, version: int | None = None) -> str | None:
        """Return prompt content by key.  Latest version if *version* is None."""
        if version is not None:
            row = self._conn.execute(
                "SELECT content FROM prompts WHERE key = ? AND version = ?",
                (key, version),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT content FROM prompts WHERE key = ? ORDER BY version DESC LIMIT 1",
                (key,),
            ).fetchone()
        return row["content"] if row else None

    def get_with_meta(self, key: str, version: int | None = None) -> dict | None:
        """Return prompt with version metadata, or None if not found."""
        if version is not None:
            row = self._conn.execute(
                "SELECT key, version, content, content_hash, created_at "
                "FROM prompts WHERE key = ? AND version = ?",
                (key, version),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT key, version, content, content_hash, created_at "
                "FROM prompts WHERE key = ? ORDER BY version DESC LIMIT 1",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "key": row["key"],
            "version": row["version"],
            "content": row["content"],
            "content_hash": row["content_hash"],
            "created_at": row["created_at"],
        }

    def list_keys(self) -> list[str]:
        """Return all unique prompt keys, sorted alphabetically."""
        rows = self._conn.execute(
            "SELECT DISTINCT key FROM prompts ORDER BY key"
        ).fetchall()
        return [r["key"] for r in rows]

    def history(self, key: str) -> list[dict]:
        """Return version history for a key, newest first."""
        rows = self._conn.execute(
            "SELECT key, version, content_hash, created_at "
            "FROM prompts WHERE key = ? ORDER BY version DESC",
            (key,),
        ).fetchall()
        return [
            {
                "key": r["key"],
                "version": r["version"],
                "content_hash": r["content_hash"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def content_hash(self, key: str) -> str | None:
        """Return content hash of latest version, or None if key doesn't exist."""
        row = self._conn.execute(
            "SELECT content_hash FROM prompts WHERE key = ? ORDER BY version DESC LIMIT 1",
            (key,),
        ).fetchone()
        return row["content_hash"] if row else None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def put(self, key: str, content: str) -> dict:
        """Create a new version of a prompt.

        No-op if content_hash matches the latest version (returns existing
        metadata without inserting a duplicate row).

        Returns dict with key, version, content_hash, created_at.
        """
        new_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Check if latest version has the same hash — skip if so
        existing = self.get_with_meta(key)
        if existing is not None and existing["content_hash"] == new_hash:
            logger.debug("Prompt %r unchanged (hash %s), skipping", key, new_hash)
            return existing

        # Compute next version number
        row = self._conn.execute(
            "SELECT MAX(version) as max_ver FROM prompts WHERE key = ?",
            (key,),
        ).fetchone()
        next_version = (row["max_ver"] or 0) + 1

        now = datetime.now(UTC).isoformat()

        with self._conn:
            self._conn.execute(
                "INSERT INTO prompts (key, version, content, content_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (key, next_version, content, new_hash, now),
            )

        logger.info("Prompt %r updated to v%d (hash %s)", key, next_version, new_hash)
        return {
            "key": key,
            "version": next_version,
            "content_hash": new_hash,
            "created_at": now,
        }

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed_from_files(self, prompts_dir: Path) -> int:
        """Import .txt files from a directory as version 1.

        Only imports keys that don't already exist in the store.
        Returns the count of prompts imported.
        """
        prompts_dir = Path(prompts_dir)
        if not prompts_dir.is_dir():
            logger.warning("Prompts directory does not exist: %s", prompts_dir)
            return 0

        existing_keys = set(self.list_keys())
        imported = 0

        for txt_file in sorted(prompts_dir.glob("*.txt")):
            key = txt_file.stem  # e.g. "episode_summary" from "episode_summary.txt"
            if key in existing_keys:
                logger.debug("Prompt %r already exists, skipping seed", key)
                continue

            content = txt_file.read_text(encoding="utf-8")
            self.put(key, content)
            imported += 1
            logger.info("Seeded prompt %r from %s", key, txt_file.name)

        if imported:
            logger.info("Seeded %d prompts from %s", imported, prompts_dir)
        return imported

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
