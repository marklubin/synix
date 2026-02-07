"""SQLite FTS5 search index — build and query."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from synix import Artifact
from synix.artifacts.provenance import ProvenanceTracker
from synix.search.results import SearchResult


class SearchIndex:
    """SQLite FTS5-backed search index."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def create(self) -> None:
        """Create the FTS5 search index table."""
        conn = self._get_conn()
        conn.execute("DROP TABLE IF EXISTS search_index")
        conn.execute("""
            CREATE VIRTUAL TABLE search_index USING fts5(
                content,
                artifact_id,
                layer_name,
                layer_level,
                metadata
            )
        """)
        conn.commit()

    def insert(self, artifact: Artifact, layer_name: str, layer_level: int) -> None:
        """Insert an artifact into the search index."""
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO search_index (content, artifact_id, layer_name, layer_level, metadata) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                artifact.content,
                artifact.artifact_id,
                layer_name,
                str(layer_level),
                json.dumps(artifact.metadata),
            ),
        )
        conn.commit()

    def query(
        self,
        q: str,
        layers: list[str] | None = None,
        provenance_tracker: ProvenanceTracker | None = None,
    ) -> list[SearchResult]:
        """Search with optional layer filtering and provenance chains."""
        conn = self._get_conn()

        if layers:
            placeholders = ",".join("?" for _ in layers)
            sql = (
                f"SELECT content, artifact_id, layer_name, layer_level, metadata, rank "
                f"FROM search_index "
                f"WHERE search_index MATCH ? AND layer_name IN ({placeholders}) "
                f"ORDER BY rank"
            )
            params = [q, *layers]
        else:
            sql = (
                "SELECT content, artifact_id, layer_name, layer_level, metadata, rank "
                "FROM search_index "
                "WHERE search_index MATCH ? "
                "ORDER BY rank"
            )
            params = [q]

        rows = conn.execute(sql, params).fetchall()

        results = []
        for row in rows:
            chain: list[str] = []
            if provenance_tracker is not None:
                records = provenance_tracker.get_chain(row["artifact_id"])
                chain = [r.artifact_id for r in records]

            metadata = json.loads(row["metadata"]) if row["metadata"] else {}

            results.append(SearchResult(
                content=row["content"],
                artifact_id=row["artifact_id"],
                layer_name=row["layer_name"],
                layer_level=int(row["layer_level"]),
                score=abs(row["rank"]),
                provenance_chain=chain,
                metadata=metadata,
            ))

        return results

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
