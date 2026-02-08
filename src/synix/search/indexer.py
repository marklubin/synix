"""SQLite FTS5 search index — build, query, projection, and shadow swap."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from pathlib import Path

from synix.core.config import EmbeddingConfig
from synix.core.models import Artifact
from synix.build.provenance import ProvenanceTracker
from synix.build.projections import BaseProjection, register_projection
from synix.search.results import SearchResult

logger = logging.getLogger(__name__)


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


class ShadowIndexManager:
    """Manages building a search index into a shadow file and atomically swapping.

    Usage:
        manager = ShadowIndexManager(build_dir)
        shadow = manager.begin_build()
        # ... populate shadow index ...
        manager.commit()  # atomic swap shadow -> main

    On failure, call manager.rollback() to clean up the shadow file
    and leave the old index unchanged.
    """

    def __init__(self, build_dir: str | Path):
        self.build_dir = Path(build_dir)
        self.main_path = self.build_dir / "search.db"
        self.shadow_path = self.build_dir / "search_shadow.db"
        self._shadow_index: SearchIndex | None = None

    def begin_build(self) -> SearchIndex:
        """Start building a new shadow index.

        Returns a SearchIndex pointing to the shadow file. The caller
        populates it, then calls commit() or rollback().
        """
        # Remove stale shadow from a previous failed build
        if self.shadow_path.exists():
            self.shadow_path.unlink()

        self._shadow_index = SearchIndex(self.shadow_path)
        self._shadow_index.create()
        return self._shadow_index

    def commit(self) -> None:
        """Atomically swap the shadow index into the main path.

        Closes the shadow connection, then uses os.replace() for an
        atomic rename on POSIX systems.
        """
        if self._shadow_index is None:
            raise RuntimeError("No shadow build in progress — call begin_build() first")

        self._shadow_index.close()
        self._shadow_index = None

        # os.replace is atomic on POSIX when src and dst are on the same filesystem
        os.replace(str(self.shadow_path), str(self.main_path))

    def rollback(self) -> None:
        """Discard the shadow index and leave the old index unchanged."""
        if self._shadow_index is not None:
            self._shadow_index.close()
            self._shadow_index = None

        if self.shadow_path.exists():
            self.shadow_path.unlink()


@register_projection("search_index")
class SearchIndexProjection(BaseProjection):
    """Materializes artifacts into a SQLite FTS5 search index.

    Uses shadow index pattern: builds into a temporary file and atomically
    swaps on success, so the old index remains available during builds and
    is preserved if the build fails.
    """

    def __init__(self, build_dir: str | Path):
        self.build_dir = Path(build_dir)
        self.db_path = self.build_dir / "search.db"
        self._index: SearchIndex | None = None

    def _get_index(self) -> SearchIndex:
        if self._index is None:
            self._index = SearchIndex(self.db_path)
        return self._index

    def materialize(self, artifacts: list[Artifact], config: dict) -> None:
        """Populate FTS5 from artifacts using shadow index pattern.

        Builds into search_shadow.db, then atomically swaps to search.db
        on success. On failure, the old search.db is preserved unchanged.

        If ``embedding_config`` is present in config, also generates and
        caches embeddings for all indexed artifacts.
        """
        manager = ShadowIndexManager(self.build_dir)
        shadow_index = manager.begin_build()

        sources = config.get("sources", [])
        layer_levels = {s["layer"]: s.get("level", 0) for s in sources}
        source_layers = set(layer_levels.keys())

        indexed_artifacts: list[Artifact] = []

        try:
            for artifact in artifacts:
                layer_name = artifact.metadata.get("layer_name", "")
                if layer_name in source_layers:
                    shadow_index.insert(artifact, layer_name, layer_levels[layer_name])
                    indexed_artifacts.append(artifact)

            manager.commit()
        except Exception:
            manager.rollback()
            raise

        # Invalidate any cached connection to the old db file
        if self._index is not None:
            self._index.close()
            self._index = None

        # Optionally generate embeddings for indexed artifacts
        embedding_config = config.get("embedding_config")
        if embedding_config is not None and indexed_artifacts:
            self._generate_embeddings(embedding_config, indexed_artifacts)

    def _generate_embeddings(
        self,
        embedding_config: EmbeddingConfig | dict,
        artifacts: list[Artifact],
    ) -> None:
        """Generate and cache embeddings for the given artifacts."""
        from synix.search.embeddings import EmbeddingProvider

        if isinstance(embedding_config, dict):
            embedding_config = EmbeddingConfig.from_dict(embedding_config)

        provider = EmbeddingProvider(embedding_config, self.build_dir)
        texts = [a.content for a in artifacts]
        try:
            provider.embed_batch(texts)
            logger.info("Generated embeddings for %d artifacts", len(artifacts))
        except Exception:
            # Embedding generation is best-effort; don't fail the whole build
            logger.warning(
                "Failed to generate embeddings for %d artifacts. "
                "Keyword search will still work.",
                len(artifacts),
                exc_info=True,
            )

    def query(
        self,
        q: str,
        layers: list[str] | None = None,
        provenance_tracker: ProvenanceTracker | None = None,
    ) -> list[SearchResult]:
        """Query the search index with optional layer filtering and provenance."""
        index = self._get_index()
        return index.query(q, layers=layers, provenance_tracker=provenance_tracker)

    def close(self) -> None:
        """Close the underlying search index."""
        if self._index is not None:
            self._index.close()
            self._index = None
