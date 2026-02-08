"""Hybrid retrieval — keyword, semantic, and combined search modes."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from synix.build.provenance import ProvenanceTracker
from synix.search.embeddings import EmbeddingProvider
from synix.search.indexer import SearchIndex
from synix.search.results import SearchResult


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Uses pure Python to avoid a hard dependency on numpy.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class HybridRetriever:
    """Combines keyword (FTS5) and semantic (embedding) search with RRF fusion.

    Three modes:
    - ``keyword``: FTS5 full-text search only.
    - ``semantic``: Embedding cosine-similarity search only.
    - ``hybrid``: Both keyword and semantic results fused via Reciprocal Rank Fusion.
    """

    # RRF constant (k) — controls how much rank position matters.
    # Typical values are 10-60; 60 is the standard from the RRF paper.
    RRF_K = 60

    def __init__(
        self,
        search_index: SearchIndex,
        embedding_provider: EmbeddingProvider | None = None,
        provenance_tracker: ProvenanceTracker | None = None,
    ):
        self.search_index = search_index
        self.embedding_provider = embedding_provider
        self.provenance_tracker = provenance_tracker
        # In-memory cache of artifact embeddings for semantic search.
        # Keyed by content hash -> embedding vector.  Populated lazily.
        self._artifact_embeddings: dict[str, list[float]] = {}

    def _get_keyword_results(
        self, query: str, layers: list[str] | None = None
    ) -> list[SearchResult]:
        """Run keyword (FTS5) search."""
        return self.search_index.query(
            query,
            layers=layers,
            provenance_tracker=self.provenance_tracker,
        )

    def _get_all_indexed_rows(
        self, layers: list[str] | None = None
    ) -> list[dict]:
        """Fetch all rows from the search index, optionally filtered by layer.

        Returns a list of dicts with keys: content, artifact_id, layer_name,
        layer_level, metadata.
        """
        import json

        conn = self.search_index._get_conn()
        if layers:
            placeholders = ",".join("?" for _ in layers)
            sql = (
                f"SELECT content, artifact_id, layer_name, layer_level, metadata "
                f"FROM search_index WHERE layer_name IN ({placeholders})"
            )
            rows = conn.execute(sql, layers).fetchall()
        else:
            sql = "SELECT content, artifact_id, layer_name, layer_level, metadata FROM search_index"
            rows = conn.execute(sql).fetchall()

        result = []
        for row in rows:
            result.append({
                "content": row["content"],
                "artifact_id": row["artifact_id"],
                "layer_name": row["layer_name"],
                "layer_level": int(row["layer_level"]),
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            })
        return result

    @staticmethod
    def _content_hash(content: str) -> str:
        """Compute a SHA-256 hash of content for cache keying."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _ensure_embeddings_loaded(self, layers: list[str] | None = None) -> dict[str, list[float]]:
        """Ensure all indexed rows have their embeddings cached in memory.

        For each row, computes a content hash and checks the in-memory cache.
        Only calls ``embed_batch`` for content that is not already cached.

        Returns:
            Mapping of content hash -> embedding vector (the cache itself).
        """
        if self.embedding_provider is None:
            return self._artifact_embeddings

        rows = self._get_all_indexed_rows(layers)
        if not rows:
            return self._artifact_embeddings

        # Find content that still needs embedding
        uncached_contents: list[str] = []
        uncached_hashes: list[str] = []
        for row in rows:
            h = self._content_hash(row["content"])
            if h not in self._artifact_embeddings:
                uncached_contents.append(row["content"])
                uncached_hashes.append(h)

        # Batch-embed only the uncached content
        if uncached_contents:
            new_embeddings = self.embedding_provider.embed_batch(uncached_contents)
            for h, emb in zip(uncached_hashes, new_embeddings):
                self._artifact_embeddings[h] = emb

        return self._artifact_embeddings

    def clear_cache(self) -> None:
        """Reset the in-memory embedding cache.

        Useful after index rebuilds when indexed content may have changed.
        """
        self._artifact_embeddings = {}

    def _get_semantic_results(
        self, query: str, layers: list[str] | None = None, top_k: int = 10
    ) -> list[SearchResult]:
        """Run semantic search using embeddings and cosine similarity."""
        if self.embedding_provider is None:
            return []

        query_embedding = self.embedding_provider.embed(query)

        # Ensure all indexed rows have their embeddings cached
        self._ensure_embeddings_loaded(layers)

        # Get all indexed rows
        rows = self._get_all_indexed_rows(layers)
        if not rows:
            return []

        # Score each document by cosine similarity using cached embeddings
        scored: list[tuple[float, dict]] = []
        for row in rows:
            h = self._content_hash(row["content"])
            emb = self._artifact_embeddings.get(h)
            if emb is None:
                continue
            sim = _cosine_similarity(query_embedding, emb)
            scored.append((sim, row))

        # Sort by similarity descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build SearchResults with provenance
        results: list[SearchResult] = []
        for sim, row in scored[:top_k]:
            chain: list[str] = []
            if self.provenance_tracker is not None:
                records = self.provenance_tracker.get_chain(row["artifact_id"])
                chain = [r.artifact_id for r in records]

            results.append(SearchResult(
                content=row["content"],
                artifact_id=row["artifact_id"],
                layer_name=row["layer_name"],
                layer_level=row["layer_level"],
                score=sim,
                provenance_chain=chain,
                metadata=row["metadata"],
            ))

        return results

    def _rrf_fuse(
        self,
        keyword_results: list[SearchResult],
        semantic_results: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Fuse keyword and semantic results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) across result lists.
        Higher RRF score = more relevant.
        """
        k = self.RRF_K

        # Map artifact_id -> (rrf_score, SearchResult)
        scores: dict[str, float] = {}
        result_map: dict[str, SearchResult] = {}

        for rank, r in enumerate(keyword_results, start=1):
            scores[r.artifact_id] = scores.get(r.artifact_id, 0.0) + 1.0 / (k + rank)
            result_map[r.artifact_id] = r

        for rank, r in enumerate(semantic_results, start=1):
            scores[r.artifact_id] = scores.get(r.artifact_id, 0.0) + 1.0 / (k + rank)
            # Prefer the semantic result object if not already present (it has sim score)
            if r.artifact_id not in result_map:
                result_map[r.artifact_id] = r

        # Sort by RRF score descending
        sorted_ids = sorted(scores, key=lambda aid: scores[aid], reverse=True)

        results: list[SearchResult] = []
        for aid in sorted_ids[:top_k]:
            base = result_map[aid]
            results.append(SearchResult(
                content=base.content,
                artifact_id=base.artifact_id,
                layer_name=base.layer_name,
                layer_level=base.layer_level,
                score=scores[aid],
                provenance_chain=base.provenance_chain,
                metadata=base.metadata,
            ))

        return results

    def query(
        self,
        q: str,
        mode: str = "hybrid",
        layers: list[str] | None = None,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search with the specified mode.

        Args:
            q: Search query text.
            mode: One of "keyword", "semantic", or "hybrid".
            layers: Optional list of layer names to filter results.
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult objects, sorted by relevance.

        Raises:
            ValueError: If mode is not one of the valid options.
            ValueError: If semantic/hybrid mode requested without embedding provider.
        """
        if mode not in ("keyword", "semantic", "hybrid"):
            raise ValueError(
                f"Invalid search mode: {mode!r}. Must be 'keyword', 'semantic', or 'hybrid'."
            )

        if mode == "keyword":
            results = self._get_keyword_results(q, layers)
            return results[:top_k]

        if mode in ("semantic", "hybrid") and self.embedding_provider is None:
            if mode == "semantic":
                raise ValueError(
                    "Semantic search requires an embedding provider. "
                    "Configure embedding_config in your pipeline."
                )
            # Hybrid without embeddings falls back to keyword
            results = self._get_keyword_results(q, layers)
            return results[:top_k]

        if mode == "semantic":
            return self._get_semantic_results(q, layers, top_k)

        # Hybrid mode
        keyword_results = self._get_keyword_results(q, layers)
        semantic_results = self._get_semantic_results(q, layers, top_k=top_k * 2)
        return self._rrf_fuse(keyword_results, semantic_results, top_k)
