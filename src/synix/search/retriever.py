"""Hybrid retrieval — keyword, semantic, and combined search modes."""

from __future__ import annotations

import math

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

    Four modes:
    - ``keyword``: FTS5 full-text search only.
    - ``semantic``: Embedding cosine-similarity search only.
    - ``hybrid``: Both keyword and semantic results fused via Reciprocal Rank Fusion.
    - ``layered``: Like hybrid, but boosts higher-level layers in semantic scoring.
    """

    # RRF constant (k) — controls how much rank position matters.
    # Typical values are 10-60; 60 is the standard from the RRF paper.
    RRF_K = 60

    # Minimum cosine similarity for semantic results. Results below this
    # threshold are filtered out to avoid returning noise matches.
    MIN_SEMANTIC_SCORE = 0.35

    # Layer boost factor for "layered" mode.  A value of 1.0 means the
    # highest layer gets a 2× multiplier while the lowest non-zero layer
    # gets a (1 + 1/max_level)× multiplier.
    LAYER_BOOST = 1.0

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

    def _get_keyword_results(self, query: str, layers: list[str] | None = None) -> list[SearchResult]:
        """Run keyword (FTS5) search."""
        return self.search_index.query(
            query,
            layers=layers,
            provenance_tracker=self.provenance_tracker,
        )

    def _get_all_indexed_rows(self, layers: list[str] | None = None) -> list[dict]:
        """Fetch all rows from the search index, optionally filtered by layer.

        Returns a list of dicts with keys: content, label, layer_name,
        layer_level, metadata.
        """
        import json

        conn = self.search_index._get_conn()
        if layers:
            placeholders = ",".join("?" for _ in layers)
            sql = (
                f"SELECT content, label, layer_name, layer_level, metadata "
                f"FROM search_index WHERE layer_name IN ({placeholders})"
            )
            rows = conn.execute(sql, layers).fetchall()
        else:
            sql = "SELECT content, label, layer_name, layer_level, metadata FROM search_index"
            rows = conn.execute(sql).fetchall()

        result = []
        for row in rows:
            result.append(
                {
                    "content": row["content"],
                    "label": row["label"],
                    "layer_name": row["layer_name"],
                    "layer_level": int(row["layer_level"]),
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                }
            )
        return result

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
            h = self.embedding_provider.content_hash(row["content"])
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
        self,
        query: str,
        layers: list[str] | None = None,
        top_k: int = 10,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Run semantic search using embeddings and cosine similarity.

        Args:
            query: Search query text.
            layers: Optional layer name filter.
            top_k: Maximum number of results.
            min_score: Minimum cosine similarity threshold.  Defaults to
                ``MIN_SEMANTIC_SCORE`` when *None*.
        """
        if self.embedding_provider is None:
            return []

        if min_score is None:
            min_score = self.MIN_SEMANTIC_SCORE

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
            h = self.embedding_provider.content_hash(row["content"])
            emb = self._artifact_embeddings.get(h)
            if emb is None:
                continue
            sim = _cosine_similarity(query_embedding, emb)
            if sim >= min_score:
                scored.append((sim, row))

        # Sort by similarity descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build SearchResults with provenance
        results: list[SearchResult] = []
        for sim, row in scored[:top_k]:
            chain: list[str] = []
            if self.provenance_tracker is not None:
                records = self.provenance_tracker.get_chain(row["label"])
                chain = [r.label for r in records]

            results.append(
                SearchResult(
                    content=row["content"],
                    label=row["label"],
                    layer_name=row["layer_name"],
                    layer_level=row["layer_level"],
                    score=sim,
                    provenance_chain=chain,
                    metadata=row["metadata"],
                    search_mode="semantic",
                    semantic_score=sim,
                )
            )

        return results

    def _rrf_fuse(
        self,
        keyword_results: list[SearchResult],
        semantic_results: list[SearchResult],
        top_k: int = 10,
        search_mode: str = "hybrid",
    ) -> list[SearchResult]:
        """Fuse keyword and semantic results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) across result lists.
        Higher RRF score = more relevant.
        """
        k = self.RRF_K

        # Map label -> (rrf_score, SearchResult)
        scores: dict[str, float] = {}
        result_map: dict[str, SearchResult] = {}
        keyword_scores: dict[str, float] = {}
        semantic_scores: dict[str, float] = {}

        for rank, r in enumerate(keyword_results, start=1):
            scores[r.label] = scores.get(r.label, 0.0) + 1.0 / (k + rank)
            result_map[r.label] = r
            if r.keyword_score is not None:
                keyword_scores[r.label] = r.keyword_score

        for rank, r in enumerate(semantic_results, start=1):
            scores[r.label] = scores.get(r.label, 0.0) + 1.0 / (k + rank)
            # Prefer the semantic result object if not already present (it has sim score)
            if r.label not in result_map:
                result_map[r.label] = r
            if r.semantic_score is not None:
                semantic_scores[r.label] = r.semantic_score

        # Sort by RRF score descending
        sorted_ids = sorted(scores, key=lambda aid: scores[aid], reverse=True)

        results: list[SearchResult] = []
        for aid in sorted_ids[:top_k]:
            base = result_map[aid]
            results.append(
                SearchResult(
                    content=base.content,
                    label=base.label,
                    layer_name=base.layer_name,
                    layer_level=base.layer_level,
                    score=scores[aid],
                    provenance_chain=base.provenance_chain,
                    metadata=base.metadata,
                    search_mode=search_mode,
                    keyword_score=keyword_scores.get(aid),
                    semantic_score=semantic_scores.get(aid),
                )
            )

        return results

    def _get_layered_results(
        self,
        query: str,
        layers: list[str] | None = None,
        top_k: int = 10,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Layer-weighted semantic search fused with keyword results via RRF.

        Boosts semantic scores for higher-level layers so that synthesised
        knowledge (core memory, monthly rollups) outweighs raw episodes when
        the cosine similarity is comparable.

        Formula per semantic result::

            weighted = similarity × (1 + LAYER_BOOST × level / max_level)

        The boosted semantic list is then RRF-fused with the keyword list.
        """
        if self.embedding_provider is None:
            # Fall back to keyword
            results = self._get_keyword_results(query, layers)
            return results[:top_k]

        if min_score is None:
            min_score = self.MIN_SEMANTIC_SCORE

        query_embedding = self.embedding_provider.embed(query)

        # Ensure embeddings are loaded
        self._ensure_embeddings_loaded(layers)

        rows = self._get_all_indexed_rows(layers)
        if not rows:
            return []

        # Determine max layer level for normalisation
        max_level = max((r["layer_level"] for r in rows), default=1) or 1

        # Score with layer boost
        scored: list[tuple[float, float, dict]] = []  # (weighted, raw_sim, row)
        for row in rows:
            h = self.embedding_provider.content_hash(row["content"])
            emb = self._artifact_embeddings.get(h)
            if emb is None:
                continue
            sim = _cosine_similarity(query_embedding, emb)
            if sim < min_score:
                continue
            level = row["layer_level"]
            boost = 1.0 + self.LAYER_BOOST * level / max_level
            scored.append((sim * boost, sim, row))

        # Sort by weighted score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build semantic SearchResult list (score = weighted, semantic_score = raw sim)
        semantic_results: list[SearchResult] = []
        for weighted, raw_sim, row in scored[: top_k * 2]:
            chain: list[str] = []
            if self.provenance_tracker is not None:
                records = self.provenance_tracker.get_chain(row["label"])
                chain = [r.label for r in records]

            semantic_results.append(
                SearchResult(
                    content=row["content"],
                    label=row["label"],
                    layer_name=row["layer_name"],
                    layer_level=row["layer_level"],
                    score=weighted,
                    provenance_chain=chain,
                    metadata=row["metadata"],
                    search_mode="layered",
                    semantic_score=raw_sim,
                )
            )

        # Fuse with keyword results
        keyword_results = self._get_keyword_results(query, layers)
        return self._rrf_fuse(keyword_results, semantic_results, top_k, search_mode="layered")

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
            mode: One of "keyword", "semantic", "hybrid", or "layered".
            layers: Optional list of layer names to filter results.
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult objects, sorted by relevance.

        Raises:
            ValueError: If mode is not one of the valid options.
            ValueError: If semantic/hybrid mode requested without embedding provider.
        """
        valid_modes = ("keyword", "semantic", "hybrid", "layered")
        if mode not in valid_modes:
            raise ValueError(f"Invalid search mode: {mode!r}. Must be one of {valid_modes!r}.")

        if mode == "keyword":
            results = self._get_keyword_results(q, layers)
            return results[:top_k]

        needs_embeddings = mode in ("semantic", "hybrid", "layered")
        if needs_embeddings and self.embedding_provider is None:
            if mode == "semantic":
                raise ValueError(
                    "Semantic search requires an embedding provider. Configure embedding_config in your pipeline."
                )
            # Hybrid/layered without embeddings falls back to keyword
            results = self._get_keyword_results(q, layers)
            return results[:top_k]

        if mode == "semantic":
            return self._get_semantic_results(q, layers, top_k)

        if mode == "layered":
            return self._get_layered_results(q, layers, top_k)

        # Hybrid mode
        keyword_results = self._get_keyword_results(q, layers)
        semantic_results = self._get_semantic_results(q, layers, top_k=top_k * 2)
        return self._rrf_fuse(keyword_results, semantic_results, top_k)
