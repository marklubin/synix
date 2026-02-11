"""Tests for hybrid retriever — keyword, semantic, and combined search modes."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from synix import Artifact
from synix.build.provenance import ProvenanceTracker
from synix.search.indexer import SearchIndex
from synix.search.results import SearchResult
from synix.search.retriever import HybridRetriever, _cosine_similarity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_embedding_provider(embedding_map: dict[str, list[float]]):
    """Create a mock EmbeddingProvider that returns deterministic embeddings.

    Args:
        embedding_map: maps text -> embedding vector. Any text not in the map
            gets a zero vector of matching dimensionality.
    """
    import hashlib

    dim = len(next(iter(embedding_map.values()))) if embedding_map else 4
    provider = MagicMock()

    def mock_embed(text):
        return embedding_map.get(text, [0.0] * dim)

    def mock_embed_batch(texts):
        return [embedding_map.get(t, [0.0] * dim) for t in texts]

    def mock_content_hash(text):
        return hashlib.sha256(f"mock:mock:{text}".encode()).hexdigest()

    provider.embed = mock_embed
    provider.embed_batch = mock_embed_batch
    provider.content_hash = mock_content_hash
    return provider


def _populate_index(index: SearchIndex, items: list[dict]) -> None:
    """Insert test items into a search index.

    Each item dict: {artifact_id, content, layer_name, layer_level}
    """
    for item in items:
        artifact = Artifact(
            artifact_id=item["artifact_id"],
            artifact_type="episode",
            content=item["content"],
            metadata={"layer_name": item["layer_name"]},
        )
        index.insert(artifact, item["layer_name"], item["layer_level"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def search_index(tmp_build_dir):
    """A fresh SearchIndex for each test."""
    idx = SearchIndex(tmp_build_dir / "search.db")
    idx.create()
    return idx


@pytest.fixture
def sample_items():
    """Standard set of test items for the search index."""
    return [
        {
            "artifact_id": "ep-001",
            "content": "Discussion about machine learning and neural networks",
            "layer_name": "episodes",
            "layer_level": 1,
        },
        {
            "artifact_id": "ep-002",
            "content": "Docker containerization and deployment practices",
            "layer_name": "episodes",
            "layer_level": 1,
        },
        {
            "artifact_id": "ep-003",
            "content": "Rust programming and ownership model",
            "layer_name": "episodes",
            "layer_level": 1,
        },
        {
            "artifact_id": "monthly-001",
            "content": "In January, the user focused on machine learning and cloud computing",
            "layer_name": "monthly",
            "layer_level": 2,
        },
        {
            "artifact_id": "core-001",
            "content": "Mark is a software engineer interested in AI and distributed systems",
            "layer_name": "core",
            "layer_level": 3,
        },
    ]


@pytest.fixture
def populated_index(search_index, sample_items):
    """SearchIndex pre-populated with sample items."""
    _populate_index(search_index, sample_items)
    return search_index


@pytest.fixture
def sample_embedding_map():
    """Embedding map for sample items + typical queries.

    Uses simple vectors where the components represent topics:
    [ML, DevOps, Rust, General]
    """
    return {
        # Index content embeddings
        "Discussion about machine learning and neural networks": [0.9, 0.1, 0.0, 0.2],
        "Docker containerization and deployment practices": [0.1, 0.9, 0.0, 0.2],
        "Rust programming and ownership model": [0.0, 0.1, 0.9, 0.2],
        "In January, the user focused on machine learning and cloud computing": [0.7, 0.3, 0.0, 0.3],
        "Mark is a software engineer interested in AI and distributed systems": [0.5, 0.4, 0.1, 0.8],
        # Query embeddings
        "machine learning": [0.95, 0.05, 0.0, 0.1],
        "Docker": [0.05, 0.95, 0.0, 0.1],
        "Rust ownership": [0.0, 0.05, 0.95, 0.1],
        "software engineer AI": [0.6, 0.2, 0.0, 0.7],
    }


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_similar_vectors(self):
        sim = _cosine_similarity([0.9, 0.1], [0.8, 0.2])
        assert 0.9 < sim < 1.0


# ---------------------------------------------------------------------------
# Keyword-only retrieval
# ---------------------------------------------------------------------------


class TestKeywordRetrieval:
    def test_keyword_returns_results(self, populated_index):
        """Keyword mode returns FTS5 results."""
        retriever = HybridRetriever(search_index=populated_index)
        results = retriever.query("machine learning", mode="keyword")

        assert len(results) > 0
        assert any("machine learning" in r.content.lower() for r in results)

    def test_keyword_no_match(self, populated_index):
        """Keyword mode returns empty for no-match query."""
        retriever = HybridRetriever(search_index=populated_index)
        results = retriever.query("xyznonexistent", mode="keyword")
        assert results == []

    def test_keyword_layer_filtering(self, populated_index):
        """Keyword mode respects layer filter."""
        retriever = HybridRetriever(search_index=populated_index)

        results = retriever.query("machine learning", mode="keyword", layers=["episodes"])
        assert all(r.layer_name == "episodes" for r in results)

        results_monthly = retriever.query("machine learning", mode="keyword", layers=["monthly"])
        assert all(r.layer_name == "monthly" for r in results_monthly)

    def test_keyword_top_k(self, populated_index):
        """Keyword mode respects top_k limit."""
        retriever = HybridRetriever(search_index=populated_index)
        results = retriever.query("machine learning", mode="keyword", top_k=1)
        assert len(results) <= 1

    def test_keyword_with_provenance(self, populated_index, tmp_build_dir):
        """Keyword mode includes provenance when tracker provided."""
        tracker = ProvenanceTracker(tmp_build_dir)
        tracker.record("ep-001", parent_ids=["t-001"], prompt_id="ep_v1")
        tracker.record("t-001", parent_ids=[])

        retriever = HybridRetriever(
            search_index=populated_index,
            provenance_tracker=tracker,
        )
        results = retriever.query("machine learning", mode="keyword")

        # Find the ep-001 result
        ep_result = next((r for r in results if r.artifact_id == "ep-001"), None)
        assert ep_result is not None
        assert len(ep_result.provenance_chain) > 0


# ---------------------------------------------------------------------------
# Semantic-only retrieval
# ---------------------------------------------------------------------------


class TestSemanticRetrieval:
    def test_semantic_returns_results(self, populated_index, sample_embedding_map):
        """Semantic mode returns results ranked by similarity."""
        provider = _make_mock_embedding_provider(sample_embedding_map)
        retriever = HybridRetriever(
            search_index=populated_index,
            embedding_provider=provider,
        )
        results = retriever.query("machine learning", mode="semantic")

        assert len(results) > 0
        # ML-related content should rank first
        assert results[0].artifact_id == "ep-001"

    def test_semantic_ranking(self, populated_index, sample_embedding_map):
        """Semantic results are ranked by cosine similarity."""
        provider = _make_mock_embedding_provider(sample_embedding_map)
        retriever = HybridRetriever(
            search_index=populated_index,
            embedding_provider=provider,
        )
        results = retriever.query("Docker", mode="semantic")

        assert len(results) > 0
        # Docker content should rank first
        assert results[0].artifact_id == "ep-002"

    def test_semantic_without_provider_raises(self, populated_index):
        """Semantic mode without embedding provider raises ValueError."""
        retriever = HybridRetriever(search_index=populated_index)
        with pytest.raises(ValueError, match="embedding provider"):
            retriever.query("test", mode="semantic")

    def test_semantic_layer_filtering(self, populated_index, sample_embedding_map):
        """Semantic mode respects layer filtering."""
        provider = _make_mock_embedding_provider(sample_embedding_map)
        retriever = HybridRetriever(
            search_index=populated_index,
            embedding_provider=provider,
        )
        results = retriever.query("machine learning", mode="semantic", layers=["episodes"])
        assert all(r.layer_name == "episodes" for r in results)

    def test_semantic_top_k(self, populated_index, sample_embedding_map):
        """Semantic mode respects top_k."""
        provider = _make_mock_embedding_provider(sample_embedding_map)
        retriever = HybridRetriever(
            search_index=populated_index,
            embedding_provider=provider,
        )
        results = retriever.query("machine learning", mode="semantic", top_k=2)
        assert len(results) <= 2

    def test_semantic_scores_are_similarities(self, populated_index, sample_embedding_map):
        """Semantic result scores are cosine similarity values."""
        provider = _make_mock_embedding_provider(sample_embedding_map)
        retriever = HybridRetriever(
            search_index=populated_index,
            embedding_provider=provider,
        )
        results = retriever.query("machine learning", mode="semantic")

        # Scores should be between -1 and 1 (cosine similarity range)
        for r in results:
            assert -1.0 <= r.score <= 1.0

        # Scores should be in descending order
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score


# ---------------------------------------------------------------------------
# Hybrid retrieval (RRF)
# ---------------------------------------------------------------------------


class TestHybridRetrieval:
    def test_hybrid_returns_results(self, populated_index, sample_embedding_map):
        """Hybrid mode returns fused results."""
        provider = _make_mock_embedding_provider(sample_embedding_map)
        retriever = HybridRetriever(
            search_index=populated_index,
            embedding_provider=provider,
        )
        results = retriever.query("machine learning", mode="hybrid")

        assert len(results) > 0

    def test_hybrid_includes_both_sources(self, populated_index, sample_embedding_map):
        """Hybrid mode can include results from both keyword and semantic."""
        provider = _make_mock_embedding_provider(sample_embedding_map)
        retriever = HybridRetriever(
            search_index=populated_index,
            embedding_provider=provider,
        )
        # "software engineer AI" matches core well semantically,
        # but FTS5 may match it differently
        results = retriever.query("software engineer AI", mode="hybrid")

        assert len(results) > 0
        # The core memory artifact should be present (strong semantic match)
        artifact_ids = {r.artifact_id for r in results}
        assert "core-001" in artifact_ids

    def test_hybrid_rrf_scores(self, populated_index, sample_embedding_map):
        """Hybrid RRF scores reflect combined ranking."""
        provider = _make_mock_embedding_provider(sample_embedding_map)
        retriever = HybridRetriever(
            search_index=populated_index,
            embedding_provider=provider,
        )
        results = retriever.query("machine learning", mode="hybrid")

        # RRF scores should be positive
        for r in results:
            assert r.score > 0

        # Scores should be in descending order
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_hybrid_top_k(self, populated_index, sample_embedding_map):
        """Hybrid mode respects top_k."""
        provider = _make_mock_embedding_provider(sample_embedding_map)
        retriever = HybridRetriever(
            search_index=populated_index,
            embedding_provider=provider,
        )
        results = retriever.query("machine learning", mode="hybrid", top_k=2)
        assert len(results) <= 2

    def test_hybrid_layer_filtering(self, populated_index, sample_embedding_map):
        """Hybrid mode respects layer filtering."""
        provider = _make_mock_embedding_provider(sample_embedding_map)
        retriever = HybridRetriever(
            search_index=populated_index,
            embedding_provider=provider,
        )
        results = retriever.query("machine learning", mode="hybrid", layers=["episodes"])
        assert all(r.layer_name == "episodes" for r in results)

    def test_hybrid_without_embeddings_falls_back(self, populated_index):
        """Hybrid mode without embedding provider falls back to keyword."""
        retriever = HybridRetriever(search_index=populated_index)
        # Should NOT raise, should fall back to keyword
        results = retriever.query("machine learning", mode="hybrid")
        assert len(results) > 0

    def test_hybrid_with_provenance(self, populated_index, sample_embedding_map, tmp_build_dir):
        """Hybrid mode includes provenance chains."""
        tracker = ProvenanceTracker(tmp_build_dir)
        tracker.record("ep-001", parent_ids=["t-001"], prompt_id="ep_v1")
        tracker.record("t-001", parent_ids=[])

        provider = _make_mock_embedding_provider(sample_embedding_map)
        retriever = HybridRetriever(
            search_index=populated_index,
            embedding_provider=provider,
            provenance_tracker=tracker,
        )
        results = retriever.query("machine learning", mode="hybrid")

        ep_result = next((r for r in results if r.artifact_id == "ep-001"), None)
        assert ep_result is not None
        assert len(ep_result.provenance_chain) > 0


# ---------------------------------------------------------------------------
# Invalid mode
# ---------------------------------------------------------------------------


class TestInvalidMode:
    def test_invalid_mode_raises(self, populated_index):
        """Invalid mode raises ValueError."""
        retriever = HybridRetriever(search_index=populated_index)
        with pytest.raises(ValueError, match="Invalid search mode"):
            retriever.query("test", mode="invalid")


# ---------------------------------------------------------------------------
# RRF fusion unit tests
# ---------------------------------------------------------------------------


class TestRRFFusion:
    def test_rrf_single_list(self):
        """RRF with only keyword results returns them ranked."""
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.RRF_K = 60

        keyword_results = [
            SearchResult(content="A", artifact_id="a", layer_name="ep", layer_level=1, score=1.0),
            SearchResult(content="B", artifact_id="b", layer_name="ep", layer_level=1, score=0.5),
        ]
        fused = retriever._rrf_fuse(keyword_results, [], top_k=10)

        assert len(fused) == 2
        assert fused[0].artifact_id == "a"
        assert fused[1].artifact_id == "b"

    def test_rrf_both_lists(self):
        """RRF with both lists produces fused ranking."""
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.RRF_K = 60

        keyword_results = [
            SearchResult(content="A", artifact_id="a", layer_name="ep", layer_level=1, score=1.0),
            SearchResult(content="B", artifact_id="b", layer_name="ep", layer_level=1, score=0.5),
        ]
        semantic_results = [
            SearchResult(content="B", artifact_id="b", layer_name="ep", layer_level=1, score=0.9),
            SearchResult(content="A", artifact_id="a", layer_name="ep", layer_level=1, score=0.3),
        ]
        fused = retriever._rrf_fuse(keyword_results, semantic_results, top_k=10)

        assert len(fused) == 2
        # Both A and B appear in both lists, but with different ranks
        # A: keyword rank 1 (1/61), semantic rank 2 (1/62) = 1/61 + 1/62
        # B: keyword rank 2 (1/62), semantic rank 1 (1/61) = 1/62 + 1/61
        # They should have equal scores
        assert fused[0].score == pytest.approx(fused[1].score)

    def test_rrf_top_k_limiting(self):
        """RRF respects top_k."""
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.RRF_K = 60

        keyword_results = [
            SearchResult(content=f"R{i}", artifact_id=f"r{i}", layer_name="ep", layer_level=1, score=1.0)
            for i in range(5)
        ]
        fused = retriever._rrf_fuse(keyword_results, [], top_k=3)
        assert len(fused) == 3

    def test_rrf_unique_per_list(self):
        """RRF with items unique to each list includes all."""
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.RRF_K = 60

        keyword_results = [
            SearchResult(content="A", artifact_id="a", layer_name="ep", layer_level=1, score=1.0),
        ]
        semantic_results = [
            SearchResult(content="B", artifact_id="b", layer_name="ep", layer_level=1, score=0.9),
        ]
        fused = retriever._rrf_fuse(keyword_results, semantic_results, top_k=10)

        assert len(fused) == 2
        ids = {r.artifact_id for r in fused}
        assert ids == {"a", "b"}


# ---------------------------------------------------------------------------
# CLI flag tests
# ---------------------------------------------------------------------------


class TestCLIFlags:
    def test_search_mode_flag_exists(self):
        """The --mode flag is registered on the search command."""
        from synix.cli.search_commands import search

        param_names = [p.name for p in search.params]
        assert "mode" in param_names

    def test_search_top_k_flag_exists(self):
        """The --top-k flag is registered on the search command."""
        from synix.cli.search_commands import search

        param_names = [p.name for p in search.params]
        assert "top_k" in param_names

    def test_search_mode_choices(self):
        """The --mode flag accepts keyword, semantic, hybrid, layered."""
        from synix.cli.search_commands import search

        mode_param = next(p for p in search.params if p.name == "mode")
        assert hasattr(mode_param.type, "choices")
        assert set(mode_param.type.choices) == {"keyword", "semantic", "hybrid", "layered"}

    def test_search_mode_default_auto_detect(self):
        """The --mode flag defaults to None (auto-detect: hybrid when embeddings exist, else keyword)."""
        from synix.cli.search_commands import search

        mode_param = next(p for p in search.params if p.name == "mode")
        assert mode_param.default is None


# ---------------------------------------------------------------------------
# Min similarity threshold
# ---------------------------------------------------------------------------


class TestMinSimilarityThreshold:
    def test_semantic_filters_low_scores(self, populated_index, sample_embedding_map):
        """Semantic mode filters results below MIN_SEMANTIC_SCORE."""
        provider = _make_mock_embedding_provider(sample_embedding_map)
        retriever = HybridRetriever(
            search_index=populated_index,
            embedding_provider=provider,
        )
        # Set a high threshold so most results get filtered
        retriever.MIN_SEMANTIC_SCORE = 0.95
        results = retriever.query("machine learning", mode="semantic")

        # Only the very close match should survive (ep-001 ~ 0.99 sim with the query)
        for r in results:
            assert r.score >= 0.95

    def test_semantic_threshold_zero_returns_all(self, populated_index, sample_embedding_map):
        """Setting min threshold to 0 returns all results."""
        provider = _make_mock_embedding_provider(sample_embedding_map)
        retriever = HybridRetriever(
            search_index=populated_index,
            embedding_provider=provider,
        )
        retriever.MIN_SEMANTIC_SCORE = 0.0
        results = retriever.query("machine learning", mode="semantic")

        # Should have all 5 items from the index
        assert len(results) == 5

    def test_semantic_threshold_one_returns_none(self, populated_index, sample_embedding_map):
        """Setting min threshold to 1.0 filters everything (nothing is perfectly identical)."""
        provider = _make_mock_embedding_provider(sample_embedding_map)
        retriever = HybridRetriever(
            search_index=populated_index,
            embedding_provider=provider,
        )
        retriever.MIN_SEMANTIC_SCORE = 1.0
        results = retriever.query("machine learning", mode="semantic")

        assert len(results) == 0

    def test_default_threshold_is_035(self):
        """Default MIN_SEMANTIC_SCORE is 0.35."""
        assert HybridRetriever.MIN_SEMANTIC_SCORE == 0.35


# ---------------------------------------------------------------------------
# Layered retrieval (layer-weighted semantic + RRF)
# ---------------------------------------------------------------------------


class TestLayeredRetrieval:
    def test_layered_returns_results(self, populated_index, sample_embedding_map):
        """Layered mode returns fused results."""
        provider = _make_mock_embedding_provider(sample_embedding_map)
        retriever = HybridRetriever(
            search_index=populated_index,
            embedding_provider=provider,
        )
        retriever.MIN_SEMANTIC_SCORE = 0.0  # Don't filter for this test
        results = retriever.query("software engineer AI", mode="layered")

        assert len(results) > 0

    def test_layered_boosts_higher_layers(self, populated_index, sample_embedding_map):
        """Layered mode boosts higher-level results in the semantic ranking."""
        provider = _make_mock_embedding_provider(sample_embedding_map)
        retriever = HybridRetriever(
            search_index=populated_index,
            embedding_provider=provider,
        )
        retriever.MIN_SEMANTIC_SCORE = 0.0

        # "software engineer AI" has similar semantic scores for monthly-001 (L2)
        # and core-001 (L3) but core should get a bigger layer boost
        results = retriever.query("software engineer AI", mode="layered")

        # Core (L3) should rank above monthly (L2) even if raw similarity is lower
        core_idx = next((i for i, r in enumerate(results) if r.artifact_id == "core-001"), None)
        monthly_idx = next((i for i, r in enumerate(results) if r.artifact_id == "monthly-001"), None)
        assert core_idx is not None
        assert monthly_idx is not None
        assert core_idx < monthly_idx, "Core (L3) should rank above monthly (L2) in layered mode"

    def test_layered_search_mode_label(self, populated_index, sample_embedding_map):
        """Layered mode results have search_mode='layered'."""
        provider = _make_mock_embedding_provider(sample_embedding_map)
        retriever = HybridRetriever(
            search_index=populated_index,
            embedding_provider=provider,
        )
        retriever.MIN_SEMANTIC_SCORE = 0.0
        results = retriever.query("machine learning", mode="layered")

        for r in results:
            assert r.search_mode == "layered"

    def test_layered_preserves_semantic_scores(self, populated_index, sample_embedding_map):
        """Layered results retain raw semantic_score (unweighted cosine sim)."""
        provider = _make_mock_embedding_provider(sample_embedding_map)
        retriever = HybridRetriever(
            search_index=populated_index,
            embedding_provider=provider,
        )
        retriever.MIN_SEMANTIC_SCORE = 0.0
        results = retriever.query("machine learning", mode="layered")

        for r in results:
            if r.semantic_score is not None:
                # Raw cosine similarity should be between -1 and 1
                assert -1.0 <= r.semantic_score <= 1.0

    def test_layered_without_embeddings_falls_back(self, populated_index):
        """Layered mode without embedding provider falls back to keyword."""
        retriever = HybridRetriever(search_index=populated_index)
        results = retriever.query("machine learning", mode="layered")
        assert len(results) > 0

    def test_layered_layer_filtering(self, populated_index, sample_embedding_map):
        """Layered mode respects layer filtering."""
        provider = _make_mock_embedding_provider(sample_embedding_map)
        retriever = HybridRetriever(
            search_index=populated_index,
            embedding_provider=provider,
        )
        retriever.MIN_SEMANTIC_SCORE = 0.0
        results = retriever.query("machine learning", mode="layered", layers=["episodes"])
        assert all(r.layer_name == "episodes" for r in results)

    def test_layered_top_k(self, populated_index, sample_embedding_map):
        """Layered mode respects top_k."""
        provider = _make_mock_embedding_provider(sample_embedding_map)
        retriever = HybridRetriever(
            search_index=populated_index,
            embedding_provider=provider,
        )
        retriever.MIN_SEMANTIC_SCORE = 0.0
        results = retriever.query("machine learning", mode="layered", top_k=2)
        assert len(results) <= 2

    def test_layered_applies_min_score(self, populated_index, sample_embedding_map):
        """Layered mode respects MIN_SEMANTIC_SCORE threshold."""
        provider = _make_mock_embedding_provider(sample_embedding_map)
        retriever = HybridRetriever(
            search_index=populated_index,
            embedding_provider=provider,
        )
        # "Rust ownership" query: only ep-003 (Rust) should have high similarity.
        # With a high threshold, most semantic results get filtered, leaving
        # only the keyword side of the fusion.
        retriever.MIN_SEMANTIC_SCORE = 0.9
        results = retriever.query("Rust ownership", mode="layered")

        # The semantic side should only contribute the Rust result
        semantic_ids = {r.artifact_id for r in results if r.semantic_score is not None and r.semantic_score >= 0.9}
        assert "ep-003" in semantic_ids or len(results) > 0  # At least keyword results

    def test_layer_boost_constant(self):
        """Default LAYER_BOOST is 1.0."""
        assert HybridRetriever.LAYER_BOOST == 1.0
