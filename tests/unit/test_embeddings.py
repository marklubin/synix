"""Tests for embedding generation and caching."""

from __future__ import annotations

import json
import struct
from unittest.mock import MagicMock

import pytest

from synix import Artifact
from synix.core.config import EmbeddingConfig
from synix.search.embeddings import EmbeddingProvider


class MockEmbeddingData:
    """Mock for a single embedding data item in OpenAI response."""

    def __init__(self, embedding: list[float], index: int = 0):
        self.embedding = embedding
        self.index = index


class MockEmbeddingResponse:
    """Mock for OpenAI embeddings.create() response."""

    def __init__(self, embeddings: list[list[float]]):
        self.data = [MockEmbeddingData(emb, idx) for idx, emb in enumerate(embeddings)]


def _deterministic_embedding(text: str) -> list[float]:
    """Generate a deterministic 4-d embedding from text."""
    n = len(text)
    return [float(n % 10) / 10, 0.5, 0.3, float(n % 5) / 10]


@pytest.fixture
def embedding_config():
    """Default embedding config for tests."""
    return EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-small",
        dimensions=4,  # small for testing
        api_key="test-key",
    )


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client that returns deterministic embeddings.

    Uses MagicMock with side_effect so call_count tracking works.
    """
    client = MagicMock()

    def _create_side_effect(**kwargs):
        texts = kwargs.get("input", [])
        embeddings = [_deterministic_embedding(t) for t in texts]
        return MockEmbeddingResponse(embeddings)

    client.embeddings.create = MagicMock(side_effect=_create_side_effect)
    return client


@pytest.fixture
def provider(embedding_config, tmp_build_dir, mock_openai_client):
    """EmbeddingProvider with mocked OpenAI client."""
    from synix.search.embeddings import OpenAIBackend

    p = EmbeddingProvider(embedding_config, tmp_build_dir)
    # Force an OpenAI backend and inject the mock client
    backend = OpenAIBackend(embedding_config)
    backend._client = mock_openai_client
    p._backend = backend
    return p


class TestEmbeddingProvider:
    def test_embed_returns_vector(self, provider):
        """embed() returns a list of floats."""
        result = provider.embed("hello world")
        assert isinstance(result, list)
        assert len(result) == 4
        assert all(isinstance(x, float) for x in result)

    def test_embed_caching_by_content_hash(self, provider):
        """Second call with same text uses cache, not API."""
        result1 = provider.embed("cached text")
        result2 = provider.embed("cached text")

        # Results should match (within float32 precision from serialization)
        for a, b in zip(result1, result2):
            assert a == pytest.approx(b, abs=1e-6)

        # The mock was called only once for this text
        assert provider._backend._client.embeddings.create.call_count == 1

    def test_embed_different_texts_call_api(self, provider):
        """Different texts each call the API."""
        provider.embed("text one")
        provider.embed("text two")

        assert provider._backend._client.embeddings.create.call_count == 2

    def test_embed_batch_returns_all(self, provider):
        """embed_batch() returns embeddings for all texts."""
        texts = ["alpha", "beta", "gamma"]
        results = provider.embed_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)
        assert all(len(r) == 4 for r in results)

    def test_embed_batch_empty(self, provider):
        """embed_batch() with empty list returns empty list."""
        results = provider.embed_batch([])
        assert results == []

    def test_embed_batch_uses_cache(self, provider):
        """embed_batch() uses cache for already-embedded texts."""
        # First, embed one text individually
        provider.embed("cached item")
        provider._backend._client.embeddings.create.reset_mock()

        # Now batch with the cached text plus a new one
        results = provider.embed_batch(["cached item", "new item"])

        assert len(results) == 2
        # Only one API call for the uncached text
        assert provider._backend._client.embeddings.create.call_count == 1
        call_kwargs = provider._backend._client.embeddings.create.call_args[1]
        assert call_kwargs["input"] == ["new item"]

    def test_embed_batch_all_cached(self, provider):
        """embed_batch() makes no API call when everything is cached."""
        provider.embed("a")
        provider.embed("b")
        provider._backend._client.embeddings.create.reset_mock()

        results = provider.embed_batch(["a", "b"])
        assert len(results) == 2
        assert provider._backend._client.embeddings.create.call_count == 0

    def test_cache_persistence(self, embedding_config, tmp_build_dir, mock_openai_client):
        """Embeddings persist across provider instances."""
        from synix.search.embeddings import OpenAIBackend

        # First provider instance
        p1 = EmbeddingProvider(embedding_config, tmp_build_dir)
        backend1 = OpenAIBackend(embedding_config)
        backend1._client = mock_openai_client
        p1._backend = backend1
        original = p1.embed("persistent text")

        # Second provider instance, same build dir
        p2 = EmbeddingProvider(embedding_config, tmp_build_dir)
        backend2 = OpenAIBackend(embedding_config)
        backend2._client = mock_openai_client
        p2._backend = backend2
        mock_openai_client.embeddings.create.reset_mock()

        cached = p2.embed("persistent text")

        # Should match within float32 precision
        for a, b in zip(cached, original):
            assert a == pytest.approx(b, abs=1e-6)

        # No API call — loaded from disk
        assert mock_openai_client.embeddings.create.call_count == 0

    def test_content_hash_deterministic(self, provider):
        """content_hash produces consistent results."""
        h1 = provider.content_hash("test")
        h2 = provider.content_hash("test")
        h3 = provider.content_hash("different")

        assert h1 == h2
        assert h1 != h3

    def test_manifest_file_created(self, provider, tmp_build_dir):
        """Embedding manifest is created on disk after embed()."""
        provider.embed("some text")

        manifest_path = tmp_build_dir / "embeddings" / "manifest.json"
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        # Manifest has _config key plus one embedding entry
        ch = provider.content_hash("some text")
        assert ch in manifest
        assert "_config" in manifest

    def test_embedding_files_created(self, provider, tmp_build_dir):
        """Binary embedding files are created on disk."""
        provider.embed("some text")

        ch = provider.content_hash("some text")
        emb_file = tmp_build_dir / "embeddings" / f"{ch}.bin"
        assert emb_file.exists()

        # Verify binary format
        data = emb_file.read_bytes()
        values = list(struct.unpack(f"<{len(data) // 4}f", data))
        assert len(values) == 4

    def test_load_nonexistent_returns_none(self, provider):
        """Loading a non-cached embedding returns None."""
        result = provider._load_embedding("nonexistent_hash")
        assert result is None

    def test_embed_batch_preserves_order(self, provider):
        """embed_batch() returns results in the same order as input texts."""
        texts = ["short", "a much longer text here", "mid"]
        results = provider.embed_batch(texts)

        # Each text should produce a different embedding based on length
        assert len(results) == 3
        # Since mock returns different values per text length, results should differ
        assert results[0] != results[1]

    def test_embed_batch_progress_callback(self, provider):
        """embed_batch() calls progress callback."""
        calls = []
        texts = ["alpha", "beta", "gamma"]
        provider.embed_batch(texts, progress_callback=lambda c, t: calls.append((c, t)))
        # At minimum, we should get a final progress report
        assert len(calls) > 0
        assert calls[-1][1] == 3  # total is always 3

    def test_content_hash_includes_model(self, tmp_build_dir):
        """Different models produce different cache keys for the same text."""
        config_a = EmbeddingConfig(provider="openai", model="model-a", dimensions=4)
        config_b = EmbeddingConfig(provider="openai", model="model-b", dimensions=4)
        provider_a = EmbeddingProvider(config_a, tmp_build_dir)
        provider_b = EmbeddingProvider(config_b, tmp_build_dir)

        h_a = provider_a.content_hash("same text")
        h_b = provider_b.content_hash("same text")
        assert h_a != h_b

    def test_content_hash_includes_provider(self, tmp_build_dir):
        """Different providers produce different cache keys for the same text."""
        config_a = EmbeddingConfig(provider="fastembed", model="BAAI/bge-small-en-v1.5")
        config_b = EmbeddingConfig(provider="openai", model="BAAI/bge-small-en-v1.5")
        provider_a = EmbeddingProvider(config_a, tmp_build_dir)
        provider_b = EmbeddingProvider(config_b, tmp_build_dir)

        h_a = provider_a.content_hash("same text")
        h_b = provider_b.content_hash("same text")
        assert h_a != h_b

    def test_manifest_stores_config_metadata(self, provider, tmp_build_dir):
        """Manifest includes _config metadata after save."""
        provider.embed("some text")

        manifest = json.loads((tmp_build_dir / "embeddings" / "manifest.json").read_text())
        config = manifest["_config"]
        assert config["provider"] == "openai"
        assert config["model"] == "text-embedding-3-small"
        assert config["dimensions"] == 4

    def test_config_mismatch_invalidates_cache(self, tmp_build_dir, mock_openai_client):
        """Switching models invalidates the manifest cache."""
        from synix.search.embeddings import OpenAIBackend

        # Build with config A
        config_a = EmbeddingConfig(provider="openai", model="model-a", dimensions=4, api_key="k")
        p1 = EmbeddingProvider(config_a, tmp_build_dir)
        backend1 = OpenAIBackend(config_a)
        backend1._client = mock_openai_client
        p1._backend = backend1
        p1.embed("hello")

        manifest_before = json.loads((tmp_build_dir / "embeddings" / "manifest.json").read_text())
        assert manifest_before["_config"]["model"] == "model-a"

        # Load with config B — should clear the manifest
        config_b = EmbeddingConfig(provider="openai", model="model-b", dimensions=4, api_key="k")
        p2 = EmbeddingProvider(config_b, tmp_build_dir)
        loaded_manifest = p2._load_manifest()
        # The old embedding hash should be gone (cache invalidated)
        hash_a = p1.content_hash("hello")
        assert hash_a not in loaded_manifest


class TestFastEmbedBackend:
    """Tests for the local FastEmbed ONNX backend."""

    @pytest.fixture
    def backend(self):
        from synix.search.embeddings import FastEmbedBackend

        config = EmbeddingConfig(
            provider="fastembed",
            model="BAAI/bge-small-en-v1.5",
            batch_size=4,
        )
        return FastEmbedBackend(config)

    def test_embed_single_text(self, backend):
        """Returns list[float] with correct dimensions (384)."""
        result = backend.embed("Hello world")
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)
        assert len(result) == 384

    def test_embed_batch(self, backend):
        """Returns correct count, all correct dimensions."""
        texts = ["Hello world", "Machine learning", "Python programming"]
        results = backend.embed_batch(texts)
        assert len(results) == 3
        for emb in results:
            assert len(emb) == 384

    def test_embed_consistency(self, backend):
        """Same text produces same embedding."""
        emb1 = backend.embed("test consistency")
        emb2 = backend.embed("test consistency")
        assert emb1 == emb2

    def test_empty_input(self, backend):
        """Empty list returns empty list."""
        assert backend.embed_batch([]) == []

    def test_progress_callback(self, backend):
        """Progress callback is called during batch embedding."""
        calls = []
        texts = ["text one", "text two", "text three", "text four", "text five"]
        backend.embed_batch(texts, progress_callback=lambda c, t: calls.append((c, t)))
        assert len(calls) > 0
        assert calls[-1][0] == len(texts)
        assert calls[-1][1] == len(texts)


class TestFastEmbedProvider:
    """Tests for EmbeddingProvider with real FastEmbed backend."""

    @pytest.fixture
    def provider(self, tmp_build_dir):
        config = EmbeddingConfig(
            provider="fastembed",
            model="BAAI/bge-small-en-v1.5",
            batch_size=4,
        )
        return EmbeddingProvider(config, tmp_build_dir)

    def test_embed_and_cache(self, provider, tmp_build_dir):
        """Embedding is generated and cached to disk."""
        emb = provider.embed("test caching")
        assert len(emb) == 384

        manifest_path = tmp_build_dir / "embeddings" / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        # 1 embedding entry + _config metadata
        assert "_config" in manifest
        assert len(manifest) == 2

    def test_cache_hit_skips_model(self, provider, tmp_build_dir):
        """Second embed call uses cache, not model."""
        text = "cache hit test"
        emb1 = provider.embed(text)

        config = EmbeddingConfig(provider="fastembed", model="BAAI/bge-small-en-v1.5")
        provider2 = EmbeddingProvider(config, tmp_build_dir)
        provider2._backend = MagicMock()

        emb2 = provider2.embed(text)
        assert emb1 == emb2
        assert not provider2._backend.embed.called

    def test_cache_persistence(self, tmp_build_dir):
        """Embed, create new provider instance, cache still works."""
        config = EmbeddingConfig(provider="fastembed", model="BAAI/bge-small-en-v1.5")
        p1 = EmbeddingProvider(config, tmp_build_dir)
        emb1 = p1.embed("persistence test")

        p2 = EmbeddingProvider(config, tmp_build_dir)
        emb2 = p2.embed("persistence test")
        assert emb1 == emb2

    def test_batch_mixed_cached_uncached(self, provider):
        """Some cached, some new — only new ones get embedded."""
        provider.embed("already cached")
        texts = ["already cached", "brand new text"]
        results = provider.embed_batch(texts)
        assert len(results) == 2
        assert len(results[0]) == 384
        assert len(results[1]) == 384


class TestHybridRetrieverWithEmbeddings:
    """Integration tests for HybridRetriever with real FastEmbed embeddings."""

    @pytest.fixture
    def index_and_retriever(self, tmp_build_dir):
        from synix.search.indexer import SearchIndex
        from synix.search.retriever import HybridRetriever

        config = EmbeddingConfig(
            provider="fastembed",
            model="BAAI/bge-small-en-v1.5",
            batch_size=4,
        )
        emb_provider = EmbeddingProvider(config, tmp_build_dir)

        index = SearchIndex(tmp_build_dir / "search.db")
        index.create()

        contents = [
            "Machine learning and deep neural networks for image classification",
            "Anthropic is building Claude, a helpful AI assistant",
            "Rust programming language has a unique ownership and borrowing model",
            "The weather in San Francisco is mild year round",
            "Python is popular for data science and web development",
        ]
        for i, content in enumerate(contents):
            index.insert(
                Artifact(
                    artifact_id=f"ep-{i:03d}",
                    artifact_type="episode",
                    content=content,
                    metadata={"layer_name": "episodes"},
                ),
                "episodes",
                1,
            )

        retriever = HybridRetriever(
            search_index=index,
            embedding_provider=emb_provider,
        )
        return index, retriever

    def test_semantic_search_returns_results(self, index_and_retriever):
        """Semantic search finds results via embedding similarity."""
        index, retriever = index_and_retriever
        results = retriever.query("artificial intelligence", mode="semantic", top_k=3)
        assert len(results) > 0
        assert any("ai" in r.content.lower() or "machine" in r.content.lower() for r in results[:2])
        index.close()

    def test_hybrid_search_fuses_results(self, index_and_retriever):
        """Hybrid mode returns results from both keyword and semantic."""
        index, retriever = index_and_retriever
        results = retriever.query("Claude AI", mode="hybrid", top_k=5)
        assert len(results) > 0
        assert results[0].search_mode == "hybrid"
        index.close()

    def test_semantic_relevance(self, index_and_retriever):
        """Semantically similar query finds related content without keyword match."""
        index, retriever = index_and_retriever
        results = retriever.query(
            "artificial intelligence safety company",
            mode="semantic",
            top_k=3,
        )
        assert len(results) > 0
        assert any("anthropic" in r.content.lower() or "claude" in r.content.lower() for r in results[:3])
        index.close()

    def test_scoring_fields_populated(self, index_and_retriever):
        """SearchResult scoring fields are set correctly per mode."""
        index, retriever = index_and_retriever

        kw_results = retriever.query("Python", mode="keyword", top_k=3)
        if kw_results:
            assert kw_results[0].search_mode == "keyword"
            assert kw_results[0].keyword_score is not None

        sem_results = retriever.query("programming", mode="semantic", top_k=3)
        if sem_results:
            assert sem_results[0].search_mode == "semantic"
            assert sem_results[0].semantic_score is not None

        hyb_results = retriever.query("Python programming", mode="hybrid", top_k=3)
        if hyb_results:
            assert hyb_results[0].search_mode == "hybrid"

        index.close()
