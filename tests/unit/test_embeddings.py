"""Tests for embedding generation and caching."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

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
        self.data = [
            MockEmbeddingData(emb, idx) for idx, emb in enumerate(embeddings)
        ]


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
    p = EmbeddingProvider(embedding_config, tmp_build_dir)
    p._client = mock_openai_client
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
        assert provider._client.embeddings.create.call_count == 1

    def test_embed_different_texts_call_api(self, provider):
        """Different texts each call the API."""
        provider.embed("text one")
        provider.embed("text two")

        assert provider._client.embeddings.create.call_count == 2

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
        provider._client.embeddings.create.reset_mock()

        # Now batch with the cached text plus a new one
        results = provider.embed_batch(["cached item", "new item"])

        assert len(results) == 2
        # Only one API call for the uncached text
        assert provider._client.embeddings.create.call_count == 1
        call_kwargs = provider._client.embeddings.create.call_args[1]
        assert call_kwargs["input"] == ["new item"]

    def test_embed_batch_all_cached(self, provider):
        """embed_batch() makes no API call when everything is cached."""
        provider.embed("a")
        provider.embed("b")
        provider._client.embeddings.create.reset_mock()

        results = provider.embed_batch(["a", "b"])
        assert len(results) == 2
        assert provider._client.embeddings.create.call_count == 0

    def test_cache_persistence(self, embedding_config, tmp_build_dir, mock_openai_client):
        """Embeddings persist across provider instances."""
        # First provider instance
        p1 = EmbeddingProvider(embedding_config, tmp_build_dir)
        p1._client = mock_openai_client
        original = p1.embed("persistent text")

        # Second provider instance, same build dir
        p2 = EmbeddingProvider(embedding_config, tmp_build_dir)
        p2._client = mock_openai_client
        mock_openai_client.embeddings.create.reset_mock()

        cached = p2.embed("persistent text")

        # Should match within float32 precision
        for a, b in zip(cached, original):
            assert a == pytest.approx(b, abs=1e-6)

        # No API call — loaded from disk
        assert mock_openai_client.embeddings.create.call_count == 0

    def test_content_hash_deterministic(self):
        """content_hash produces consistent results."""
        h1 = EmbeddingProvider.content_hash("test")
        h2 = EmbeddingProvider.content_hash("test")
        h3 = EmbeddingProvider.content_hash("different")

        assert h1 == h2
        assert h1 != h3

    def test_manifest_file_created(self, provider, tmp_build_dir):
        """Embedding manifest is created on disk after embed()."""
        provider.embed("some text")

        manifest_path = tmp_build_dir / "embeddings" / "manifest.json"
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        assert len(manifest) == 1
        ch = EmbeddingProvider.content_hash("some text")
        assert ch in manifest

    def test_embedding_files_created(self, provider, tmp_build_dir):
        """Binary embedding files are created on disk."""
        provider.embed("some text")

        ch = EmbeddingProvider.content_hash("some text")
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
