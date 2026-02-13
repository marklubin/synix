"""Tests for the cassette layer — LLM and embedding record/replay."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from synix.build.cassette import (
    CassetteClientWrapper,
    CassetteEmbeddingWrapper,
    CassetteEntry,
    CassetteMiss,
    CassetteStore,
    compute_cassette_key,
    maybe_wrap_client,
    maybe_wrap_embedding_provider,
)
from synix.build.llm_client import LLMClient, LLMResponse
from synix.core.config import EmbeddingConfig, LLMConfig

# ---------------------------------------------------------------------------
# compute_cassette_key tests
# ---------------------------------------------------------------------------


class TestComputeCassetteKey:
    def test_deterministic(self):
        """Same inputs produce the same key."""
        msgs = [{"role": "user", "content": "Hello"}]
        k1 = compute_cassette_key("anthropic", "claude-3", msgs, 1024, 0.3)
        k2 = compute_cassette_key("anthropic", "claude-3", msgs, 1024, 0.3)
        assert k1 == k2

    def test_different_messages_different_key(self):
        msgs1 = [{"role": "user", "content": "Hello"}]
        msgs2 = [{"role": "user", "content": "Goodbye"}]
        k1 = compute_cassette_key("anthropic", "claude-3", msgs1, 1024, 0.3)
        k2 = compute_cassette_key("anthropic", "claude-3", msgs2, 1024, 0.3)
        assert k1 != k2

    def test_different_model_different_key(self):
        msgs = [{"role": "user", "content": "Hello"}]
        k1 = compute_cassette_key("anthropic", "claude-3", msgs, 1024, 0.3)
        k2 = compute_cassette_key("anthropic", "claude-4", msgs, 1024, 0.3)
        assert k1 != k2

    def test_whitespace_normalization(self):
        """Trailing whitespace and \\r\\n are normalized."""
        msgs1 = [{"role": "user", "content": "Hello  \r\n"}]
        msgs2 = [{"role": "user", "content": "Hello"}]
        k1 = compute_cassette_key("anthropic", "claude-3", msgs1, 1024, 0.3)
        k2 = compute_cassette_key("anthropic", "claude-3", msgs2, 1024, 0.3)
        assert k1 == k2

    def test_key_is_hex_sha256(self):
        msgs = [{"role": "user", "content": "test"}]
        key = compute_cassette_key("anthropic", "claude-3", msgs, 1024, 0.3)
        assert len(key) == 64
        int(key, 16)  # should not raise


# ---------------------------------------------------------------------------
# CassetteStore tests
# ---------------------------------------------------------------------------


class TestCassetteStore:
    def test_empty_dir_no_crash(self, tmp_path):
        store = CassetteStore(tmp_path / "cassettes")
        assert store.get("nonexistent") is None

    def test_put_and_get(self, tmp_path):
        store = CassetteStore(tmp_path)
        entry = CassetteEntry(
            key="abc123",
            request={"model": "claude-3"},
            response={"text": "Hello world"},
        )
        store.put(entry)
        result = store.get("abc123")
        assert result is not None
        assert result.response["text"] == "Hello world"

    def test_save_and_reload(self, tmp_path):
        """Save to YAML, create new store from same dir, entries persist."""
        store1 = CassetteStore(tmp_path)
        store1.put(
            CassetteEntry(
                key="key1",
                request={"model": "test"},
                response={"text": "response1"},
                meta={"note": "test entry"},
            )
        )
        store1.save()

        # Reload from disk
        store2 = CassetteStore(tmp_path)
        result = store2.get("key1")
        assert result is not None
        assert result.response["text"] == "response1"
        assert result.meta["note"] == "test entry"

    def test_multiple_entries_roundtrip(self, tmp_path):
        store = CassetteStore(tmp_path)
        for i in range(5):
            store.put(
                CassetteEntry(
                    key=f"key{i}",
                    response={"text": f"response{i}"},
                )
            )
        store.save()

        store2 = CassetteStore(tmp_path)
        for i in range(5):
            entry = store2.get(f"key{i}")
            assert entry is not None
            assert entry.response["text"] == f"response{i}"


# ---------------------------------------------------------------------------
# CassetteClientWrapper tests
# ---------------------------------------------------------------------------


def _make_mock_client(response_text="Mock response"):
    """Create a mock LLMClient that returns a fixed response."""
    config = LLMConfig(
        provider="anthropic",
        model="claude-test",
        temperature=0.3,
        max_tokens=1024,
        api_key="test-key",
    )
    client = MagicMock(spec=LLMClient)
    client.config = config
    client.complete.return_value = LLMResponse(
        content=response_text,
        model="claude-test",
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
    )
    return client


class TestCassetteClientWrapper:
    def test_replay_hit(self, tmp_path):
        """Replay mode returns cached response."""
        mock_client = _make_mock_client()
        store = CassetteStore(tmp_path)

        # Pre-populate store
        messages = [{"role": "user", "content": "Hello"}]
        key = compute_cassette_key("anthropic", "claude-test", messages, 1024, 0.3)
        store.put(
            CassetteEntry(
                key=key,
                response={
                    "text": "Cached hello",
                    "model": "claude-test",
                    "input_tokens": 5,
                    "output_tokens": 10,
                },
            )
        )

        wrapper = CassetteClientWrapper(mock_client, "replay", store)
        result = wrapper.complete(messages=messages)

        assert result.content == "Cached hello"
        assert result.input_tokens == 5
        mock_client.complete.assert_not_called()

    def test_replay_miss_raises(self, tmp_path):
        """Replay mode raises CassetteMiss when key not found."""
        mock_client = _make_mock_client()
        store = CassetteStore(tmp_path)
        wrapper = CassetteClientWrapper(mock_client, "replay", store)

        with pytest.raises(CassetteMiss) as exc_info:
            wrapper.complete(messages=[{"role": "user", "content": "Unknown"}])

        assert "SYNIX_CASSETTE_MODE=record" in str(exc_info.value)

    def test_record_calls_real_then_saves(self, tmp_path):
        """Record mode calls real API and saves to store."""
        mock_client = _make_mock_client("Real response")
        store = CassetteStore(tmp_path)
        wrapper = CassetteClientWrapper(mock_client, "record", store)

        messages = [{"role": "user", "content": "Test prompt"}]
        result = wrapper.complete(messages=messages)

        assert result.content == "Real response"
        mock_client.complete.assert_called_once()

        # Verify it was saved
        key = compute_cassette_key("anthropic", "claude-test", messages, 1024, 0.3)
        entry = store.get(key)
        assert entry is not None
        assert entry.response["text"] == "Real response"

    def test_record_deduplicates(self, tmp_path):
        """Record mode uses cache to avoid duplicate API calls."""
        mock_client = _make_mock_client("First call")
        store = CassetteStore(tmp_path)
        wrapper = CassetteClientWrapper(mock_client, "record", store)

        messages = [{"role": "user", "content": "Dedup test"}]
        wrapper.complete(messages=messages)
        wrapper.complete(messages=messages)

        # Should only call real API once
        assert mock_client.complete.call_count == 1

    def test_off_mode_passthrough(self, tmp_path):
        """Off mode passes through directly to real client."""
        mock_client = _make_mock_client("Direct response")

        # Verify maybe_wrap_client returns original when mode=off
        with patch.dict(os.environ, {"SYNIX_CASSETTE_MODE": "off"}):
            result = maybe_wrap_client(mock_client)
            assert result is mock_client

    def test_wrapper_exposes_config(self, tmp_path):
        """Wrapper exposes the real client's config."""
        mock_client = _make_mock_client()
        store = CassetteStore(tmp_path)
        wrapper = CassetteClientWrapper(mock_client, "replay", store)
        assert wrapper.config is mock_client.config


# ---------------------------------------------------------------------------
# maybe_wrap_client tests
# ---------------------------------------------------------------------------


class TestMaybeWrapClient:
    def test_off_returns_original(self):
        mock_client = _make_mock_client()
        with patch.dict(os.environ, {"SYNIX_CASSETTE_MODE": "off"}, clear=False):
            result = maybe_wrap_client(mock_client)
            assert result is mock_client

    def test_unset_returns_original(self):
        mock_client = _make_mock_client()
        env = dict(os.environ)
        env.pop("SYNIX_CASSETTE_MODE", None)
        env.pop("SYNIX_CASSETTE_DIR", None)
        with patch.dict(os.environ, env, clear=True):
            result = maybe_wrap_client(mock_client)
            assert result is mock_client

    def test_record_without_dir_raises(self):
        mock_client = _make_mock_client()
        env = {"SYNIX_CASSETTE_MODE": "record"}
        # Remove SYNIX_CASSETTE_DIR
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="SYNIX_CASSETTE_DIR"):
                maybe_wrap_client(mock_client)

    def test_record_with_dir_wraps(self, tmp_path):
        mock_client = _make_mock_client()
        env = {
            "SYNIX_CASSETTE_MODE": "record",
            "SYNIX_CASSETTE_DIR": str(tmp_path),
        }
        with patch.dict(os.environ, env, clear=False):
            result = maybe_wrap_client(mock_client)
            assert isinstance(result, CassetteClientWrapper)
            assert result.mode == "record"


# ---------------------------------------------------------------------------
# Embedding cassette tests
# ---------------------------------------------------------------------------


def _make_mock_embedding_provider(tmp_path):
    """Create a mock EmbeddingProvider."""
    config = EmbeddingConfig(
        provider="fastembed",
        model="test-model",
        dimensions=3,
    )
    provider = MagicMock()
    provider.config = config
    provider.build_dir = tmp_path
    provider.embeddings_dir = tmp_path / "embeddings"
    provider.manifest_path = tmp_path / "embeddings" / "manifest.json"
    provider.content_hash.side_effect = lambda text: (
        __import__("hashlib").sha256(f"fastembed:test-model:{text}".encode()).hexdigest()
    )
    provider.embed.side_effect = lambda text: [0.1, 0.2, 0.3]
    provider.embed_batch.side_effect = lambda texts, cb=None: [[0.1, 0.2, 0.3]] * len(texts)
    return provider


class TestCassetteEmbeddingWrapper:
    def test_record_and_replay_single(self, tmp_path):
        """Record an embedding, then replay it."""
        cassette_dir = tmp_path / "cassettes"
        provider = _make_mock_embedding_provider(tmp_path / "build")

        # Record
        wrapper = CassetteEmbeddingWrapper(provider, "record", cassette_dir)
        result = wrapper.embed("test text")
        assert result == [0.1, 0.2, 0.3]
        provider.embed.assert_called_once()

        # Replay — create new wrapper pointing to same cassette dir
        provider2 = _make_mock_embedding_provider(tmp_path / "build2")
        wrapper2 = CassetteEmbeddingWrapper(provider2, "replay", cassette_dir)
        result2 = wrapper2.embed("test text")
        assert len(result2) == 3
        # Real provider should not have been called
        provider2.embed.assert_not_called()

    def test_record_and_replay_batch(self, tmp_path):
        """Record batch embeddings, then replay."""
        cassette_dir = tmp_path / "cassettes"
        provider = _make_mock_embedding_provider(tmp_path / "build")

        # Record
        wrapper = CassetteEmbeddingWrapper(provider, "record", cassette_dir)
        results = wrapper.embed_batch(["text1", "text2"])
        assert len(results) == 2

        # Replay
        provider2 = _make_mock_embedding_provider(tmp_path / "build2")
        wrapper2 = CassetteEmbeddingWrapper(provider2, "replay", cassette_dir)
        results2 = wrapper2.embed_batch(["text1", "text2"])
        assert len(results2) == 2
        provider2.embed_batch.assert_not_called()

    def test_replay_miss_raises(self, tmp_path):
        cassette_dir = tmp_path / "cassettes"
        provider = _make_mock_embedding_provider(tmp_path / "build")
        wrapper = CassetteEmbeddingWrapper(provider, "replay", cassette_dir)

        with pytest.raises(CassetteMiss):
            wrapper.embed("unknown text")

    def test_content_hash_passthrough(self, tmp_path):
        """Wrapper preserves content_hash() method from real provider."""
        cassette_dir = tmp_path / "cassettes"
        provider = _make_mock_embedding_provider(tmp_path / "build")
        wrapper = CassetteEmbeddingWrapper(provider, "record", cassette_dir)

        h1 = wrapper.content_hash("test")
        h2 = provider.content_hash("test")
        assert h1 == h2


# ---------------------------------------------------------------------------
# maybe_wrap_embedding_provider tests
# ---------------------------------------------------------------------------


class TestMaybeWrapEmbeddingProvider:
    def test_off_returns_original(self, tmp_path):
        provider = _make_mock_embedding_provider(tmp_path)
        with patch.dict(os.environ, {"SYNIX_CASSETTE_MODE": "off"}, clear=False):
            result = maybe_wrap_embedding_provider(provider)
            assert result is provider

    def test_unset_returns_original(self, tmp_path):
        provider = _make_mock_embedding_provider(tmp_path)
        env = dict(os.environ)
        env.pop("SYNIX_CASSETTE_MODE", None)
        env.pop("SYNIX_CASSETTE_DIR", None)
        with patch.dict(os.environ, env, clear=True):
            result = maybe_wrap_embedding_provider(provider)
            assert result is provider

    def test_record_without_dir_raises(self, tmp_path):
        provider = _make_mock_embedding_provider(tmp_path)
        env = {"SYNIX_CASSETTE_MODE": "record"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ValueError, match="SYNIX_CASSETTE_DIR"):
                maybe_wrap_embedding_provider(provider)
