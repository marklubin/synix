"""Embedding generation and caching via OpenAI-compatible API."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

from synix.core.config import EmbeddingConfig
from synix.core.errors import atomic_write


class EmbeddingProvider:
    """Generates and caches embeddings using an OpenAI-compatible API.

    Embeddings are cached by content hash to avoid re-embedding unchanged
    artifacts. Cache is stored as raw binary files in build_dir/embeddings/
    with a JSON manifest mapping content_hash -> embedding filename.
    """

    def __init__(self, config: EmbeddingConfig, build_dir: str | Path):
        self.config = config
        self.build_dir = Path(build_dir)
        self.embeddings_dir = self.build_dir / "embeddings"
        self.manifest_path = self.embeddings_dir / "manifest.json"
        self._client = None
        self._manifest: dict[str, str] | None = None

    def _get_client(self):
        """Lazily create the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "openai package is required for embeddings. "
                    "Install with: pip install openai"
                ) from e

            kwargs: dict = {}
            api_key = self.config.resolve_api_key()
            if api_key:
                kwargs["api_key"] = api_key
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _load_manifest(self) -> dict[str, str]:
        """Load the embedding cache manifest from disk."""
        if self._manifest is not None:
            return self._manifest
        if self.manifest_path.exists():
            self._manifest = json.loads(self.manifest_path.read_text())
        else:
            self._manifest = {}
        return self._manifest

    def _save_manifest(self) -> None:
        """Write the embedding cache manifest to disk."""
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        atomic_write(self.manifest_path, json.dumps(self._manifest or {}, indent=2))

    @staticmethod
    def content_hash(text: str) -> str:
        """Compute SHA256 hash of text for cache keying."""
        return hashlib.sha256(text.encode()).hexdigest()

    def _save_embedding(self, content_hash: str, embedding: list[float]) -> None:
        """Save a single embedding vector to disk as raw binary."""
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{content_hash}.bin"
        filepath = self.embeddings_dir / filename
        # Pack as little-endian floats
        data = struct.pack(f"<{len(embedding)}f", *embedding)
        filepath.write_bytes(data)
        manifest = self._load_manifest()
        manifest[content_hash] = filename
        self._manifest = manifest

    def _load_embedding(self, content_hash: str) -> list[float] | None:
        """Load a cached embedding from disk. Returns None if not cached."""
        manifest = self._load_manifest()
        filename = manifest.get(content_hash)
        if filename is None:
            return None
        filepath = self.embeddings_dir / filename
        if not filepath.exists():
            return None
        data = filepath.read_bytes()
        count = len(data) // 4  # 4 bytes per float32
        return list(struct.unpack(f"<{count}f", data))

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text, using cache when available.

        Returns a list of floats (the embedding vector).
        """
        ch = self.content_hash(text)
        cached = self._load_embedding(ch)
        if cached is not None:
            return cached

        client = self._get_client()
        kwargs: dict = {
            "model": self.config.model,
            "input": [text],
        }
        if self.config.dimensions:
            kwargs["dimensions"] = self.config.dimensions

        response = client.embeddings.create(**kwargs)
        embedding = response.data[0].embedding

        self._save_embedding(ch, embedding)
        self._save_manifest()
        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts, using cache when available.

        Texts that are already cached are not re-sent to the API. Returns
        embeddings in the same order as the input texts.
        """
        if not texts:
            return []

        hashes = [self.content_hash(t) for t in texts]
        results: list[list[float] | None] = [None] * len(texts)

        # Check cache first
        uncached_indices: list[int] = []
        for i, ch in enumerate(hashes):
            cached = self._load_embedding(ch)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)

        # Batch-embed the uncached texts
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            client = self._get_client()
            kwargs: dict = {
                "model": self.config.model,
                "input": uncached_texts,
            }
            if self.config.dimensions:
                kwargs["dimensions"] = self.config.dimensions

            response = client.embeddings.create(**kwargs)

            # Response data may not be in order; sort by index
            sorted_data = sorted(response.data, key=lambda d: d.index)

            for j, idx in enumerate(uncached_indices):
                embedding = sorted_data[j].embedding
                results[idx] = embedding
                self._save_embedding(hashes[idx], embedding)

            self._save_manifest()

        return results  # type: ignore[return-value]
