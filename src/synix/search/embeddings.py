"""Embedding generation and caching with dual backend (FastEmbed + OpenAI)."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _suppress_hf_warnings() -> None:
    """Suppress noisy HuggingFace/tokenizers warnings during embedding model load."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    warnings.filterwarnings("ignore", message=".*huggingface.*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*tokenizers.*")
    warnings.filterwarnings("ignore", message=".*progress bar.*", category=UserWarning)
    warnings.filterwarnings("ignore", module="huggingface_hub")


from synix.core.config import EmbeddingConfig
from synix.core.errors import atomic_write


class FastEmbedBackend:
    """Local embedding backend using FastEmbed (ONNX Runtime)."""

    _model_cache: dict[str, object] = {}  # class-level cache for model instances

    def __init__(self, config: EmbeddingConfig):
        self.model_name = config.model
        self.batch_size = config.batch_size

    def _get_model(self):
        if self.model_name not in self._model_cache:
            _suppress_hf_warnings()
            from fastembed import TextEmbedding

            self._model_cache[self.model_name] = TextEmbedding(model_name=self.model_name)
        return self._model_cache[self.model_name]

    def embed(self, text: str) -> list[float]:
        model = self._get_model()
        result = list(model.embed([text]))
        return result[0].tolist()

    def embed_batch(
        self,
        texts: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        results: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            embeddings = list(model.embed(chunk))
            results.extend(e.tolist() for e in embeddings)
            if progress_callback:
                progress_callback(len(results), len(texts))
        return results


class OpenAIBackend:
    """Remote embedding backend using OpenAI-compatible API."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.batch_size = config.batch_size
        self.concurrency = config.concurrency
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            kwargs: dict = {}
            api_key = self.config.resolve_api_key()
            if api_key:
                kwargs["api_key"] = api_key
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _embed_chunk(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        kwargs: dict = {
            "model": self.config.model,
            "input": texts,
        }
        if self.config.dimensions:
            kwargs["dimensions"] = self.config.dimensions
        response = client.embeddings.create(**kwargs)
        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [d.embedding for d in sorted_data]

    def embed(self, text: str) -> list[float]:
        return self._embed_chunk([text])[0]

    def embed_batch(
        self,
        texts: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []

        chunks = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]

        if len(chunks) == 1:
            result = self._embed_chunk(chunks[0])
            if progress_callback:
                progress_callback(len(result), len(texts))
            return result

        results: list[list[float] | None] = [None] * len(texts)
        completed = 0

        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {}
            for chunk_idx, chunk in enumerate(chunks):
                future = pool.submit(self._embed_chunk, chunk)
                futures[future] = chunk_idx

            for future in as_completed(futures):
                chunk_idx = futures[future]
                embeddings = future.result()
                start = chunk_idx * self.batch_size
                for j, emb in enumerate(embeddings):
                    results[start + j] = emb
                completed += len(embeddings)
                if progress_callback:
                    progress_callback(completed, len(texts))

        return results  # type: ignore[return-value]


class EmbeddingProvider:
    """Generates and caches embeddings using FastEmbed (local) or OpenAI API.

    Dispatches to the appropriate backend based on config.provider.
    Embeddings are cached by content hash to avoid re-embedding unchanged
    artifacts. Cache is stored as raw binary files in build_dir/embeddings/
    with a JSON manifest mapping content_hash -> embedding filename.
    """

    def __init__(self, config: EmbeddingConfig, build_dir: str | Path):
        self.config = config
        self.build_dir = Path(build_dir)
        self.embeddings_dir = self.build_dir / "embeddings"
        self.manifest_path = self.embeddings_dir / "manifest.json"
        self._backend: FastEmbedBackend | OpenAIBackend | None = None
        self._manifest: dict[str, str] | None = None

    def _get_backend(self) -> FastEmbedBackend | OpenAIBackend:
        """Lazily create the embedding backend."""
        if self._backend is None:
            if self.config.provider == "fastembed":
                self._backend = FastEmbedBackend(self.config)
            else:
                self._backend = OpenAIBackend(self.config)
        return self._backend

    def _current_config_meta(self) -> dict:
        """Return a dict describing the current embedding config for manifest storage."""
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "dimensions": self.config.dimensions,
        }

    def _load_manifest(self) -> dict[str, str]:
        """Load the embedding cache manifest from disk.

        If the stored ``_config`` metadata doesn't match the current config,
        the manifest is cleared so all embeddings are regenerated.
        """
        if self._manifest is not None:
            return self._manifest
        if self.manifest_path.exists():
            raw = json.loads(self.manifest_path.read_text())
            stored_config = raw.get("_config")
            if stored_config is not None and stored_config != self._current_config_meta():
                # Config mismatch — invalidate cache
                self._manifest = {}
            else:
                self._manifest = raw
        else:
            self._manifest = {}
        return self._manifest

    def _save_manifest(self) -> None:
        """Write the embedding cache manifest to disk, including config metadata."""
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        data = dict(self._manifest or {})
        data["_config"] = self._current_config_meta()
        atomic_write(self.manifest_path, json.dumps(data, indent=2))

    def content_hash(self, text: str) -> str:
        """Compute SHA256 hash of text for cache keying.

        Includes provider and model in the key so switching models
        invalidates all cached embeddings.
        """
        key = f"{self.config.provider}:{self.config.model}:{text}"
        return hashlib.sha256(key.encode()).hexdigest()

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
        """Generate embedding for a single text, using cache when available."""
        ch = self.content_hash(text)
        cached = self._load_embedding(ch)
        if cached is not None:
            return cached

        backend = self._get_backend()
        embedding = backend.embed(text)

        self._save_embedding(ch, embedding)
        self._save_manifest()
        return embedding

    def embed_batch(
        self,
        texts: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts, using cache when available.

        Texts that are already cached are not re-sent to the backend. Returns
        embeddings in the same order as the input texts.

        Args:
            texts: List of texts to embed.
            progress_callback: Optional callback(completed, total) for progress updates.
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

        cached_count = len(texts) - len(uncached_indices)

        # Report cached items as initial progress
        if progress_callback and cached_count > 0:
            progress_callback(cached_count, len(texts))

        # Batch-embed the uncached texts
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            backend = self._get_backend()

            def _backend_progress(completed: int, total: int) -> None:
                if progress_callback:
                    progress_callback(cached_count + completed, len(texts))

            new_embeddings = backend.embed_batch(uncached_texts, _backend_progress)

            for j, idx in enumerate(uncached_indices):
                results[idx] = new_embeddings[j]
                self._save_embedding(hashes[idx], new_embeddings[j])

            self._save_manifest()

        return results  # type: ignore[return-value]
