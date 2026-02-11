"""Cassette layer — record and replay LLM + embedding calls for deterministic demos.

Provides wrappers around LLMClient and EmbeddingProvider that intercept external
calls. In ``record`` mode, calls pass through to the real backend and responses are
saved to disk. In ``replay`` mode, responses are loaded from disk and the real
backend is never contacted.

Configuration via environment variables:
  SYNIX_CASSETTE_MODE  — "record", "replay", or "off" (default: "off")
  SYNIX_CASSETTE_DIR   — path to cassette directory (required when mode != "off")
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time as _time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from synix.build.llm_client import LLMClient, LLMResponse

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CassetteMiss(Exception):
    """Raised in replay mode when a cassette key is not found."""

    def __init__(self, key: str, preview: str = ""):
        self.key = key
        self.preview = preview
        msg = f"Cassette miss for key {key[:12]}..."
        if preview:
            msg += f" (prompt: {preview[:80]})"
        msg += "\nRun with SYNIX_CASSETTE_MODE=record to capture this call."
        super().__init__(msg)


# ---------------------------------------------------------------------------
# LLM Cassette
# ---------------------------------------------------------------------------

def compute_cassette_key(
    provider: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> str:
    """Compute a deterministic key for an LLM request.

    Normalizes whitespace and sorts JSON keys. Excludes api_key and
    artifact_desc which don't affect the response.
    """
    # Normalize messages: strip trailing whitespace, normalize line endings
    normalized_msgs = []
    for msg in messages:
        norm = {}
        for k, v in sorted(msg.items()):
            if isinstance(v, str):
                v = v.replace("\r\n", "\n").rstrip()
            norm[k] = v
        normalized_msgs.append(norm)

    payload = {
        "provider": provider,
        "model": model,
        "messages": normalized_msgs,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class CassetteEntry:
    """A single recorded LLM interaction."""

    key: str
    request: dict = field(default_factory=dict)
    response: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


class CassetteStore:
    """Thread-safe store for LLM cassette entries, backed by a YAML file."""

    def __init__(self, cassette_dir: Path):
        self.cassette_dir = Path(cassette_dir)
        self._entries: dict[str, CassetteEntry] = {}
        self._lock = threading.Lock()
        self._load()

    def _yaml_path(self) -> Path:
        return self.cassette_dir / "llm.yaml"

    def _load(self) -> None:
        path = self._yaml_path()
        if not path.exists():
            return
        try:
            data = yaml.safe_load(path.read_text()) or []
        except (yaml.YAMLError, OSError):
            return
        if not isinstance(data, list):
            return
        for item in data:
            if not isinstance(item, dict):
                continue
            key = item.get("key", "")
            if key:
                self._entries[key] = CassetteEntry(
                    key=key,
                    request=item.get("request", {}),
                    response=item.get("response", {}),
                    meta=item.get("meta", {}),
                )

    def get(self, key: str) -> CassetteEntry | None:
        with self._lock:
            return self._entries.get(key)

    def put(self, entry: CassetteEntry) -> None:
        with self._lock:
            self._entries[entry.key] = entry

    def save(self) -> None:
        self.cassette_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = []
            for entry in self._entries.values():
                data.append({
                    "key": entry.key,
                    "request": entry.request,
                    "response": entry.response,
                    "meta": entry.meta,
                })
        # Write outside lock
        self._yaml_path().write_text(
            yaml.dump(data, default_flow_style=False, allow_unicode=True, width=120)
        )


class CassetteClientWrapper:
    """Wraps an LLMClient, intercepting complete() for record/replay.

    Exposes the same interface as LLMClient so it can be used as a
    drop-in replacement.
    """

    def __init__(self, real_client: LLMClient, mode: str, store: CassetteStore):
        self.real_client = real_client
        self.mode = mode  # "record" or "replay"
        self.store = store
        # Expose config for code that reads client.config
        self.config = real_client.config

    def complete(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        artifact_desc: str = "artifact",
    ) -> LLMResponse:
        resolved_max_tokens = (
            max_tokens if max_tokens is not None else self.config.max_tokens
        )
        resolved_temperature = (
            temperature if temperature is not None else self.config.temperature
        )

        key = compute_cassette_key(
            provider=self.config.provider,
            model=self.config.model,
            messages=messages,
            max_tokens=resolved_max_tokens,
            temperature=resolved_temperature,
        )

        if self.mode == "replay":
            entry = self.store.get(key)
            if entry is None:
                preview = ""
                if messages:
                    content = messages[0].get("content", "")
                    preview = content[:80] if isinstance(content, str) else ""
                raise CassetteMiss(key, preview)
            # Simulate LLM latency in demo mode
            if os.environ.get("SYNIX_DEMO") == "1":
                output_tokens = entry.response.get("output_tokens", 100)
                jitter = random.uniform(0.3, 0.8) + (output_tokens / 500)
                _time.sleep(min(jitter, 3.0))
            resp = entry.response
            return LLMResponse(
                content=resp.get("text", ""),
                model=resp.get("model", self.config.model),
                input_tokens=resp.get("input_tokens", 0),
                output_tokens=resp.get("output_tokens", 0),
                total_tokens=(
                    resp.get("input_tokens", 0) + resp.get("output_tokens", 0)
                ),
            )

        # record mode: check cache first to avoid duplicate API calls
        entry = self.store.get(key)
        if entry is not None:
            resp = entry.response
            return LLMResponse(
                content=resp.get("text", ""),
                model=resp.get("model", self.config.model),
                input_tokens=resp.get("input_tokens", 0),
                output_tokens=resp.get("output_tokens", 0),
                total_tokens=(
                    resp.get("input_tokens", 0) + resp.get("output_tokens", 0)
                ),
            )

        # Call real API
        response = self.real_client.complete(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            artifact_desc=artifact_desc,
        )

        # Build preview for request metadata
        preview = ""
        if messages:
            content = messages[0].get("content", "")
            if isinstance(content, str):
                preview = content[:200]

        self.store.put(CassetteEntry(
            key=key,
            request={
                "provider": self.config.provider,
                "model": self.config.model,
                "prompt_preview": preview,
                "params": {
                    "max_tokens": resolved_max_tokens,
                    "temperature": resolved_temperature,
                },
            },
            response={
                "text": response.content,
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        ))
        self.store.save()

        return response


def maybe_wrap_client(client: LLMClient) -> LLMClient | CassetteClientWrapper:
    """Wrap an LLMClient with cassette support if env vars are set.

    Returns the original client if SYNIX_CASSETTE_MODE is "off" or unset.
    """
    mode = os.environ.get("SYNIX_CASSETTE_MODE", "off").lower()
    if mode == "off" or not mode:
        return client

    cassette_dir = os.environ.get("SYNIX_CASSETTE_DIR")
    if not cassette_dir:
        raise ValueError(
            f"SYNIX_CASSETTE_MODE={mode} requires SYNIX_CASSETTE_DIR to be set"
        )

    store = CassetteStore(Path(cassette_dir))
    return CassetteClientWrapper(client, mode, store)


# ---------------------------------------------------------------------------
# Embedding Cassette
# ---------------------------------------------------------------------------

class CassetteEmbeddingWrapper:
    """Wraps an EmbeddingProvider for record/replay of embedding calls.

    Stores cached embeddings in the cassette directory using the same binary
    format as EmbeddingProvider (manifest.json + .bin files), reusing the
    existing serialization logic.
    """

    def __init__(self, real_provider, mode: str, cassette_dir: Path):
        self.real_provider = real_provider
        self.mode = mode
        self.cassette_dir = Path(cassette_dir)
        # Expose config for code that reads provider.config
        self.config = real_provider.config
        self.build_dir = real_provider.build_dir

        # Create a shadow EmbeddingProvider that uses the cassette dir for storage
        # Lazy import to avoid build→search dependency (FR-2.3)
        import importlib
        _embeddings = importlib.import_module("synix.search.embeddings")

        self._shadow = _embeddings.EmbeddingProvider(
            self.config,
            self.cassette_dir,
        )
        # Override embeddings_dir to use cassette location
        self._shadow.embeddings_dir = self.cassette_dir / "embeddings"
        self._shadow.manifest_path = self._shadow.embeddings_dir / "manifest.json"

    def content_hash(self, text: str) -> str:
        return self.real_provider.content_hash(text)

    def embed(self, text: str) -> list[float]:
        ch = self.content_hash(text)

        # Check cassette cache
        cached = self._shadow._load_embedding(ch)
        if cached is not None:
            return cached

        if self.mode == "replay":
            raise CassetteMiss(ch, text[:80])

        # record mode: call real provider, save to cassette
        embedding = self.real_provider.embed(text)
        self._shadow._save_embedding(ch, embedding)
        self._shadow._save_manifest()
        return embedding

    def embed_batch(
        self,
        texts: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []

        hashes = [self.content_hash(t) for t in texts]
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []

        # Check cassette cache first
        for i, ch in enumerate(hashes):
            cached = self._shadow._load_embedding(ch)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)

        cached_count = len(texts) - len(uncached_indices)
        if progress_callback and cached_count > 0:
            progress_callback(cached_count, len(texts))

        if uncached_indices:
            if self.mode == "replay":
                raise CassetteMiss(
                    hashes[uncached_indices[0]],
                    texts[uncached_indices[0]][:80],
                )

            # record mode: call real provider for uncached texts
            uncached_texts = [texts[i] for i in uncached_indices]

            def _backend_progress(completed: int, total: int) -> None:
                if progress_callback:
                    progress_callback(cached_count + completed, len(texts))

            new_embeddings = self.real_provider.embed_batch(
                uncached_texts, _backend_progress
            )
            for j, idx in enumerate(uncached_indices):
                results[idx] = new_embeddings[j]
                self._shadow._save_embedding(hashes[idx], new_embeddings[j])

            self._shadow._save_manifest()

        return results  # type: ignore[return-value]


def maybe_wrap_embedding_provider(provider) -> object:
    """Wrap an EmbeddingProvider with cassette support if env vars are set.

    Returns the original provider if SYNIX_CASSETTE_MODE is "off" or unset.
    """
    mode = os.environ.get("SYNIX_CASSETTE_MODE", "off").lower()
    if mode == "off" or not mode:
        return provider

    cassette_dir = os.environ.get("SYNIX_CASSETTE_DIR")
    if not cassette_dir:
        raise ValueError(
            f"SYNIX_CASSETTE_MODE={mode} requires SYNIX_CASSETTE_DIR to be set"
        )

    return CassetteEmbeddingWrapper(provider, mode, Path(cassette_dir))
