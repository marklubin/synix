"""Chunk — configurable 1:N text splitting transform.

Splits each input artifact into multiple smaller chunks. No LLM call —
pure text processing. Each chunk tracks provenance to the source artifact
and carries metadata for downstream grouping.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from collections.abc import Callable

from synix.core.models import Artifact, Transform, TransformContext
from synix.ext._util import stable_callable_repr

logger = logging.getLogger(__name__)


class Chunk(Transform):
    """1:N transform — split each input artifact into chunks.

    Example::

        chunks = Chunk(
            "doc-chunks",
            depends_on=[documents],
            chunk_size=1000,
            chunk_overlap=200,
            artifact_type="chunk",
        )

    Chunking strategies (in priority order):

    1. ``chunker`` callable — ``Callable[[str], list[str]]``, full control
    2. ``separator`` — split on a delimiter string, filter empty segments
    3. ``chunk_size`` + ``chunk_overlap`` — fixed-size character windowing (default)

    Output metadata per chunk: ``source_label``, ``chunk_index``, ``chunk_total``,
    plus all metadata from the input artifact (propagated).
    """

    def __init__(
        self,
        name: str,
        *,
        depends_on: list | None = None,
        chunker: Callable[[str], list[str]] | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str | None = None,
        label_fn: Callable | None = None,
        metadata_fn: Callable | None = None,
        artifact_type: str = "chunk",
        config: dict | None = None,
    ):
        super().__init__(name, depends_on=depends_on, config=config)
        self.chunker = chunker
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.label_fn = label_fn
        self.metadata_fn = metadata_fn
        self.artifact_type = artifact_type

        if chunker is None and separator is None:
            if chunk_size <= 0:
                raise ValueError(f"chunk_size must be positive, got {chunk_size}")
            if chunk_overlap < 0:
                raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
            if chunk_overlap >= chunk_size:
                raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")

    def get_cache_key(self, config: dict) -> str:
        """Include chunking config and callables in cache key."""
        chunker_str = stable_callable_repr(self.chunker) if self.chunker is not None else ""
        metadata_fn_str = stable_callable_repr(self.metadata_fn) if self.metadata_fn is not None else ""
        label_fn_str = stable_callable_repr(self.label_fn) if self.label_fn is not None else ""
        parts = (
            f"{self.chunk_size}\x00{self.chunk_overlap}\x00{self.separator or ''}"
            f"\x00{self.artifact_type}\x00{chunker_str}\x00{metadata_fn_str}\x00{label_fn_str}"
        )
        return hashlib.sha256(parts.encode()).hexdigest()[:16]

    def compute_fingerprint(self, config: dict):
        """Add callable fingerprint components for chunker, label_fn, metadata_fn."""
        fp = super().compute_fingerprint(config)
        callables = {}
        if self.chunker is not None:
            callables["chunker"] = self.chunker
        if self.label_fn is not None:
            callables["label_fn"] = self.label_fn
        if self.metadata_fn is not None:
            callables["metadata_fn"] = self.metadata_fn
        if callables:
            from synix.build.fingerprint import Fingerprint, compute_digest, fingerprint_value

            components = dict(fp.components)
            for key, fn in callables.items():
                try:
                    components[key] = fingerprint_value(inspect.getsource(fn))
                except (OSError, TypeError):
                    components[key] = fingerprint_value(repr(fn))
            return Fingerprint(scheme=fp.scheme, digest=compute_digest(components), components=components)
        return fp

    def execute(self, inputs: list[Artifact], ctx: TransformContext) -> list[Artifact]:
        inp = inputs[0]
        chunks = self._chunk(inp.content)
        total = len(chunks)

        results = []
        for i, chunk_text in enumerate(chunks):
            meta = dict(inp.metadata)
            meta.update(
                {
                    "source_label": inp.label,
                    "chunk_index": i,
                    "chunk_total": total,
                }
            )
            if self.metadata_fn is not None:
                meta.update(self.metadata_fn(inp, i, total))

            if self.label_fn is not None:
                label = self.label_fn(inp, i, total)
            else:
                label = f"{self.name}-{inp.label}-{i}"

            results.append(
                Artifact(
                    label=label,
                    artifact_type=self.artifact_type,
                    content=chunk_text,
                    input_ids=[inp.artifact_id],
                    metadata=meta,
                )
            )
        return results

    def estimate_output_count(self, input_count: int) -> int:
        return input_count * 3

    def _chunk(self, text: str) -> list[str]:
        """Dispatch to the configured chunking strategy."""
        if self.chunker is not None:
            result = self.chunker(text)
            if not result:
                return [text]
            return result
        if self.separator is not None:
            return self._separator_chunk(text)
        return self._fixed_chunk(text)

    def _fixed_chunk(self, text: str) -> list[str]:
        """Sliding window: chunk_size chars with chunk_overlap overlap."""
        if not text:
            return [""]
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += step
        return chunks

    def _separator_chunk(self, text: str) -> list[str]:
        """Split on separator, filter empty segments."""
        if not text:
            return [""]
        parts = [seg for seg in text.split(self.separator) if seg.strip()]
        if not parts:
            return [""]
        return parts
