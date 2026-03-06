"""Core data models for Synix.

Layer hierarchy:
    Layer (abstract base — name, depends_on)
    ├── Source          → reads source files (root nodes, no depends_on)
    ├── Transform       → processes inputs via split()/execute()
    │   ├── MapSynthesis, GroupSynthesis, ReduceSynthesis, FoldSynthesis, Merge
    │   ├── bundled memory transforms under synix.ext
    │   └── (user-defined subclasses)
    ├── SearchSurface   → build-time searchable capability
    ├── SearchIndex     → projection compatibility output
    └── FlatFile        → renders artifacts into a markdown file
"""

from __future__ import annotations

import hashlib
import inspect
import json
from abc import abstractmethod
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Artifact:
    """Immutable, versioned build output."""

    label: str
    artifact_type: str  # "transcript", "episode", "rollup", "core_memory", "search_index"
    content: str
    artifact_id: str = ""
    input_ids: list[str] = field(default_factory=list)
    prompt_id: str | None = None
    model_config: dict | None = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.artifact_id and isinstance(self.content, str):
            self.artifact_id = f"sha256:{hashlib.sha256(self.content.encode()).hexdigest()}"


@dataclass
class ProvenanceRecord:
    """Lineage record for an artifact."""

    label: str
    parent_labels: list[str] = field(default_factory=list)
    prompt_id: str | None = None
    model_config: dict | None = None
    created_at: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Layer hierarchy
# ---------------------------------------------------------------------------


class Layer:
    """Abstract base for all pipeline nodes.

    All pipeline nodes (sources, transforms, projections) are Layers.
    Layers form a DAG via depends_on references. Level is computed
    from DAG depth, never user-specified.
    """

    def __init__(
        self,
        name: str,
        *,
        depends_on: list[Layer] | None = None,
        config: dict | None = None,
    ):
        self.name = name
        self.depends_on: list[Layer] = depends_on or []
        self.config: dict = config or {}
        self._level: int = 0  # computed by compute_levels()


class Source(Layer):
    """Root layer — reads source files. No dependencies.

    Default implementation reads files via built-in Parse logic.
    Subclass and override load() for custom source formats (JSON, CSV, API).
    """

    def __init__(
        self,
        name: str,
        *,
        dir: str | None = None,
        config: dict | None = None,
    ):
        super().__init__(name, config=config)
        self.dir = dir  # override pipeline.source_dir

    def load(self, config: dict) -> list[Artifact]:
        """Read source files and return artifacts.

        Default: delegates to the built-in parse logic (adapter registry).
        Override for custom source formats (JSON, CSV, API).
        """
        from synix.build.parse_transform import ParseTransform

        transform = ParseTransform()
        load_config = dict(config)
        if self.dir:
            load_config["source_dir"] = self.dir
        return transform.execute([], load_config)


class Transform(Layer):
    """Layer that processes inputs into new artifacts.

    Transforms must be stateless during execution. Do not modify ``self``
    in ``execute()`` or ``split()``. Constructor state is read-only after init.

    The runner uses ``copy.copy(transform)`` per thread as a best-effort
    guard against accidental top-level ``self`` mutation, but this is shallow
    and does NOT isolate nested mutable state.
    """

    prompt_name: str | None = None  # prompt template file (built-ins set this)

    def __init__(
        self,
        name: str,
        *,
        depends_on: list[Layer] | None = None,
        uses: list[Layer] | None = None,
        config: dict | None = None,
        context_budget: int | None = None,
        batch: bool | None = None,
    ):
        super().__init__(name, depends_on=depends_on, config=config)
        self.uses: list[Layer] = uses or []
        self.context_budget = context_budget
        self.batch = batch

    @abstractmethod
    def execute(self, inputs: list[Artifact], ctx: TransformContext) -> list[Artifact]:
        """Transform input artifacts into output artifacts.

        Returns a list because some transforms produce multiple outputs
        (e.g., one episode per conversation, one rollup per month).
        """
        ...

    def split(self, inputs: list[Artifact], ctx: TransformContext) -> list[tuple[list[Artifact], dict]]:
        """Split inputs into independently-processable work units.

        Each unit is (unit_inputs, config_extras). The runner calls split()
        to determine parallelism, then executes each unit (potentially
        concurrently) via execute(unit_inputs, ctx.with_updates(config_extras)).

        Default: 1:1 — one unit per input artifact. When inputs is empty
        (e.g., source/parse transforms), returns a single unit so execute()
        is still called. Override for transforms that need different
        decomposition (e.g., N:1 for core synthesis, group-by-month for
        monthly rollup).
        """
        if not inputs:
            return [(inputs, {})]
        return [([inp], {}) for inp in inputs]

    def estimate_output_count(self, input_count: int) -> int:
        """Estimate number of output artifacts for plan mode. Default: 1:1."""
        return input_count

    def get_context(self, value: TransformContext | dict | None = None) -> TransformContext:
        """Normalize runtime config into a public TransformContext."""
        return TransformContext.from_value(value)

    def get_search_surface(
        self,
        ctx: TransformContext | dict | None,
        surface: str | Layer | None = None,
        *,
        required: bool = False,
    ):
        """Resolve a declared build-time search surface from the runtime context."""
        context = self.get_context(ctx)
        return context.search(surface=surface, transform=self, required=required)

    def load_prompt(self, name: str) -> str:
        """Load a prompt template from the prompts/ directory."""
        from synix.build.transforms import PROMPTS_DIR

        path = PROMPTS_DIR / f"{name}.txt"
        return path.read_text()

    def get_prompt_id(self, template_name: str) -> str:
        """Generate a versioned prompt ID from the template content hash."""
        content = self.load_prompt(template_name)
        hash_prefix = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"{template_name}_v{hash_prefix}"

    def get_cache_key(self, config: dict) -> str:
        """Return a hash of transform-specific config that affects output.

        Override in subclasses to include config values that should
        invalidate the cache when changed (e.g., topics list, context_budget).
        Default returns empty string (no extra cache key).
        """
        return ""

    def compute_fingerprint(self, config: dict):
        """Compute this transform's identity fingerprint.

        Components:
          transform_id -- hash of module + qualname (class identity)
          source       -- hash of this class's source code
          prompt       -- hash of prompt template (if prompt_name set)
          config       -- result of get_cache_key() (transform-specific config)
          model        -- hash of llm_config dict (if present)

        Subclasses can override to add/change components.
        """
        from synix.build.fingerprint import Fingerprint, compute_digest, fingerprint_value

        components: dict[str, str] = {}

        # Transform class identity (module + qualname)
        components["transform_id"] = fingerprint_value(f"{type(self).__module__}.{type(self).__qualname__}")

        # Source code of the concrete transform class
        try:
            components["source"] = fingerprint_value(inspect.getsource(type(self)))
        except (OSError, TypeError):
            components["source"] = fingerprint_value(type(self).__qualname__)

        # Prompt template
        if self.prompt_name:
            try:
                components["prompt"] = fingerprint_value(self.load_prompt(self.prompt_name))
            except (FileNotFoundError, OSError):
                pass

        # Transform-specific config (reuses existing get_cache_key override point)
        cache_key = self.get_cache_key(config)
        if cache_key:
            components["config"] = cache_key  # already a hash from subclass

        # Model config
        llm_config = config.get("llm_config")
        if llm_config:
            components["model"] = fingerprint_value(llm_config)

        if self.uses:
            use_signatures = []
            for layer in self.uses:
                if hasattr(layer, "usage_signature"):
                    use_signatures.append(json.dumps(layer.usage_signature(), sort_keys=True, default=str))
                else:
                    use_signatures.append(f"{type(layer).__module__}.{type(layer).__qualname__}:{layer.name}")
            components["uses"] = fingerprint_value(use_signatures)

        return Fingerprint(
            scheme="synix:transform:v2",
            digest=compute_digest(components),
            components=components,
        )


class SearchSurface(Layer):
    """Named searchable build-time capability over a set of source layers."""

    def __init__(
        self,
        name: str,
        *,
        sources: list[Layer],
        modes: list[str] | None = None,
        embedding_config: dict | None = None,
        config: dict | None = None,
    ):
        super().__init__(name, depends_on=list(sources), config=config or {})
        self.sources = sources
        self.modes = modes or ["fulltext"]
        self.search = self.modes  # compatibility alias for older config paths
        self.embedding_config = embedding_config or {}

    def usage_signature(self) -> dict:
        """Stable identity for transforms that declare uses=[this surface]."""
        return {
            "type": f"{type(self).__module__}.{type(self).__qualname__}",
            "name": self.name,
            "sources": [source.name for source in self.sources],
            "modes": list(self.modes),
            "embedding_config": dict(self.embedding_config),
            "config": dict(self.config),
        }


class SearchIndex(Layer):
    """Projection compatibility layer for ``build/search.db``.

    This remains a projection output, not a build-time capability, and cannot
    satisfy ``uses=[...]`` declarations.
    """

    def __init__(
        self,
        name: str,
        *,
        sources: list[Layer],
        search: list[str] | None = None,
        embedding_config: dict | None = None,
        config: dict | None = None,
    ):
        super().__init__(name, depends_on=list(sources), config=config or {})
        self.sources = sources
        self.search = search or ["fulltext"]
        self.embedding_config = embedding_config or {}


class FlatFile(Layer):
    """Projection — renders artifacts into a markdown context document."""

    def __init__(
        self,
        name: str,
        *,
        sources: list[Layer],
        output_path: str = "./build/context.md",
        config: dict | None = None,
    ):
        super().__init__(name, depends_on=list(sources), config=config or {})
        self.sources = sources
        self.output_path = output_path


class TransformContext(MutableMapping[str, Any]):
    """Runtime context passed to transforms.

    It behaves like the legacy config dict for compatibility, but also exposes
    explicit capability interfaces such as ``ctx.search(...)``.
    """

    _RUNTIME_ONLY_KEYS = frozenset(
        {
            "llm_config",
            "search_surface",
            "search_surfaces",
            "search_db_path",
            "workspace",
            "_logger",
            "_layer_name",
            "_shared_llm_client",
        }
    )

    def __init__(self, data: dict[str, Any] | None = None):
        self._data: dict[str, Any] = dict(data or {})

    @classmethod
    def from_value(cls, value: TransformContext | dict | None = None) -> TransformContext:
        """Wrap a dict-like config into a TransformContext."""
        if isinstance(value, cls):
            return value
        return cls(value)

    def with_updates(self, updates: dict[str, Any] | None = None) -> TransformContext:
        """Return a new context with ``updates`` merged on top."""
        merged = dict(self._data)
        if updates:
            merged.update(updates)
        return TransformContext(merged)

    def search(
        self,
        surface: str | Layer | None = None,
        *,
        transform: Transform | None = None,
        required: bool = False,
    ):
        """Resolve an explicit search surface handle for transform code."""
        from synix.core.search_handles import resolve_search_surface_handle

        return resolve_search_surface_handle(self._data, surface=surface, transform=transform, required=required)

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow dict copy of the underlying data."""
        return dict(self._data)

    def copy(self) -> dict[str, Any]:
        """Return a shallow dict copy for legacy ``config.copy()`` callers."""
        return dict(self._data)

    @property
    def config(self) -> dict[str, Any]:
        """User-facing transform config without runtime-injected capabilities."""
        return {k: v for k, v in self._data.items() if not self._is_runtime_only_key(k)}

    @property
    def llm_config(self) -> dict[str, Any]:
        """Resolved LLM configuration for this invocation."""
        llm_config = self._data.get("llm_config", {})
        return dict(llm_config) if isinstance(llm_config, dict) else {}

    @property
    def workspace(self) -> dict[str, Any]:
        """Build-scoped workspace metadata for this invocation."""
        workspace = self._data.get("workspace", {})
        return dict(workspace) if isinstance(workspace, dict) else {}

    @property
    def logger(self) -> Any:
        """Structured logger injected by the runner when available."""
        return self._data.get("_logger")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    @classmethod
    def _is_runtime_only_key(cls, key: str) -> bool:
        """Return True when ``key`` is injected by the runner rather than user config."""
        return key.startswith("_") or key in cls._RUNTIME_ONLY_KEYS


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline:
    """Pipeline definition. Separates the DAG (layers) from post-build operations."""

    def __init__(
        self,
        name: str,
        *,
        source_dir: str = "./sources",
        build_dir: str = "./build",
        synix_dir: str | None = None,
        llm_config: dict | None = None,
        concurrency: int = 5,
    ):
        self.name = name
        self.source_dir = source_dir
        self.build_dir = build_dir
        self.synix_dir = synix_dir
        self.llm_config: dict = llm_config or {}
        self.concurrency = concurrency
        self.layers: list[Layer] = []  # Source + Transform
        self.surfaces: list[Layer] = []  # SearchSurface
        self.projections: list[Layer] = []  # SearchIndex + FlatFile
        self.validators: list = []  # untyped to avoid circular import with validators.py
        self.fixers: list = []  # untyped to avoid circular import with fixers.py

    def add(self, *layers: Layer) -> None:
        """Add layers to the pipeline.

        Source and Transform go into the build DAG.
        SearchSurface goes into build-time search surfaces.
        SearchIndex and FlatFile go into projections (separate lifecycle).
        """
        for layer in layers:
            if isinstance(layer, (SearchIndex, FlatFile)):
                self.projections.append(layer)
            elif isinstance(layer, SearchSurface):
                self.surfaces.append(layer)
            else:
                self.layers.append(layer)

    def add_validator(self, validator) -> None:
        """Add a post-build domain validator."""
        self.validators.append(validator)

    def add_fixer(self, fixer) -> None:
        """Add a violation fixer."""
        self.fixers.append(fixer)
