"""Core data models for Synix.

Layer hierarchy:
    Layer (abstract base — name, depends_on)
    ├── Source          → reads source files (root nodes, no depends_on)
    ├── Transform       → processes inputs via split()/execute()
    │   ├── EpisodeSummary, MonthlyRollup, TopicalRollup, CoreSynthesis, Merge (built-in)
    │   └── (user-defined subclasses)
    ├── SearchIndex     → materializes artifacts into FTS5 + embeddings
    └── FlatFile        → renders artifacts into a markdown file
"""

from __future__ import annotations

import hashlib
import inspect
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


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
        if not isinstance(self.content, str):
            msg = f"Artifact content must be a string, got {type(self.content).__name__}"
            raise TypeError(msg)
        if not self.artifact_id:
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
        config: dict | None = None,
        context_budget: int | None = None,
        batch: bool | None = None,
    ):
        super().__init__(name, depends_on=depends_on, config=config)
        self.context_budget = context_budget
        self.batch = batch

    @abstractmethod
    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        """Transform input artifacts into output artifacts.

        Returns a list because some transforms produce multiple outputs
        (e.g., one episode per conversation, one rollup per month).
        """
        ...

    def split(self, inputs: list[Artifact], config: dict) -> list[tuple[list[Artifact], dict]]:
        """Split inputs into independently-processable work units.

        Each unit is (unit_inputs, config_extras). The runner calls split()
        to determine parallelism, then executes each unit (potentially
        concurrently) via execute(unit_inputs, {**config, **config_extras}).

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

        return Fingerprint(
            scheme="synix:transform:v2",
            digest=compute_digest(components),
            components=components,
        )


class SearchIndex(Layer):
    """Projection — materializes artifacts into a searchable index."""

    def __init__(
        self,
        name: str,
        *,
        sources: list[Layer],
        search: list[str] | None = None,
        embedding_config: dict | None = None,
        config: dict | None = None,
    ):
        # SearchIndex depends on its source layers
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
        self.projections: list[Layer] = []  # SearchIndex + FlatFile
        self.validators: list = []  # untyped to avoid circular import with validators.py
        self.fixers: list = []  # untyped to avoid circular import with fixers.py

    def add(self, *layers: Layer) -> None:
        """Add layers to the pipeline.

        Source and Transform go into the build DAG.
        SearchIndex and FlatFile go into projections (separate lifecycle).
        """
        for layer in layers:
            if isinstance(layer, (SearchIndex, FlatFile)):
                self.projections.append(layer)
            else:
                self.layers.append(layer)

    def add_validator(self, validator) -> None:
        """Add a post-build domain validator."""
        self.validators.append(validator)

    def add_fixer(self, fixer) -> None:
        """Add a violation fixer."""
        self.fixers.append(fixer)
