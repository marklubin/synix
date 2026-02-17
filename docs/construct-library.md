# Construct Library — Scaling Synix's Building Blocks

> Design document for a unified pattern to safely expand Synix along every dimension: source adapters, projection targets, transforms, validators, fixers, pipeline fragments, and template pipelines.

## Problem

Synix needs to grow from a handful of built-in integrations to a broad library of composable building blocks. Today we have 3 source adapters (ChatGPT, Claude, text), 2 projection types (SearchIndex, FlatFile), 4 ext transforms (Map, Group, Reduce, Fold), a few validators (PII, SemanticConflict, RequiredField, MutualExclusion, Citation), and a few fixers (SemanticEnrichment, CitationEnrichment). Each of these was hand-built with ad-hoc conventions.

What we need:

1. A **scalable pattern** for adding n+1 of any component type with minimal per-component effort
2. **Conformance test suites** that automatically validate new components without writing per-component tests
3. **Composition primitives** (pipeline fragments, template pipelines) for higher-level reuse
4. **Claude Code automation** so an agent can scaffold, implement, test, and validate a new component end-to-end
5. All of this without introducing a universal base class or framework that fights the existing architecture

## Design Principles

**Shared lifecycle, separate interfaces.** Borrowed from CDK/Terraform/dbt: components share metadata conventions and testing infrastructure, but each component type keeps its own interface. No universal `Component` base class — that would fight the existing `Layer` hierarchy.

**L1/L2/L3 construct levels.** Inspired by CDK:

| Level | What it is | Synix examples |
|-------|-----------|----------------|
| L1 — Primitives | Core abstractions, changed rarely | `Source`, `Transform`, `SearchIndex`, `FlatFile`, `BaseValidator`, `BaseFixer` |
| L2 — Constructs | Opinionated implementations of L1 | `SlackAdapter`, `MapSynthesis`, `JsonApiProjection`, `PIIValidator` |
| L3 — Patterns | Compositions of L2s | `AgentMemoryFragment`, `CustomerServiceTemplate` |

This document covers the **infrastructure** for L2 and L3. L1 primitives are already stable.

**Test the output, not the internals.** Every component ultimately feeds into the build pipeline and produces artifacts. The conformance framework tests observable behavior (artifact structure, metadata, content properties) rather than internal implementation details.

## Component Types

### Overview

| Component Type | L1 Base | Registration | Key Contract |
|---------------|---------|-------------|-------------|
| Source Adapter | `adapters/registry.py` | `@register_adapter(extensions)` | `(Path) -> list[Artifact]` |
| Transform | `core/models.Transform` | `Pipeline.add()` | `execute(inputs, config) -> list[Artifact]` |
| Projection Target | `build/projections.BaseProjection` | `get_projection()` dispatch | `materialize(artifacts, config) -> None` |
| Validator | `build/validators.BaseValidator` | `Pipeline.add_validator()` | `validate(artifacts, ctx) -> list[Violation]` |
| Fixer | `build/fixers.BaseFixer` | `Pipeline.add_fixer()` | `fix(violation, ctx) -> FixAction` |
| Pipeline Fragment | new: `PipelineFragment` | `fragment.instantiate(pipeline)` | `instantiate(pipeline, **kwargs) -> list[Layer]` |
| Template Pipeline | `templates/` directory | `synix init --template` | Complete `pipeline.py` + sources + golden files |

### Source Adapters

**Current pattern.** Adapter functions are registered via `@register_adapter(extensions)` in `adapters/registry.py`. Each adapter is a function `(Path) -> list[Artifact]`. JSON files use auto-detection (ChatGPT vs Claude sniffing).

**Target pattern.** Keep the function-based registry for simple file-format adapters. For richer sources (API-based, multi-file, stateful auth), introduce an optional `SourceAdapter` class:

```python
# synix/adapters/base.py

class SourceAdapter(ABC):
    """Base class for source adapters that need more than a parse function.

    Simple file parsers should use @register_adapter instead.
    SourceAdapter is for sources that need configuration, authentication,
    or multi-step ingestion (e.g., Slack API, Notion export).
    """

    name: str = ""                    # e.g., "slack", "notion"
    extensions: list[str] = []        # file extensions this adapter handles
    requires: list[str] = []          # optional dependency packages

    @abstractmethod
    def parse(self, filepath: Path) -> list[Artifact]:
        """Parse a file into artifacts."""
        ...

    def configure(self, config: dict) -> None:
        """Optional: configure the adapter (API keys, pagination, etc.)."""
        pass

    @classmethod
    def create_sample(cls, target_dir: Path) -> None:
        """Create sample source files for demos/testing."""
        pass
```

Both patterns (function and class) coexist. `@register_adapter` remains the recommended path for file-format parsers. `SourceAdapter` subclasses auto-register via `__init_subclass__` or an explicit `register()` call.

**Required metadata contract.** Every source adapter must produce artifacts with at minimum:

```python
metadata = {
    "source": "<adapter-name>",           # "chatgpt", "claude", "slack", etc.
    "source_conversation_id": "<id>",      # unique within source
    "title": "<title>",
    "date": "<YYYY-MM-DD>",
    "message_count": <int>,
    "last_message_date": "<YYYY-MM-DD>",   # for freshness sorting
}
```

And artifact labels must follow the `t-{source}-{id}` convention.

### Transforms (ext library)

**Current pattern.** `synix.ext` provides `MapSynthesis`, `GroupSynthesis`, `ReduceSynthesis`, `FoldSynthesis` — all subclasses of `Transform` with configurable prompts and label functions.

**Target pattern.** Continue this approach. New ext transforms follow the existing conventions:

- Subclass `Transform`
- Accept `prompt`, `artifact_type`, `label_fn` in constructor
- Override `get_cache_key()` to include prompt/config in fingerprint
- Override `split()` when parallelism strategy differs from 1:1
- Use `_get_llm_client(config)` and `_logged_complete()` for LLM calls

**What's new:** A `TransformConstruct` metadata mixin (see [Shared Metadata](#shared-metadata)) and conformance tests that auto-discover all ext transforms.

### Projection Targets

**Current pattern.** `BaseProjection` ABC with `materialize(artifacts, config)`. Two implementations: `FlatFileProjection` and `SearchIndexProjection`. Dispatch is via hardcoded `get_projection()`.

**Target pattern.** Registry-based dispatch (like adapters) with a `ProjectionTarget` base:

```python
# build/projections.py additions

_PROJECTION_REGISTRY: dict[str, type[BaseProjection]] = {}

def register_projection(name: str):
    """Decorator to register a projection implementation."""
    def decorator(cls):
        _PROJECTION_REGISTRY[name] = cls
        return cls
    return decorator

def get_projection(name: str, *args, **kwargs) -> BaseProjection:
    cls = _PROJECTION_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown projection: {name}. Available: {list(_PROJECTION_REGISTRY)}")
    return cls(*args, **kwargs)
```

New projection types (JSON API, webhook, CLAUDE.md renderer, etc.) register via decorator, then correspond to a new `Layer` subclass in `core/models.py` — or reuse `FlatFile`/`SearchIndex` with different materialization logic.

### Validators

**Current pattern.** `BaseValidator` ABC with `validate(artifacts, ctx) -> list[Violation]`. Typed constructors (layers, patterns, severity). Fail-closed by default.

**Target pattern.** No structural changes needed — the validator framework is already well-abstracted. New validators subclass `BaseValidator`, get conformance-tested automatically.

### Fixers

**Current pattern.** `BaseFixer` ABC with `fix(violation, ctx) -> FixAction`. Typed constructors. Handles specific violation types.

**Target pattern.** Same as validators — the framework is solid. New fixers subclass `BaseFixer`, get conformance-tested automatically.

### Pipeline Fragments (L3)

**New concept.** A `PipelineFragment` is a reusable composition of multiple layers — analogous to CDK L3 constructs or Terraform modules. It wires together L2 components into a higher-level building block.

```python
# synix/fragments.py

class PipelineFragment(ABC):
    """Reusable composition of layers.

    A fragment encapsulates a common pipeline pattern (e.g., "ingest + summarize + index")
    and instantiates the concrete layers when added to a pipeline.
    """

    name: str = ""
    description: str = ""
    requires: list[str] = []  # optional dependency packages

    @abstractmethod
    def instantiate(self, pipeline: Pipeline, **kwargs) -> list[Layer]:
        """Create and wire layers, returning the output layers.

        Callers can use the returned layers as depends_on for downstream layers.
        The fragment is responsible for calling pipeline.add() on internal layers.
        """
        ...
```

Example:

```python
class AgentMemoryFragment(PipelineFragment):
    """Standard agent memory pipeline: transcripts -> episodes -> monthly -> core."""

    name = "agent-memory"
    description = "Full agent memory pipeline from transcripts to core memory"

    def instantiate(self, pipeline, *, context_budget=10000):
        transcripts = Source("transcripts")
        episodes = EpisodeSummary("episodes", depends_on=[transcripts])
        monthly = MonthlyRollup("monthly", depends_on=[episodes])
        core = CoreSynthesis("core", depends_on=[monthly], context_budget=context_budget)

        pipeline.add(transcripts, episodes, monthly, core)
        return [core]  # downstream layers depend on core
```

Usage:

```python
pipeline = Pipeline("my-pipeline")

# Fragment wires up the standard memory layers
memory = AgentMemoryFragment()
[core] = memory.instantiate(pipeline, context_budget=15000)

# User adds their own layers on top
pipeline.add(SearchIndex("search", sources=[core], search=["fulltext"]))
```

### Template Pipelines

**Current pattern.** Complete demo pipelines live in `src/synix/templates/`. Each template has `pipeline.py`, `sources/`, and `golden/` for verified output.

**Target pattern.** No structural change. Templates are the top-level "L3" — complete working examples. The construct library makes templates thinner: they compose fragments and configure L2 constructs rather than defining everything from scratch.

## Shared Metadata

Rather than a universal base class, shared metadata is mixed into each component type via a convention and optional mixin:

```python
# synix/constructs.py

class ComponentMeta:
    """Mixin for component metadata — discovery, documentation, testing.

    Mixed into each component type's base class. Not a standalone base.
    Provides a consistent interface for tooling (CLI discovery, scaffolding,
    conformance tests) without changing the component's primary interface.
    """

    component_name: str = ""          # human-readable name
    component_type: str = ""          # "adapter", "transform", "projection", "validator", "fixer", "fragment"
    description: str = ""             # one-line description
    requires: list[str] = []          # optional pip extras needed
    version: str = "0.1.0"           # semver for the component

    @classmethod
    def create_sample(cls, target_dir: Path) -> None:
        """Create sample input files for testing/demos. Optional."""
        pass
```

This is opt-in. Existing components (PII validator, SemanticConflict, etc.) don't need to add it immediately. New components should include it for discoverability.

## Conformance Test Framework

The key insight: **parametrized test suites that auto-discover components**. Adding a new source adapter or transform automatically gets test coverage without writing per-component test files.

### Per-Type Conformance Suites

```python
# tests/conformance/test_adapters.py

import pytest
from synix.adapters.registry import _ADAPTERS

def discover_adapters():
    """Yield (extension, parser_fn) for each registered adapter."""
    for ext, fn in _ADAPTERS.items():
        yield pytest.param(ext, fn, id=f"adapter-{ext}")

@pytest.mark.parametrize("ext,parse_fn", discover_adapters())
class TestAdapterConformance:
    """Every registered adapter must pass these tests."""

    def test_returns_list_of_artifacts(self, ext, parse_fn, adapter_sample_file):
        """parse_fn returns list[Artifact], possibly empty."""
        result = parse_fn(adapter_sample_file(ext))
        assert isinstance(result, list)
        assert all(isinstance(a, Artifact) for a in result)

    def test_required_metadata(self, ext, parse_fn, adapter_sample_file):
        """Every artifact has required metadata keys."""
        result = parse_fn(adapter_sample_file(ext))
        required_keys = {"source", "source_conversation_id", "title", "date", "message_count"}
        for artifact in result:
            missing = required_keys - set(artifact.metadata.keys())
            assert not missing, f"Adapter {ext} missing metadata: {missing}"

    def test_label_convention(self, ext, parse_fn, adapter_sample_file):
        """Labels follow t-{source}-{id} convention."""
        result = parse_fn(adapter_sample_file(ext))
        for artifact in result:
            assert artifact.label.startswith("t-"), f"Bad label: {artifact.label}"

    def test_content_not_empty(self, ext, parse_fn, adapter_sample_file):
        """Artifacts have non-empty content."""
        result = parse_fn(adapter_sample_file(ext))
        for artifact in result:
            assert artifact.content.strip(), f"Empty content in {artifact.label}"

    def test_artifact_id_computed(self, ext, parse_fn, adapter_sample_file):
        """Artifact IDs are SHA256 content hashes."""
        result = parse_fn(adapter_sample_file(ext))
        for artifact in result:
            assert artifact.artifact_id.startswith("sha256:")
```

Similar suites for transforms, projections, validators, and fixers.

### BuildHarness

Inspired by CDK's `Template.from_stack()` — test build outputs without running the full pipeline:

```python
# tests/harness.py

class BuildHarness:
    """Test harness for pipeline components.

    Runs a minimal pipeline through the build runner and captures outputs
    for assertion. Handles tmp directories, mock LLM, and cleanup.
    """

    def __init__(self, layers, *, source_files=None, llm_responses=None):
        self.layers = layers
        self.source_files = source_files or {}
        self.llm_responses = llm_responses or {}
        self._artifacts = None
        self._projections = None

    def build(self) -> BuildHarness:
        """Execute the pipeline and capture results."""
        # Set up tmp dirs, mock LLM, run build, capture artifacts
        ...
        return self

    def artifacts(self, layer_name=None) -> list[Artifact]:
        """Get output artifacts, optionally filtered by layer."""
        ...

    def assert_artifact_count(self, layer_name, expected):
        """Assert number of artifacts from a layer."""
        ...

    def assert_metadata_present(self, layer_name, *keys):
        """Assert all artifacts from layer have these metadata keys."""
        ...

    def assert_no_violations(self, validator):
        """Assert a validator produces no violations on the output."""
        ...
```

Usage:

```python
def test_slack_adapter_in_pipeline():
    harness = BuildHarness(
        layers=[Source("transcripts")],
        source_files={"slack_export.json": SLACK_SAMPLE_DATA},
    ).build()

    harness.assert_artifact_count("transcripts", 5)
    harness.assert_metadata_present("transcripts", "source", "date", "message_count")
```

### Sample Data Convention

Every adapter and transform must provide sample data for conformance tests:

```
tests/
  fixtures/
    adapters/
      chatgpt_sample.json      # minimal valid ChatGPT export
      claude_sample.json        # minimal valid Claude export
      slack_sample.json         # minimal valid Slack export (when added)
      text_sample.txt           # minimal text file
    transforms/
      map_input.json            # sample input for MapSynthesis
      ...
```

The `adapter_sample_file` fixture resolves extension to the right sample file. New adapters must include a sample file — the conformance test fails without one.

## Implementation Roadmap

### Phase 1: Conformance Infrastructure

1. **Adapter conformance suite** — parametrized tests over `_ADAPTERS` registry. Write sample data fixtures for existing adapters (chatgpt, claude, text). Verify all 3 existing adapters pass.

2. **Transform conformance suite** — parametrized tests over `synix.ext` transforms. Mock LLM. Verify MapSynthesis, GroupSynthesis, ReduceSynthesis, FoldSynthesis pass.

3. **Validator conformance suite** — parametrized tests over built-in validators. Verify PII, SemanticConflict, RequiredField, MutualExclusion, Citation pass.

4. **Fixer conformance suite** — parametrized tests over built-in fixers. Verify SemanticEnrichment, CitationEnrichment pass.

### Phase 2: Registration Infrastructure

5. **Projection registry** — replace hardcoded `get_projection()` with decorator-based registry. Register existing FlatFileProjection and SearchIndexProjection.

6. **SourceAdapter base class** — add `adapters/base.py` with optional class-based adapter pattern. Existing function adapters continue to work unchanged.

7. **ComponentMeta mixin** — add `constructs.py` with metadata mixin. Apply to new components going forward, retrofit existing ones opportunistically.

### Phase 3: Composition

8. **PipelineFragment base** — add `fragments.py`. Extract the existing personal memory pipeline pattern into `AgentMemoryFragment` as the reference implementation.

9. **BuildHarness** — add `tests/harness.py`. Use it in existing integration tests as proof of concept.

### Phase 4: First Expansions

10. **First new source adapter** — implement using the pattern (e.g., Slack, JSONL, CSV). Verify it passes conformance automatically.

11. **First new projection** — implement using the registry (e.g., JSON API context document). Verify it passes conformance automatically.

### Phase 5: Automation

12. **Claude Code scaffolding docs** — document the per-component-type implementation checklist in CLAUDE.md so Claude Code can scaffold new components end-to-end:
    - Create adapter file from reference template
    - Create sample data fixture
    - Run conformance tests
    - Update `__init__.py` exports
    - Add optional dependency to pyproject.toml extras

## What NOT to Do

- **No universal `Component` base class.** Each component type has its own interface — forcing them into a common base would fight the existing `Layer` hierarchy and add complexity with no real benefit.

- **No plugin system or entry_points.** Synix components live in-tree. Third-party plugins are a v2 concern. The registry pattern is sufficient for now.

- **No auto-discovery via filesystem scanning.** Components register explicitly via decorators or `Pipeline.add()`. Magic import scanning creates hard-to-debug ordering issues.

- **No config schema validation framework.** Components use typed constructors with Python type hints. Adding JSON Schema or Pydantic validation for component configs is premature — the typed constructor pattern is clear and testable.

- **No backwards-compatibility shims.** When a component interface changes, update all implementations. The codebase is small enough that this is cheaper than maintaining compatibility layers.

- **No separate packages per component.** Everything stays in `synix`. Optional heavy dependencies (fastembed, slack-sdk) use pyproject.toml extras (`synix[slack]`, `synix[embeddings]`).

## Dependency Management

Heavy optional dependencies are declared in `pyproject.toml` extras:

```toml
[project.optional-dependencies]
slack = ["slack-sdk>=3.0"]
notion = ["notion-client>=2.0"]
embeddings = ["fastembed>=0.2"]
```

Components that require optional dependencies:
- Import lazily (inside methods, not at module top)
- Declare `requires = ["slack"]` in their metadata
- Raise `ImportError` with a helpful message: `"pip install synix[slack]" or "uvx synix[slack]"`

Built-in components (ChatGPT, Claude, text, FTS5, FlatFile) must never require optional dependencies.

## File Layout

```
src/synix/
├── constructs.py              # ComponentMeta mixin
├── fragments.py               # PipelineFragment ABC
├── adapters/
│   ├── base.py                # SourceAdapter class (optional, for rich adapters)
│   ├── registry.py            # existing function registry (unchanged)
│   ├── chatgpt.py             # existing
│   ├── claude.py              # existing
│   ├── text.py                # existing
│   ├── slack.py               # future L2
│   └── csv_adapter.py         # future L2
├── ext/
│   ├── map_synthesis.py       # existing L2
│   ├── group_synthesis.py     # existing L2
│   ├── reduce_synthesis.py    # existing L2
│   ├── fold_synthesis.py      # existing L2
│   └── ...                    # future L2 transforms
├── build/
│   ├── projections.py         # existing + projection registry
│   ├── validators.py          # existing (unchanged)
│   └── fixers.py              # existing (unchanged)
├── fragments/
│   └── agent_memory.py        # reference L3 fragment
└── templates/
    ├── 01-personal-memory/    # existing template
    ├── 02-tv-returns/         # existing template
    └── 03-team-report/        # existing template

tests/
├── conformance/
│   ├── conftest.py            # shared fixtures (sample data resolver, build harness)
│   ├── test_adapters.py       # adapter conformance suite
│   ├── test_transforms.py     # transform conformance suite
│   ├── test_validators.py     # validator conformance suite
│   └── test_fixers.py         # fixer conformance suite
├── fixtures/
│   └── adapters/              # sample data per adapter
└── harness.py                 # BuildHarness utility
```

## Success Criteria

The pattern is working when:

1. **Adding a new source adapter** requires: one Python file, one sample data fixture, zero test files. Conformance tests catch regressions automatically.
2. **Adding a new ext transform** requires: one Python file, zero test files. Conformance tests verify the Transform contract.
3. **Adding a new projection** requires: one Python file + `@register_projection`, zero test files beyond the conformance suite.
4. **Adding a new validator/fixer** requires: one Python file, zero test files beyond the conformance suite.
5. **Composing a pipeline fragment** requires: one Python file that wires L2 constructs. The BuildHarness tests the composition end-to-end.
6. **Claude Code can scaffold** a new component from just a name and type, run conformance tests, and verify it passes — without human intervention for the mechanical parts.
