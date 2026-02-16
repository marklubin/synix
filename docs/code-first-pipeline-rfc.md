# RFC: Code-First Pipeline Declaration

**Status**: Draft (rev 2 — addresses staff review)
**Author**: Mark Lubin
**Date**: 2026-02-15
**Revision**: 2026-02-16 — incorporates staff review findings

## Problem

Synix pipeline definitions currently use a string-based dispatch DSL. Transforms, validators, fixers, and projections are referenced by name strings that get resolved at runtime via decorator-populated registries. This creates unnecessary indirection:

```python
# 1. Define the class
@register_transform("work_style")
class WorkStyleTransform(BaseTransform):
    def execute(self, inputs, config): ...

# 2. Reference it by string (why?)
pipeline.add_layer(Layer(name="work_styles", level=1, depends_on=["bios"], transform="work_style"))
```

The string dispatch adds ceremony (`@register_transform` decorator), indirection (the runner resolves `"work_style"` back to `WorkStyleTransform` at runtime), and makes pipelines harder to navigate (you can't click through `"work_style"` to the class definition). The same pattern repeats for validators (`ValidatorDecl(name="citation", config={...})`), fixers (`FixerDecl(name="citation_enrichment", config={...})`), and projections (`Projection(projection_type="search_index", ...)`).

## Proposal

Make pipeline definitions code-first: pass actual implementation objects directly. Keep string dispatch as a backward-compatible fallback.

### Before

```python
from synix import Layer, Pipeline, Projection, ValidatorDecl
from synix.build.transforms import BaseTransform, register_transform
from synix.build.validators import BaseValidator, register_validator

@register_transform("work_style")
class WorkStyleTransform(BaseTransform):
    def execute(self, inputs, config): ...

@register_validator("max_length")
class MaxLengthValidator(BaseValidator):
    def validate(self, artifacts, ctx): ...

pipeline = Pipeline("my-pipeline")
pipeline.llm_config = {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"}

pipeline.add_layer(Layer(name="bios", level=0, transform="parse"))
pipeline.add_layer(Layer(name="work_styles", level=1, depends_on=["bios"], transform="work_style"))
pipeline.add_validator(ValidatorDecl(name="citation", config={"layers": ["work_styles"]}))
pipeline.add_validator(ValidatorDecl(name="max_length", config={"layers": ["work_styles"], "max_chars": 5000}))
pipeline.add_projection(Projection(name="search", projection_type="search_index", sources=[...]))
```

### After

```python
from synix import Pipeline, Layer
from synix.transforms import BaseTransform, Parse
from synix.validators import Citation, BaseValidator
from synix.projections import SearchIndex

class WorkStyleTransform(BaseTransform):  # No decorator needed
    def execute(self, inputs, config): ...

class MaxLengthValidator(BaseValidator):  # No decorator needed
    def validate(self, artifacts, ctx): ...

pipeline = Pipeline("my-pipeline")
pipeline.llm_config = {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"}

pipeline.add_layer(Layer(name="bios", level=0, transform=Parse(source_dir="./sources/bios")))
pipeline.add_layer(Layer(name="work_styles", level=1, depends_on=["bios"], transform=WorkStyleTransform()))
pipeline.add_validator(Citation(layers=["work_styles"]))
pipeline.add_validator(MaxLengthValidator(layers=["work_styles"], max_chars=5000))
pipeline.add_projection(SearchIndex("search", sources=[...]))
```

## Design

### Transform resolution

`Layer.transform` accepts two forms:

| Form | Example | Resolution |
|------|---------|------------|
| **Instance** (preferred) | `transform=WorkStyleTransform()` | Used directly |
| **String** (backward compat) | `transform="work_style"` | Registry lookup via `get_transform()` |

> **Design note (staff review #5)**: Class-form (`transform=WorkStyleTransform`) is intentionally not supported. It implies a zero-arg constructor contract that conflicts with typed constructors. Use instance form for explicit configuration, string form for backward compat.

A new `resolve_transform()` function handles both:

```python
def resolve_transform(ref):
    if isinstance(ref, BaseTransform):
        return ref
    if isinstance(ref, str):
        return get_transform(ref)  # existing registry
    raise TypeError(
        f"Layer.transform must be a BaseTransform instance or string name. Got: {type(ref).__name__}"
    )
```

The runner and plan module call `resolve_transform(layer.transform)` instead of `get_transform(layer.transform)`. Existing string-based pipelines continue to work unchanged.

### Eliminating `_transform_to_prompt_name()`

Currently the runner maps transform names to prompt files via a hardcoded dictionary duplicated in both `runner.py` and `plan.py`:

```python
def _transform_to_prompt_name(transform_name: str) -> str:
    mapping = {
        "episode_summary": "episode_summary",
        "monthly_rollup": "monthly_rollup",
        "topical_rollup": "topical_rollup",
        "core_synthesis": "core_memory",  # the mismatch that requires this mapping
    }
    return mapping.get(transform_name, transform_name)
```

With code-first, each transform carries its own `prompt_name` class attribute:

```python
class BaseTransform(ABC):
    prompt_name: str | None = None  # Override in subclasses that use prompts

class EpisodeSummaryTransform(BaseTransform):
    prompt_name = "episode_summary"

class CoreSynthesisTransform(BaseTransform):
    prompt_name = "core_memory"
```

The runner reads `transform.prompt_name` directly. The `_transform_to_prompt_name()` function and its duplicate are deleted.

### Lazy builtin registration

Currently `runner.py` and `plan.py` have side-effect imports solely to populate the transform registry:

```python
import synix.build.llm_transforms  # noqa: F401 — triggers @register_transform
import synix.build.merge_transform  # noqa: F401
import synix.build.parse_transform  # noqa: F401
```

These move into a thread-safe lazy guard called by `get_transform()`:

```python
import threading

_builtins_lock = threading.Lock()
_builtins_loaded = False

def _ensure_builtins_registered():
    global _builtins_loaded
    if _builtins_loaded:
        return
    with _builtins_lock:
        if _builtins_loaded:  # double-check after acquiring lock
            return
        import synix.build.parse_transform   # noqa: F401
        import synix.build.llm_transforms    # noqa: F401
        import synix.build.merge_transform   # noqa: F401
        _builtins_loaded = True  # set AFTER imports succeed

def get_transform(name: str) -> BaseTransform:
    _ensure_builtins_registered()
    if name not in _TRANSFORMS:
        raise ValueError(...)
    return _TRANSFORMS[name]()
```

> **Design note (staff review #1)**: The flag is set after imports complete, inside the lock. Double-checked locking pattern ensures thread safety without contention on the hot path. If an import fails, `_builtins_loaded` remains `False` so the next call retries.

When transforms are passed as instances, the registry is never touched and these imports never fire.

### Typed validator/fixer constructors

Validators and fixers currently receive configuration via monkey-patching:

```python
validator = get_validator(decl.name)
validator._config = decl.config  # type: ignore[attr-defined]
validator._field_name = decl.config.get("field", "")  # type: ignore[attr-defined]
```

With code-first, validators get proper typed constructors and a public `to_config_dict()` method that returns the config dict. This replaces the private `_config` monkey-patching:

> **Design note (staff review #4)**: `to_config_dict()` is the public API contract. The `_config` attribute is deprecated — internal code transitions to `to_config_dict()`. The legacy `ValidatorDecl` path still sets `_config` for backward compat; `to_config_dict()` falls back to it.

```python
class BaseValidator(ABC):
    """Abstract base for validators."""
    name: str = ""

    def to_config_dict(self) -> dict:
        """Return config dict for this validator.

        Override in subclasses with typed constructors. The execution engine
        uses this to determine artifact scoping (layers, scope, artifact_ids).
        """
        return getattr(self, "_config", {})  # Fallback for legacy ValidatorDecl path

class CitationValidator(BaseValidator):
    name = "citation"

    def __init__(self, *, layers=None, llm_config=None, max_artifacts=None, fail_open=False):
        self.layers = layers or []
        self.llm_config = llm_config or {}
        self.max_artifacts = max_artifacts
        self.fail_open = fail_open

    def to_config_dict(self) -> dict:
        d: dict = {"layers": self.layers, "fail_open": self.fail_open}
        if self.llm_config:
            d["llm_config"] = self.llm_config
        if self.max_artifacts is not None:
            d["max_artifacts"] = self.max_artifacts
        return d
```

`run_validators()` handles both `ValidatorDecl` (legacy) and `BaseValidator` (code-first):

```python
for entry in pipeline.validators:
    if isinstance(entry, BaseValidator):
        validator = entry
        config = validator.to_config_dict()
    else:
        # Legacy ValidatorDecl path — instantiate and inject config
        validator = get_validator(entry.name)
        validator._config = entry.config
        validator._field_name = entry.config.get("field", "")
        config = entry.config
    artifacts = _gather_artifacts(store, config)
    violations = validator.validate(artifacts, ctx)
```

Same pattern for fixers (`BaseFixer.to_config_dict()`).

**Custom validator contract**: Custom validators that use typed constructors MUST override `to_config_dict()` to return a dict with at minimum `layers` (for artifact scoping). Custom validators that don't override `to_config_dict()` validate all artifacts in the build.

### Public import surface

> **Design note (staff review #6)**: Re-export modules use `__getattr__` lazy loading for built-in classes. This avoids importing the full transform/validator/fixer/projection stack at `import synix.transforms` time. Base classes and utilities are eagerly imported (they're lightweight); concrete built-in implementations are lazy.

```python
# src/synix/transforms/__init__.py
from synix.build.transforms import BaseTransform, register_transform, resolve_transform

# Lazy imports for built-in transforms — avoids import-time side effects
_LAZY_IMPORTS = {
    "Parse": ("synix.build.parse_transform", "ParseTransform"),
    "EpisodeSummary": ("synix.build.llm_transforms", "EpisodeSummaryTransform"),
    "MonthlyRollup": ("synix.build.llm_transforms", "MonthlyRollupTransform"),
    "TopicalRollup": ("synix.build.llm_transforms", "TopicalRollupTransform"),
    "CoreSynthesis": ("synix.build.llm_transforms", "CoreSynthesisTransform"),
    "Merge": ("synix.build.merge_transform", "MergeTransform"),
}

def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, class_name = _LAZY_IMPORTS[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    raise AttributeError(f"module 'synix.transforms' has no attribute {name!r}")

__all__ = ["BaseTransform", "register_transform", "resolve_transform", *_LAZY_IMPORTS]
```

Same pattern for validators, fixers, projections:

```python
# src/synix/validators/__init__.py — base classes eager, built-ins lazy
from synix.build.validators import BaseValidator, register_validator
# Lazy: Citation, SemanticConflict, MutualExclusion, RequiredField, PII
```

```python
# src/synix/fixers/__init__.py — base classes eager, built-ins lazy
from synix.build.fixers import BaseFixer, register_fixer
# Lazy: SemanticEnrichment, CitationEnrichment
```

```python
# src/synix/projections/__init__.py — base classes eager, built-ins lazy
from synix.build.projections import BaseProjection
# Lazy: FlatFile, SearchIndex
```

### Plugin discovery (future work)

> **Design note (staff review #7)**: Third-party transforms/validators/fixers currently require import side effects. A future version should support Python entry-point discovery (`[project.entry-points."synix.transforms"]`) for zero-config plugin loading. This is out of scope for the initial code-first refactor.

## Backward Compatibility

Everything is additive. No existing code breaks:

| Feature | Status |
|---------|--------|
| `Layer(transform="parse")` | Still works (string resolved via registry) |
| `@register_transform("name")` | Still works (populates registry for string dispatch) |
| `ValidatorDecl(name="citation", config={...})` | Still works |
| `FixerDecl(name="citation_enrichment", config={...})` | Still works |
| `Projection(projection_type="search_index", ...)` | Still works |
| `get_transform()` / `get_validator()` / `get_fixer()` | Still works |
| Existing fingerprint caches | Not invalidated (same content hashes) |

## Fingerprint Compatibility

> **Design note (staff review #2)**: Fingerprint correctness requires that any state affecting transform output is captured in the fingerprint.

### Canonical fingerprint payload

Every transform fingerprint MUST include these components:

| Component | Source | Purpose |
|-----------|--------|---------|
| `transform_id` | `module + qualname` of the transform class | Identifies the transform implementation |
| `source` | SHA256 of `inspect.getsource(type(self))` | Detects code changes |
| `prompt` | SHA256 of prompt template content (if `prompt_name` set) | Detects prompt changes |
| `config` | Output of `get_cache_key(config)` | Captures behavior-affecting config |
| `model` | SHA256 of `llm_config` dict | Detects model changes |

The existing `compute_fingerprint()` already computes `source`, `prompt`, `config`, and `model`. The `transform_id` component (`module + qualname`) is new — it distinguishes two transforms with identical source code but different class identities.

### Config fingerprinting contract

**For custom transforms with constructor state**: Override `get_cache_key(config)` to include instance attributes that affect output:

```python
class TopicFilterTransform(BaseTransform):
    def __init__(self, topics: list[str]):
        self.topics = topics

    def get_cache_key(self, config: dict) -> str:
        # Include instance state in fingerprint
        return hashlib.sha256(json.dumps(sorted(self.topics)).encode()).hexdigest()
```

This is the same pattern already used by `TopicalRollupTransform` and `MergeTransform`.

### Fingerprint compatibility matrix

| Scenario | Cache reuse? | Why |
|----------|-------------|-----|
| String pipeline → same string pipeline | Yes | All components identical |
| String pipeline → code-first, same class + config | Yes | Same source, prompt, config, model hashes |
| Code-first → same instance, same constructor args | Yes | Deterministic |
| Code-first → same class, different constructor args | No | `get_cache_key` returns different hash |
| Modified transform source code | No | Source hash changes |
| Same pipeline across processes | Yes | All hash inputs are deterministic |

## Concurrency Safety

> **Design note (staff review #3)**: Transform instances must be safe to share across threads.

**Thread-safety contract**: Transform `execute()` and `split()` methods must not modify `self`. They receive inputs and config as arguments and return new values. This is the existing implicit contract — now made explicit.

**Enforcement**: The runner uses `copy.copy(transform)` before dispatching to each concurrent worker in `_execute_transform_concurrent()`. This provides per-worker isolation so even accidental `self` mutation cannot cause cross-thread corruption:

```python
def _execute_transform_concurrent(transform, units, config, concurrency, on_complete):
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        for unit_inputs, config_extras in units:
            worker_transform = copy.copy(transform)  # per-worker isolation
            worker_config = {**config, **config_extras}
            future = executor.submit(worker_transform.execute, unit_inputs, worker_config)
            futures[future] = unit_inputs
        ...
```

**Documentation requirement**: `BaseTransform` docstring must state: "Transforms must be stateless during execution. Do not modify `self` in `execute()` or `split()`. Constructor state is read-only after initialization."

## Normative Contracts

### Transform thread-safety and mutability

1. Transforms MUST be stateless during execution. `execute()` and `split()` MUST NOT modify `self`.
2. Constructor state is read-only after `__init__` returns. The runner MAY `copy.copy()` transforms before dispatch.
3. Built-in transforms MUST have concurrent execution tests.
4. Custom transforms SHOULD document any thread-safety assumptions beyond the base contract.

### Fingerprint determinism and identity inputs

1. Every transform fingerprint MUST include all five canonical components: `transform_id`, `source`, `prompt`, `config`, `model`.
2. `get_cache_key(config)` MUST be deterministic: same inputs produce same hash across processes and interpreter restarts.
3. Custom transforms with constructor state MUST override `get_cache_key(config)` to include instance attributes that affect output.
4. Transforms SHOULD NOT use non-deterministic values (timestamps, random seeds, process IDs) in cache keys.

### Validator/fixer config serialization

1. Custom validators with typed constructors MUST override `to_config_dict()` to return a dict with at minimum `layers` (for artifact scoping).
2. `to_config_dict()` is the public API contract. The `_config` attribute is deprecated and MUST NOT be relied upon by new code.
3. `to_config_dict()` MUST return a JSON-serializable dict.
4. Custom validators that don't override `to_config_dict()` validate all artifacts in the build (no scoping).

## Compatibility Matrix

### Runtime resolution

| Pipeline style | Transform resolution | Validator resolution | Fixer resolution |
|---------------|---------------------|---------------------|-----------------|
| **Legacy (string dispatch)** | `get_transform(name)` via registry | `get_validator(decl.name)` + `_config` injection | `get_fixer(decl.name)` + `_config` injection |
| **Code-first (instances)** | Used directly via `resolve_transform()` | Used directly, `to_config_dict()` for scoping | Used directly, `to_config_dict()` for scoping |
| **Mixed mode** | Per-layer: instance or string resolved independently | Per-entry: `BaseValidator` or `ValidatorDecl` dispatched independently | Per-entry: `BaseFixer` or `FixerDecl` dispatched independently |

### Fingerprint/caching behavior

| Pipeline style | Fingerprint inputs | Cache reuse across styles? |
|---------------|-------------------|---------------------------|
| **Legacy** | source hash, prompt hash, config hash, model hash | Yes — if migrated to code-first with same class + config |
| **Code-first** | transform_id + source hash, prompt hash, config hash, model hash | Yes — stable across processes |
| **Mixed** | Each layer fingerprinted independently using its resolution path | Yes — layers are independent |

### Error semantics

| Error condition | Legacy behavior | Code-first behavior |
|----------------|----------------|-------------------|
| Unknown transform name | `ValueError` from `get_transform()` | Same (string path unchanged) |
| Invalid transform type | N/A (always string) | `TypeError` from `resolve_transform()` |
| Missing `to_config_dict()` override | N/A (`_config` injected) | Validates all artifacts (no scoping) |
| Import failure in lazy loading | `ValueError` (transform not found) | Same, but retryable (flag stays `False`) |

## Failure Modes and Mitigations

### Startup concurrency race

**Failure**: Two threads call `get_transform()` simultaneously on first use. Thread B observes incomplete registry.

**Mitigation**: Double-checked locking with `threading.Lock`. The `_builtins_loaded` flag is set AFTER all imports complete, inside the lock. Thread B blocks on the lock until imports finish.

### Import failure in lazy loading

**Failure**: One of the builtin transform modules fails to import (e.g., missing dependency).

**Mitigation**: `_builtins_loaded` remains `False` because it's set after all imports succeed. The next `get_transform()` call retries the imports. The import error propagates to the caller — fail fast, no silent degradation.

### Cache mismatch from code-first migration

**Failure**: User migrates from string dispatch to code-first. Different fingerprint causes unnecessary rebuild.

**Mitigation**: The fingerprint components (source hash, prompt hash, config hash, model hash) are identical for the same class regardless of dispatch path. The new `transform_id` component is additive — it only creates cache misses when transform identity genuinely differs. Migration from string to code-first with the same class and config does NOT invalidate caches.

### Accidental `self` mutation in concurrent transforms

**Failure**: A custom transform modifies `self` during `execute()`, causing cross-thread corruption.

**Mitigation**: The runner creates `copy.copy(transform)` per worker before dispatch. Even if a transform violates the statelessness contract, per-worker isolation prevents cross-thread corruption. The `BaseTransform` docstring documents the contract.

### Custom validator missing `to_config_dict()`

**Failure**: User creates a code-first validator with typed constructor but forgets to override `to_config_dict()`. The default falls back to `getattr(self, "_config", {})`, which returns `{}` — validator runs against all artifacts.

**Mitigation**: This is documented as explicit behavior in the "Custom validator contract" section. Validators that don't provide scoping run against the full build. This is safe (over-validates, never under-validates) and matches the legacy behavior for validators without `layers` config.

## Test Plan Delta

The following tests are required before merge:

1. **Concurrent cold-start transform lookup**: Spawn N threads calling `get_transform()` simultaneously with no prior import. All threads must succeed and return the correct transform. Verify built-in modules import exactly once.

2. **Fingerprint differentiation for constructor config**: Two instances of the same transform class with different constructor args must produce different fingerprints. Same constructor args must produce identical fingerprints.

3. **Fingerprint stability across interpreter runs**: Run the same pipeline definition in two separate subprocess invocations. Verify fingerprint hashes are identical.

4. **Mixed-mode pipeline execution equivalence**: Build a pipeline with some layers using string dispatch and some using instance dispatch. Verify artifacts are identical to a pure string-dispatch version of the same pipeline.

5. **Validator/fixer public config API compatibility**: Create a custom `BaseValidator` subclass with typed constructor and `to_config_dict()`. Run through `run_validators()`. Verify scoping works correctly and no `_config` monkey-patching occurs.

## Implementation Phases

### Phase 1: Foundation (no behavior change)

Add `resolve_transform()`, `prompt_name` attribute, lazy builtin loading. All additive — zero existing behavior changes.

**Files**: `build/transforms.py`, `build/llm_transforms.py`, `build/parse_transform.py`, `build/merge_transform.py`, `core/models.py`

### Phase 2: Wire into runner + plan

Replace `get_transform()` + `_transform_to_prompt_name()` with `resolve_transform()` + `transform.prompt_name`. Delete the duplicated mapping functions.

**Files**: `build/runner.py`, `build/plan.py`

### Phase 3: Typed validator/fixer constructors

Add `__init__` to built-in validators and fixers. Update `run_validators()` and `run_fixers()` to handle both declaration types.

**Files**: `build/validators.py`, `build/fixers.py`, `core/models.py`, `cli/validate_commands.py`

### Phase 4: Public imports + template migration

Populate re-export modules. Migrate all 4 templates to code-first style. Regenerate golden files.

**Files**: `transforms/__init__.py`, `validators/__init__.py` (new), `fixers/__init__.py` (new), `projections/__init__.py`, all template `pipeline.py` + `transforms.py` files

### Phase 5: Documentation

Update `CLAUDE.md`, `docs/pipeline-api.md`, `llms.txt`.

## What Stays Unchanged

- Decorator registries (`@register_transform`, etc.) — still available, just optional
- `get_transform()` / `get_validator()` / `get_fixer()` — still work
- `ValidatorDecl` / `FixerDecl` / `Projection` dataclasses — still accepted
- `load_pipeline()` — unchanged
- Fingerprint scheme (`synix:transform:v1`) — unchanged
- Artifact storage format — unchanged
