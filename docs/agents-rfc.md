# Agent Gateway Interface for Synix Transforms

## Context

Today the generic synthesis transforms are coupled to Synix's built-in LLM execution path:

- The transform owns prompt rendering.
- The transform calls `LLMClient` through `_logged_complete()`.
- The transform records prompt/model provenance and participates in fingerprint-based caching.

That coupling makes it hard to use the same transform abstractions with an externally managed agent or generation service. But the thing we want to decouple is the **execution gateway**, not the transform semantics.

## Goal

Define a minimal `Agent` interface that generic transforms can call to obtain output text.

The external agent implementation is **managed outside Synix**. Synix does not own its lifecycle, registry, config, or runtime surface. Synix only knows:

1. how to render the transform prompt
2. how to pass that rendered prompt to an injected agent
3. how to fingerprint that agent for cache correctness

## Non-Goals

- Synix does **not** become an agent framework.
- No agent registry in config or TOML.
- No `run_agent`, `list_agents`, or standalone agent MCP surface.
- No transform-shape-specific callbacks like `for_map()` or `for_reduce()`.
- No movement of grouping, sorting, fold checkpointing, search-surface access, or `context_budget` logic out of Synix transforms.
- No overloading of `prompt_id` to mean agent identity.

---

## Design

### Boundary

Synix continues to own transform behavior. The injected agent is just a writer:

```
Artifact inputs
    |
    v
Transform
  - sort/group/fold logic
  - prompt rendering
  - context_budget
  - provenance bookkeeping
    |
    v
Agent.write(request)
    |
    v
text output
    |
    v
Artifact output
```

This preserves the current architecture: Synix remains an offline build system with explicit transform semantics. The new interface only decouples the final execution step from the built-in LLM client.

### New: `src/synix/agents.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class AgentRequest:
    prompt: str
    max_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResult:
    content: str


@runtime_checkable
class Agent(Protocol):
    def write(self, request: AgentRequest) -> AgentResult:
        """Execute a rendered synthesis request and return output text."""

    def fingerprint_value(self) -> str:
        """Return a deterministic fingerprint for output-affecting behavior."""
```

### Why this shape

- `prompt` is plain text, not chat-message dicts. That keeps the interface transport-neutral.
- `metadata` gives the external implementation optional context without forcing Synix to expose transform-shape-specific methods.
- `write()` is generic. Synix transforms keep owning their own shape semantics.
- `fingerprint_value()` is the cache/provenance contract between Synix and the external implementation.

### Metadata passed in `AgentRequest`

Synix may populate `metadata` with execution context such as:

- `transform_name`
- `artifact_type`
- `input_labels`
- `group_key`
- `step`
- `total`

This metadata is advisory only. It does not replace prompt rendering and it does not become part of the public transform API.

---

## Fingerprint Contract

The injected agent implementation supplies its own fingerprint logic.

That fingerprint is not a hint. It is a **cache-correctness contract**.

Requirements:

- Deterministic for the same effective behavior.
- Changes whenever output-affecting behavior changes.
- Safe to use as part of transform fingerprinting.

Examples of things an external implementation may include:

- model/version
- instructions owned by the external agent
- tool set or tool schema revision
- endpoint revision
- decoding parameters
- externally managed prompt/template version

If an implementation cannot provide a trustworthy fingerprint, Synix should not treat it as cache-safe.

Recommended v1 behavior:

- `fingerprint_value()` is required for any agent-backed transform.
- If it is missing or empty, transform construction raises `ValueError`.

---

## Artifact Provenance

### Recommended: modify `src/synix/core/models.py`

Add a separate field:

```python
agent_fingerprint: str | None = None
```

This field is distinct from `prompt_id`.

- `prompt_id` remains prompt/template provenance.
- `agent_fingerprint` records the execution backend identity used for this artifact.

We should **not** set `prompt_id = agent_fingerprint`. Those are different provenance dimensions and they are already used separately in diffing, snapshots, and artifact storage.

Because this is a persisted field, the RFC implementation must update all artifact serialization paths, snapshot/object-store payloads, and any SDK/viewer shims that materialize `Artifact`.

---

## Phase 1: Generic transform integration

All four generic synthesis transforms gain:

```python
agent: Agent | None = None
```

### Constructor

- `prompt` stays required.
- `agent` is optional.

This is the key design difference from the earlier draft: using an injected agent does **not** make prompt ownership leave the transform.

### Execution

When `agent is None`:

- Existing behavior is unchanged.
- The transform uses the built-in LLM client path.

When `agent is not None`:

1. The transform renders the prompt exactly as it does today.
2. The transform builds an `AgentRequest`.
3. The transform calls `agent.write(request)`.
4. The returned `content` becomes the artifact body.

Example for `MapSynthesis`:

```python
rendered = render_template(
    self.prompt,
    artifact=inp.content,
    label=inp.label,
    artifact_type=inp.artifact_type,
)

if self.agent is None:
    response = _logged_complete(...)
    content = response.content
else:
    result = self.agent.write(
        AgentRequest(
            prompt=rendered,
            metadata={
                "transform_name": self.name,
                "shape": "map",
                "input_labels": [inp.label],
                "artifact_type": self.artifact_type,
            },
        )
    )
    content = result.content
```

The same pattern applies to Reduce, Group, and Fold.

### Fingerprinting

When `agent is None`:

- existing fingerprint behavior remains unchanged

When `agent is not None`:

- `compute_fingerprint()` adds an `"agent"` component from `agent.fingerprint_value()`
- `get_cache_key()` includes `agent.fingerprint_value()`
- the built-in `"model"` fingerprint component from `llm_config` is omitted for that transform, because Synix did not execute the call

Prompt hashing and other transform-owned config hashing remain intact.

### Artifact output

When `agent is not None`:

- set `agent_fingerprint = agent.fingerprint_value()`
- keep `prompt_id` as the transform prompt version/hash
- leave `model_config` as `None` unless Synix itself performed the LLM call

This preserves provenance meaning:

- prompt provenance says what Synix rendered
- agent fingerprint says what external executor produced the text

### Files modified

- `src/synix/agents.py`
- `src/synix/ext/map_synthesis.py`
- `src/synix/ext/reduce_synthesis.py`
- `src/synix/ext/group_synthesis.py`
- `src/synix/ext/fold_synthesis.py`
- `src/synix/core/models.py`
- artifact persistence/snapshot/diff layers that serialize `Artifact`

---

## Phase 2: Tests

### New: `tests/unit/test_agents.py`

- Protocol compliance with a fake external agent
- `AgentRequest` and `AgentResult` construction
- `fingerprint_value()` determinism requirement documented via tests

### New: `tests/unit/test_agent_transforms.py`

- Each generic synthesis transform with `agent=` calls `agent.write()`
- Rendered prompt remains transform-owned
- Fingerprint changes when agent fingerprint changes
- Prompt remains required even when `agent=` is set
- Backward compatibility: existing `prompt=` path behaves identically

### Update existing artifact/snapshot tests

- `agent_fingerprint` persists through artifact store and snapshot/object store
- diffing does not confuse `prompt_id` and `agent_fingerprint`

### Optional e2e

- Small pipeline using `MapSynthesis(..., agent=fake_agent)` and `ReduceSynthesis(..., agent=fake_agent)`

---

## Deliberately Out of Scope

These ideas are explicitly removed from this RFC:

- `SynixLLMAgent` as a first-class public runtime persona
- standalone `agent.run()`
- `agents_config.py`
- `[agents.*]` config sections
- `ServerConfig` changes
- MCP tools for agent lifecycle
- template/docs changes that imply Synix now manages agents

If we later want a convenience adapter that wraps `LLMClient` behind the `Agent` protocol, that can be a separate follow-on RFC. It is not needed to establish the boundary.

---

## Open Questions

1. Should `agent_fingerprint` be a first-class `Artifact` field in v1, or should we stage it in `metadata` first to reduce schema churn?
2. Should bundled transforms in `src/synix/build/llm_transforms.py` gain the same `agent=` escape hatch in the initial slice, or only the generic transforms?
3. Do we want a tiny internal adapter like `LLMClientAgent` for testability, or is `agent=None` enough for the built-in path?

---

## Verification

1. `uv run pytest tests/unit/test_agents.py -v`
2. `uv run pytest tests/unit/test_agent_transforms.py -v`
3. `uv run pytest tests/unit/test_ext_transforms.py -v`
4. `uv run pytest tests/unit/test_diff.py -v`
5. `uv run release`
