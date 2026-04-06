# Agents as First-Class Citizens in Synix

## Context

Transforms are coupled to inline prompt strings. A MapSynthesis is defined with `prompt=` and its identity is the class + prompt hash. There's no reusable "agent" concept — the same summarization behavior in two pipelines means duplicating the prompt string. Using that same persona outside a pipeline (MCP tool, script, ad-hoc query) has no portable unit.

**Goal:** Define `Agent` as a first-class interface — a contract that any agent implementation can satisfy. Transforms and MCP tools program to the interface, not a concrete class. Our initial implementation (`SynixLLMAgent`) is a single-turn LLM persona, but the abstraction supports plugging in entirely different agent backends later.

## Architecture

```
              ┌──────────────────────────────────┐
              │        Agent (Protocol)           │
              │  name, agent_id                   │
              │  fingerprint_value()              │
              │  for_map(artifact) → messages     │
              │  for_reduce(artifacts) → messages │
              │  for_group(key, artifacts) → msgs │
              │  for_fold(accumulated, art) → msgs│
              └───────────────┬──────────────────┘
                              │ implements
              ┌───────────────▼──────────────────┐
              │        SynixLLMAgent              │
              │  instructions, llm_config         │
              │  (system prompt + user content)   │
              └───────────────┬──────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           │                  │                  │
    ┌──────▼──────┐   ┌──────▼──────┐   ┌───────▼───────┐
    │  Transform  │   │  MCP Tool   │   │   Standalone  │
    │  agent=     │   │  run_agent  │   │  agent.run()  │
    └─────────────┘   └─────────────┘   └───────────────┘
```

Each transform calls the corresponding typed method — MapSynthesis calls `agent.for_map()`, ReduceSynthesis calls `agent.for_reduce()`, etc. The agent controls how instructions and content are combined into messages for each shape.

---

## Phase 1: Agent interface + SynixLLMAgent (standalone, no transform changes)

### New: `src/synix/agents.py` (~140 lines)

**1. Agent Protocol** — typed per pipeline integration point:

```python
from typing import Protocol, runtime_checkable
from synix.core.models import Artifact

@runtime_checkable
class Agent(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def agent_id(self) -> str: ...
    def fingerprint_value(self) -> str: ...

    def for_map(self, artifact: Artifact) -> list[dict]:
        """Produce messages for a 1:1 map transform."""

    def for_reduce(self, artifacts: list[Artifact]) -> list[dict]:
        """Produce messages for an N:1 reduce transform."""

    def for_group(self, group_key: str, artifacts: list[Artifact]) -> list[dict]:
        """Produce messages for a group transform (one group)."""

    def for_fold(self, accumulated: str, artifact: Artifact, step: int, total: int) -> list[dict]:
        """Produce messages for one step of a fold transform."""
```

Each method receives typed inputs matching the transform shape, returns `list[dict]` (messages) ready for the LLM client. No generic `build_messages()`.

**2. SynixLLMAgent dataclass** — concrete V1 implementation:

```python
@dataclass
class SynixLLMAgent:
    name: str
    instructions: str          # system prompt / template with {placeholders}
    llm_config: dict | None = None
    description: str = ""
```

Properties and methods:
- `instructions_hash` → SHA256[:16] of instructions
- `agent_id` → `"{name}_v{instructions_hash}"` — stable identity
- `for_map(artifact)` → renders instructions with `{artifact}`, `{label}`, `{artifact_type}`, returns `[system, user]`
- `for_reduce(artifacts)` → sorts, joins as `### {label}\n{content}` with `---` separators, renders with `{artifacts}`, `{count}`, returns `[system, user]`
- `for_group(key, artifacts)` → like reduce but also renders `{group_key}`, `{artifact_type}`
- `for_fold(accumulated, artifact, step, total)` → renders with `{accumulated}`, `{artifact}`, `{label}`, `{step}`, `{total}`, returns `[system, user]`
- `run(input_text)` → standalone convenience (**NOT part of protocol**). Creates own LLMClient, calls `for_map()` with synthetic Artifact, returns string.
- `fingerprint_value()` → hash of name + instructions + llm_config
- `render(**kwargs)` → template substitution via `ext/_render.render_template()`
- `from_file(cls, name, path, ...)` → classmethod
- Validation: `__post_init__` raises ValueError if name or instructions empty

Key design decisions:
- Protocol with typed methods per transform shape — no ambiguity about what the agent receives
- `runtime_checkable` so `isinstance(obj, Agent)` works for validation
- `run()` is NOT part of protocol — convenience on concrete class only
- Message construction lives in the agent, not the transform — agent decides how to frame inputs as messages

### Modify: `src/synix/build/llm_client.py` — system message support

In `_complete_anthropic()`, before `self._client.messages.create()`:

```python
system_text = None
user_messages = []
for msg in messages:
    if msg["role"] == "system":
        system_text = msg["content"]
    else:
        user_messages.append(msg)
kwargs = {"model": ..., "max_tokens": ..., "messages": user_messages}
if system_text is not None:
    kwargs["system"] = system_text
```

`_complete_openai()` already handles system messages natively.

### Modify: `src/synix/core/models.py` — add `agent_id` to Artifact

`agent_id: str | None = None` field (after `prompt_id`).

### Modify: `src/synix/__init__.py` — export both

```python
from synix.agents import Agent, SynixLLMAgent
```

### New: `tests/unit/test_agents.py` (~160 lines)

- SynixLLMAgent creation, agent_id stability, instructions_hash
- `render()` with placeholders
- `for_map(artifact)` — produces `[system, user]` with correct rendering
- `for_reduce(artifacts)` — sorts, joins, produces `[system, user]` with `{artifacts}`, `{count}`
- `for_group(key, artifacts)` — like reduce but includes `{group_key}`
- `for_fold(accumulated, artifact, step, total)` — fold-specific placeholders
- `run()` with mocked LLMClient
- `from_file()` reads from disk
- Validation: empty name, empty instructions → ValueError
- `fingerprint_value()` determinism
- Protocol compliance: `isinstance(SynixLLMAgent(...), Agent)` is True
- Custom implementation satisfying Agent protocol accepted by isinstance check

**Gate:** `uv run release` passes. SynixLLMAgent usable standalone via `agent.run()`.

---

## Phase 2: Transform integration

All four generic transforms gain `agent: Agent | None = None`:

**Constructor:** `prompt` becomes optional (default `""`). Validate exactly one of prompt or agent is set.

**execute():** When agent set, call the corresponding typed method:
- `map_synthesis.py`: `messages = self.agent.for_map(inp)` — agent gets full Artifact
- `reduce_synthesis.py`: `messages = self.agent.for_reduce(sorted_inputs)` — agent gets list[Artifact]
- `group_synthesis.py`: `messages = self.agent.for_group(group_key, sorted_inputs)`
- `fold_synthesis.py`: `messages = self.agent.for_fold(accumulated, inp, step, total)`

When agent is None: completely unchanged behavior (backward compatible).

**get_cache_key():** Use `agent.fingerprint_value()` in place of prompt hash when agent set.

**compute_fingerprint():** Add `"agent"` component using `agent.fingerprint_value()` when set.

**Artifact output:** Set `agent_id = self.agent.agent_id` and `prompt_id = self.agent.agent_id` (backward compat).

**No changes to `src/synix/build/llm_transforms.py`** — transforms use pipeline LLM client but agent messages. Agent's `llm_config` only matters for standalone `agent.run()`.

### Files modified:
- `src/synix/ext/map_synthesis.py`
- `src/synix/ext/reduce_synthesis.py`
- `src/synix/ext/group_synthesis.py`
- `src/synix/ext/fold_synthesis.py`

### New: `tests/unit/test_agent_transforms.py` (~150 lines)

- Each shape with agent param → verify correct typed method called, messages passed to LLM
- Verify `agent_id` set on output artifacts for all shapes
- Verify fingerprint changes when agent instructions change
- Verify mutual exclusion: both prompt and agent raises ValueError
- Verify backward compat: `prompt=` still works identically
- Test with a custom Agent protocol implementation (not SynixLLMAgent)

### New: `tests/e2e/test_agent_pipeline.py` (~100 lines)

- Pipeline with agent-backed MapSynthesis + ReduceSynthesis, build, verify artifacts have `agent_id`

**Gate:** `uv run release` passes.

---

## Phase 3: Agent config/lifecycle management + MCP

### New: `src/synix/agents_config.py` (~60 lines)

Separate module — decoupled from server config:

```python
@dataclass
class AgentEntry:
    name: str
    description: str = ""
    instructions_file: str = ""
    model: str | None = None
    temperature: float | None = None

class AgentRegistry:
    def __init__(self, entries: list[AgentEntry], base_dir: str = "."): ...
    @classmethod
    def from_toml_dict(cls, raw: dict, base_dir: str = ".") -> AgentRegistry: ...
    def list_agents(self) -> list[str]: ...
    def get_agent(self, name: str) -> SynixLLMAgent: ...
```

TOML format:
```toml
[agents.summarizer]
description = "Summarizes conversation sessions"
instructions_file = "prompts/summarizer.txt"

[agents.analyst]
description = "Analyzes patterns across episodes"
instructions_file = "prompts/analyst.txt"
model = "claude-sonnet-4-20250514"
temperature = 0.5
```

### Modify: `src/synix/server/config.py`
- Add `agents_raw: dict` to ServerConfig, store `raw.get("agents", {})` without parsing

### Modify: `src/synix/server/mcp_tools.py`
- Add `list_agents`, `run_agent` tools using AgentRegistry from `_state`

### New: `tests/unit/test_agents_config.py`

**Gate:** `uv run release` passes.

---

## Phase 4: Template + docs

- Update template 08-agent-memory/pipeline.py to show agent usage
- Add `docs/agents.md`
- Update CLAUDE.md module structure

---

## Files changed

| File | Change |
|------|--------|
| `src/synix/agents.py` | NEW — Agent protocol + SynixLLMAgent |
| `src/synix/agents_config.py` | NEW — AgentEntry, AgentRegistry |
| `src/synix/__init__.py` | Export Agent, SynixLLMAgent |
| `src/synix/core/models.py` | agent_id field on Artifact |
| `src/synix/build/llm_client.py` | System message extraction in _complete_anthropic() |
| `src/synix/ext/map_synthesis.py` | agent= param + execute/fingerprint/cache_key |
| `src/synix/ext/reduce_synthesis.py` | Same pattern |
| `src/synix/ext/group_synthesis.py` | Same pattern |
| `src/synix/ext/fold_synthesis.py` | Same pattern |
| `src/synix/server/config.py` | agents_raw passthrough field |
| `src/synix/server/mcp_tools.py` | list_agents, run_agent tools |
| `tests/unit/test_agents.py` | NEW |
| `tests/unit/test_agent_transforms.py` | NEW |
| `tests/unit/test_agents_config.py` | NEW |
| `tests/e2e/test_agent_pipeline.py` | NEW |
| `docs/agents.md` | NEW |

## Existing code to reuse

- `build/fingerprint.py:fingerprint_value()` — for instructions_hash and fingerprint_value()
- `ext/_render.py:render_template()` — for SynixLLMAgent.render() placeholder substitution
- `core/config.py:LLMConfig` — for SynixLLMAgent.run() client creation
- `build/llm_client.py:LLMClient` — for SynixLLMAgent.run() standalone execution
- `server/config.py` patterns — for TOML parsing in AgentRegistry

## Verification

1. `uv run pytest tests/unit/test_agents.py -v` — protocol + standalone agent tests
2. `uv run pytest tests/unit/test_agent_transforms.py -v` — transform integration
3. `uv run pytest tests/unit/test_agents_config.py -v` — registry/config
4. `uv run pytest tests/e2e/test_agent_pipeline.py -v` — full pipeline e2e
5. `uv run pytest tests/unit/test_ext_transforms.py -v` — existing tests still pass (backward compat)
6. `uv run pytest tests/unit/test_llm_client.py -v` — system message extraction
7. `uv run release` — full gate
