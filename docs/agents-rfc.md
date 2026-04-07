# Typed Agent Protocol + SynixLLMAgent

## Context

PR #126 shipped a generic `write(AgentRequest) → AgentResult` gateway. Too generic — it doesn't know about transform shapes, and the agent doesn't own rendering or execution. The original intent was typed operations matching pipeline transforms, with agents that have identity, own their execution, and can be configured in workspaces.

This replaces the generic gateway with a typed protocol and adds `SynixLLMAgent` as the built-in implementation.

## Agent Protocol (replaces current `agents.py`)

```python
@dataclass
class Group:
    """Result of a group operation — key, member artifacts, and synthesized content."""
    key: str
    artifacts: list[Artifact]
    content: str

@runtime_checkable
class Agent(Protocol):
    """Named execution agent for Synix pipeline operations.
    
    agent_id is stable identity (who). fingerprint_value() is config
    snapshot (how it behaves now). Separate lifecycles.
    """

    @property
    def agent_id(self) -> str:
        """Stable identity — who this agent is. For lineage/provenance.
        Changes only when the agent is fundamentally replaced."""

    def fingerprint_value(self) -> str:
        """Config snapshot — current behavior hash. For cache invalidation.
        Changes when instructions/model/temperature change."""

    def map(self, artifact: Artifact) -> str:
        """1:1 — process single artifact. Used by MapSynthesis."""

    def reduce(self, artifacts: list[Artifact]) -> str:
        """N:1 — combine artifacts into one. Used by ReduceSynthesis."""

    def group(self, artifacts: list[Artifact]) -> list[Group]:
        """N:M — assign artifacts to groups and synthesize each.
        Returns list of Group(key, artifacts, content).
        Used by GroupSynthesis.
        Note: parallel feed-forward variant tracked in #127."""

    def fold(self, accumulated: str, artifact: Artifact, step: int, total: int) -> str:
        """Sequential — one fold step. Used by FoldSynthesis."""
```

## SynixLLMAgent (built-in implementation)

```python
@dataclass
class SynixLLMAgent:
    """Built-in agent backed by PromptStore + LLMClient."""
    name: str                          # stable identity (= agent_id)
    prompt_key: str                    # key in PromptStore for instructions
    llm_config: dict | None = None     # provider/model/temperature
    description: str = ""
    _prompt_store: Any = None          # injected at runtime (PromptStore)

    @property
    def agent_id(self) -> str:
        return self.name

    @property
    def instructions(self) -> str:
        """Load current instructions from PromptStore."""
        if self._prompt_store is None:
            raise ValueError(f"Agent {self.name!r} has no prompt store — call bind_prompt_store() first")
        content = self._prompt_store.get(self.prompt_key)
        if content is None:
            raise ValueError(f"Prompt key {self.prompt_key!r} not found in store")
        return content

    def fingerprint_value(self) -> str:
        """Hash of prompt content (from store) + llm_config.
        Auto-invalidates when prompt is edited in viewer."""
        content_hash = self._prompt_store.content_hash(self.prompt_key) if self._prompt_store else ""
        # hash content_hash + llm_config

    def bind_prompt_store(self, store) -> SynixLLMAgent:
        """Bind a PromptStore to this agent. Returns self for chaining."""
        self._prompt_store = store
        return self

    def map(self, artifact) -> str:
        rendered = render_template(self.instructions,
            artifact=artifact.content, label=artifact.label,
            artifact_type=artifact.artifact_type)
        return self._call(rendered)

    def reduce(self, artifacts) -> str:
        joined = "\n---\n".join(f"### {a.label}\n{a.content}" for a in artifacts)
        rendered = render_template(self.instructions,
            artifacts=joined, count=str(len(artifacts)))
        return self._call(rendered)

    def group(self, artifacts) -> list[Group]:
        # Not yet implemented for SynixLLMAgent (tracked in #127)
        raise NotImplementedError("SynixLLMAgent.group() not yet implemented")

    def fold(self, accumulated, artifact, step, total) -> str:
        rendered = render_template(self.instructions,
            accumulated=accumulated, artifact=artifact.content,
            label=artifact.label, step=str(step), total=str(total))
        return self._call(rendered)

    def _call(self, user_content: str) -> str:
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": user_content},
        ]
        return self._get_client().complete(messages=messages).content

    def _get_client(self):
        from synix.build.llm_client import LLMClient
        from synix.core.config import LLMConfig
        return LLMClient(LLMConfig.from_dict(self.llm_config or {}))
```

**Prompt Store integration:**
- Instructions loaded from `PromptStore.get(prompt_key)` at call time (not cached — picks up edits)
- `fingerprint_value()` uses `PromptStore.content_hash(prompt_key)` — auto-invalidates cache when prompt is edited in the viewer
- `bind_prompt_store()` injects the store at runtime (during workspace/server startup)
- Workspace `load_agents()` auto-binds the workspace's PromptStore to each agent

## Artifact provenance (already on Artifact from PR #126)

- `agent_id: str | None` — from `agent.agent_id` (stable: "summarizer")
- `agent_fingerprint: str | None` — from `agent.fingerprint_value()` (config snapshot hash)

## Transform integration (replace write() with typed calls)

```python
# MapSynthesis.execute():
if self.agent is not None:
    content = self.agent.map(inp)
    agent_id_val = self.agent.agent_id
    agent_fp = self.agent.fingerprint_value()
    model_config = None
else:
    # existing _logged_complete() path unchanged

# ReduceSynthesis.execute():
if self.agent is not None:
    content = self.agent.reduce(sorted_inputs)

# GroupSynthesis.execute():
if self.agent is not None:
    groups = self.agent.group(inputs)
    # create one artifact per Group

# FoldSynthesis fold loop:
if self.agent is not None:
    accumulated = self.agent.fold(accumulated, inp, step, total)
```

## Workspace config + load_agents()

```toml
# synix.toml
[agents.writer]
instructions_file = "prompts/writer.txt"
provider = "openai-compatible"
model = "Qwen/Qwen3.5-2B"
base_url = "http://localhost:8100/v1"
temperature = 0.3
max_tokens = 2048
```

```python
# In workspace.py
def load_agents(config_path=None) -> dict[str, SynixLLMAgent]:
    """Load agents from synix.toml. For use in pipeline.py."""
```

## Implementation — 3 parallel agents

### Agent 1: Rewrite agents.py + tests
- Replace AgentRequest/AgentResult/write() with Group + typed protocol
- Add SynixLLMAgent with map/reduce/fold/_call/_get_client
- Update __init__.py exports
- Rewrite tests/unit/test_agents.py

### Agent 2: Update all 4 transforms + tests
- Replace agent.write(AgentRequest(...)) with typed calls
- Remove prompt rendering from agent path (agent owns it)
- Set agent_id on artifacts from agent.agent_id
- Update tests/unit/test_agent_transforms.py

### Agent 3: Workspace config + e2e
- Add [agents.*] parsing to workspace.py
- Add load_agents() convenience
- Update tests/e2e/test_agent_pipeline.py
- New tests/unit/test_workspace_agents.py

## Files

| File | Change |
|------|--------|
| `src/synix/agents.py` | Rewrite — Group, Agent Protocol, SynixLLMAgent |
| `src/synix/__init__.py` | Update exports |
| `src/synix/ext/map_synthesis.py` | agent.map(inp) |
| `src/synix/ext/reduce_synthesis.py` | agent.reduce(inputs) |
| `src/synix/ext/group_synthesis.py` | agent.group(inputs) → list[Group] |
| `src/synix/ext/fold_synthesis.py` | agent.fold(accumulated, inp, step, total) |
| `src/synix/workspace.py` | [agents.*] config, load_agents() |
| `tests/unit/test_agents.py` | Rewrite |
| `tests/unit/test_agent_transforms.py` | Update |
| `tests/e2e/test_agent_pipeline.py` | Update |
| `tests/unit/test_workspace_agents.py` | NEW |

## Verification

1. `uv run pytest tests/unit/test_agents.py -v`
2. `uv run pytest tests/unit/test_agent_transforms.py -v`
3. `uv run pytest tests/e2e/test_agent_pipeline.py -v`
4. `uv run pytest tests/unit/test_workspace_agents.py -v`
5. `uv run pytest tests/unit/test_ext_transforms.py -v` — backward compat
6. `uv run release` — full gate
