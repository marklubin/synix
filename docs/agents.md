# Agents

Agents are named execution units for Synix pipeline transforms. They replace anonymous LLM calls with identifiable, reusable personas that compose with transform task prompts.

## Concepts

**Transform prompt** defines the task structure — WHAT to do:
```python
"Infer this person's work style from their background.\n\n{artifact}"
```

**Agent instructions** define the persona — HOW to do it:
```
You are a concise analytical writer. Focus on patterns, dynamics,
and actionable insights. Write in 2-4 sentences.
```

Both compose: the transform renders its prompt as the user message, the agent provides its instructions as the system message.

## The Agent Protocol

```python
from synix.agents import Agent

class Agent(Protocol):
    @property
    def agent_id(self) -> str:
        """Stable identity — who this agent is."""

    def fingerprint_value(self) -> str:
        """Config snapshot — drives cache invalidation."""

    def map(self, artifact: Artifact, task_prompt: str) -> str:
        """1:1 transform."""

    def reduce(self, artifacts: list[Artifact], task_prompt: str) -> str:
        """N:1 transform."""

    def group(self, artifacts: list[Artifact], task_prompt: str) -> list[Group]:
        """N:M transform."""

    def fold(self, accumulated: str, artifact: Artifact, step: int, total: int, task_prompt: str) -> str:
        """Sequential fold transform."""
```

Each method matches a pipeline transform shape. The agent receives typed inputs plus the rendered task prompt from the transform.

### Identity vs Fingerprint

- **`agent_id`** — stable identity ("analyst", "reporter"). Changes only when the agent is replaced. Recorded in artifact provenance for lineage.
- **`fingerprint_value()`** — config snapshot hash. Changes when instructions, model, or temperature change. Drives cache invalidation. Different lifecycle from `agent_id`.

## SynixLLMAgent

The built-in implementation backed by PromptStore + LLMClient:

```python
from synix.agents import SynixLLMAgent

analyst = SynixLLMAgent(
    name="analyst",
    prompt_key="analyst",           # key in PromptStore
    llm_config={
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",
    },
    description="Concise analytical writer",
)
analyst.bind_prompt_store(store)    # inject PromptStore for instructions
```

Instructions are loaded from the PromptStore at call time — edits in the viewer are picked up automatically. The fingerprint uses the prompt store's content hash, so cache invalidates on edit.

## Using Agents in Pipelines

```python
from synix.agents import SynixLLMAgent
from synix.transforms import MapSynthesis, ReduceSynthesis, FoldSynthesis

analyst = SynixLLMAgent(name="analyst", prompt_key="analyst")
reporter = SynixLLMAgent(name="reporter", prompt_key="reporter")

# Same agent, different tasks
work_styles = MapSynthesis(
    "work_styles",
    depends_on=[bios],
    prompt="Infer this person's work style:\n\n{artifact}",
    agent=analyst,
)

team_dynamics = ReduceSynthesis(
    "team_dynamics",
    depends_on=[work_styles],
    prompt="Analyze team dynamics from these profiles:\n\n{artifacts}",
    agent=analyst,
)

# Different agent, different task
final_report = FoldSynthesis(
    "final_report",
    depends_on=[team_dynamics, brief],
    prompt="Update the report:\n\nDraft:\n{accumulated}\n\nNew:\n{artifact}",
    agent=reporter,
)
```

The `prompt=` defines the task template (rendered by the transform with shape-specific placeholders). The `agent=` defines who executes it. Both are required when using agents.

## Workspace Configuration

Define agents in `synix.toml`:

```toml
[agents.analyst]
prompt_key = "analyst"
instructions_file = "prompts/analyst.txt"
description = "Concise analytical writer"

[agents.reporter]
prompt_key = "reporter"
instructions_file = "prompts/reporter.txt"
provider = "openai-compatible"
model = "Qwen/Qwen3.5-2B"
base_url = "http://localhost:8100/v1"
```

Load in pipeline.py:

```python
from synix.workspace import load_agents

agents = load_agents()
analyst = agents["analyst"]
reporter = agents["reporter"]
```

`load_agents()` reads `synix.toml`, creates `SynixLLMAgent` instances, and binds the workspace's PromptStore.

## Artifact Provenance

When an agent executes a transform step, the output artifact records:

- **`agent_id`** — which agent produced this ("analyst")
- **`agent_fingerprint`** — the agent's config snapshot hash at build time
- **`prompt_id`** — the transform's prompt template hash (separate from agent)

This enables lineage tracking: "this artifact was produced by the analyst agent (v3a8f) using the work-style task template (v1b2c)."

## Custom Agent Implementations

Any object satisfying the `Agent` protocol works:

```python
class MyCustomAgent:
    @property
    def agent_id(self):
        return "my-agent"

    def fingerprint_value(self):
        return "v1-stable"

    def map(self, artifact, task_prompt):
        # Call your own API, local model, or anything else
        return my_api.generate(task_prompt)

    def reduce(self, artifacts, task_prompt):
        return my_api.generate(task_prompt)

    def group(self, artifacts, task_prompt):
        # Return list[Group] with key, artifacts, content
        ...

    def fold(self, accumulated, artifact, step, total, task_prompt):
        return my_api.generate(task_prompt)
```

Pass it to any transform: `MapSynthesis("layer", prompt="...", agent=MyCustomAgent())`.
