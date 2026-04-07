"""Agent interface for Synix pipeline operations.

Agents are named execution units with typed methods matching transform
shapes (map, reduce, group, fold). Each agent has stable identity
(agent_id) and a config snapshot fingerprint for cache invalidation.

The transform renders its prompt (task structure) and passes it as
task_prompt. The agent provides persona/semantic instructions and
executes the LLM call. Both compose: transform defines WHAT to do,
agent defines HOW to do it.

SynixLLMAgent is the built-in implementation backed by PromptStore
for versioned instructions and LLMClient for execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from synix.core.models import Artifact


@dataclass
class Group:
    """Result of a group operation."""

    key: str
    artifacts: list[Artifact]
    content: str


@runtime_checkable
class Agent(Protocol):
    """Named execution agent for Synix pipeline operations.

    Each method receives typed transform inputs plus a task_prompt
    rendered by the transform. The agent composes its own instructions
    (persona/semantics) with the task prompt and executes.

    agent_id is stable identity (who this agent is).
    fingerprint_value() is config snapshot (how it behaves now).
    These have separate lifecycles.
    """

    @property
    def agent_id(self) -> str:
        """Stable identity. Changes only when the agent is replaced."""
        ...

    def fingerprint_value(self) -> str:
        """Config snapshot hash. Changes when instructions/model/config change.
        Drives cache invalidation."""
        ...

    def map(self, artifact: Artifact, task_prompt: str) -> str:
        """1:1 — process single artifact.

        task_prompt: rendered by the transform with {artifact}, {label}, etc.
        """
        ...

    def reduce(self, artifacts: list[Artifact], task_prompt: str) -> str:
        """N:1 — combine artifacts into one.

        task_prompt: rendered by the transform with {artifacts}, {count}, etc.
        """
        ...

    def group(self, artifacts: list[Artifact], task_prompt: str) -> list[Group]:
        """N:M — assign artifacts to groups and synthesize each.

        task_prompt: rendered by the transform (may be used per-group).
        """
        ...

    def fold(self, accumulated: str, artifact: Artifact, step: int, total: int, task_prompt: str) -> str:
        """Sequential — one fold step.

        task_prompt: rendered by the transform with {accumulated}, {artifact}, {step}, {total}, etc.
        """
        ...


@dataclass
class SynixLLMAgent:
    """Built-in agent backed by PromptStore + LLMClient.

    Instructions (persona/semantics) are loaded from PromptStore by
    prompt_key at call time, so edits in the viewer are picked up
    automatically. The transform's task_prompt becomes the user message,
    agent instructions become the system message.
    """

    name: str
    prompt_key: str
    llm_config: dict | None = None
    description: str = ""
    _prompt_store: Any = field(default=None, repr=False)

    def __post_init__(self):
        if not self.name:
            raise ValueError("SynixLLMAgent must have a name")
        if not self.prompt_key:
            raise ValueError("SynixLLMAgent must have a prompt_key")

    @property
    def agent_id(self) -> str:
        return self.name

    @property
    def instructions(self) -> str:
        """Load current instructions from PromptStore."""
        if self._prompt_store is None:
            raise ValueError(
                f"Agent {self.name!r} has no prompt store — call bind_prompt_store() first"
            )
        content = self._prompt_store.get(self.prompt_key)
        if content is None:
            raise ValueError(f"Prompt key {self.prompt_key!r} not found in store")
        return content

    def fingerprint_value(self) -> str:
        """Hash of prompt content (from store) + llm_config.

        Raises ValueError if prompt store is not bound — fingerprinting
        requires a known prompt state for cache correctness.
        """
        from synix.build.fingerprint import compute_digest, fingerprint_value

        if self._prompt_store is None:
            raise ValueError(
                f"Agent {self.name!r} has no prompt store — "
                "bind_prompt_store() before fingerprinting"
            )
        content_hash = self._prompt_store.content_hash(self.prompt_key) or ""

        components = {"prompt_content": content_hash}
        if self.llm_config:
            components["llm_config"] = fingerprint_value(self.llm_config)
        return compute_digest(components)

    def bind_prompt_store(self, store) -> SynixLLMAgent:
        """Bind a PromptStore. Returns self for chaining."""
        self._prompt_store = store
        return self

    def map(self, artifact: Artifact, task_prompt: str) -> str:
        return self._call(task_prompt)

    def reduce(self, artifacts: list[Artifact], task_prompt: str) -> str:
        return self._call(task_prompt)

    def group(self, artifacts: list[Artifact], task_prompt: str) -> list[Group]:
        raise NotImplementedError(
            f"SynixLLMAgent {self.name!r} does not implement group(). "
            "See issue #127 for agent-driven grouping."
        )

    def fold(self, accumulated: str, artifact: Artifact, step: int, total: int, task_prompt: str) -> str:
        return self._call(task_prompt)

    def _call(self, task_prompt: str) -> str:
        """Execute LLM call: instructions as system, task_prompt as user."""
        from synix.build.llm_client import LLMClient
        from synix.core.config import LLMConfig

        config = LLMConfig.from_dict(self.llm_config or {})
        client = LLMClient(config)
        response = client.complete(
            messages=[
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": task_prompt},
            ],
            artifact_desc=f"agent:{self.name}",
        )
        return response.content

    @classmethod
    def from_file(
        cls,
        name: str,
        prompt_key: str,
        instructions_path: str | Path,
        prompt_store,
        **kwargs,
    ) -> SynixLLMAgent:
        """Create agent and seed its instructions into the PromptStore from a file."""
        content = Path(instructions_path).read_text()
        prompt_store.put(prompt_key, content)
        agent = cls(name=name, prompt_key=prompt_key, **kwargs)
        agent.bind_prompt_store(prompt_store)
        return agent
