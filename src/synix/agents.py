"""Agent interface for Synix pipeline operations.

Agents are named execution units with typed methods matching transform
shapes (map, reduce, group, fold). Each agent has stable identity
(agent_id) and a config snapshot fingerprint for cache invalidation.

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

    def map(self, artifact: Artifact) -> str:
        """1:1 — process single artifact."""
        ...

    def reduce(self, artifacts: list[Artifact]) -> str:
        """N:1 — combine artifacts into one."""
        ...

    def group(self, artifacts: list[Artifact]) -> list[Group]:
        """N:M — assign artifacts to groups and synthesize each."""
        ...

    def fold(self, accumulated: str, artifact: Artifact, step: int, total: int) -> str:
        """Sequential — one fold step."""
        ...


@dataclass
class SynixLLMAgent:
    """Built-in agent backed by PromptStore + LLMClient.

    Instructions are loaded from PromptStore by prompt_key at call time,
    so edits in the viewer are picked up automatically. fingerprint_value()
    uses the prompt store's content hash, so cache invalidates on edit.
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
        """Hash of prompt content (from store) + llm_config."""
        from synix.build.fingerprint import compute_digest, fingerprint_value

        content_hash = ""
        if self._prompt_store is not None:
            content_hash = self._prompt_store.content_hash(self.prompt_key) or ""

        components = {"prompt_content": content_hash}
        if self.llm_config:
            components["llm_config"] = fingerprint_value(self.llm_config)
        return compute_digest(components)

    def bind_prompt_store(self, store) -> SynixLLMAgent:
        """Bind a PromptStore. Returns self for chaining."""
        self._prompt_store = store
        return self

    def map(self, artifact: Artifact) -> str:
        from synix.ext._render import render_template

        rendered = render_template(
            self.instructions,
            artifact=artifact.content,
            label=artifact.label,
            artifact_type=artifact.artifact_type,
        )
        return self._call(rendered)

    def reduce(self, artifacts: list[Artifact]) -> str:
        from synix.ext._render import render_template

        joined = "\n---\n".join(f"### {a.label}\n{a.content}" for a in artifacts)
        rendered = render_template(
            self.instructions,
            artifacts=joined,
            count=str(len(artifacts)),
        )
        return self._call(rendered)

    def group(self, artifacts: list[Artifact]) -> list[Group]:
        raise NotImplementedError(
            f"SynixLLMAgent {self.name!r} does not implement group(). "
            "See issue #127 for agent-driven grouping."
        )

    def fold(self, accumulated: str, artifact: Artifact, step: int, total: int) -> str:
        from synix.ext._render import render_template

        rendered = render_template(
            self.instructions,
            accumulated=accumulated,
            artifact=artifact.content,
            label=artifact.label,
            step=str(step),
            total=str(total),
        )
        return self._call(rendered)

    def _call(self, user_content: str) -> str:
        """Execute an LLM call with instructions as system prompt."""
        from synix.build.llm_client import LLMClient
        from synix.core.config import LLMConfig

        config = LLMConfig.from_dict(self.llm_config or {})
        client = LLMClient(config)
        response = client.complete(
            messages=[
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": user_content},
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
