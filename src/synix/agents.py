"""Agent gateway interface for Synix transforms.

Synix owns prompt rendering and transform semantics. The Agent protocol
is an execution gateway --- the transform renders the prompt, the agent
produces the output text.

The Agent does NOT own prompts, grouping, sorting, fold checkpointing,
search-surface access, or context_budget logic. Those stay in transforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class AgentRequest:
    """A rendered synthesis request ready for execution.

    The transform has already rendered the prompt. The agent just needs
    to produce output text.
    """

    prompt: str
    max_tokens: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResult:
    """Output from an agent execution."""

    content: str


@runtime_checkable
class Agent(Protocol):
    """Execution gateway for synthesis transforms.

    Any object with write() and fingerprint_value() methods satisfies
    this protocol. Synix does not own the agent's lifecycle, config,
    or runtime surface.
    """

    def write(self, request: AgentRequest) -> AgentResult:
        """Execute a rendered synthesis request and return output text."""
        ...

    def fingerprint_value(self) -> str:
        """Return a deterministic fingerprint for output-affecting behavior.

        This is a cache-correctness contract. The fingerprint must:
        - Be deterministic for the same effective behavior
        - Change whenever output-affecting behavior changes
        - Be safe to use as part of transform fingerprinting

        Examples of what an implementation may include:
        model/version, instructions, tool set, endpoint revision,
        decoding parameters, externally managed prompt version.
        """
        ...
