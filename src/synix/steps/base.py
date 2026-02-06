"""Abstract base class for pipeline steps."""

from __future__ import annotations

import hashlib
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from synix.db.artifacts import Record
    from synix.llm.client import LLMClient


class PromptFunction(Protocol):
    """Protocol for prompt functions."""

    def __call__(self, *args: Any, **kwargs: Any) -> str: ...


@dataclass
class StepResult:
    """Result from executing a step."""

    record: "Record"
    was_cached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Step(ABC):
    """Abstract base class for pipeline steps.

    Steps transform input records into output records. Each step type
    has its own logic for:
    - Computing materialization keys (for caching)
    - Executing the transformation
    """

    name: str
    from_: str | None  # Upstream step name (None for sources)
    prompt: PromptFunction | None = None
    model: str = "deepseek-chat"
    step_type: str = field(init=False)

    @abstractmethod
    def compute_materialization_key(
        self,
        inputs: list["Record"],
        step_version_hash: str,
        branch: str = "main",
    ) -> str:
        """Compute a unique cache key for this step's output.

        The key should uniquely identify the output based on:
        - The inputs
        - The step configuration (captured in step_version_hash)
        - The branch

        Args:
            inputs: Input records to this step.
            step_version_hash: Hash of step type + prompt source + model.
            branch: Current branch name.

        Returns:
            A unique string key for caching.
        """
        ...

    @abstractmethod
    def execute(
        self,
        inputs: list["Record"],
        llm: "LLMClient",
        run_id: str,
    ) -> "Record":
        """Execute the step transformation.

        Args:
            inputs: Input records to transform.
            llm: LLM client for calling the model.
            run_id: Current run ID for tracking.

        Returns:
            The output Record (not yet persisted).
        """
        ...

    def compute_version_hash(self) -> str:
        """Compute hash of step type + prompt source + model.

        This hash changes when the step configuration changes,
        triggering reprocessing.
        """
        components = [self.step_type, self.model]

        if self.prompt is not None:
            try:
                source = inspect.getsource(self.prompt)
                components.append(source)
            except (OSError, TypeError):
                # Can't get source (built-in, lambda, etc.)
                components.append(str(self.prompt))

        combined = ":".join(components)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]


def compute_combined_fingerprint(records: list["Record"]) -> str:
    """Compute combined fingerprint for multiple records.

    Used by aggregate steps to detect when group membership changes.
    """
    fingerprints = sorted(r.content_fingerprint for r in records)
    combined = ":".join(fingerprints)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]
