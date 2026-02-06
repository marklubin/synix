"""Transform step implementation (1:1 processing)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

from synix.steps.base import PromptFunction, Step

if TYPE_CHECKING:
    from synix.db.artifacts import Record
    from synix.llm.client import LLMClient


@dataclass
class TransformStep(Step):
    """Transform step: processes one input record into one output record.

    Materialization key format:
        (branch, step_name, input_record_id, step_version_hash)
    """

    step_type: str = field(init=False, default="transform")

    def compute_materialization_key(
        self,
        inputs: list["Record"],
        step_version_hash: str,
        branch: str = "main",
    ) -> str:
        """Compute cache key for transform output.

        Transform processes exactly one input, so the key includes
        the input record ID.
        """
        if len(inputs) != 1:
            msg = f"Transform step requires exactly 1 input, got {len(inputs)}"
            raise ValueError(msg)

        input_record = inputs[0]
        components = [branch, self.name, str(input_record.id), step_version_hash]
        return ":".join(components)

    def execute(
        self,
        inputs: list["Record"],
        llm: "LLMClient",
        run_id: str,
    ) -> "Record":
        """Execute the transform on a single input record.

        Args:
            inputs: Single-element list with the input record.
            llm: LLM client for generation.
            run_id: Current run ID.

        Returns:
            New Record with transformed content.
        """
        from synix.db.artifacts import Record

        if len(inputs) != 1:
            msg = f"Transform step requires exactly 1 input, got {len(inputs)}"
            raise ValueError(msg)

        if self.prompt is None:
            msg = "Transform step requires a prompt function"
            raise ValueError(msg)

        input_record = inputs[0]

        # Generate prompt from input
        prompt_text = self.prompt(input_record)

        # Call LLM
        response = llm.complete(prompt_text)

        # Create output record
        record = Record(
            id=str(uuid4()),
            content=response.content,
            content_fingerprint=Record.compute_fingerprint(response.content),
            step_name=self.name,
            branch="main",  # v0.1 always main
            materialization_key="",  # Set by caller
            run_id=run_id,
        )

        # Set audit info
        record.audit = {
            "prompt_hash": self.compute_version_hash(),
            "model": llm.config.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        }

        # Propagate relevant metadata from input
        input_meta = input_record.metadata_
        record.metadata_ = {
            "meta.time.created_at": input_meta.get("meta.time.created_at"),
            "meta.source.record_id": str(input_record.id),
            "meta.source.step": input_record.step_name,
        }

        return record


def create_transform_step(
    name: str,
    from_: str,
    prompt: PromptFunction,
    model: str = "deepseek-chat",
) -> TransformStep:
    """Factory function to create a transform step.

    Args:
        name: Step name.
        from_: Name of upstream step.
        prompt: Function that takes a Record and returns prompt string.
        model: LLM model to use.

    Returns:
        Configured TransformStep.
    """
    return TransformStep(
        name=name,
        from_=from_,
        prompt=prompt,
        model=model,
    )
