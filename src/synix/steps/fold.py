"""Fold step implementation (sequential processing with state accumulation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

from synix.steps.base import PromptFunction, Step, compute_combined_fingerprint

if TYPE_CHECKING:
    from synix.db.artifacts import Record
    from synix.llm.client import LLMClient


@dataclass
class FoldStep(Step):
    """Fold step: processes records sequentially with carried state.

    Each record transforms the accumulated state via LLM.
    Final output is the accumulated result.

    Key differences from Aggregate:
    - Order matters — records processed sequentially (sorted by created_at)
    - State carried forward — each iteration builds on previous
    - Multiple LLM calls — one per input record (not batched)
    - Single output — final accumulated state becomes the record

    Materialization key format:
        (branch, step_name, input_count, combined_fingerprint, step_version_hash)

    Prompt signature: (accumulated_state: str, current_record: Record) -> str
    """

    initial_state: str = ""  # Starting accumulator value
    step_type: str = field(init=False, default="fold")

    def compute_materialization_key(
        self,
        inputs: list["Record"],
        step_version_hash: str,
        branch: str = "main",
    ) -> str:
        """Compute cache key for fold output.

        The key includes input count and combined fingerprint to detect:
        - Content changes in any input
        - Changes in group membership (additions/removals)

        Order is preserved in combined fingerprint via sorted fingerprints.
        """
        if not inputs:
            msg = "Fold step requires at least 1 input"
            raise ValueError(msg)

        combined_fp = compute_combined_fingerprint(inputs)
        components = [branch, self.name, str(len(inputs)), combined_fp, step_version_hash]
        return ":".join(components)

    def execute(
        self,
        inputs: list["Record"],
        llm: "LLMClient",
        run_id: str,
    ) -> "Record":
        """Execute the fold over all input records.

        Processes records sequentially (sorted by created_at), carrying
        state forward. Returns a single record with the final state.

        Args:
            inputs: List of input records to fold over.
            llm: LLM client for generation.
            run_id: Current run ID.

        Returns:
            New Record with final accumulated state.
        """
        from synix.db.artifacts import Record

        if not inputs:
            msg = "Fold step requires at least 1 input"
            raise ValueError(msg)

        if self.prompt is None:
            msg = "Fold step requires a prompt function"
            raise ValueError(msg)

        # Sort inputs by created_at
        def get_created_at(r: "Record") -> str:
            return str(r.metadata_.get("meta.time.created_at", ""))

        sorted_inputs = sorted(inputs, key=get_created_at)

        # Fold through all inputs
        state = self.initial_state
        total_input_tokens = 0
        total_output_tokens = 0

        for record in sorted_inputs:
            prompt_text = self.prompt(state, record)
            response = llm.complete(prompt_text)
            state = response.content
            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens

        # Create output record with final state
        output = Record(
            id=str(uuid4()),
            content=state,
            content_fingerprint=Record.compute_fingerprint(state),
            step_name=self.name,
            branch="main",  # v0.1 always main
            materialization_key="",  # Set by caller
            run_id=run_id,
        )

        # Set audit info
        output.audit = {
            "prompt_hash": self.compute_version_hash(),
            "model": llm.config.model,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "input_count": len(inputs),
            "iterations": len(sorted_inputs),
        }

        # Set metadata
        output.metadata_ = {
            "meta.fold.input_count": len(inputs),
            "meta.fold.initial_state_empty": self.initial_state == "",
        }

        return output


def create_fold_step(
    name: str,
    from_: str,
    prompt: PromptFunction,
    initial_state: str = "",
    model: str = "deepseek-chat",
) -> FoldStep:
    """Factory function to create a fold step.

    Args:
        name: Step name.
        from_: Name of upstream step.
        prompt: Function that takes (state, Record) and returns prompt string.
        initial_state: Initial accumulator value.
        model: LLM model to use.

    Returns:
        Configured FoldStep.
    """
    return FoldStep(
        name=name,
        from_=from_,
        prompt=prompt,
        initial_state=initial_state,
        model=model,
    )
