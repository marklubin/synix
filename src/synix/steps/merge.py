"""Merge step implementation (multi-source combination)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

from synix.steps.base import PromptFunction, Step, compute_combined_fingerprint

if TYPE_CHECKING:
    from synix.db.artifacts import Record
    from synix.llm.client import LLMClient


@dataclass
class MergeStep(Step):
    """Merge step: combines records from multiple upstream sources.

    Fan-in operation that unifies outputs from different steps into
    a single record.

    Key differences from Transform/Aggregate:
    - Multiple upstream sources — not just one from_
    - Input organized by source — dict mapping step names to records
    - Single output — unified record from all sources

    Materialization key format:
        (branch, step_name, source_fingerprints, step_version_hash)

    Where source_fingerprints is: "source1=fp1|source2=fp2|..."

    Prompt signature: (sources: dict[str, list[Record]]) -> str
    """

    sources: list[str] = field(default_factory=list)  # Multiple upstream step names
    step_type: str = field(init=False, default="merge")

    def __post_init__(self) -> None:
        """Validate configuration."""
        # MergeStep uses sources, not from_
        # Set from_ to empty string to satisfy base class
        if not self.from_:
            object.__setattr__(self, "from_", "")

    def compute_materialization_key(
        self,
        inputs: list["Record"],
        step_version_hash: str,
        branch: str = "main",
    ) -> str:
        """Compute cache key for merge output.

        Includes fingerprints from all sources to detect changes in any upstream.
        """
        if not inputs:
            msg = "Merge step requires at least 1 input"
            raise ValueError(msg)

        # Organize inputs by source
        sources_dict = self._organize_by_source(inputs)

        # Build combined key with per-source fingerprints
        source_fps = []
        for source_name in sorted(sources_dict.keys()):
            source_records = sources_dict[source_name]
            fp = compute_combined_fingerprint(source_records)
            source_fps.append(f"{source_name}={fp}")

        combined_key = "|".join(source_fps)

        # Hash the combined key to keep it manageable
        combined_hash = hashlib.sha256(combined_key.encode()).hexdigest()[:16]

        components = [branch, self.name, combined_hash, step_version_hash]
        return ":".join(components)

    def _organize_by_source(self, inputs: list["Record"]) -> dict[str, list["Record"]]:
        """Organize input records by their source step name.

        Args:
            inputs: All input records from various upstream steps.

        Returns:
            Dictionary mapping step names to lists of records.
        """
        sources_dict: dict[str, list["Record"]] = {}
        for record in inputs:
            step = record.step_name
            if step not in sources_dict:
                sources_dict[step] = []
            sources_dict[step].append(record)
        return sources_dict

    def execute(
        self,
        inputs: list["Record"],
        llm: "LLMClient",
        run_id: str,
    ) -> "Record":
        """Execute the merge on records from multiple sources.

        Args:
            inputs: List of records from all upstream sources.
            llm: LLM client for generation.
            run_id: Current run ID.

        Returns:
            New Record with merged content.
        """
        from synix.db.artifacts import Record

        if not inputs:
            msg = "Merge step requires at least 1 input"
            raise ValueError(msg)

        if self.prompt is None:
            msg = "Merge step requires a prompt function"
            raise ValueError(msg)

        # Organize inputs by source step
        sources_dict = self._organize_by_source(inputs)

        # Call prompt with organized dict
        prompt_text = self.prompt(sources_dict)
        response = llm.complete(prompt_text)

        # Create output record
        output = Record(
            id=str(uuid4()),
            content=response.content,
            content_fingerprint=Record.compute_fingerprint(response.content),
            step_name=self.name,
            branch="main",  # v0.1 always main
            materialization_key="",  # Set by caller
            run_id=run_id,
        )

        # Set audit info
        output.audit = {
            "prompt_hash": self.compute_version_hash(),
            "model": llm.config.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "input_count": len(inputs),
            "source_count": len(sources_dict),
        }

        # Set metadata
        output.metadata_ = {
            "meta.merge.source_steps": list(sources_dict.keys()),
            "meta.merge.source_counts": {k: len(v) for k, v in sources_dict.items()},
            "meta.merge.total_inputs": len(inputs),
        }

        return output


def create_merge_step(
    name: str,
    sources: list[str],
    prompt: PromptFunction,
    model: str = "deepseek-chat",
) -> MergeStep:
    """Factory function to create a merge step.

    Args:
        name: Step name.
        sources: Names of upstream steps to merge.
        prompt: Function that takes dict[str, list[Record]] and returns prompt string.
        model: LLM model to use.

    Returns:
        Configured MergeStep.
    """
    return MergeStep(
        name=name,
        from_=None,  # MergeStep doesn't use from_
        sources=sources,
        prompt=prompt,
        model=model,
    )
