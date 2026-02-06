"""Aggregate step implementation (N:1 processing)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from synix.steps.base import PromptFunction, Step, compute_combined_fingerprint

if TYPE_CHECKING:
    from synix.db.artifacts import Record
    from synix.llm.client import LLMClient


@dataclass
class AggregateStep(Step):
    """Aggregate step: processes N input records into 1 output per group.

    Groups records by a time period (month, week, day) and generates
    one aggregated output per group.

    Materialization key format:
        (branch, step_name, group_key, combined_input_fingerprint, step_version_hash)
    """

    period: str = "month"  # month, week, day
    step_type: str = field(init=False, default="aggregate")

    def compute_materialization_key(
        self,
        inputs: list["Record"],
        step_version_hash: str,
        branch: str = "main",
    ) -> str:
        """Compute cache key for aggregate output.

        Aggregate uses a combined fingerprint of all inputs to detect
        when group membership or content changes.
        """
        if not inputs:
            msg = "Aggregate step requires at least 1 input"
            raise ValueError(msg)

        # Get group key from first input (all should have same group)
        group_key = self._extract_group_key(inputs[0])
        combined_fp = compute_combined_fingerprint(inputs)

        components = [branch, self.name, group_key, combined_fp, step_version_hash]
        return ":".join(components)

    def _extract_group_key(self, record: "Record") -> str:
        """Extract group key from record based on period.

        Looks for metadata in order:
        1. meta.time.period (explicit period value)
        2. meta.time.created_at (parse and format)
        """
        meta = record.metadata_

        # Check for explicit period
        period_val = meta.get("meta.time.period")
        if period_val:
            return str(period_val)

        # Parse created_at and format based on period
        created_at = meta.get("meta.time.created_at")
        if created_at:
            if isinstance(created_at, str):
                # Parse ISO format
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            elif isinstance(created_at, datetime):
                dt = created_at
            else:
                dt = datetime.now()

            if self.period == "month":
                return dt.strftime("%Y-%m")
            elif self.period == "week":
                return dt.strftime("%Y-W%W")
            elif self.period == "day":
                return dt.strftime("%Y-%m-%d")
            else:
                return dt.strftime("%Y-%m")

        # Fallback
        return "unknown"

    def group_records(self, records: list["Record"]) -> dict[str, list["Record"]]:
        """Group records by period.

        Args:
            records: All input records from upstream step.

        Returns:
            Dictionary mapping group_key -> list of records.
        """
        groups: dict[str, list["Record"]] = {}
        for record in records:
            key = self._extract_group_key(record)
            if key not in groups:
                groups[key] = []
            groups[key].append(record)
        return groups

    def execute(
        self,
        inputs: list["Record"],
        llm: "LLMClient",
        run_id: str,
    ) -> "Record":
        """Execute the aggregate on multiple input records.

        Args:
            inputs: List of input records for one group.
            llm: LLM client for generation.
            run_id: Current run ID.

        Returns:
            New Record with aggregated content.
        """
        from synix.db.artifacts import Record

        if not inputs:
            msg = "Aggregate step requires at least 1 input"
            raise ValueError(msg)

        if self.prompt is None:
            msg = "Aggregate step requires a prompt function"
            raise ValueError(msg)

        # Get group key for period label
        group_key = self._extract_group_key(inputs[0])

        # Sort inputs by created_at
        def get_created_at(r: "Record") -> str:
            return str(r.metadata_.get("meta.time.created_at", ""))

        sorted_inputs = sorted(inputs, key=get_created_at)

        # Generate prompt from inputs - prompt takes (records, period)
        prompt_text = self.prompt(sorted_inputs, group_key)

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
            "input_count": len(inputs),
        }

        # Set metadata
        record.metadata_ = {
            "meta.time.period": group_key,
            "meta.aggregate.input_count": len(inputs),
            "meta.aggregate.period_type": self.period,
        }

        return record


def create_aggregate_step(
    name: str,
    from_: str,
    prompt: PromptFunction,
    period: str = "month",
    model: str = "deepseek-chat",
) -> AggregateStep:
    """Factory function to create an aggregate step.

    Args:
        name: Step name.
        from_: Name of upstream step.
        prompt: Function that takes (list[Record], period) and returns prompt string.
        period: Time period for grouping ('month', 'week', 'day').
        model: LLM model to use.

    Returns:
        Configured AggregateStep.
    """
    return AggregateStep(
        name=name,
        from_=from_,
        prompt=prompt,
        period=period,
        model=model,
    )
