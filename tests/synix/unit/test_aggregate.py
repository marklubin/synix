"""Unit tests for Aggregate step."""

from uuid import uuid4

import pytest


class TestAggregateStep:
    """Tests for AggregateStep."""

    def test_create_aggregate_step(self, sample_aggregate_prompt):
        """AggregateStep can be created."""
        from synix.steps.aggregate import AggregateStep

        step = AggregateStep(
            name="monthly",
            from_="summaries",
            prompt=sample_aggregate_prompt,
            period="month",
        )

        assert step.name == "monthly"
        assert step.from_ == "summaries"
        assert step.step_type == "aggregate"
        assert step.period == "month"

    def test_group_records_by_month(self, sample_aggregate_prompt):
        """Records are grouped by month correctly."""
        from synix.db.artifacts import Record
        from synix.steps.aggregate import AggregateStep

        step = AggregateStep(
            name="monthly",
            from_="summaries",
            prompt=sample_aggregate_prompt,
            period="month",
        )

        records = []
        for i, date in enumerate(
            ["2024-03-15T10:00:00Z", "2024-03-20T14:00:00Z", "2024-04-05T09:00:00Z"]
        ):
            record = Record(
                id=str(uuid4()),
                content=f"Content {i}",
                content_fingerprint=f"fp{i}",
                step_name="summaries",
                branch="main",
                materialization_key=f"key{i}",
                run_id="run",
            )
            record.metadata_ = {"meta.time.created_at": date}
            records.append(record)

        groups = step.group_records(records)

        assert "2024-03" in groups
        assert "2024-04" in groups
        assert len(groups["2024-03"]) == 2
        assert len(groups["2024-04"]) == 1

    def test_group_records_by_week(self, sample_aggregate_prompt):
        """Records are grouped by week correctly."""
        from synix.db.artifacts import Record
        from synix.steps.aggregate import AggregateStep

        step = AggregateStep(
            name="weekly",
            from_="summaries",
            prompt=sample_aggregate_prompt,
            period="week",
        )

        records = []
        for i, date in enumerate(["2024-03-15T10:00:00Z", "2024-03-16T14:00:00Z"]):
            record = Record(
                id=str(uuid4()),
                content=f"Content {i}",
                content_fingerprint=f"fp{i}",
                step_name="summaries",
                branch="main",
                materialization_key=f"key{i}",
                run_id="run",
            )
            record.metadata_ = {"meta.time.created_at": date}
            records.append(record)

        groups = step.group_records(records)

        # Both dates should be in the same week
        assert len(groups) == 1

    def test_materialization_key_uses_combined_fingerprint(self, sample_aggregate_prompt):
        """Materialization key includes combined fingerprint."""
        from synix.db.artifacts import Record
        from synix.steps.aggregate import AggregateStep

        step = AggregateStep(
            name="monthly",
            from_="summaries",
            prompt=sample_aggregate_prompt,
            period="month",
        )

        records = []
        for i in range(2):
            record = Record(
                id=str(uuid4()),
                content=f"Content {i}",
                content_fingerprint=f"fingerprint{i}",
                step_name="summaries",
                branch="main",
                materialization_key=f"key{i}",
                run_id="run",
            )
            record.metadata_ = {"meta.time.created_at": "2024-03-15T10:00:00Z"}
            records.append(record)

        version_hash = step.compute_version_hash()
        mat_key = step.compute_materialization_key(records, version_hash)

        # Format: branch:step_name:group_key:combined_fingerprint:version_hash
        parts = mat_key.split(":")
        assert len(parts) == 5
        assert parts[0] == "main"
        assert parts[1] == "monthly"
        assert parts[2] == "2024-03"
        # parts[3] is combined fingerprint
        assert parts[4] == version_hash

    def test_combined_fingerprint_changes_with_membership(self, sample_aggregate_prompt):
        """Combined fingerprint changes when group membership changes."""
        from synix.db.artifacts import Record
        from synix.steps.aggregate import AggregateStep
        from synix.steps.base import compute_combined_fingerprint

        step = AggregateStep(
            name="monthly",
            from_="summaries",
            prompt=sample_aggregate_prompt,
            period="month",
        )

        # Create base record
        record1 = Record(
            id=str(uuid4()),
            content="Content 1",
            content_fingerprint="fp1",
            step_name="summaries",
            branch="main",
            materialization_key="key1",
            run_id="run",
        )

        record2 = Record(
            id=str(uuid4()),
            content="Content 2",
            content_fingerprint="fp2",
            step_name="summaries",
            branch="main",
            materialization_key="key2",
            run_id="run",
        )

        fp_single = compute_combined_fingerprint([record1])
        fp_both = compute_combined_fingerprint([record1, record2])

        assert fp_single != fp_both

    def test_execute_produces_record(self, sample_aggregate_prompt, mock_llm):
        """Execute produces a new record with group metadata."""
        from synix.db.artifacts import Record
        from synix.steps.aggregate import AggregateStep

        step = AggregateStep(
            name="monthly",
            from_="summaries",
            prompt=sample_aggregate_prompt,
            period="month",
        )

        records = []
        for i in range(2):
            record = Record(
                id=str(uuid4()),
                content=f"Summary {i}",
                content_fingerprint=f"fp{i}",
                step_name="summaries",
                branch="main",
                materialization_key=f"key{i}",
                run_id="run",
            )
            record.metadata_ = {"meta.time.created_at": "2024-03-15T10:00:00Z"}
            records.append(record)

        output = step.execute(records, mock_llm, "run-456")

        assert output.step_name == "monthly"
        assert output.metadata_["meta.time.period"] == "2024-03"
        assert output.metadata_["meta.aggregate.input_count"] == 2
        assert output.audit["input_count"] == 2
