"""Unit tests for Fold step."""

from uuid import uuid4

import pytest


class TestFoldStep:
    """Tests for FoldStep."""

    def test_create_fold_step(self):
        """FoldStep can be created."""
        from synix.steps.fold import FoldStep

        def prompt(state: str, record) -> str:
            return f"State: {state}\nRecord: {record.content}"

        step = FoldStep(
            name="narrative",
            from_="summaries",
            prompt=prompt,
            initial_state="",
        )

        assert step.name == "narrative"
        assert step.from_ == "summaries"
        assert step.step_type == "fold"
        assert step.initial_state == ""

    def test_fold_step_with_initial_state(self):
        """FoldStep can have a non-empty initial state."""
        from synix.steps.fold import FoldStep

        def prompt(state: str, record) -> str:
            return f"{state}\n{record.content}"

        step = FoldStep(
            name="narrative",
            from_="summaries",
            prompt=prompt,
            initial_state="# My Story\n\n",
        )

        assert step.initial_state == "# My Story\n\n"

    def test_materialization_key_includes_input_count(self):
        """Materialization key includes input count."""
        from synix.db.artifacts import Record
        from synix.steps.fold import FoldStep

        def prompt(state: str, record) -> str:
            return f"{state} {record.content}"

        step = FoldStep(
            name="narrative",
            from_="summaries",
            prompt=prompt,
        )

        records = []
        for i in range(3):
            record = Record(
                id=str(uuid4()),
                content=f"Content {i}",
                content_fingerprint=f"fp{i}",
                step_name="summaries",
                branch="main",
                materialization_key=f"key{i}",
                run_id="run",
            )
            record.metadata_ = {"meta.time.created_at": f"2024-03-{15+i:02d}T10:00:00Z"}
            records.append(record)

        version_hash = step.compute_version_hash()
        mat_key = step.compute_materialization_key(records, version_hash)

        # Format: branch:step_name:count:combined_fp:version_hash
        parts = mat_key.split(":")
        assert len(parts) == 5
        assert parts[0] == "main"
        assert parts[1] == "narrative"
        assert parts[2] == "3"  # count
        assert parts[4] == version_hash

    def test_materialization_key_changes_with_content(self):
        """Materialization key changes when content changes."""
        from synix.db.artifacts import Record
        from synix.steps.fold import FoldStep

        def prompt(state: str, record) -> str:
            return f"{state} {record.content}"

        step = FoldStep(
            name="narrative",
            from_="summaries",
            prompt=prompt,
        )

        record1 = Record(
            id=str(uuid4()),
            content="Content A",
            content_fingerprint="fpA",
            step_name="summaries",
            branch="main",
            materialization_key="keyA",
            run_id="run",
        )
        record1.metadata_ = {"meta.time.created_at": "2024-03-15T10:00:00Z"}

        record2 = Record(
            id=str(uuid4()),
            content="Content B",
            content_fingerprint="fpB",  # Different fingerprint
            step_name="summaries",
            branch="main",
            materialization_key="keyB",
            run_id="run",
        )
        record2.metadata_ = {"meta.time.created_at": "2024-03-15T10:00:00Z"}

        version_hash = step.compute_version_hash()
        mat_key1 = step.compute_materialization_key([record1], version_hash)
        mat_key2 = step.compute_materialization_key([record2], version_hash)

        assert mat_key1 != mat_key2

    def test_execute_processes_in_order(self, mock_llm):
        """Execute processes records in chronological order."""
        from synix.db.artifacts import Record
        from synix.steps.fold import FoldStep

        call_order = []

        def prompt(state: str, record) -> str:
            call_order.append(record.content)
            return f"Updated: {state} + {record.content}"

        step = FoldStep(
            name="narrative",
            from_="summaries",
            prompt=prompt,
        )

        # Create records in reverse chronological order
        records = []
        for i in [2, 0, 1]:  # Out of order
            record = Record(
                id=str(uuid4()),
                content=f"Item-{i}",
                content_fingerprint=f"fp{i}",
                step_name="summaries",
                branch="main",
                materialization_key=f"key{i}",
                run_id="run",
            )
            record.metadata_ = {"meta.time.created_at": f"2024-03-{10+i:02d}T10:00:00Z"}
            records.append(record)

        # Set up mock to return incrementing responses
        mock_llm.default_response = "accumulated"

        output = step.execute(records, mock_llm, "run-123")

        # Should be processed in chronological order (0, 1, 2)
        assert call_order == ["Item-0", "Item-1", "Item-2"]

    def test_execute_accumulates_state(self, mock_llm):
        """Execute accumulates state through iterations."""
        from synix.db.artifacts import Record
        from synix.steps.fold import FoldStep

        def prompt(state: str, record) -> str:
            return f"Current: {state}\nAdd: {record.content}"

        step = FoldStep(
            name="narrative",
            from_="summaries",
            prompt=prompt,
            initial_state="START",
        )

        records = []
        for i in range(2):
            record = Record(
                id=str(uuid4()),
                content=f"Item-{i}",
                content_fingerprint=f"fp{i}",
                step_name="summaries",
                branch="main",
                materialization_key=f"key{i}",
                run_id="run",
            )
            record.metadata_ = {"meta.time.created_at": f"2024-03-{10+i:02d}T10:00:00Z"}
            records.append(record)

        # Mock returns different responses for each call
        mock_llm.responses = {
            "Current: START": "STATE_1",
            "Current: STATE_1": "STATE_2",
        }
        mock_llm.default_response = "FINAL_STATE"

        output = step.execute(records, mock_llm, "run-123")

        # Should have called LLM twice (once per record)
        assert len(mock_llm.calls) == 2

    def test_execute_produces_record_with_metadata(self, mock_llm):
        """Execute produces a record with fold metadata."""
        from synix.db.artifacts import Record
        from synix.steps.fold import FoldStep

        def prompt(state: str, record) -> str:
            return f"{state} {record.content}"

        step = FoldStep(
            name="narrative",
            from_="summaries",
            prompt=prompt,
        )

        records = []
        for i in range(3):
            record = Record(
                id=str(uuid4()),
                content=f"Content {i}",
                content_fingerprint=f"fp{i}",
                step_name="summaries",
                branch="main",
                materialization_key=f"key{i}",
                run_id="run",
            )
            record.metadata_ = {"meta.time.created_at": f"2024-03-{10+i:02d}T10:00:00Z"}
            records.append(record)

        output = step.execute(records, mock_llm, "run-123")

        assert output.step_name == "narrative"
        assert output.metadata_["meta.fold.input_count"] == 3
        assert output.metadata_["meta.fold.initial_state_empty"] is True
        assert output.audit["input_count"] == 3
        assert output.audit["iterations"] == 3

    def test_execute_aggregates_token_counts(self, mock_llm):
        """Execute aggregates token counts from all iterations."""
        from synix.db.artifacts import Record
        from synix.steps.fold import FoldStep

        def prompt(state: str, record) -> str:
            return f"{state} {record.content}"

        step = FoldStep(
            name="narrative",
            from_="summaries",
            prompt=prompt,
        )

        records = []
        for i in range(3):
            record = Record(
                id=str(uuid4()),
                content=f"Content {i}",
                content_fingerprint=f"fp{i}",
                step_name="summaries",
                branch="main",
                materialization_key=f"key{i}",
                run_id="run",
            )
            record.metadata_ = {"meta.time.created_at": f"2024-03-{10+i:02d}T10:00:00Z"}
            records.append(record)

        output = step.execute(records, mock_llm, "run-123")

        # Token counts should be aggregated from all 3 calls
        assert output.audit["input_tokens"] > 0
        assert output.audit["output_tokens"] > 0

    def test_execute_requires_inputs(self, mock_llm):
        """Execute raises error with no inputs."""
        from synix.steps.fold import FoldStep

        def prompt(state: str, record) -> str:
            return f"{state} {record.content}"

        step = FoldStep(
            name="narrative",
            from_="summaries",
            prompt=prompt,
        )

        with pytest.raises(ValueError, match="requires at least 1 input"):
            step.execute([], mock_llm, "run-123")

    def test_version_hash_changes_with_prompt(self):
        """Version hash changes when prompt changes."""
        from synix.steps.fold import FoldStep

        def prompt1(state: str, record) -> str:
            return f"{state} {record.content}"

        def prompt2(state: str, record) -> str:
            return f"Different: {state} - {record.content}"

        step1 = FoldStep(name="narrative", from_="summaries", prompt=prompt1)
        step2 = FoldStep(name="narrative", from_="summaries", prompt=prompt2)

        assert step1.compute_version_hash() != step2.compute_version_hash()

    def test_create_fold_step_factory(self):
        """Factory function creates FoldStep correctly."""
        from synix.steps.fold import create_fold_step

        def prompt(state: str, record) -> str:
            return f"{state} {record.content}"

        step = create_fold_step(
            name="narrative",
            from_="summaries",
            prompt=prompt,
            initial_state="BEGIN",
            model="gpt-4",
        )

        assert step.name == "narrative"
        assert step.from_ == "summaries"
        assert step.initial_state == "BEGIN"
        assert step.model == "gpt-4"
