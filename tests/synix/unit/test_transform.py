"""Unit tests for Transform step."""

from uuid import uuid4

import pytest


class TestTransformStep:
    """Tests for TransformStep."""

    def test_create_transform_step(self, sample_prompt):
        """TransformStep can be created."""
        from synix.steps.transform import TransformStep

        step = TransformStep(
            name="summarize",
            from_="source",
            prompt=sample_prompt,
        )

        assert step.name == "summarize"
        assert step.from_ == "source"
        assert step.step_type == "transform"

    def test_materialization_key_format(self, sample_prompt):
        """Materialization key has correct format."""
        from synix.db.artifacts import Record
        from synix.steps.transform import TransformStep

        step = TransformStep(name="summarize", from_="source", prompt=sample_prompt)

        record = Record(
            id="record-123",
            content="Test content",
            content_fingerprint="fp123",
            step_name="source",
            branch="main",
            materialization_key="old-key",
            run_id="run-123",
        )

        version_hash = step.compute_version_hash()
        mat_key = step.compute_materialization_key([record], version_hash)

        # Format: branch:step_name:input_id:version_hash
        parts = mat_key.split(":")
        assert len(parts) == 4
        assert parts[0] == "main"
        assert parts[1] == "summarize"
        assert parts[2] == "record-123"
        assert parts[3] == version_hash

    def test_version_hash_changes_with_prompt(self):
        """Version hash changes when prompt changes."""
        from synix.steps.transform import TransformStep

        def prompt1(r):
            return f"Summarize: {r.content}"

        def prompt2(r):
            return f"Describe: {r.content}"

        step1 = TransformStep(name="step", from_="source", prompt=prompt1)
        step2 = TransformStep(name="step", from_="source", prompt=prompt2)

        hash1 = step1.compute_version_hash()
        hash2 = step2.compute_version_hash()

        assert hash1 != hash2

    def test_version_hash_changes_with_model(self, sample_prompt):
        """Version hash changes when model changes."""
        from synix.steps.transform import TransformStep

        step1 = TransformStep(name="step", from_="source", prompt=sample_prompt, model="gpt-4")
        step2 = TransformStep(name="step", from_="source", prompt=sample_prompt, model="gpt-3.5-turbo")

        hash1 = step1.compute_version_hash()
        hash2 = step2.compute_version_hash()

        assert hash1 != hash2

    def test_execute_requires_single_input(self, sample_prompt, mock_llm):
        """Execute raises error if not exactly 1 input."""
        from synix.db.artifacts import Record
        from synix.steps.transform import TransformStep

        step = TransformStep(name="step", from_="source", prompt=sample_prompt)

        # No inputs
        with pytest.raises(ValueError, match="requires exactly 1 input"):
            step.execute([], mock_llm, "run-123")

        # Multiple inputs
        records = [
            Record(
                id=str(uuid4()),
                content="content",
                content_fingerprint="fp",
                step_name="source",
                branch="main",
                materialization_key="key1",
                run_id="run",
            ),
            Record(
                id=str(uuid4()),
                content="content2",
                content_fingerprint="fp2",
                step_name="source",
                branch="main",
                materialization_key="key2",
                run_id="run",
            ),
        ]

        with pytest.raises(ValueError, match="requires exactly 1 input"):
            step.execute(records, mock_llm, "run-123")

    def test_execute_produces_record(self, sample_prompt, mock_llm):
        """Execute produces a new record."""
        from synix.db.artifacts import Record
        from synix.steps.transform import TransformStep

        step = TransformStep(name="summarize", from_="source", prompt=sample_prompt)

        input_record = Record(
            id=str(uuid4()),
            content="Original content to summarize",
            content_fingerprint=Record.compute_fingerprint("Original content"),
            step_name="source",
            branch="main",
            materialization_key="source:key",
            run_id="run-123",
        )
        input_record.metadata_ = {"meta.time.created_at": "2024-03-15T10:00:00Z"}

        output = step.execute([input_record], mock_llm, "run-456")

        assert output.content == mock_llm.default_response
        assert output.step_name == "summarize"
        assert "prompt_hash" in output.audit
        assert output.audit["model"] == "mock-model"
