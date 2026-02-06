"""Unit tests for Merge step."""

from uuid import uuid4

import pytest


class TestMergeStep:
    """Tests for MergeStep."""

    def test_create_merge_step(self):
        """MergeStep can be created."""
        from synix.steps.merge import MergeStep

        def prompt(sources: dict) -> str:
            return f"Merge: {list(sources.keys())}"

        step = MergeStep(
            name="combined",
            from_=None,
            sources=["step_a", "step_b"],
            prompt=prompt,
        )

        assert step.name == "combined"
        assert step.sources == ["step_a", "step_b"]
        assert step.step_type == "merge"

    def test_merge_step_from_is_empty_string(self):
        """MergeStep sets from_ to empty string."""
        from synix.steps.merge import MergeStep

        def prompt(sources: dict) -> str:
            return str(sources)

        step = MergeStep(
            name="combined",
            from_=None,
            sources=["a", "b"],
            prompt=prompt,
        )

        # from_ should be empty string (not None)
        assert step.from_ == ""

    def test_organize_by_source(self):
        """Records are organized by source step name."""
        from synix.db.artifacts import Record
        from synix.steps.merge import MergeStep

        def prompt(sources: dict) -> str:
            return str(sources)

        step = MergeStep(
            name="combined",
            from_=None,
            sources=["step_a", "step_b"],
            prompt=prompt,
        )

        records = []
        # 2 records from step_a
        for i in range(2):
            record = Record(
                id=str(uuid4()),
                content=f"A content {i}",
                content_fingerprint=f"fpA{i}",
                step_name="step_a",
                branch="main",
                materialization_key=f"keyA{i}",
                run_id="run",
            )
            records.append(record)

        # 3 records from step_b
        for i in range(3):
            record = Record(
                id=str(uuid4()),
                content=f"B content {i}",
                content_fingerprint=f"fpB{i}",
                step_name="step_b",
                branch="main",
                materialization_key=f"keyB{i}",
                run_id="run",
            )
            records.append(record)

        organized = step._organize_by_source(records)

        assert "step_a" in organized
        assert "step_b" in organized
        assert len(organized["step_a"]) == 2
        assert len(organized["step_b"]) == 3

    def test_materialization_key_includes_source_fingerprints(self):
        """Materialization key includes fingerprints from all sources."""
        from synix.db.artifacts import Record
        from synix.steps.merge import MergeStep

        def prompt(sources: dict) -> str:
            return str(sources)

        step = MergeStep(
            name="combined",
            from_=None,
            sources=["step_a", "step_b"],
            prompt=prompt,
        )

        records = []
        for step_name in ["step_a", "step_b"]:
            record = Record(
                id=str(uuid4()),
                content=f"{step_name} content",
                content_fingerprint=f"fp_{step_name}",
                step_name=step_name,
                branch="main",
                materialization_key=f"key_{step_name}",
                run_id="run",
            )
            records.append(record)

        version_hash = step.compute_version_hash()
        mat_key = step.compute_materialization_key(records, version_hash)

        # Format: branch:step_name:combined_hash:version_hash
        parts = mat_key.split(":")
        assert len(parts) == 4
        assert parts[0] == "main"
        assert parts[1] == "combined"
        assert parts[3] == version_hash

    def test_materialization_key_changes_with_source_content(self):
        """Materialization key changes when source content changes."""
        from synix.db.artifacts import Record
        from synix.steps.merge import MergeStep

        def prompt(sources: dict) -> str:
            return str(sources)

        step = MergeStep(
            name="combined",
            from_=None,
            sources=["step_a"],
            prompt=prompt,
        )

        record1 = Record(
            id=str(uuid4()),
            content="Content A",
            content_fingerprint="fpA",
            step_name="step_a",
            branch="main",
            materialization_key="keyA",
            run_id="run",
        )

        record2 = Record(
            id=str(uuid4()),
            content="Content B",
            content_fingerprint="fpB",  # Different fingerprint
            step_name="step_a",
            branch="main",
            materialization_key="keyB",
            run_id="run",
        )

        version_hash = step.compute_version_hash()
        mat_key1 = step.compute_materialization_key([record1], version_hash)
        mat_key2 = step.compute_materialization_key([record2], version_hash)

        assert mat_key1 != mat_key2

    def test_execute_produces_record_with_metadata(self, mock_llm):
        """Execute produces a record with merge metadata."""
        from synix.db.artifacts import Record
        from synix.steps.merge import MergeStep

        def prompt(sources: dict) -> str:
            parts = []
            for step_name, recs in sources.items():
                parts.append(f"{step_name}: {len(recs)} records")
            return "Combine:\n" + "\n".join(parts)

        step = MergeStep(
            name="combined",
            from_=None,
            sources=["step_a", "step_b"],
            prompt=prompt,
        )

        records = []
        for step_name, count in [("step_a", 2), ("step_b", 3)]:
            for i in range(count):
                record = Record(
                    id=str(uuid4()),
                    content=f"{step_name} content {i}",
                    content_fingerprint=f"fp_{step_name}_{i}",
                    step_name=step_name,
                    branch="main",
                    materialization_key=f"key_{step_name}_{i}",
                    run_id="run",
                )
                records.append(record)

        output = step.execute(records, mock_llm, "run-123")

        assert output.step_name == "combined"
        assert set(output.metadata_["meta.merge.source_steps"]) == {"step_a", "step_b"}
        assert output.metadata_["meta.merge.source_counts"]["step_a"] == 2
        assert output.metadata_["meta.merge.source_counts"]["step_b"] == 3
        assert output.metadata_["meta.merge.total_inputs"] == 5
        assert output.audit["source_count"] == 2
        assert output.audit["input_count"] == 5

    def test_execute_passes_organized_sources_to_prompt(self, mock_llm):
        """Execute passes correctly organized sources to prompt."""
        from synix.db.artifacts import Record
        from synix.steps.merge import MergeStep

        received_sources = {}

        def prompt(sources: dict) -> str:
            received_sources.update(sources)
            return "merged"

        step = MergeStep(
            name="combined",
            from_=None,
            sources=["step_a", "step_b"],
            prompt=prompt,
        )

        records = []
        for step_name in ["step_a", "step_b"]:
            record = Record(
                id=str(uuid4()),
                content=f"{step_name} content",
                content_fingerprint=f"fp_{step_name}",
                step_name=step_name,
                branch="main",
                materialization_key=f"key_{step_name}",
                run_id="run",
            )
            records.append(record)

        step.execute(records, mock_llm, "run-123")

        assert "step_a" in received_sources
        assert "step_b" in received_sources
        assert len(received_sources["step_a"]) == 1
        assert len(received_sources["step_b"]) == 1

    def test_execute_requires_inputs(self, mock_llm):
        """Execute raises error with no inputs."""
        from synix.steps.merge import MergeStep

        def prompt(sources: dict) -> str:
            return str(sources)

        step = MergeStep(
            name="combined",
            from_=None,
            sources=["step_a", "step_b"],
            prompt=prompt,
        )

        with pytest.raises(ValueError, match="requires at least 1 input"):
            step.execute([], mock_llm, "run-123")

    def test_version_hash_changes_with_prompt(self):
        """Version hash changes when prompt changes."""
        from synix.steps.merge import MergeStep

        def prompt1(sources: dict) -> str:
            return f"Merge: {sources}"

        def prompt2(sources: dict) -> str:
            return f"Combine: {sources}"

        step1 = MergeStep(name="combined", from_=None, sources=["a"], prompt=prompt1)
        step2 = MergeStep(name="combined", from_=None, sources=["a"], prompt=prompt2)

        assert step1.compute_version_hash() != step2.compute_version_hash()

    def test_create_merge_step_factory(self):
        """Factory function creates MergeStep correctly."""
        from synix.steps.merge import create_merge_step

        def prompt(sources: dict) -> str:
            return str(sources)

        step = create_merge_step(
            name="combined",
            sources=["step_a", "step_b", "step_c"],
            prompt=prompt,
            model="gpt-4",
        )

        assert step.name == "combined"
        assert step.sources == ["step_a", "step_b", "step_c"]
        assert step.model == "gpt-4"
