"""End-to-end tests for fold pipelines."""

import pytest


class TestFoldPipeline:
    """E2E tests for pipelines with fold steps."""

    def test_fold_step_processes_records_sequentially(
        self, initialized_db, claude_export_file, mock_llm
    ):
        """Fold step processes records in order and accumulates state."""
        from synix import Pipeline

        call_count = [0]
        states = []

        def evolve_narrative(state: str, record) -> str:
            call_count[0] += 1
            states.append(state)
            return f"{state}\n- {record.content[:50]}"

        pipeline = Pipeline("fold-test", agent="test", settings=initialized_db)
        pipeline.llm = mock_llm
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.fold("narrative", from_="claude", prompt=evolve_narrative, initial_state="# Story")

        # Set up mock to return evolving state
        mock_llm.default_response = "Narrative updated"

        result = pipeline.run()

        assert result.status == "completed"
        # Should have made one call per source record
        assert call_count[0] == 3  # 3 conversations in fixture
        # First call should have initial state
        assert states[0] == "# Story"

    def test_fold_pipeline_caches_output(
        self, initialized_db, claude_export_file, mock_llm
    ):
        """Fold step output is cached on subsequent runs."""
        from synix import Pipeline

        def evolve(state: str, record) -> str:
            return f"{state} + {record.content[:20]}"

        pipeline = Pipeline("fold-cache-test", agent="test", settings=initialized_db)
        pipeline.llm = mock_llm
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.fold("narrative", from_="claude", prompt=evolve)

        # First run
        result1 = pipeline.run()
        assert result1.status == "completed"
        first_output = result1.stats["output"]
        first_calls = len(mock_llm.calls)

        # Second run - should skip
        result2 = pipeline.run()
        assert result2.status == "completed"
        assert result2.stats["skipped"] > 0
        # No additional LLM calls for cached output
        assert len(mock_llm.calls) == first_calls

    def test_fold_step_regenerates_on_prompt_change(
        self, initialized_db, claude_export_file, mock_llm
    ):
        """Fold step regenerates when prompt changes."""
        from synix import Pipeline

        def evolve_v1(state: str, record) -> str:
            return f"V1: {state}"

        pipeline1 = Pipeline("fold-regen-test", agent="test", settings=initialized_db)
        pipeline1.llm = mock_llm
        pipeline1.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline1.fold("narrative", from_="claude", prompt=evolve_v1)

        result1 = pipeline1.run()
        assert result1.status == "completed"
        calls_v1 = len(mock_llm.calls)

        # Change prompt
        def evolve_v2(state: str, record) -> str:
            return f"V2: {state}"

        pipeline2 = Pipeline("fold-regen-test", agent="test", settings=initialized_db)
        pipeline2.llm = mock_llm
        pipeline2.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline2.fold("narrative", from_="claude", prompt=evolve_v2)

        result2 = pipeline2.run()
        assert result2.status == "completed"
        # Should have made new LLM calls
        assert len(mock_llm.calls) > calls_v1

    def test_fold_with_transform_chain(
        self, initialized_db, claude_export_file, mock_llm
    ):
        """Fold can follow transform in pipeline."""
        from synix import Pipeline

        def summarize(record) -> str:
            return f"Summarize: {record.content[:100]}"

        def evolve(state: str, record) -> str:
            return f"{state}\n{record.content}"

        pipeline = Pipeline("transform-fold-test", agent="test", settings=initialized_db)
        pipeline.llm = mock_llm
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.transform("summaries", from_="claude", prompt=summarize)
        pipeline.fold("narrative", from_="summaries", prompt=evolve)

        result = pipeline.run()

        assert result.status == "completed"
        # Should have output from both transform and fold
        assert result.stats["output"] > 0

    def test_fold_output_has_correct_metadata(
        self, initialized_db, claude_export_file, mock_llm
    ):
        """Fold output record has fold-specific metadata."""
        from synix import Pipeline

        def evolve(state: str, record) -> str:
            return f"{state} updated"

        pipeline = Pipeline("fold-meta-test", agent="test", settings=initialized_db)
        pipeline.llm = mock_llm
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.fold("narrative", from_="claude", prompt=evolve, initial_state="")

        result = pipeline.run()
        assert result.status == "completed"

        # Get the fold output record
        records = pipeline.get_records("narrative")
        assert len(records) == 1  # Fold produces single output

        record = records[0]
        assert record.metadata_.get("meta.fold.input_count") == 3
        assert record.metadata_.get("meta.fold.initial_state_empty") is True
        assert record.audit.get("iterations") == 3
