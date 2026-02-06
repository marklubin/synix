"""End-to-end tests for complete pipeline flows."""

import pytest


class TestFullPipelineFlow:
    """Tests for complete source → transform → aggregate pipeline."""

    def test_complete_pipeline_with_mock_llm(
        self, initialized_db, claude_export_file, mock_llm, sample_prompt, sample_aggregate_prompt
    ):
        """Full pipeline runs with mock LLM and creates all records."""
        from synix.pipeline import Pipeline

        pipeline = Pipeline("test", agent="test", settings=initialized_db, llm=mock_llm)
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.transform("summaries", from_="claude", prompt=sample_prompt)
        pipeline.aggregate("monthly", from_="summaries", period="month", prompt=sample_aggregate_prompt)

        result = pipeline.run()

        assert result.status == "completed"
        assert result.stats["output"] > 0
        assert result.stats["errors"] == 0

        # Verify records were created at each step
        source_records = pipeline.get_records("claude")
        summary_records = pipeline.get_records("summaries")
        monthly_records = pipeline.get_records("monthly")

        assert len(source_records) == 3  # 3 conversations in fixture
        assert len(summary_records) == 3  # 1:1 transform
        assert len(monthly_records) >= 1  # At least 1 month aggregate

    def test_provenance_chain(
        self, initialized_db, claude_export_file, mock_llm, sample_prompt, sample_aggregate_prompt
    ):
        """Records maintain provenance chain."""
        from synix.pipeline import Pipeline

        pipeline = Pipeline("test", agent="test", settings=initialized_db, llm=mock_llm)
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.transform("summaries", from_="claude", prompt=sample_prompt)
        pipeline.aggregate("monthly", from_="summaries", period="month", prompt=sample_aggregate_prompt)

        pipeline.run()

        # Get a monthly record
        monthly_records = pipeline.get_records("monthly")
        assert len(monthly_records) >= 1

        # Check provenance
        monthly = monthly_records[0]
        sources = pipeline.get_sources(monthly.id)

        # Monthly should link to summaries
        assert len(sources) > 0
        assert all(s.step_name == "summaries" for s in sources)

        # Each summary should link to source
        for summary in sources:
            summary_sources = pipeline.get_sources(summary.id)
            assert len(summary_sources) == 1
            assert summary_sources[0].step_name == "claude"

    def test_llm_called_for_transforms(
        self, initialized_db, claude_export_file, mock_llm, sample_prompt
    ):
        """LLM is called for each transform."""
        from synix.pipeline import Pipeline

        pipeline = Pipeline("test", agent="test", settings=initialized_db, llm=mock_llm)
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.transform("summaries", from_="claude", prompt=sample_prompt)

        pipeline.run()

        # LLM should have been called for each source record
        source_records = pipeline.get_records("claude")
        assert len(mock_llm.calls) == len(source_records)

    def test_search_works_after_run(
        self, initialized_db, claude_export_file, mock_llm, sample_prompt
    ):
        """FTS search works on pipeline records."""
        from synix.pipeline import Pipeline

        pipeline = Pipeline("test", agent="test", settings=initialized_db, llm=mock_llm)
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.transform("summaries", from_="claude", prompt=sample_prompt)

        pipeline.run()

        # Search in source records
        hits = pipeline.search("Rust")
        assert len(hits) >= 1

    def test_run_result_has_stats(
        self, initialized_db, claude_export_file, mock_llm, sample_prompt
    ):
        """Run result includes accurate statistics."""
        from synix.pipeline import Pipeline

        pipeline = Pipeline("test", agent="test", settings=initialized_db, llm=mock_llm)
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.transform("summaries", from_="claude", prompt=sample_prompt)

        result = pipeline.run()

        assert "input" in result.stats
        assert "output" in result.stats
        assert "skipped" in result.stats
        assert "errors" in result.stats
        assert "tokens" in result.stats

        # First run should have outputs, no skips
        assert result.stats["output"] > 0

    def test_run_specific_step(
        self, initialized_db, claude_export_file, mock_llm, sample_prompt, sample_aggregate_prompt
    ):
        """Can run only a specific step and its dependencies."""
        from synix.pipeline import Pipeline

        pipeline = Pipeline("test", agent="test", settings=initialized_db, llm=mock_llm)
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.transform("summaries", from_="claude", prompt=sample_prompt)
        pipeline.aggregate("monthly", from_="summaries", period="month", prompt=sample_aggregate_prompt)

        # Run only up to summaries
        result = pipeline.run(step="summaries")

        assert result.status == "completed"

        # Should have source and summary records
        source_records = pipeline.get_records("claude")
        summary_records = pipeline.get_records("summaries")
        monthly_records = pipeline.get_records("monthly")

        assert len(source_records) > 0
        assert len(summary_records) > 0
        assert len(monthly_records) == 0  # Monthly shouldn't run

    def test_plan_shows_execution_info(
        self, initialized_db, claude_export_file, sample_prompt, sample_aggregate_prompt
    ):
        """Plan shows what would execute."""
        from synix.pipeline import Pipeline

        pipeline = Pipeline("test", agent="test", settings=initialized_db)
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.transform("summaries", from_="claude", prompt=sample_prompt)
        pipeline.aggregate("monthly", from_="summaries", period="month", prompt=sample_aggregate_prompt)

        plan = pipeline.plan()

        assert len(plan.steps) == 3
        step_names = [s["name"] for s in plan.steps]
        assert "claude" in step_names
        assert "summaries" in step_names
        assert "monthly" in step_names
