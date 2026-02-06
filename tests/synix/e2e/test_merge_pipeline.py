"""End-to-end tests for merge pipelines."""

import pytest


class TestMergePipeline:
    """E2E tests for pipelines with merge steps."""

    def test_merge_combines_multiple_sources(
        self, initialized_db, claude_export_file, chatgpt_export_file, mock_llm
    ):
        """Merge step combines records from multiple upstream steps."""
        from synix import Pipeline

        received_sources = {}

        def combine(sources: dict) -> str:
            received_sources.update(sources)
            return "Combined output"

        pipeline = Pipeline("merge-test", agent="test", settings=initialized_db)
        pipeline.llm = mock_llm
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.source("chatgpt", file=str(chatgpt_export_file), format="chatgpt-export")
        pipeline.merge("combined", sources=["claude", "chatgpt"], prompt=combine)

        result = pipeline.run()

        assert result.status == "completed"
        # Should have received records from both sources
        assert "claude" in received_sources
        assert "chatgpt" in received_sources
        assert len(received_sources["claude"]) == 3  # 3 Claude conversations
        assert len(received_sources["chatgpt"]) == 2  # 2 ChatGPT conversations

    def test_merge_with_transform_branches(
        self, initialized_db, claude_export_file, chatgpt_export_file, mock_llm
    ):
        """Merge can combine outputs from parallel transform branches."""
        from synix import Pipeline

        def summarize(record) -> str:
            return f"Summarize: {record.content[:50]}"

        def combine(sources: dict) -> str:
            parts = []
            for step, records in sources.items():
                parts.append(f"{step}: {len(records)} summaries")
            return "\n".join(parts)

        pipeline = Pipeline("branch-merge-test", agent="test", settings=initialized_db)
        pipeline.llm = mock_llm

        # Two parallel branches
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.source("chatgpt", file=str(chatgpt_export_file), format="chatgpt-export")

        pipeline.transform("claude-sum", from_="claude", prompt=summarize)
        pipeline.transform("gpt-sum", from_="chatgpt", prompt=summarize)

        # Merge the summaries
        pipeline.merge("all-summaries", sources=["claude-sum", "gpt-sum"], prompt=combine)

        result = pipeline.run()

        assert result.status == "completed"
        # Check merge output exists
        records = pipeline.get_records("all-summaries")
        assert len(records) == 1

    def test_merge_respects_execution_order(
        self, initialized_db, claude_export_file, chatgpt_export_file, mock_llm
    ):
        """Merge step executes after all its sources."""
        from synix import Pipeline

        execution_order = []

        def track_transform(record) -> str:
            execution_order.append(f"transform:{record.step_name}")
            return "transformed"

        def track_merge(sources: dict) -> str:
            execution_order.append(f"merge:{list(sources.keys())}")
            return "merged"

        pipeline = Pipeline("order-test", agent="test", settings=initialized_db)
        pipeline.llm = mock_llm
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.source("chatgpt", file=str(chatgpt_export_file), format="chatgpt-export")
        pipeline.transform("claude-sum", from_="claude", prompt=track_transform)
        pipeline.transform("gpt-sum", from_="chatgpt", prompt=track_transform)
        pipeline.merge("combined", sources=["claude-sum", "gpt-sum"], prompt=track_merge)

        result = pipeline.run()

        assert result.status == "completed"
        # Merge should be last
        assert "merge:" in execution_order[-1]

    def test_merge_pipeline_caches_output(
        self, initialized_db, claude_export_file, chatgpt_export_file, mock_llm
    ):
        """Merge step output is cached on subsequent runs."""
        from synix import Pipeline

        def combine(sources: dict) -> str:
            return "combined"

        pipeline = Pipeline("merge-cache-test", agent="test", settings=initialized_db)
        pipeline.llm = mock_llm
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.source("chatgpt", file=str(chatgpt_export_file), format="chatgpt-export")
        pipeline.merge("combined", sources=["claude", "chatgpt"], prompt=combine)

        # First run
        result1 = pipeline.run()
        first_calls = len(mock_llm.calls)

        # Second run - merge should be skipped
        result2 = pipeline.run()
        assert result2.stats["skipped"] > 0
        # No additional merge LLM call
        assert len(mock_llm.calls) == first_calls

    def test_merge_output_has_correct_metadata(
        self, initialized_db, claude_export_file, chatgpt_export_file, mock_llm
    ):
        """Merge output record has merge-specific metadata."""
        from synix import Pipeline

        def combine(sources: dict) -> str:
            return "combined"

        pipeline = Pipeline("merge-meta-test", agent="test", settings=initialized_db)
        pipeline.llm = mock_llm
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.source("chatgpt", file=str(chatgpt_export_file), format="chatgpt-export")
        pipeline.merge("combined", sources=["claude", "chatgpt"], prompt=combine)

        result = pipeline.run()
        assert result.status == "completed"

        records = pipeline.get_records("combined")
        assert len(records) == 1

        record = records[0]
        assert set(record.metadata_["meta.merge.source_steps"]) == {"claude", "chatgpt"}
        assert record.metadata_["meta.merge.source_counts"]["claude"] == 3
        assert record.metadata_["meta.merge.source_counts"]["chatgpt"] == 2
        assert record.metadata_["meta.merge.total_inputs"] == 5

    def test_plan_shows_merge_sources(
        self, initialized_db, claude_export_file, chatgpt_export_file, mock_llm
    ):
        """Plan result shows merge step sources."""
        from synix import Pipeline

        def combine(sources: dict) -> str:
            return "combined"

        pipeline = Pipeline("merge-plan-test", agent="test", settings=initialized_db)
        pipeline.llm = mock_llm
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.source("chatgpt", file=str(chatgpt_export_file), format="chatgpt-export")
        pipeline.merge("combined", sources=["claude", "chatgpt"], prompt=combine)

        plan = pipeline.plan()

        # Find merge step in plan
        merge_step = next(s for s in plan.steps if s["name"] == "combined")
        assert merge_step["type"] == "merge"
        assert merge_step["from"] == ["claude", "chatgpt"]

    def test_filter_order_for_merge_includes_all_sources(
        self, initialized_db, claude_export_file, chatgpt_export_file, mock_llm
    ):
        """Running specific merge step includes all its sources."""
        from synix import Pipeline

        def summarize(record) -> str:
            return "summary"

        def combine(sources: dict) -> str:
            return "combined"

        pipeline = Pipeline("filter-test", agent="test", settings=initialized_db)
        pipeline.llm = mock_llm
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.source("chatgpt", file=str(chatgpt_export_file), format="chatgpt-export")
        pipeline.transform("claude-sum", from_="claude", prompt=summarize)
        pipeline.transform("gpt-sum", from_="chatgpt", prompt=summarize)
        pipeline.merge("combined", sources=["claude-sum", "gpt-sum"], prompt=combine)

        # Run only the merge step
        result = pipeline.run(step="combined")

        assert result.status == "completed"
        # Should have processed both sources and transforms
        records = pipeline.get_records("combined")
        assert len(records) == 1
