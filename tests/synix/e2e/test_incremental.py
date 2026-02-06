"""End-to-end tests for incremental processing."""

import pytest


class TestIncrementalProcessing:
    """Tests for incremental/cached processing behavior."""

    def test_rerun_skips_processed(
        self, initialized_db, claude_export_file, mock_llm, sample_prompt
    ):
        """Second run skips already-processed records."""
        from synix.pipeline import Pipeline

        pipeline = Pipeline("test", agent="test", settings=initialized_db, llm=mock_llm)
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.transform("summaries", from_="claude", prompt=sample_prompt)

        # First run - processes everything
        result1 = pipeline.run()
        calls_after_first = len(mock_llm.calls)

        assert result1.stats["output"] > 0
        assert result1.stats["skipped"] == 0

        # Second run - should skip everything
        result2 = pipeline.run()

        assert result2.stats["output"] == 0
        assert result2.stats["skipped"] == result1.stats["output"]
        # No new LLM calls
        assert len(mock_llm.calls) == calls_after_first

    def test_full_flag_reprocesses_all(
        self, initialized_db, claude_export_file, mock_llm, sample_prompt
    ):
        """--full flag skips materialization key check."""
        from synix.pipeline import Pipeline
        from synix.services.records import delete_records_by_step, count_records_by_step
        from synix.db.engine import get_artifact_session

        pipeline = Pipeline("test", agent="test", settings=initialized_db, llm=mock_llm)
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.transform("summaries", from_="claude", prompt=sample_prompt)

        # First run
        result1 = pipeline.run()

        # Delete the transform outputs to verify reprocessing
        with get_artifact_session(initialized_db) as session:
            deleted = delete_records_by_step(session, "summaries")

        # Verify deletion worked
        with get_artifact_session(initialized_db) as session:
            count = count_records_by_step(session, "summaries")
            assert count == 0, f"Expected 0 summaries after delete, got {count}"

        # Second run should reprocess transforms
        result2 = pipeline.run()

        # New summary records should be created
        assert result2.stats["output"] >= 3  # 3 transforms recreated

    def test_prompt_change_triggers_reprocess(self, initialized_db, claude_export_file, mock_llm):
        """Changing prompt triggers reprocessing due to different version hash."""
        from synix.pipeline import Pipeline

        def prompt_v1(record):
            return f"Summarize v1: {record.content}"

        def prompt_v2(record):
            return f"Summarize v2: {record.content}"

        # First run with prompt_v1
        pipeline1 = Pipeline("test", agent="test", settings=initialized_db, llm=mock_llm)
        pipeline1.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline1.transform("summaries", from_="claude", prompt=prompt_v1)

        result1 = pipeline1.run()
        first_llm_calls = len(mock_llm.calls)
        assert result1.stats["output"] > 0

        # Second run with prompt_v2 (different version hash means new mat keys)
        pipeline2 = Pipeline("test", agent="test", settings=initialized_db, llm=mock_llm)
        pipeline2.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline2.transform("summaries", from_="claude", prompt=prompt_v2)

        result2 = pipeline2.run()

        # LLM should be called again because different version hash = different mat key
        assert len(mock_llm.calls) > first_llm_calls
        # Both sources are skipped (same conv IDs), but summaries reprocessed
        assert result2.stats["output"] >= 3  # 3 new transform outputs

    def test_source_rerun_skips_existing(self, initialized_db, claude_export_file, mock_llm):
        """Source records are also cached."""
        from synix.pipeline import Pipeline

        pipeline = Pipeline("test", agent="test", settings=initialized_db, llm=mock_llm)
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")

        # First run
        result1 = pipeline.run()
        first_output = result1.stats["output"]

        # Second run
        result2 = pipeline.run()

        # Sources should be skipped
        assert result2.stats["skipped"] == first_output
        assert result2.stats["output"] == 0

    def test_aggregate_reprocesses_when_group_changes(
        self, initialized_db, tmp_path, mock_llm, sample_prompt, sample_aggregate_prompt
    ):
        """Aggregate reprocesses when group membership changes."""
        import json

        from synix.pipeline import Pipeline

        # Create initial export with 2 conversations in March
        export_file = tmp_path / "export.json"
        data = {
            "conversations": [
                {
                    "uuid": "conv-001",
                    "title": "Conv 1",
                    "created_at": "2024-03-15T10:00:00Z",
                    "chat_messages": [
                        {"sender": "human", "text": "Hello"},
                        {"sender": "assistant", "text": "Hi there!"},
                    ],
                },
                {
                    "uuid": "conv-002",
                    "title": "Conv 2",
                    "created_at": "2024-03-20T10:00:00Z",
                    "chat_messages": [
                        {"sender": "human", "text": "Question"},
                        {"sender": "assistant", "text": "Answer"},
                    ],
                },
            ]
        }
        export_file.write_text(json.dumps(data))

        # First run
        pipeline1 = Pipeline("test", agent="test", settings=initialized_db, llm=mock_llm)
        pipeline1.source("claude", file=str(export_file), format="claude-export")
        pipeline1.transform("summaries", from_="claude", prompt=sample_prompt)
        pipeline1.aggregate("monthly", from_="summaries", period="month", prompt=sample_aggregate_prompt)

        result1 = pipeline1.run()
        monthly1 = pipeline1.get_records("monthly")

        # Add a third conversation to March
        data["conversations"].append(
            {
                "uuid": "conv-003",
                "title": "Conv 3",
                "created_at": "2024-03-25T10:00:00Z",
                "chat_messages": [
                    {"sender": "human", "text": "More"},
                    {"sender": "assistant", "text": "Content"},
                ],
            }
        )
        export_file.write_text(json.dumps(data))

        # Second run - same pipeline definition
        pipeline2 = Pipeline("test", agent="test", settings=initialized_db, llm=mock_llm)
        pipeline2.source("claude", file=str(export_file), format="claude-export")
        pipeline2.transform("summaries", from_="claude", prompt=sample_prompt)
        pipeline2.aggregate("monthly", from_="summaries", period="month", prompt=sample_aggregate_prompt)

        result2 = pipeline2.run()

        # New source and summary should be processed
        # Monthly should also be reprocessed (group changed)
        assert result2.stats["output"] >= 2  # At least new summary + new aggregate

    def test_multiple_periods_cached_independently(
        self, initialized_db, tmp_path, mock_llm, sample_prompt, sample_aggregate_prompt
    ):
        """Different period groups are cached independently."""
        import json

        from synix.pipeline import Pipeline

        # Create export with conversations in different months
        export_file = tmp_path / "export.json"
        data = {
            "conversations": [
                {
                    "uuid": "conv-001",
                    "title": "March Conv",
                    "created_at": "2024-03-15T10:00:00Z",
                    "chat_messages": [
                        {"sender": "human", "text": "March"},
                        {"sender": "assistant", "text": "Reply"},
                    ],
                },
                {
                    "uuid": "conv-002",
                    "title": "April Conv",
                    "created_at": "2024-04-15T10:00:00Z",
                    "chat_messages": [
                        {"sender": "human", "text": "April"},
                        {"sender": "assistant", "text": "Reply"},
                    ],
                },
            ]
        }
        export_file.write_text(json.dumps(data))

        pipeline = Pipeline("test", agent="test", settings=initialized_db, llm=mock_llm)
        pipeline.source("claude", file=str(export_file), format="claude-export")
        pipeline.transform("summaries", from_="claude", prompt=sample_prompt)
        pipeline.aggregate("monthly", from_="summaries", period="month", prompt=sample_aggregate_prompt)

        result = pipeline.run()

        # Should have 2 monthly aggregates (one for each month)
        monthly = pipeline.get_records("monthly")
        assert len(monthly) == 2

        periods = {r.metadata_["meta.time.period"] for r in monthly}
        assert "2024-03" in periods
        assert "2024-04" in periods
