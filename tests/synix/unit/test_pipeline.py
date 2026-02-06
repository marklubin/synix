"""Unit tests for Pipeline class."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestPipelineDAG:
    """Tests for pipeline DAG building and resolution."""

    def test_source_registration(self, test_settings, claude_export_file):
        """Sources can be registered."""
        from synix.pipeline import Pipeline

        pipeline = Pipeline("test", agent="test", settings=test_settings)
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")

        assert "claude" in pipeline._sources
        assert pipeline._sources["claude"].name == "claude"

    def test_transform_registration(self, test_settings, claude_export_file, sample_prompt):
        """Transform steps can be registered."""
        from synix.pipeline import Pipeline

        pipeline = Pipeline("test", agent="test", settings=test_settings)
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.transform("summaries", from_="claude", prompt=sample_prompt)

        assert "summaries" in pipeline._steps
        assert pipeline._steps["summaries"].from_ == "claude"
        assert pipeline._steps["summaries"].step_type == "transform"

    def test_aggregate_registration(
        self, test_settings, claude_export_file, sample_prompt, sample_aggregate_prompt
    ):
        """Aggregate steps can be registered."""
        from synix.pipeline import Pipeline

        pipeline = Pipeline("test", agent="test", settings=test_settings)
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.transform("summaries", from_="claude", prompt=sample_prompt)
        pipeline.aggregate("monthly", from_="summaries", period="month", prompt=sample_aggregate_prompt)

        assert "monthly" in pipeline._steps
        assert pipeline._steps["monthly"].from_ == "summaries"
        assert pipeline._steps["monthly"].step_type == "aggregate"
        assert pipeline._steps["monthly"].period == "month"

    def test_execution_order_simple(self, test_settings, claude_export_file, sample_prompt):
        """Execution order follows dependencies."""
        from synix.pipeline import Pipeline

        pipeline = Pipeline("test", agent="test", settings=test_settings)
        pipeline.source("source1", file=str(claude_export_file), format="claude-export")
        pipeline.transform("step1", from_="source1", prompt=sample_prompt)

        order = pipeline._resolve_order()

        assert order.index("source1") < order.index("step1")

    def test_execution_order_complex(
        self, test_settings, claude_export_file, sample_prompt, sample_aggregate_prompt
    ):
        """Complex DAG resolves correctly."""
        from synix.pipeline import Pipeline

        pipeline = Pipeline("test", agent="test", settings=test_settings)
        pipeline.source("source1", file=str(claude_export_file), format="claude-export")
        pipeline.transform("transform1", from_="source1", prompt=sample_prompt)
        pipeline.aggregate("aggregate1", from_="transform1", period="month", prompt=sample_aggregate_prompt)

        order = pipeline._resolve_order()

        assert order.index("source1") < order.index("transform1")
        assert order.index("transform1") < order.index("aggregate1")

    def test_cycle_detection(self, test_settings, sample_prompt):
        """Cycles in DAG raise error."""
        from synix.pipeline import Pipeline

        pipeline = Pipeline("test", agent="test", settings=test_settings)

        # Manually create a cycle (this shouldn't happen in normal use)
        from synix.steps.transform import TransformStep

        step1 = TransformStep(name="step1", from_="step2", prompt=sample_prompt)
        step2 = TransformStep(name="step2", from_="step1", prompt=sample_prompt)
        pipeline._steps = {"step1": step1, "step2": step2}

        with pytest.raises(ValueError, match="Cycle detected"):
            pipeline._resolve_order()

    def test_unknown_format_raises_error(self, test_settings, claude_export_file):
        """Unknown source format raises error."""
        from synix.pipeline import Pipeline

        pipeline = Pipeline("test", agent="test", settings=test_settings)

        with pytest.raises(ValueError, match="Unknown source format"):
            pipeline.source("test", file=str(claude_export_file), format="unknown-format")

    def test_filter_order_for_step(
        self, test_settings, claude_export_file, sample_prompt, sample_aggregate_prompt
    ):
        """Filtering order includes only needed steps."""
        from synix.pipeline import Pipeline

        pipeline = Pipeline("test", agent="test", settings=test_settings)
        pipeline.source("source1", file=str(claude_export_file), format="claude-export")
        pipeline.transform("transform1", from_="source1", prompt=sample_prompt)
        pipeline.aggregate("aggregate1", from_="transform1", period="month", prompt=sample_aggregate_prompt)

        # Ask only for transform1
        full_order = pipeline._resolve_order()
        filtered = pipeline._filter_order_for_step(full_order, "transform1")

        assert "source1" in filtered
        assert "transform1" in filtered
        assert "aggregate1" not in filtered
