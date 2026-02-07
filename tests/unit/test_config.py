"""Tests for pipeline config loading and validation."""

from __future__ import annotations

import textwrap

import pytest

from synix import Layer, Pipeline
from synix.pipeline.config import load_pipeline, validate_pipeline


class TestPipelineConfig:
    def test_load_pipeline_module(self, tmp_path):
        """Create a temp pipeline.py file, load it, verify Pipeline object."""
        pipeline_file = tmp_path / "test_pipeline.py"
        pipeline_file.write_text(textwrap.dedent("""\
            from synix import Pipeline, Layer

            pipeline = Pipeline("test")
            pipeline.add_layer(Layer(name="transcripts", level=0, transform="parse"))
            pipeline.add_layer(Layer(
                name="episodes", level=1, depends_on=["transcripts"],
                transform="episode_summary", grouping="by_conversation",
            ))
        """))

        result = load_pipeline(str(pipeline_file))
        assert isinstance(result, Pipeline)
        assert result.name == "test"
        assert len(result.layers) == 2
        assert result.layers[0].name == "transcripts"
        assert result.layers[1].name == "episodes"

    def test_validate_acyclic(self):
        """Valid DAG passes validation."""
        pipeline = Pipeline("test")
        pipeline.add_layer(Layer(name="transcripts", level=0, transform="parse"))
        pipeline.add_layer(Layer(name="episodes", level=1, depends_on=["transcripts"], transform="summarize"))
        pipeline.add_layer(Layer(name="core", level=2, depends_on=["episodes"], transform="synthesize"))

        # Should not raise
        validate_pipeline(pipeline)

    def test_validate_cyclic_rejected(self):
        """Circular depends_on raises ValueError."""
        pipeline = Pipeline("test")
        pipeline.add_layer(Layer(name="transcripts", level=0, depends_on=["core"], transform="parse"))
        pipeline.add_layer(Layer(name="episodes", level=1, depends_on=["transcripts"], transform="summarize"))
        pipeline.add_layer(Layer(name="core", level=2, depends_on=["episodes"], transform="synthesize"))

        with pytest.raises(ValueError, match="circular"):

            validate_pipeline(pipeline)

    def test_validate_missing_dependency(self):
        """depends_on references nonexistent layer, raises ValueError."""
        pipeline = Pipeline("test")
        pipeline.add_layer(Layer(name="transcripts", level=0, transform="parse"))
        pipeline.add_layer(Layer(name="episodes", level=1, depends_on=["nonexistent"], transform="summarize"))

        with pytest.raises(ValueError, match="does not exist"):
            validate_pipeline(pipeline)

    def test_validate_single_root(self):
        """Must have exactly one level-0 layer."""
        # No level-0 layer
        pipeline = Pipeline("test")
        pipeline.add_layer(Layer(name="episodes", level=1, transform="summarize"))

        with pytest.raises(ValueError, match="level-0"):
            validate_pipeline(pipeline)

        # Two level-0 layers
        pipeline2 = Pipeline("test2")
        pipeline2.add_layer(Layer(name="transcripts1", level=0, transform="parse"))
        pipeline2.add_layer(Layer(name="transcripts2", level=0, transform="parse"))

        with pytest.raises(ValueError, match="level-0"):
            validate_pipeline(pipeline2)

    def test_load_missing_file(self):
        """Loading a nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_pipeline("/nonexistent/pipeline.py")

    def test_load_module_without_pipeline_var(self, tmp_path):
        """Module without 'pipeline' variable raises ValueError."""
        bad_file = tmp_path / "bad_pipeline.py"
        bad_file.write_text("x = 42\n")

        with pytest.raises(ValueError, match="must define"):
            load_pipeline(str(bad_file))
