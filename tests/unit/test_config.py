"""Tests for pipeline config loading and validation."""

from __future__ import annotations

import textwrap

import pytest

from synix import Pipeline, Source
from synix.build.pipeline import load_pipeline, validate_pipeline
from synix.transforms import CoreSynthesis, EpisodeSummary


class TestPipelineConfig:
    def test_load_pipeline_module(self, tmp_path):
        """Create a temp pipeline.py file, load it, verify Pipeline object."""
        pipeline_file = tmp_path / "test_pipeline.py"
        pipeline_file.write_text(
            textwrap.dedent("""\
            from synix import Pipeline, Source
            from synix.transforms import EpisodeSummary

            pipeline = Pipeline("test")
            transcripts = Source("transcripts")
            episodes = EpisodeSummary("episodes", depends_on=[transcripts])
            pipeline.add(transcripts, episodes)
        """)
        )

        result = load_pipeline(str(pipeline_file))
        assert isinstance(result, Pipeline)
        assert result.name == "test"
        assert len(result.layers) == 2
        assert result.layers[0].name == "transcripts"
        assert result.layers[1].name == "episodes"

    def test_validate_acyclic(self):
        """Valid DAG passes validation."""
        pipeline = Pipeline("test")
        transcripts = Source("transcripts")
        episodes = EpisodeSummary("episodes", depends_on=[transcripts])
        core = CoreSynthesis("core", depends_on=[episodes])
        pipeline.add(transcripts, episodes, core)

        # Should not raise
        validate_pipeline(pipeline)

    def test_validate_cyclic_rejected(self):
        """Circular depends_on raises an error."""
        pipeline = Pipeline("test")
        # Create a cycle: transcripts -> episodes -> core -> transcripts
        # We have to manually wire the cycle since the API won't let us
        # easily create forward references. We'll create them and then
        # manually set depends_on.
        transcripts = Source("transcripts")
        episodes = EpisodeSummary("episodes", depends_on=[transcripts])
        core = CoreSynthesis("core", depends_on=[episodes])
        # Force a cycle by adding core as a dependency of transcripts
        transcripts.depends_on = [core]
        pipeline.add(transcripts, episodes, core)

        with pytest.raises((ValueError, RecursionError)):
            validate_pipeline(pipeline)

    def test_validate_missing_dependency(self):
        """depends_on references nonexistent layer, raises ValueError."""
        pipeline = Pipeline("test")
        transcripts = Source("transcripts")
        # Create an episode that depends on a layer not in the pipeline
        phantom = Source("nonexistent")
        episodes = EpisodeSummary("episodes", depends_on=[phantom])
        pipeline.add(transcripts, episodes)

        with pytest.raises(ValueError, match="does not exist"):
            validate_pipeline(pipeline)

    def test_validate_no_root_rejected(self):
        """Must have at least one Source layer."""
        pipeline = Pipeline("test")
        # Add only a Transform (no Source)
        episodes = EpisodeSummary("episodes")
        pipeline.add(episodes)

        with pytest.raises(ValueError, match="Source"):
            validate_pipeline(pipeline)

    def test_validate_multiple_roots_accepted(self):
        """Multiple Source layers are allowed for multi-source pipelines."""
        pipeline = Pipeline("test")
        chatgpt = Source("chatgpt_transcripts")
        claude = Source("claude_transcripts")
        episodes = EpisodeSummary("episodes", depends_on=[chatgpt, claude])
        pipeline.add(chatgpt, claude, episodes)

        # Should not raise
        validate_pipeline(pipeline)

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
