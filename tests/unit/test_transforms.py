"""Tests for transform system — base, parse, and LLM transforms."""

from __future__ import annotations

from synix import Artifact
from synix.transforms.base import BaseTransform, get_transform, register_transform

# Import to trigger registration
import synix.transforms.parse  # noqa: F401
import synix.transforms.summarize  # noqa: F401


class TestBaseTransform:
    """Tests for base transform infrastructure."""

    def test_prompt_template_loading(self):
        """Templates load from prompts/ directory without error."""
        transform = get_transform("episode_summary")
        template = transform.load_prompt("episode_summary")
        assert "{transcript}" in template
        assert "episode summary" in template.lower() or "summarizing" in template.lower()

    def test_prompt_id_versioning(self):
        """Same template produces same prompt_id; different template produces different id."""
        transform = get_transform("episode_summary")

        id1 = transform.get_prompt_id("episode_summary")
        id2 = transform.get_prompt_id("episode_summary")
        assert id1 == id2  # deterministic

        id_monthly = transform.get_prompt_id("monthly_rollup")
        assert id1 != id_monthly  # different templates → different ids

    def test_all_templates_loadable(self):
        """All four prompt templates load without error."""
        transform = get_transform("episode_summary")
        for name in ["episode_summary", "monthly_rollup", "topical_rollup", "core_memory"]:
            content = transform.load_prompt(name)
            assert len(content) > 0

    def test_get_transform_unknown_raises(self):
        """Requesting unknown transform raises ValueError."""
        import pytest
        with pytest.raises(ValueError, match="Unknown transform"):
            get_transform("nonexistent_transform")


class TestTransformRegistry:
    """Tests for transform registration."""

    def test_parse_transform_registered(self):
        """ParseTransform is available via registry."""
        transform = get_transform("parse")
        assert transform is not None

    def test_all_transforms_registered(self):
        """All expected transforms are registered."""
        for name in ["parse", "episode_summary", "monthly_rollup", "topical_rollup", "core_synthesis"]:
            transform = get_transform(name)
            assert isinstance(transform, BaseTransform)


class TestEpisodeSummaryTransform:
    """Tests for episode summary LLM transform."""

    def test_episode_summary_prompt_construction(self, mock_llm, sample_artifacts):
        """Verify prompt includes transcript content."""
        transform = get_transform("episode_summary")
        transcripts = [a for a in sample_artifacts if a.artifact_type == "transcript"]

        results = transform.execute(transcripts[:1], {"llm_config": {}})

        # Verify the LLM was called with the transcript content in the prompt
        assert len(mock_llm) == 1
        prompt_content = mock_llm[0]["messages"][0]["content"]
        assert transcripts[0].content in prompt_content

    def test_episode_summary_output_artifact(self, mock_llm, sample_artifacts):
        """Output has correct type, metadata, prompt_id."""
        transform = get_transform("episode_summary")
        transcripts = [a for a in sample_artifacts if a.artifact_type == "transcript"]

        results = transform.execute(transcripts[:1], {"llm_config": {}})

        assert len(results) == 1
        ep = results[0]
        assert ep.artifact_type == "episode"
        assert ep.artifact_id.startswith("ep-")
        assert ep.prompt_id is not None
        assert ep.prompt_id.startswith("episode_summary_v")
        assert ep.content_hash.startswith("sha256:")
        assert ep.metadata["source_conversation_id"] == transcripts[0].metadata["source_conversation_id"]

    def test_episode_summary_multiple_inputs(self, mock_llm, sample_artifacts):
        """Multiple transcripts produce multiple episodes."""
        transform = get_transform("episode_summary")
        transcripts = [a for a in sample_artifacts if a.artifact_type == "transcript"]

        results = transform.execute(transcripts, {"llm_config": {}})
        assert len(results) == len(transcripts)
        assert len(mock_llm) == len(transcripts)


class TestMonthlyRollupTransform:
    """Tests for monthly rollup LLM transform."""

    def test_monthly_rollup_groups_by_month(self, mock_llm):
        """6 episodes across 2 months → 2 rollups."""
        episodes = [
            Artifact(
                artifact_id=f"ep-{i}",
                artifact_type="episode",
                content=f"Episode {i} content about technical topics.",
                metadata={"date": date, "title": f"Episode {i}"},
            )
            for i, date in enumerate([
                "2024-03-10", "2024-03-15", "2024-03-20",
                "2024-04-05", "2024-04-10", "2024-04-15",
            ])
        ]

        transform = get_transform("monthly_rollup")
        results = transform.execute(episodes, {"llm_config": {}})

        assert len(results) == 2
        assert len(mock_llm) == 2
        months = {r.metadata["month"] for r in results}
        assert months == {"2024-03", "2024-04"}
        # Check episode counts
        for r in results:
            assert r.metadata["episode_count"] == 3

    def test_monthly_rollup_artifact_type(self, mock_llm):
        """Rollup artifacts have correct type and ID format."""
        episodes = [
            Artifact(
                artifact_id="ep-1",
                artifact_type="episode",
                content="Content here.",
                metadata={"date": "2024-03-15", "title": "Test"},
            )
        ]
        transform = get_transform("monthly_rollup")
        results = transform.execute(episodes, {"llm_config": {}})

        assert len(results) == 1
        assert results[0].artifact_type == "rollup"
        assert results[0].artifact_id == "monthly-2024-03"


class TestTopicalRollupTransform:
    """Tests for topical rollup LLM transform."""

    def test_topical_rollup_produces_per_topic(self, mock_llm):
        """3 topics configured → 3 topic artifacts."""
        episodes = [
            Artifact(
                artifact_id="ep-1",
                artifact_type="episode",
                content="Discussion about career and AI projects.",
                metadata={"date": "2024-03-15", "title": "Career chat"},
            ),
            Artifact(
                artifact_id="ep-2",
                artifact_type="episode",
                content="Discussion about health and exercise.",
                metadata={"date": "2024-03-16", "title": "Health chat"},
            ),
        ]
        topics = ["career", "health", "ai-projects"]

        transform = get_transform("topical_rollup")
        results = transform.execute(episodes, {
            "llm_config": {},
            "topics": topics,
        })

        assert len(results) == 3
        assert len(mock_llm) == 3
        topic_ids = {r.artifact_id for r in results}
        assert topic_ids == {"topic-career", "topic-health", "topic-ai-projects"}

    def test_topical_rollup_uses_all_episodes_without_search(self, mock_llm):
        """Without search_db_path, all episodes used for each topic."""
        episodes = [
            Artifact(
                artifact_id=f"ep-{i}",
                artifact_type="episode",
                content=f"Content {i}",
                metadata={"date": "2024-03-15", "title": f"Ep {i}"},
            )
            for i in range(3)
        ]

        transform = get_transform("topical_rollup")
        results = transform.execute(episodes, {
            "llm_config": {},
            "topics": ["test-topic"],
        })

        assert len(results) == 1
        # All 3 episodes should be in input_hashes
        assert len(results[0].input_hashes) == 3


class TestCoreSynthesisTransform:
    """Tests for core synthesis LLM transform."""

    def test_core_synthesis_single_output(self, mock_llm):
        """Always produces exactly 1 artifact."""
        rollups = [
            Artifact(
                artifact_id=f"monthly-2024-0{i}",
                artifact_type="rollup",
                content=f"Rollup for month {i}.",
                metadata={"month": f"2024-0{i}"},
            )
            for i in range(1, 4)
        ]

        transform = get_transform("core_synthesis")
        results = transform.execute(rollups, {"llm_config": {}, "context_budget": 5000})

        assert len(results) == 1
        assert results[0].artifact_id == "core-memory"
        assert results[0].artifact_type == "core_memory"
        assert results[0].metadata["context_budget"] == 5000
        assert results[0].metadata["input_count"] == 3

    def test_core_synthesis_includes_prompt_id(self, mock_llm):
        """Core synthesis artifact has a valid prompt_id."""
        rollups = [
            Artifact(
                artifact_id="monthly-2024-01",
                artifact_type="rollup",
                content="Rollup content.",
                metadata={"month": "2024-01"},
            )
        ]

        transform = get_transform("core_synthesis")
        results = transform.execute(rollups, {"llm_config": {}})

        assert results[0].prompt_id is not None
        assert results[0].prompt_id.startswith("core_memory_v")
