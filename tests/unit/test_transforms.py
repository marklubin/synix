"""Tests for transform system — base, parse, and LLM transforms."""

from __future__ import annotations

import pytest

from synix import (
    Artifact,
    Pipeline,
    SearchSurface,
    SearchSurfaceUnavailableError,
    Source,
    Transform,
    TransformContext,
)
from synix import (
    SearchSurfaceHandle as PublicSearchSurfaceHandle,
)
from synix.build.artifacts import ArtifactStore
from synix.build.llm_transforms import (
    CoreSynthesis,
    EpisodeSummary,
    MonthlyRollup,
    TopicalRollup,
)
from synix.build.parse_transform import ParseTransform
from synix.build.plan import plan_build
from synix.build.runner import run
from synix.search.indexer import SearchIndex


class TestBaseTransform:
    """Tests for base transform infrastructure."""

    def test_prompt_template_loading(self):
        """Templates load from prompts/ directory without error."""
        transform = EpisodeSummary("test")
        template = transform.load_prompt("episode_summary")
        assert "{transcript}" in template
        assert "episode summary" in template.lower() or "summarizing" in template.lower()

    def test_prompt_id_versioning(self):
        """Same template produces same prompt_id; different template produces different id."""
        transform = EpisodeSummary("test")

        id1 = transform.get_prompt_id("episode_summary")
        id2 = transform.get_prompt_id("episode_summary")
        assert id1 == id2  # deterministic

        id_monthly = transform.get_prompt_id("monthly_rollup")
        assert id1 != id_monthly  # different templates → different ids

    def test_all_templates_loadable(self):
        """All four prompt templates load without error."""
        transform = EpisodeSummary("test")
        for name in ["episode_summary", "monthly_rollup", "topical_rollup", "core_memory"]:
            content = transform.load_prompt(name)
            assert len(content) > 0


class TestTransformInstantiation:
    """Tests for transform direct instantiation."""

    def test_parse_transform_instantiates(self):
        """ParseTransform is directly instantiable."""
        transform = ParseTransform()
        assert transform is not None

    def test_all_transforms_instantiate(self):
        """All expected transforms are directly instantiable."""
        transforms = [
            ParseTransform(),
            EpisodeSummary("test"),
            MonthlyRollup("test"),
            TopicalRollup("test"),
            CoreSynthesis("test"),
        ]
        for t in transforms:
            assert t is not None


class TestParseTransformSourcePath:
    """Tests for source_path metadata on parsed artifacts."""

    def test_source_path_set_on_flat_dir(self, tmp_path):
        """Artifacts from a flat source dir get source_path = filename."""
        src = tmp_path / "sources"
        src.mkdir()
        (src / "alpha.md").write_text("Alpha content\n")
        (src / "beta.md").write_text("Beta content\n")

        transform = ParseTransform()
        artifacts = transform.execute([], {"source_dir": str(src)})

        for art in artifacts:
            assert "source_path" in art.metadata
        paths = {a.metadata["source_path"] for a in artifacts}
        assert "alpha.md" in paths
        assert "beta.md" in paths

    def test_source_path_relative_to_source_dir(self, tmp_path):
        """Artifacts from nested dirs get relative paths."""
        src = tmp_path / "sources"
        sub = src / "team-a"
        sub.mkdir(parents=True)
        (sub / "alice.md").write_text("Alice bio\n")

        transform = ParseTransform()
        artifacts = transform.execute([], {"source_dir": str(src)})

        assert len(artifacts) == 1
        assert artifacts[0].metadata["source_path"] == "team-a/alice.md"

    def test_source_path_preserves_deep_nesting(self, tmp_path):
        """Deep nesting is preserved in source_path."""
        src = tmp_path / "sources"
        deep = src / "dept" / "eng" / "backend"
        deep.mkdir(parents=True)
        (deep / "notes.md").write_text("Backend notes\n")

        transform = ParseTransform()
        artifacts = transform.execute([], {"source_dir": str(src)})

        assert len(artifacts) == 1
        assert artifacts[0].metadata["source_path"] == "dept/eng/backend/notes.md"

    def test_source_path_on_json_exports(self, tmp_path):
        """JSON export artifacts also get source_path."""
        src = tmp_path / "sources"
        src.mkdir()
        # Minimal Claude export
        export = {
            "conversations": [
                {
                    "uuid": "conv-001",
                    "title": "Test",
                    "created_at": "2024-01-01T00:00:00Z",
                    "chat_messages": [
                        {
                            "uuid": "msg-001",
                            "sender": "human",
                            "text": "Hello",
                            "created_at": "2024-01-01T00:00:00Z",
                        },
                        {
                            "uuid": "msg-002",
                            "sender": "assistant",
                            "text": "Hi there",
                            "created_at": "2024-01-01T00:01:00Z",
                        },
                    ],
                }
            ]
        }
        import json

        (src / "export.json").write_text(json.dumps(export))

        transform = ParseTransform()
        artifacts = transform.execute([], {"source_dir": str(src)})

        assert len(artifacts) >= 1
        for art in artifacts:
            assert art.metadata["source_path"] == "export.json"


class TestLegacyCustomTransformCompatibility:
    """Legacy custom transforms keep working with the typed runtime context."""

    def test_transform_context_config_filters_runtime_keys(self):
        """User config excludes injected runtime-only and underscore-prefixed keys."""
        ctx = TransformContext(
            {
                "topics": ["career"],
                "llm_config": {"model": "test"},
                "_logger": object(),
                "_future_runtime_key": "hidden",
                "search_db_path": "/tmp/search.db",
            }
        )

        assert ctx.config == {"topics": ["career"]}

    def test_runner_supports_legacy_config_copy_and_mutation(self, tmp_path):
        """Runner passes a mapping-compatible context to old config-style transforms."""

        class LegacyConfigTransform(Transform):
            def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
                copied = config.copy()
                copied["copied"] = True
                config["_legacy_seen"] = True
                assert copied["prefix"] == "legacy"
                assert config.get("_legacy_seen") is True
                return [
                    Artifact(
                        label=f"{config['prefix']}-{inp.label}",
                        artifact_type="legacy_summary",
                        content=f"{inp.content}\nlegacy={copied['copied']}",
                        input_ids=[inp.artifact_id],
                    )
                    for inp in inputs
                ]

        source_dir = tmp_path / "sources"
        build_dir = tmp_path / "build"
        source_dir.mkdir()
        (source_dir / "alpha.md").write_text("Alpha content\n")

        pipeline = Pipeline("legacy-config-runner")
        pipeline.source_dir = str(source_dir)
        pipeline.build_dir = str(build_dir)
        transcripts = Source("transcripts")
        legacy = LegacyConfigTransform("legacy", depends_on=[transcripts], config={"prefix": "legacy"})
        pipeline.add(transcripts, legacy)

        result = run(pipeline)

        assert result.built == 2
        artifacts = ArtifactStore(build_dir).list_artifacts("legacy")
        assert len(artifacts) == 1
        assert "legacy=True" in artifacts[0].content

    def test_plan_supports_legacy_config_style_split(self, tmp_path):
        """Planner still supports custom split() implementations written against config dicts."""

        class LegacySplitTransform(Transform):
            def split(self, inputs: list[Artifact], config: dict) -> list[tuple[list[Artifact], dict]]:
                preview = config.copy()
                fanout = preview.get("fanout", 1)
                return [([inp], {"_slot": str(i)}) for i, inp in enumerate(inputs[:fanout])]

            def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
                return [
                    Artifact(
                        label=f"slot-{config.get('_slot', '0')}-{inputs[0].label}",
                        artifact_type="legacy_slot",
                        content=inputs[0].content,
                        input_ids=[inputs[0].artifact_id],
                    )
                ]

        source_dir = tmp_path / "sources"
        build_dir = tmp_path / "build"
        source_dir.mkdir()
        (source_dir / "alpha.md").write_text("Alpha content\n")
        (source_dir / "beta.md").write_text("Beta content\n")

        pipeline = Pipeline("legacy-config-plan")
        pipeline.source_dir = str(source_dir)
        pipeline.build_dir = str(build_dir)
        transcripts = Source("transcripts")
        legacy = LegacySplitTransform("legacy", depends_on=[transcripts], config={"fanout": 2})
        pipeline.add(transcripts, legacy)

        plan = plan_build(pipeline)
        legacy_step = next(step for step in plan.steps if step.name == "legacy")

        assert legacy_step.parallel_units == 2
        assert legacy_step.status == "new"


class TestEpisodeSummaryTransform:
    """Tests for episode summary LLM transform."""

    def test_episode_summary_prompt_construction(self, mock_llm, sample_artifacts):
        """Verify prompt includes transcript content."""
        transform = EpisodeSummary("test")
        transcripts = [a for a in sample_artifacts if a.artifact_type == "transcript"]

        results = transform.execute(transcripts[:1], {"llm_config": {}})

        # Verify the LLM was called with the transcript content in the prompt
        assert len(mock_llm) == 1
        prompt_content = mock_llm[0]["messages"][0]["content"]
        assert transcripts[0].content in prompt_content

    def test_episode_summary_output_artifact(self, mock_llm, sample_artifacts):
        """Output has correct type, metadata, prompt_id."""
        transform = EpisodeSummary("test")
        transcripts = [a for a in sample_artifacts if a.artifact_type == "transcript"]

        results = transform.execute(transcripts[:1], {"llm_config": {}})

        assert len(results) == 1
        ep = results[0]
        assert ep.artifact_type == "episode"
        assert ep.label.startswith("ep-")
        assert ep.prompt_id is not None
        assert ep.prompt_id.startswith("episode_summary_v")
        assert ep.artifact_id.startswith("sha256:")
        assert ep.metadata["source_conversation_id"] == transcripts[0].metadata["source_conversation_id"]

    def test_episode_summary_multiple_inputs(self, mock_llm, sample_artifacts):
        """Multiple transcripts produce multiple episodes."""
        transform = EpisodeSummary("test")
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
                label=f"ep-{i}",
                artifact_type="episode",
                content=f"Episode {i} content about technical topics.",
                metadata={"date": date, "title": f"Episode {i}"},
            )
            for i, date in enumerate(
                [
                    "2024-03-10",
                    "2024-03-15",
                    "2024-03-20",
                    "2024-04-05",
                    "2024-04-10",
                    "2024-04-15",
                ]
            )
        ]

        transform = MonthlyRollup("test")
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
                label="ep-1",
                artifact_type="episode",
                content="Content here.",
                metadata={"date": "2024-03-15", "title": "Test"},
            )
        ]
        transform = MonthlyRollup("test")
        results = transform.execute(episodes, {"llm_config": {}})

        assert len(results) == 1
        assert results[0].artifact_type == "rollup"
        assert results[0].label == "monthly-2024-03"


class TestTopicalRollupTransform:
    """Tests for topical rollup LLM transform."""

    def test_transform_context_exposes_public_search_handle(self, tmp_path):
        """Transforms resolve declared search surfaces through a typed public handle."""
        db_path = tmp_path / "episode-search.db"
        index = SearchIndex(db_path)
        index.create()
        index.insert(
            Artifact(
                label="ep-1",
                artifact_type="episode",
                content="Discussion about career planning and AI projects.",
                metadata={"date": "2024-03-15", "title": "Career chat"},
            ),
            "episodes",
            1,
        )
        index.close()

        surface = SearchSurface("episode-search", sources=[])
        transform = TopicalRollup("test-topics", uses=[surface])
        ctx = TransformContext(
            {
                "topics": ["career"],
                "search_surfaces": {
                    "episode-search": {
                        "name": "episode-search",
                        "kind": "search_surface",
                        "db_path": str(db_path),
                        "modes": ["fulltext"],
                        "sources": ["episodes"],
                    }
                },
            }
        )

        handle = transform.get_search_surface(ctx, required=True)

        assert isinstance(handle, PublicSearchSurfaceHandle)
        assert handle is not None
        results = handle.query("career", layers=["episodes"])
        assert [result.label for result in results] == ["ep-1"]

    def test_topical_rollup_produces_per_topic(self, mock_llm):
        """3 topics configured → 3 topic artifacts."""
        episodes = [
            Artifact(
                label="ep-1",
                artifact_type="episode",
                content="Discussion about career and AI projects.",
                metadata={"date": "2024-03-15", "title": "Career chat"},
            ),
            Artifact(
                label="ep-2",
                artifact_type="episode",
                content="Discussion about health and exercise.",
                metadata={"date": "2024-03-16", "title": "Health chat"},
            ),
        ]
        topics = ["career", "health", "ai-projects"]

        transform = TopicalRollup("test")
        results = transform.execute(
            episodes,
            {
                "llm_config": {},
                "topics": topics,
            },
        )

        assert len(results) == 3
        assert len(mock_llm) == 3
        topic_labels = {r.label for r in results}
        assert topic_labels == {"topic-career", "topic-health", "topic-ai-projects"}

    def test_topical_rollup_uses_all_episodes_without_declared_surface(self, mock_llm):
        """Without a declared search surface, all episodes are used for each topic."""
        episodes = [
            Artifact(
                label=f"ep-{i}",
                artifact_type="episode",
                content=f"Content {i}",
                metadata={"date": "2024-03-15", "title": f"Ep {i}"},
            )
            for i in range(3)
        ]

        transform = TopicalRollup("test")
        results = transform.execute(
            episodes,
            {
                "llm_config": {},
                "topics": ["test-topic"],
            },
        )

        assert len(results) == 1
        # All 3 episodes should be in input_ids
        assert len(results[0].input_ids) == 3

    def test_topical_rollup_requires_declared_surface_when_unavailable(self, mock_llm, tmp_path):
        """A declared-but-missing search surface is a hard error, not a silent fallback."""
        episodes = [
            Artifact(
                label="ep-1",
                artifact_type="episode",
                content="Discussion about career and AI projects.",
                metadata={"date": "2024-03-15", "title": "Career chat"},
            )
        ]
        surface = SearchSurface("episode-search", sources=[])
        transform = TopicalRollup("test-topics", uses=[surface], config={"topics": ["career"]})

        with pytest.raises(SearchSurfaceUnavailableError):
            transform.execute(
                episodes,
                {
                    "llm_config": {},
                    "topics": ["career"],
                    "search_surfaces": {
                        "episode-search": {
                            "name": "episode-search",
                            "kind": "search_surface",
                            "db_path": str(tmp_path / "missing.db"),
                            "modes": ["fulltext"],
                            "sources": ["episodes"],
                        }
                    },
                },
            )


class TestCoreSynthesisTransform:
    """Tests for core synthesis LLM transform."""

    def test_core_synthesis_single_output(self, mock_llm):
        """Always produces exactly 1 artifact."""
        rollups = [
            Artifact(
                label=f"monthly-2024-0{i}",
                artifact_type="rollup",
                content=f"Rollup for month {i}.",
                metadata={"month": f"2024-0{i}"},
            )
            for i in range(1, 4)
        ]

        transform = CoreSynthesis("test")
        results = transform.execute(rollups, {"llm_config": {}, "context_budget": 5000})

        assert len(results) == 1
        assert results[0].label == "core-memory"
        assert results[0].artifact_type == "core_memory"
        assert results[0].metadata["context_budget"] == 5000
        assert results[0].metadata["input_count"] == 3

    def test_core_synthesis_includes_prompt_id(self, mock_llm):
        """Core synthesis artifact has a valid prompt_id."""
        rollups = [
            Artifact(
                label="monthly-2024-01",
                artifact_type="rollup",
                content="Rollup content.",
                metadata={"month": "2024-01"},
            )
        ]

        transform = CoreSynthesis("test")
        results = transform.execute(rollups, {"llm_config": {}})

        assert results[0].prompt_id is not None
        assert results[0].prompt_id.startswith("core_memory_v")
