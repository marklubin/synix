"""Tests for projections as first-class build steps.

Covers: search surface materialization, plan visibility, build-time search
via surfaces, progressive projection progress tracking, topical rollup
surface requirements, and logger events.

Phase 12 changes: projections (SynixSearch, FlatFile, SearchIndex) are no
longer materialized during build — they are recorded as declarations in the
manifest only, materialized via ``synix release``.  Search surfaces ARE still
materialized at build time, but to ``.synix/work/surfaces/`` instead of
``build/surfaces/``.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from synix import Artifact, FlatFile, Pipeline, SearchSurface, SearchSurfaceUnavailableError, Source, SynixSearch
from synix import SearchIndex as SearchIndexLayer
from synix.build.plan import ProjectionPlan, plan_build
from synix.build.refs import synix_dir_for_build_dir
from synix.build.runner import (
    _materialize_layer_search_surfaces,
    run,
)
from synix.build.snapshot_view import SnapshotArtifactCache
from synix.ext import CoreSynthesis, EpisodeSummary, MonthlyRollup, TopicalRollup

FIXTURES_DIR = Path(__file__).parent.parent / "synix" / "fixtures"


def _layers_in_index(db_path: Path) -> set[str]:
    """Return the distinct layer_name values present in search.db."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute("SELECT DISTINCT layer_name FROM search_index").fetchall()
    conn.close()
    return {r[0] for r in rows}


def _artifact_count_in_index(db_path: Path) -> int:
    """Return total row count in the search index."""
    conn = sqlite3.connect(str(db_path))
    (count,) = conn.execute("SELECT count(*) FROM search_index").fetchone()
    conn.close()
    return count


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def source_dir(tmp_path):
    """Source directory with both export fixtures."""
    src = tmp_path / "exports"
    src.mkdir()
    shutil.copy(FIXTURES_DIR / "chatgpt_export.json", src / "chatgpt_export.json")
    shutil.copy(FIXTURES_DIR / "claude_export.json", src / "claude_export.json")
    return src


@pytest.fixture
def build_dir(tmp_path):
    return tmp_path / "build"


@pytest.fixture
def work_dir(tmp_path):
    """The .synix/work directory where surfaces are materialized."""
    synix_dir = tmp_path / ".synix"
    w = synix_dir / "work"
    w.mkdir(parents=True, exist_ok=True)
    return w


def _monthly_pipeline(build_dir: Path) -> Pipeline:
    """Pipeline with monthly rollups."""
    p = Pipeline("monthly-pipeline")
    p.build_dir = str(build_dir)
    p.llm_config = {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}

    transcripts = Source("transcripts")
    episodes = EpisodeSummary("episodes", depends_on=[transcripts])
    monthly = MonthlyRollup("monthly", depends_on=[episodes])
    core = CoreSynthesis("core", depends_on=[monthly], context_budget=10000)

    p.add(transcripts, episodes, monthly, core)
    p.add(SearchIndexLayer("memory-index", sources=[episodes, monthly, core], search=["fulltext"]))
    p.add(FlatFile("context-doc", sources=[core], output_path=str(build_dir / "context.md")))
    return p


def _topical_pipeline(build_dir: Path) -> Pipeline:
    """Pipeline with topical rollups instead of monthly."""
    p = Pipeline("topical-pipeline")
    p.build_dir = str(build_dir)
    p.llm_config = {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}

    transcripts = Source("transcripts")
    episodes = EpisodeSummary("episodes", depends_on=[transcripts])
    topics = TopicalRollup("topics", depends_on=[episodes], config={"topics": ["programming", "machine-learning"]})
    core = CoreSynthesis("core", depends_on=[topics], context_budget=10000)

    p.add(transcripts, episodes, topics, core)
    p.add(SearchIndexLayer("memory-index", sources=[episodes, topics, core], search=["fulltext"]))
    p.add(FlatFile("context-doc", sources=[core], output_path=str(build_dir / "context.md")))
    return p


def _surface_topical_pipeline(build_dir: Path) -> Pipeline:
    """Pipeline with a named search surface used by topical rollups."""
    p = Pipeline("surface-topical-pipeline")
    p.build_dir = str(build_dir)
    p.llm_config = {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}

    transcripts = Source("transcripts")
    episodes = EpisodeSummary("episodes", depends_on=[transcripts])
    episode_search = SearchSurface("episode-search", sources=[episodes], modes=["fulltext"])
    topics = TopicalRollup(
        "topics",
        depends_on=[episodes],
        uses=[episode_search],
        config={"topics": ["programming", "machine-learning"]},
    )
    core = CoreSynthesis("core", depends_on=[topics], context_budget=10000)

    p.add(transcripts, episodes, episode_search, topics, core)
    p.add(FlatFile("context-doc", sources=[core], output_path=str(build_dir / "context.md")))
    return p


@pytest.fixture
def pipeline_obj(build_dir):
    return _monthly_pipeline(build_dir)


# ---------------------------------------------------------------------------
# 1. Search Surface Materialization
#
# These tests verify that _materialize_layer_search_surfaces correctly
# materializes search surfaces to .synix/work/surfaces/ when their full
# source set is available.
# ---------------------------------------------------------------------------


class TestSearchSurfaceMaterialization:
    """Verify search surfaces materialize to work_dir once all sources are ready."""

    @pytest.fixture
    def episode_artifacts(self):
        """Fake episode artifacts with the metadata the index needs."""
        return [
            Artifact(
                label="ep-conv001",
                artifact_type="episode",
                content="Discussion about Python programming and web development.",
                metadata={"layer_name": "episodes", "layer_level": 1, "date": "2024-03", "title": "Python chat"},
            ),
            Artifact(
                label="ep-conv002",
                artifact_type="episode",
                content="Machine learning model training and evaluation.",
                metadata={"layer_name": "episodes", "layer_level": 1, "date": "2024-03", "title": "ML chat"},
            ),
        ]

    @pytest.fixture
    def monthly_artifacts(self):
        return [
            Artifact(
                label="monthly-2024-03",
                artifact_type="rollup",
                content="March themes: programming and ML.",
                metadata={"layer_name": "monthly", "layer_level": 2, "month": "2024-03"},
            ),
        ]

    def test_search_surface_waits_for_all_source_layers(
        self,
        work_dir,
        episode_artifacts,
        monthly_artifacts,
    ):
        """Search surfaces materialize only once their full source set is available."""
        episodes = Source("episodes")
        monthly = Source("monthly")
        surface = SearchSurface("episode-monthly-search", sources=[episodes, monthly], modes=["fulltext"])

        pipeline = Pipeline("surface-pipeline")
        pipeline.build_dir = str(work_dir.parent.parent / "build")
        pipeline.add(episodes, monthly, surface)

        layer_artifacts: dict[str, list[Artifact]] = {"episodes": episode_artifacts}
        db_path = work_dir / "surfaces" / "episode-monthly-search.db"

        _materialize_layer_search_surfaces(
            pipeline,
            "episodes",
            layer_artifacts,
            work_dir,
        )
        assert not db_path.exists(), "surface DB should not exist until all source layers are available"

        layer_artifacts["monthly"] = monthly_artifacts
        _materialize_layer_search_surfaces(
            pipeline,
            "monthly",
            layer_artifacts,
            work_dir,
        )
        assert db_path.exists(), "surface DB should exist once all source layers are available"
        assert _layers_in_index(db_path) == {"episodes", "monthly"}

    def test_single_source_surface_materializes_immediately(
        self,
        work_dir,
        episode_artifacts,
    ):
        """A surface with a single source materializes as soon as that source completes."""
        episodes = Source("episodes")
        surface = SearchSurface("episode-search", sources=[episodes], modes=["fulltext"])

        pipeline = Pipeline("single-surface-pipeline")
        pipeline.build_dir = str(work_dir.parent.parent / "build")
        pipeline.add(episodes, surface)

        layer_artifacts: dict[str, list[Artifact]] = {"episodes": episode_artifacts}
        db_path = work_dir / "surfaces" / "episode-search.db"

        _materialize_layer_search_surfaces(
            pipeline,
            "episodes",
            layer_artifacts,
            work_dir,
        )
        assert db_path.exists(), "surface DB should exist after its single source completes"
        assert _layers_in_index(db_path) == {"episodes"}


# ---------------------------------------------------------------------------
# 2. End-to-end progressive materialization (full pipeline)
# ---------------------------------------------------------------------------


class TestProgressiveMaterialization:
    def test_topical_pipeline_with_cached_episodes(self, source_dir, build_dir, mock_llm):
        """Build monthly then topical -- episodes cached, topical succeeds (no 'no such table')."""
        monthly = _monthly_pipeline(build_dir)
        result1 = run(monthly, source_dir=str(source_dir))
        assert result1.built > 0

        topical = _topical_pipeline(build_dir)
        result2 = run(topical, source_dir=str(source_dir))

        # Episodes: all cached
        e_stats = next(s for s in result2.layer_stats if s.name == "episodes")
        assert e_stats.cached > 0
        assert e_stats.built == 0

        # Topics: all built (new layer)
        topic_stats = next(s for s in result2.layer_stats if s.name == "topics")
        assert topic_stats.built > 0

    def test_topical_transform_queries_surface_at_build_time(
        self,
        source_dir,
        build_dir,
        mock_llm,
    ):
        """Topical rollup queries the episode search surface built at build time.

        This is the critical dependency: episodes -> [search surface] -> topics.
        The surface is materialized to .synix/work/surfaces/.
        """
        pipeline = _surface_topical_pipeline(build_dir)
        result = run(pipeline, source_dir=str(source_dir))

        # Topics should have been built (not errored due to missing surface)
        topic_stats = next(s for s in result.layer_stats if s.name == "topics")
        assert topic_stats.built > 0

        # The surface DB should exist in the work directory
        synix_dir = synix_dir_for_build_dir(build_dir)
        surface_db = synix_dir / "work" / "surfaces" / "episode-search.db"
        assert surface_db.exists(), "Surface DB should be materialized in .synix/work/surfaces/"
        assert "episodes" in _layers_in_index(surface_db)


class TestProgressProjectionFinish:
    """Regression test: progressive projection_finish must update the running entry, not a stale done one."""

    def test_repeated_projection_finish_updates_running_entry(self):
        """When the same projection is started multiple times (progressive search_index),
        projection_finish must mark the currently-running entry as done, not re-mark
        a previously-completed one."""
        from synix.cli.progress import BuildProgress

        progress = BuildProgress()

        # Simulate progressive materialization: same projection started twice
        # (once after layer "bios", once after layer "work_styles")
        progress.layer_finish("bios", built=3, cached=0)
        progress.projection_start("search", triggered_by="bios")
        progress.projection_finish("search", triggered_by="bios")

        progress.layer_finish("work_styles", built=3, cached=0)
        progress.projection_start("search", triggered_by="work_styles")
        progress.projection_finish("search", triggered_by="work_styles")

        # Both entries should be "done", not stuck at "running"
        for ps in progress._projection_states:
            assert ps["status"] == "done", f"Projection entry stuck at '{ps['status']}' (expected 'done')"

        # Verify entries filed under correct layers
        assert len(progress._layer_projections["bios"]) == 1
        assert progress._layer_projections["bios"][0]["status"] == "done"
        assert len(progress._layer_projections["work_styles"]) == 1
        assert progress._layer_projections["work_styles"][0]["status"] == "done"


# ---------------------------------------------------------------------------
# 3. Projection Plan
# ---------------------------------------------------------------------------


class TestProjectionPlan:
    def test_plan_includes_projections_as_declared(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """plan_build() on a fresh pipeline includes projections with status='declared'."""
        plan = plan_build(pipeline_obj, source_dir=str(source_dir))

        assert len(plan.projections) == 2
        for pp in plan.projections:
            assert pp.status == "declared"
            assert "manifest declaration" in pp.reason
            assert isinstance(pp, ProjectionPlan)

    def test_plan_projections_remain_declared_after_build(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """After running, plan_build() still reports projections as 'declared' (not cached)."""
        run(pipeline_obj, source_dir=str(source_dir))
        plan = plan_build(pipeline_obj, source_dir=str(source_dir))

        for pp in plan.projections:
            assert pp.status == "declared", f"Projection {pp.name} was {pp.status}, expected declared"

    def test_plan_projections_in_json_output(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """plan.to_dict() has a 'projections' key with correct structure."""
        plan = plan_build(pipeline_obj, source_dir=str(source_dir))
        d = plan.to_dict()

        assert "projections" in d
        assert len(d["projections"]) == 2
        for proj_dict in d["projections"]:
            assert "name" in proj_dict
            assert "projection_type" in proj_dict
            assert "source_layers" in proj_dict
            assert "status" in proj_dict
            assert "artifact_count" in proj_dict
            assert "reason" in proj_dict

    def test_plan_includes_search_surfaces(self, source_dir, build_dir, mock_llm):
        """plan_build() exposes build-time search surfaces separately."""
        plan = plan_build(_surface_topical_pipeline(build_dir), source_dir=str(source_dir))

        assert len(plan.surfaces) == 1
        assert plan.surfaces[0].name == "episode-search"
        assert plan.surfaces[0].projection_type == "search_surface"
        assert plan.surfaces[0].status == "rebuild"

    def test_plan_search_surfaces_status_rebuild(self, source_dir, build_dir, mock_llm):
        """After running, plan_build() still reports search surfaces as 'rebuild'."""
        pipeline = _surface_topical_pipeline(build_dir)
        run(pipeline, source_dir=str(source_dir))
        plan = plan_build(pipeline, source_dir=str(source_dir))

        assert len(plan.surfaces) == 1
        assert plan.surfaces[0].status == "rebuild"

    def test_plan_includes_synix_search_projection_type(self, source_dir, build_dir):
        """plan_build() exposes SynixSearch as the canonical search projection type."""
        pipeline = Pipeline("synix-search-plan")
        pipeline.build_dir = str(build_dir)

        transcripts = Source("transcripts")
        episodes = EpisodeSummary("episodes", depends_on=[transcripts])
        surface = SearchSurface("memory-search", sources=[episodes], modes=["fulltext"])
        search_output = SynixSearch("search", surface=surface)

        pipeline.add(transcripts, episodes, surface, search_output)

        plan = plan_build(pipeline, source_dir=str(source_dir))
        assert len(plan.projections) == 1
        assert plan.projections[0].projection_type == "synix_search"
        assert plan.projections[0].status == "declared"

    def test_plan_rejects_surface_that_depends_on_future_layers(self, source_dir, build_dir, mock_llm):
        """A transform cannot use a search surface whose sources are not all upstream."""
        pipeline = Pipeline("invalid-surface-order")
        pipeline.build_dir = str(build_dir)
        pipeline.llm_config = {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}

        transcripts = Source("transcripts")
        episodes = EpisodeSummary("episodes", depends_on=[transcripts])
        monthly = MonthlyRollup("monthly", depends_on=[episodes])
        future_surface = SearchSurface("future-surface", sources=[episodes, monthly], modes=["fulltext"])
        topics = TopicalRollup(
            "topics",
            depends_on=[episodes],
            uses=[future_surface],
            config={"topics": ["programming"]},
        )

        pipeline.add(transcripts, episodes, monthly, future_surface, topics)

        with pytest.raises(ValueError, match="before all of its source layers are built"):
            plan_build(pipeline, source_dir=str(source_dir))
        with pytest.raises(ValueError, match="before all of its source layers are built"):
            run(pipeline, source_dir=str(source_dir))

    def test_plan_accepts_equivalent_surface_declared_as_separate_object(self, source_dir, build_dir, mock_llm):
        """Validation resolves declared surfaces against the pipeline by name, not object identity."""
        pipeline = Pipeline("equivalent-surface")
        pipeline.build_dir = str(build_dir)
        pipeline.llm_config = {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}

        transcripts = Source("transcripts")
        episodes = EpisodeSummary("episodes", depends_on=[transcripts])
        registered_surface = SearchSurface("episode-search", sources=[episodes], modes=["fulltext"])
        equivalent_surface = SearchSurface("episode-search", sources=[episodes], modes=["fulltext"])
        topics = TopicalRollup(
            "topics",
            depends_on=[episodes],
            uses=[equivalent_surface],
            config={"topics": ["programming"]},
        )

        pipeline.add(transcripts, episodes, registered_surface, topics)

        plan = plan_build(pipeline, source_dir=str(source_dir))
        assert next(step for step in plan.steps if step.name == "topics").status == "new"

    def test_plan_rejects_mismatched_surface_declaration(self, source_dir, build_dir, mock_llm):
        """A same-name surface with different semantics is rejected during validation."""
        pipeline = Pipeline("mismatched-surface")
        pipeline.build_dir = str(build_dir)
        pipeline.llm_config = {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}

        transcripts = Source("transcripts")
        episodes = EpisodeSummary("episodes", depends_on=[transcripts])
        registered_surface = SearchSurface("episode-search", sources=[episodes], modes=["fulltext"])
        mismatched_surface = SearchSurface("episode-search", sources=[episodes], modes=["semantic"])
        topics = TopicalRollup(
            "topics",
            depends_on=[episodes],
            uses=[mismatched_surface],
            config={"topics": ["programming"]},
        )

        pipeline.add(transcripts, episodes, registered_surface, topics)

        with pytest.raises(ValueError, match="does not match the surface added to the pipeline"):
            plan_build(pipeline, source_dir=str(source_dir))

    def test_search_index_projection_does_not_satisfy_surface_use(self, source_dir, build_dir, mock_llm):
        """A SearchIndex projection with the same name is not a substitute for SearchSurface."""
        pipeline = Pipeline("projection-is-not-surface")
        pipeline.build_dir = str(build_dir)
        pipeline.llm_config = {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}

        transcripts = Source("transcripts")
        episodes = EpisodeSummary("episodes", depends_on=[transcripts])
        declared_surface = SearchSurface("episode-search", sources=[episodes], modes=["fulltext"])
        topics = TopicalRollup(
            "topics",
            depends_on=[episodes],
            uses=[declared_surface],
            config={"topics": ["programming"]},
        )

        pipeline.add(transcripts, episodes, topics)
        pipeline.add(SearchIndexLayer("episode-search", sources=[episodes], search=["fulltext"]))

        with pytest.raises(ValueError, match="belongs to a projection"):
            plan_build(pipeline, source_dir=str(source_dir))
        with pytest.raises(ValueError, match="belongs to a projection"):
            run(pipeline, source_dir=str(source_dir))


# ---------------------------------------------------------------------------
# 4. Topical Rollup Surface Requirements
# ---------------------------------------------------------------------------


class TestTopicalRollupSurfaceRequirements:
    def test_topical_rollup_missing_declared_surface(self, source_dir, build_dir, mock_llm):
        """TopicalRollup fails when a declared search surface path is missing."""
        # First, run parse + episodes to get episode artifacts
        monthly = _monthly_pipeline(build_dir)
        run(monthly, source_dir=str(source_dir))

        synix_dir = synix_dir_for_build_dir(build_dir)
        store = SnapshotArtifactCache(synix_dir)
        episodes = store.list_artifacts("episodes")
        assert len(episodes) > 0

        surface = SearchSurface("episode-search", sources=[])
        transform = TopicalRollup("test-topics", uses=[surface])
        config = {
            "llm_config": {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024},
            "topics": ["programming"],
            "search_surfaces": {
                "episode-search": {
                    "name": "episode-search",
                    "kind": "search_surface",
                    "db_path": str(build_dir / "nonexistent.db"),
                    "modes": ["fulltext"],
                    "sources": ["episodes"],
                }
            },
        }
        with pytest.raises(SearchSurfaceUnavailableError):
            transform.execute(episodes, config)

    def test_topical_rollup_invalid_declared_surface(self, source_dir, build_dir, mock_llm):
        """TopicalRollup fails when the declared search surface lacks an FTS table."""
        # Run a build to get episodes
        monthly = _monthly_pipeline(build_dir)
        run(monthly, source_dir=str(source_dir))

        synix_dir = synix_dir_for_build_dir(build_dir)
        store = SnapshotArtifactCache(synix_dir)
        episodes = store.list_artifacts("episodes")
        assert len(episodes) > 0

        # Create an empty SQLite file (no search_index table)
        empty_db = build_dir / "empty_search.db"
        empty_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(empty_db))
        conn.execute("CREATE TABLE dummy (id INTEGER)")
        conn.commit()
        conn.close()

        surface = SearchSurface("episode-search", sources=[])
        transform = TopicalRollup("test-topics", uses=[surface])
        config = {
            "llm_config": {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024},
            "topics": ["programming"],
            "search_surfaces": {
                "episode-search": {
                    "name": "episode-search",
                    "kind": "search_surface",
                    "db_path": str(empty_db),
                    "modes": ["fulltext"],
                    "sources": ["episodes"],
                }
            },
        }
        with pytest.raises(SearchSurfaceUnavailableError):
            transform.execute(episodes, config)


# ---------------------------------------------------------------------------
# 5. Projection Logger Events
# ---------------------------------------------------------------------------


class TestProjectionLoggerEvents:
    def test_projection_events_in_log_file(self, source_dir, build_dir, mock_llm):
        """JSONL log file contains projection_start and projection_finish events.

        Uses a pipeline with a SearchSurface since only surfaces are materialized
        at build time (projections like SearchIndex/FlatFile are declaration-only).
        """
        pipeline = _surface_topical_pipeline(build_dir)
        run(pipeline, source_dir=str(source_dir))

        # Logs are written to .synix/logs/ (not build/logs/)
        synix_dir = synix_dir_for_build_dir(build_dir)
        logs_dir = synix_dir / "logs"
        assert logs_dir.exists()

        log_files = list(logs_dir.glob("*.jsonl"))
        assert len(log_files) > 0

        events = []
        for log_file in log_files:
            for line in log_file.read_text().splitlines():
                if line.strip():
                    events.append(json.loads(line))

        event_types = [e["event"] for e in events]
        assert "projection_start" in event_types, f"No projection_start in events: {event_types}"
        assert "projection_finish" in event_types, f"No projection_finish in events: {event_types}"

        # Verify projection names are present (the surface name)
        proj_start_events = [e for e in events if e["event"] == "projection_start"]
        proj_names = {e["projection"] for e in proj_start_events}
        assert "episode-search" in proj_names


# ---------------------------------------------------------------------------
# 6. SynixSearch defensive behavior
# ---------------------------------------------------------------------------


class TestSynixSearchDefensive:
    def test_synix_search_tolerates_surface_without_embedding_config(self):
        """SynixSearch defensively handles surfaces whose embedding_config is None."""
        transcripts = Source("transcripts")
        surface = SearchSurface("memory-search", sources=[transcripts], modes=["fulltext"])
        surface.embedding_config = None

        search_output = SynixSearch("search", surface=surface)
        assert search_output.embedding_config == {}
