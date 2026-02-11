"""Tests for projections as first-class build steps.

Covers: layer→projection dependency chain, progressive materialization,
projection caching, plan visibility, build stats, defensive fallback,
and logger events.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from synix import Artifact, Layer, Pipeline, Projection
from synix.build.artifacts import ArtifactStore
from synix.build.plan import ProjectionPlan, plan_build
from synix.build.runner import (
    _materialize_layer_projections,
    run,
)
from synix.search.indexer import SearchIndex

FIXTURES_DIR = Path(__file__).parent.parent / "synix" / "fixtures"


def _layers_in_index(db_path: Path) -> set[str]:
    """Return the distinct layer_name values present in search.db."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT DISTINCT layer_name FROM search_index"
    ).fetchall()
    conn.close()
    return {r[0] for r in rows}


def _artifact_count_in_index(db_path: Path) -> int:
    """Return total row count in the search index."""
    conn = sqlite3.connect(str(db_path))
    (count,) = conn.execute(
        "SELECT count(*) FROM search_index"
    ).fetchone()
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


def _monthly_pipeline(build_dir: Path) -> Pipeline:
    """Pipeline with monthly rollups."""
    p = Pipeline("monthly-pipeline")
    p.build_dir = str(build_dir)
    p.llm_config = {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}

    p.add_layer(Layer(name="transcripts", level=0, transform="parse"))
    p.add_layer(Layer(name="episodes", level=1, depends_on=["transcripts"],
                      transform="episode_summary", grouping="by_conversation"))
    p.add_layer(Layer(name="monthly", level=2, depends_on=["episodes"],
                      transform="monthly_rollup", grouping="by_month"))
    p.add_layer(Layer(name="core", level=3, depends_on=["monthly"],
                      transform="core_synthesis", grouping="single", context_budget=10000))

    p.add_projection(Projection(
        name="memory-index",
        projection_type="search_index",
        sources=[
            {"layer": "episodes", "search": ["fulltext"]},
            {"layer": "monthly", "search": ["fulltext"]},
            {"layer": "core", "search": ["fulltext"]},
        ],
    ))
    p.add_projection(Projection(
        name="context-doc",
        projection_type="flat_file",
        sources=[{"layer": "core"}],
        config={"output_path": str(build_dir / "context.md")},
    ))
    return p


def _topical_pipeline(build_dir: Path) -> Pipeline:
    """Pipeline with topical rollups instead of monthly."""
    p = Pipeline("topical-pipeline")
    p.build_dir = str(build_dir)
    p.llm_config = {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}

    p.add_layer(Layer(name="transcripts", level=0, transform="parse"))
    p.add_layer(Layer(name="episodes", level=1, depends_on=["transcripts"],
                      transform="episode_summary", grouping="by_conversation"))
    p.add_layer(Layer(
        name="topics", level=2, depends_on=["episodes"],
        transform="topical_rollup", grouping="by_topic",
        config={"topics": ["programming", "machine-learning"]},
    ))
    p.add_layer(Layer(name="core", level=3, depends_on=["topics"],
                      transform="core_synthesis", grouping="single", context_budget=10000))

    p.add_projection(Projection(
        name="memory-index",
        projection_type="search_index",
        sources=[
            {"layer": "episodes", "search": ["fulltext"]},
            {"layer": "topics", "search": ["fulltext"]},
            {"layer": "core", "search": ["fulltext"]},
        ],
    ))
    p.add_projection(Projection(
        name="context-doc",
        projection_type="flat_file",
        sources=[{"layer": "core"}],
        config={"output_path": str(build_dir / "context.md")},
    ))
    return p


@pytest.fixture
def pipeline_obj(build_dir):
    return _monthly_pipeline(build_dir)


# ---------------------------------------------------------------------------
# 1. Layer → Projection Dependency Chain
#
# These tests call _materialize_layer_projections directly with incrementally
# populated layer_artifacts, so you can see exactly which layer triggers
# which projection work and what the index contains at each step.
# ---------------------------------------------------------------------------

class TestLayerProjectionChain:
    """Verify the progressive relationship: each layer completion triggers
    its projection step, and downstream layers can query the result."""

    @pytest.fixture
    def episode_artifacts(self):
        """Fake episode artifacts with the metadata the index needs."""
        return [
            Artifact(
                artifact_id="ep-conv001", artifact_type="episode",
                content="Discussion about Python programming and web development.",
                metadata={"layer_name": "episodes", "layer_level": 1,
                          "date": "2024-03", "title": "Python chat"},
            ),
            Artifact(
                artifact_id="ep-conv002", artifact_type="episode",
                content="Machine learning model training and evaluation.",
                metadata={"layer_name": "episodes", "layer_level": 1,
                          "date": "2024-03", "title": "ML chat"},
            ),
        ]

    @pytest.fixture
    def monthly_artifacts(self):
        return [
            Artifact(
                artifact_id="monthly-2024-03", artifact_type="rollup",
                content="March themes: programming and ML.",
                metadata={"layer_name": "monthly", "layer_level": 2, "month": "2024-03"},
            ),
        ]

    @pytest.fixture
    def core_artifacts(self):
        return [
            Artifact(
                artifact_id="core-memory", artifact_type="core_memory",
                content="## Identity\nSoftware engineer focused on AI.",
                metadata={"layer_name": "core", "layer_level": 3},
            ),
        ]

    def test_after_episodes_index_has_only_episodes(
        self, build_dir, pipeline_obj, episode_artifacts,
    ):
        """After episodes complete, search index contains episode data only."""
        build_dir.mkdir(parents=True, exist_ok=True)
        store = ArtifactStore(build_dir)
        layer_artifacts = {"episodes": episode_artifacts}

        _materialize_layer_projections(
            pipeline_obj, "episodes", layer_artifacts, store, build_dir,
        )

        db_path = build_dir / "search.db"
        assert db_path.exists(), "search.db should exist after episodes projection"
        assert _layers_in_index(db_path) == {"episodes"}

    def test_after_monthly_index_has_episodes_and_monthly(
        self, build_dir, pipeline_obj, episode_artifacts, monthly_artifacts,
    ):
        """After monthly completes, search index contains both episodes and monthly."""
        build_dir.mkdir(parents=True, exist_ok=True)
        store = ArtifactStore(build_dir)
        layer_artifacts = {"episodes": episode_artifacts}

        # Step 1: episodes complete → first projection
        _materialize_layer_projections(
            pipeline_obj, "episodes", layer_artifacts, store, build_dir,
        )
        db_path = build_dir / "search.db"
        assert _layers_in_index(db_path) == {"episodes"}

        # Step 2: monthly complete → index rebuilt with both
        layer_artifacts["monthly"] = monthly_artifacts
        _materialize_layer_projections(
            pipeline_obj, "monthly", layer_artifacts, store, build_dir,
        )
        assert _layers_in_index(db_path) == {"episodes", "monthly"}

    def test_after_core_index_has_all_three_layers(
        self, build_dir, pipeline_obj,
        episode_artifacts, monthly_artifacts, core_artifacts,
    ):
        """After core completes, search index contains episodes + monthly + core."""
        build_dir.mkdir(parents=True, exist_ok=True)
        store = ArtifactStore(build_dir)
        layer_artifacts: dict[str, list[Artifact]] = {}

        db_path = build_dir / "search.db"

        # Simulate the runner's per-layer projection calls
        layer_artifacts["episodes"] = episode_artifacts
        _materialize_layer_projections(
            pipeline_obj, "episodes", layer_artifacts, store, build_dir,
        )
        assert _layers_in_index(db_path) == {"episodes"}

        layer_artifacts["monthly"] = monthly_artifacts
        _materialize_layer_projections(
            pipeline_obj, "monthly", layer_artifacts, store, build_dir,
        )
        assert _layers_in_index(db_path) == {"episodes", "monthly"}

        layer_artifacts["core"] = core_artifacts
        _materialize_layer_projections(
            pipeline_obj, "core", layer_artifacts, store, build_dir,
        )
        assert _layers_in_index(db_path) == {"episodes", "monthly", "core"}

    def test_flat_file_waits_for_all_sources(
        self, build_dir, pipeline_obj, episode_artifacts, monthly_artifacts,
    ):
        """context-doc (flat_file) is NOT created until core layer is available."""
        build_dir.mkdir(parents=True, exist_ok=True)
        store = ArtifactStore(build_dir)
        context_path = build_dir / "context.md"

        layer_artifacts: dict[str, list[Artifact]] = {"episodes": episode_artifacts}
        _materialize_layer_projections(
            pipeline_obj, "episodes", layer_artifacts, store, build_dir,
        )
        assert not context_path.exists(), "context.md should not exist after only episodes"

        layer_artifacts["monthly"] = monthly_artifacts
        _materialize_layer_projections(
            pipeline_obj, "monthly", layer_artifacts, store, build_dir,
        )
        assert not context_path.exists(), "context.md should not exist after only monthly"

        # Only after core layer is present should the flat file materialize
        layer_artifacts["core"] = [Artifact(
            artifact_id="core-memory", artifact_type="core_memory",
            content="## Identity\nSoftware engineer.",
            metadata={"layer_name": "core", "layer_level": 3},
        )]
        _materialize_layer_projections(
            pipeline_obj, "core", layer_artifacts, store, build_dir,
        )
        assert context_path.exists(), "context.md should exist after core completes"

    def test_topical_transform_queries_intermediate_index(
        self, source_dir, build_dir, mock_llm,
    ):
        """Topical rollup can query the episode search index built after episodes complete.

        This is the critical dependency: episodes → [search_index projection] → topics.
        """
        topical = _topical_pipeline(build_dir)
        result = run(topical, source_dir=str(source_dir))

        # Topics should have been built (not errored due to missing index)
        topic_stats = next(s for s in result.layer_stats if s.name == "topics")
        assert topic_stats.built > 0

        # The search index should contain data from all three source layers
        db_path = build_dir / "search.db"
        layers = _layers_in_index(db_path)
        assert "episodes" in layers
        assert "topics" in layers
        assert "core" in layers


# ---------------------------------------------------------------------------
# 2. End-to-end progressive materialization (full pipeline)
# ---------------------------------------------------------------------------

class TestProgressiveMaterialization:
    def test_topical_pipeline_with_cached_episodes(self, source_dir, build_dir, mock_llm):
        """Build monthly then topical — episodes cached, topical succeeds (no 'no such table')."""
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

    def test_search_index_queryable_after_run(self, source_dir, build_dir, mock_llm):
        """After a pipeline run, search.db exists and returns results."""
        pipeline = _monthly_pipeline(build_dir)
        run(pipeline, source_dir=str(source_dir))

        db_path = build_dir / "search.db"
        assert db_path.exists()

        index = SearchIndex(db_path)
        results = index.query("programming")
        assert isinstance(results, list)
        assert len(results) > 0
        index.close()


# ---------------------------------------------------------------------------
# 2. Projection Caching
# ---------------------------------------------------------------------------

class TestProjectionCaching:
    def test_projection_cache_file_created(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """After a run, .projection_cache.json exists with entries per projection."""
        run(pipeline_obj, source_dir=str(source_dir))

        cache_path = build_dir / ".projection_cache.json"
        assert cache_path.exists()

        cache = json.loads(cache_path.read_text())
        assert "memory-index" in cache
        assert "context-doc" in cache

    def test_projection_cached_on_second_run(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """Second run: all projection_stats have status='cached'."""
        run(pipeline_obj, source_dir=str(source_dir))
        result2 = run(pipeline_obj, source_dir=str(source_dir))

        assert len(result2.projection_stats) > 0
        for ps in result2.projection_stats:
            assert ps.status == "cached", f"Projection {ps.name} was {ps.status}, expected cached"

    def test_projection_rebuilds_on_new_artifacts(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """After adding a new conversation, at least one projection rebuilds."""
        run(pipeline_obj, source_dir=str(source_dir))

        # Add a new conversation to force new artifacts
        claude_path = source_dir / "claude_export.json"
        data = json.loads(claude_path.read_text())
        data["conversations"].append({
            "uuid": "conv-proj-test-001",
            "title": "Projection rebuild test",
            "created_at": "2024-05-01T10:00:00Z",
            "chat_messages": [
                {"uuid": "msg-p1", "sender": "human",
                 "text": "Hello, testing projection rebuild.", "created_at": "2024-05-01T10:00:00Z"},
                {"uuid": "msg-p2", "sender": "assistant",
                 "text": "Acknowledged, this is a test conversation.",
                 "created_at": "2024-05-01T10:01:00Z"},
            ],
        })
        claude_path.write_text(json.dumps(data))

        result2 = run(pipeline_obj, source_dir=str(source_dir))
        statuses = {ps.name: ps.status for ps in result2.projection_stats}
        assert "built" in statuses.values(), f"Expected at least one 'built' projection, got {statuses}"


# ---------------------------------------------------------------------------
# 3. Projection Plan
# ---------------------------------------------------------------------------

class TestProjectionPlan:
    def test_plan_includes_projections(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """plan_build() on a fresh pipeline includes projections with status='new'."""
        plan = plan_build(pipeline_obj, source_dir=str(source_dir))

        assert len(plan.projections) == 2
        for pp in plan.projections:
            assert pp.status == "new"
            assert isinstance(pp, ProjectionPlan)

    def test_plan_projections_cached_after_build(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """After running, plan_build() reports projections as 'cached'."""
        run(pipeline_obj, source_dir=str(source_dir))
        plan = plan_build(pipeline_obj, source_dir=str(source_dir))

        for pp in plan.projections:
            assert pp.status == "cached", f"Projection {pp.name} was {pp.status}, expected cached"

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


# ---------------------------------------------------------------------------
# 4. Projection Stats
# ---------------------------------------------------------------------------

class TestProjectionStats:
    def test_run_result_has_projection_stats(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """RunResult.projection_stats has one entry per projection."""
        result = run(pipeline_obj, source_dir=str(source_dir))
        assert len(result.projection_stats) == 2

    def test_projection_stats_first_run_built(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """First run: all projection stats have status='built'."""
        result = run(pipeline_obj, source_dir=str(source_dir))

        for ps in result.projection_stats:
            assert ps.status == "built", f"Projection {ps.name} was {ps.status}, expected built"

    def test_projection_stats_second_run_cached(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """Second run: all projection stats have status='cached'."""
        run(pipeline_obj, source_dir=str(source_dir))
        result2 = run(pipeline_obj, source_dir=str(source_dir))

        for ps in result2.projection_stats:
            assert ps.status == "cached", f"Projection {ps.name} was {ps.status}, expected cached"


# ---------------------------------------------------------------------------
# 5. Topical Rollup Defensive Fallback
# ---------------------------------------------------------------------------

class TestTopicalRollupDefensiveFallback:
    def test_topical_rollup_no_search_db(self, source_dir, build_dir, mock_llm):
        """TopicalRollupTransform succeeds when search_db_path doesn't exist."""
        from synix.build.transforms import get_transform

        # First, run parse + episodes to get episode artifacts
        monthly = _monthly_pipeline(build_dir)
        result = run(monthly, source_dir=str(source_dir))

        from synix.build.artifacts import ArtifactStore
        store = ArtifactStore(build_dir)
        episodes = store.list_artifacts("episodes")
        assert len(episodes) > 0

        transform = get_transform("topical_rollup")
        config = {
            "llm_config": {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024},
            "topics": ["programming"],
            "search_db_path": str(build_dir / "nonexistent.db"),
        }
        results = transform.execute(episodes, config)
        assert len(results) > 0
        assert results[0].artifact_type == "rollup"

    def test_topical_rollup_empty_search_db(self, source_dir, build_dir, mock_llm):
        """TopicalRollupTransform succeeds when search db has no FTS5 table."""
        # Run a build to get episodes
        monthly = _monthly_pipeline(build_dir)
        run(monthly, source_dir=str(source_dir))

        from synix.build.artifacts import ArtifactStore
        store = ArtifactStore(build_dir)
        episodes = store.list_artifacts("episodes")
        assert len(episodes) > 0

        # Create an empty SQLite file (no search_index table)
        empty_db = build_dir / "empty_search.db"
        conn = sqlite3.connect(str(empty_db))
        conn.execute("CREATE TABLE dummy (id INTEGER)")
        conn.commit()
        conn.close()

        from synix.build.transforms import get_transform
        transform = get_transform("topical_rollup")
        config = {
            "llm_config": {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024},
            "topics": ["programming"],
            "search_db_path": str(empty_db),
        }
        results = transform.execute(episodes, config)
        assert len(results) > 0
        assert results[0].artifact_type == "rollup"


# ---------------------------------------------------------------------------
# 6. Projection Logger Events
# ---------------------------------------------------------------------------

class TestProjectionLoggerEvents:
    def test_projection_events_in_log_file(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """JSONL log file contains projection_start and projection_finish events."""
        run(pipeline_obj, source_dir=str(source_dir))

        logs_dir = build_dir / "logs"
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

        # Verify projection names are present
        proj_start_events = [e for e in events if e["event"] == "projection_start"]
        proj_names = {e["projection"] for e in proj_start_events}
        assert "memory-index" in proj_names
