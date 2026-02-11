"""Integration test — full pipeline with mock LLM."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from synix import Layer, Pipeline, Projection
from synix.artifacts.provenance import ProvenanceTracker
from synix.artifacts.store import ArtifactStore
from synix.pipeline.runner import run
from synix.search.index import SearchIndex

FIXTURES_DIR = Path(__file__).parent.parent / "synix" / "fixtures"


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
def pipeline_obj(build_dir):
    """Standard monthly pipeline."""
    p = Pipeline("test-pipeline")
    p.build_dir = str(build_dir)
    p.llm_config = {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}

    p.add_layer(Layer(name="transcripts", level=0, transform="parse"))
    p.add_layer(
        Layer(
            name="episodes",
            level=1,
            depends_on=["transcripts"],
            transform="episode_summary",
            grouping="by_conversation",
        )
    )
    p.add_layer(
        Layer(name="monthly", level=2, depends_on=["episodes"], transform="monthly_rollup", grouping="by_month")
    )
    p.add_layer(
        Layer(
            name="core",
            level=3,
            depends_on=["monthly"],
            transform="core_synthesis",
            grouping="single",
            context_budget=10000,
        )
    )

    p.add_projection(
        Projection(
            name="memory-index",
            projection_type="search_index",
            sources=[
                {"layer": "episodes", "search": ["fulltext"]},
                {"layer": "monthly", "search": ["fulltext"]},
                {"layer": "core", "search": ["fulltext"]},
            ],
        )
    )
    p.add_projection(
        Projection(
            name="context-doc",
            projection_type="flat_file",
            sources=[{"layer": "core"}],
            config={"output_path": str(build_dir / "context.md")},
        )
    )

    return p


class TestFullPipelineRun:
    def test_full_pipeline_mock_llm(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """Run complete pipeline — all layers built, all projections materialized."""
        result = run(pipeline_obj, source_dir=str(source_dir))

        assert result.built > 0
        assert result.total_time > 0
        assert len(result.layer_stats) == 4  # transcripts, episodes, monthly, core

        # All layers have stats
        layer_names = [s.name for s in result.layer_stats]
        assert "transcripts" in layer_names
        assert "episodes" in layer_names
        assert "monthly" in layer_names
        assert "core" in layer_names

    def test_artifact_count_matches_expectations(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """Correct number of artifacts per layer."""
        run(pipeline_obj, source_dir=str(source_dir))
        store = ArtifactStore(build_dir)

        # 3 ChatGPT + 5 Claude = 8 transcripts
        transcripts = store.list_artifacts("transcripts")
        assert len(transcripts) == 8

        # 8 episodes (one per conversation)
        episodes = store.list_artifacts("episodes")
        assert len(episodes) == 8

        # Monthly rollups: March 2024 + April 2024 = 2 months
        monthly = store.list_artifacts("monthly")
        assert len(monthly) >= 1  # at least 1 month

        # Core: exactly 1
        core = store.list_artifacts("core")
        assert len(core) == 1

    def test_all_artifacts_have_provenance(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """Every non-root artifact has provenance records."""
        run(pipeline_obj, source_dir=str(source_dir))
        store = ArtifactStore(build_dir)
        provenance = ProvenanceTracker(build_dir)

        # Episodes should have provenance pointing to transcripts
        episodes = store.list_artifacts("episodes")
        for ep in episodes:
            parents = provenance.get_parents(ep.artifact_id)
            assert len(parents) > 0, f"Episode {ep.artifact_id} has no provenance parents"

        # Core should have provenance
        core = store.list_artifacts("core")
        for c in core:
            parents = provenance.get_parents(c.artifact_id)
            assert len(parents) > 0, f"Core {c.artifact_id} has no provenance parents"

    def test_search_returns_results_after_run(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """Pipeline run → search works."""
        run(pipeline_obj, source_dir=str(source_dir))

        index = SearchIndex(build_dir / "search.db")
        results = index.query("programming")
        # Should find results from mock LLM responses
        assert len(results) > 0
        index.close()

    def test_context_doc_exists_after_run(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """Pipeline run → context.md exists."""
        run(pipeline_obj, source_dir=str(source_dir))

        context_path = build_dir / "context.md"
        assert context_path.exists()
        content = context_path.read_text()
        assert len(content) > 0
