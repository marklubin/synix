"""Integration test — full pipeline with mock LLM."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from synix import FlatFile, Pipeline, SearchIndex, Source
from synix.build.runner import run
from synix.build.snapshot_view import SnapshotArtifactCache
from synix.ext import CoreSynthesis, EpisodeSummary, MonthlyRollup
from synix.search.index import SearchIndex as SearchIdx

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

    transcripts = Source("transcripts")
    episodes = EpisodeSummary("episodes", depends_on=[transcripts])
    monthly = MonthlyRollup("monthly", depends_on=[episodes])
    core = CoreSynthesis("core", depends_on=[monthly], context_budget=10000)

    p.add(transcripts, episodes, monthly, core)
    p.add(SearchIndex("memory-index", sources=[episodes, monthly, core], search=["fulltext"]))
    p.add(FlatFile("context-doc", sources=[core], output_path=str(build_dir / "context.md")))

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
        synix_dir = build_dir.parent / ".synix"
        store = SnapshotArtifactCache(synix_dir)

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
        synix_dir = build_dir.parent / ".synix"
        store = SnapshotArtifactCache(synix_dir)

        # Episodes should have provenance pointing to transcripts
        episodes = store.list_artifacts("episodes")
        for ep in episodes:
            parents = store.get_parents(ep.label)
            assert len(parents) > 0, f"Episode {ep.label} has no provenance parents"

        # Core should have provenance
        core = store.list_artifacts("core")
        for c in core:
            parents = store.get_parents(c.label)
            assert len(parents) > 0, f"Core {c.label} has no provenance parents"

    def test_search_returns_results_after_run(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """Pipeline run → release → search works."""
        from synix.build.release_engine import execute_release

        run(pipeline_obj, source_dir=str(source_dir))

        synix_dir = build_dir.parent / ".synix"
        execute_release(synix_dir, release_name="local")
        release_dir = synix_dir / "releases" / "local"

        index = SearchIdx(release_dir / "search.db")
        results = index.query("programming")
        # Should find results from mock LLM responses
        assert len(results) > 0
        index.close()

    def test_context_doc_exists_after_run(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """Pipeline run → release → context.md exists."""
        from synix.build.release_engine import execute_release

        run(pipeline_obj, source_dir=str(source_dir))

        synix_dir = build_dir.parent / ".synix"
        execute_release(synix_dir, release_name="local")
        release_dir = synix_dir / "releases" / "local"

        context_path = release_dir / "context.md"
        assert context_path.exists()
        content = context_path.read_text()
        assert len(content) > 0
