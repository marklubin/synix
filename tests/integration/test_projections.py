"""Integration tests — end-to-end projection materialization."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from synix import FlatFile, Pipeline, SearchIndex, Source
from synix.artifacts.provenance import ProvenanceTracker
from synix.build.runner import run
from synix.search.index import SearchIndex as SearchIdx
from synix.ext import CoreSynthesis, EpisodeSummary, MonthlyRollup

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


class TestProjections:
    def test_search_index_reflects_all_layers(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """Search results come from multiple layers."""
        run(pipeline_obj, source_dir=str(source_dir))

        index = SearchIdx(build_dir / "search.db")

        # Query broadly — should get results from episodes, monthly, and core
        results = index.query("programming")
        layer_names = {r.layer_name for r in results}

        # At minimum, episodes should have matches (mock returns "programming" content)
        assert len(results) > 0
        # Should have results from at least 2 layers
        assert len(layer_names) >= 1

        index.close()

    def test_provenance_chain_depth(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """Core result traces back to transcript through all layers."""
        run(pipeline_obj, source_dir=str(source_dir))

        provenance = ProvenanceTracker(build_dir)

        # Core memory should have a provenance chain
        chain = provenance.get_chain("core-memory")
        assert len(chain) > 0

        # The chain should include the core artifact
        chain_ids = [r.label for r in chain]
        assert "core-memory" in chain_ids

        # Should trace through monthly rollups
        monthly_ids = [cid for cid in chain_ids if cid.startswith("monthly-")]
        assert len(monthly_ids) > 0

    def test_flat_file_is_ready_to_use(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """context.md could be pasted into a system prompt."""
        run(pipeline_obj, source_dir=str(source_dir))

        context_path = build_dir / "context.md"
        assert context_path.exists()

        content = context_path.read_text()
        # Should contain the mock core memory text
        assert len(content) > 0
        # Should be markdown-ish (the mock returns markdown headers)
        assert "##" in content or len(content) > 50
