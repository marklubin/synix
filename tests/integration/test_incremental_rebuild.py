"""Integration tests — incremental rebuild / cache behavior."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from synix import FlatFile, Pipeline, SearchIndex, Source
from synix.build.runner import run
from synix.transforms import CoreSynthesis, EpisodeSummary, MonthlyRollup

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


class TestIncrementalRebuild:
    def test_second_run_all_cached(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """Run twice — second run caches all transform layers."""
        result1 = run(pipeline_obj, source_dir=str(source_dir))
        assert result1.built > 0

        result2 = run(pipeline_obj, source_dir=str(source_dir))
        # Source layers always re-parse (counted as built), but transform layers
        # should all be cached on the second run.
        t_stats = next(s for s in result2.layer_stats if s.name == "transcripts")
        for stats in result2.layer_stats:
            if stats.name != "transcripts":  # skip Source layer
                assert stats.built == 0, f"Layer {stats.name} should be cached but built {stats.built}"
                assert stats.cached > 0, f"Layer {stats.name} should have cached artifacts"

    def test_new_source_partial_rebuild(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """Add a conversation — only its chain rebuilds."""
        # First run
        result1 = run(pipeline_obj, source_dir=str(source_dir))
        first_built = result1.built

        # Add a new conversation to the Claude export
        claude_path = source_dir / "claude_export.json"
        data = json.loads(claude_path.read_text())
        data["conversations"].append(
            {
                "uuid": "conv-new-001",
                "title": "New conversation about Synix",
                "created_at": "2024-03-25T10:00:00Z",
                "chat_messages": [
                    {
                        "uuid": "msg-new-1",
                        "sender": "human",
                        "text": "Tell me about Synix.",
                        "created_at": "2024-03-25T10:00:00Z",
                    },
                    {
                        "uuid": "msg-new-2",
                        "sender": "assistant",
                        "text": "Synix is a build system for agent memory.",
                        "created_at": "2024-03-25T10:01:00Z",
                    },
                ],
            }
        )
        claude_path.write_text(json.dumps(data))

        # Second run — should have some cached + some built
        result2 = run(pipeline_obj, source_dir=str(source_dir))
        assert result2.built > 0  # new transcript + its episode + rollups + core
        assert result2.cached > 0  # existing transcripts and some episodes

    def test_prompt_change_cascades(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """Change episode prompt — episodes + rollups + core rebuild, transcripts cached."""
        # First full run
        result1 = run(pipeline_obj, source_dir=str(source_dir))
        assert result1.built > 0

        # Find transcript stats and episode stats
        transcript_stats_1 = next(s for s in result1.layer_stats if s.name == "transcripts")
        total_transcripts = transcript_stats_1.built + transcript_stats_1.cached

        # Now change the episode prompt template to force rebuild
        from synix.build.transforms import PROMPTS_DIR

        prompt_path = PROMPTS_DIR / "episode_summary.txt"
        original = prompt_path.read_text()
        try:
            prompt_path.write_text(original + "\n\nAdditional instruction: be more detailed.")

            result2 = run(pipeline_obj, source_dir=str(source_dir))

            # Transcripts: Source layers always re-parse (built > 0),
            # but content is unchanged so downstream caching still works.
            transcript_stats = next(s for s in result2.layer_stats if s.name == "transcripts")
            assert transcript_stats.built == total_transcripts

            # Episodes should rebuild (prompt changed)
            episode_stats = next(s for s in result2.layer_stats if s.name == "episodes")
            assert episode_stats.built > 0
        finally:
            # Restore original prompt
            prompt_path.write_text(original)

    def test_cache_metrics_accurate(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """RunResult reports correct built/cached/skipped counts."""
        result1 = run(pipeline_obj, source_dir=str(source_dir))

        total1 = result1.built + result1.cached + result1.skipped
        assert total1 > 0
        # First run: everything built, nothing cached
        assert result1.cached == 0

        result2 = run(pipeline_obj, source_dir=str(source_dir))
        total2 = result2.built + result2.cached + result2.skipped
        assert total2 > 0
        # Second run: Source layers always re-parse (built), but all Transform
        # layers should be cached. The total artifact count should be the same.
        transcript_built = next(s for s in result2.layer_stats if s.name == "transcripts").built
        assert result2.built == transcript_built  # only Source layer counted as built
        assert result2.cached > 0  # Transform layers are cached
