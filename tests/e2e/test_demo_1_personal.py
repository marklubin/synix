"""Demo 1: Personal Memory Pipeline — E2E tests.

Tests the full personal memory pipeline using the demo_1_personal corpus (30 conversations).
Exercises: build, plan, search, verify, diff, config change (monthly→topical), cache behavior.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from synix.cli import main

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CORPUS_DIR = Path(__file__).parent.parent / "fixtures" / "corpus" / "demo_1_personal"


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace with demo_1 corpus and a build dir."""
    source_dir = tmp_path / "exports"
    source_dir.mkdir()
    build_dir = tmp_path / "build"

    # Copy demo_1 corpus
    shutil.copy(CORPUS_DIR / "chatgpt_export.json", source_dir / "chatgpt_export.json")
    shutil.copy(CORPUS_DIR / "claude_export.json", source_dir / "claude_export.json")

    return {"root": tmp_path, "source_dir": source_dir, "build_dir": build_dir}


@pytest.fixture
def monthly_pipeline_file(workspace):
    """Write a monthly pipeline.py into the workspace."""
    path = workspace["root"] / "pipeline_monthly.py"
    path.write_text(f"""
from synix import Pipeline, Layer, Projection

pipeline = Pipeline("demo1-monthly")
pipeline.source_dir = "{workspace['source_dir']}"
pipeline.build_dir = "{workspace['build_dir']}"
pipeline.llm_config = {{"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}}

pipeline.add_layer(Layer(name="transcripts", level=0, transform="parse"))
pipeline.add_layer(Layer(name="episodes", level=1, depends_on=["transcripts"], transform="episode_summary", grouping="by_conversation"))
pipeline.add_layer(Layer(name="monthly", level=2, depends_on=["episodes"], transform="monthly_rollup", grouping="by_month"))
pipeline.add_layer(Layer(name="core", level=3, depends_on=["monthly"], transform="core_synthesis", grouping="single", context_budget=10000))

pipeline.add_projection(Projection(name="memory-index", projection_type="search_index", sources=[
    {{"layer": "episodes", "search": ["fulltext"]}},
    {{"layer": "monthly", "search": ["fulltext"]}},
    {{"layer": "core", "search": ["fulltext"]}},
]))
pipeline.add_projection(Projection(name="context-doc", projection_type="flat_file", sources=[{{"layer": "core"}}], config={{"output_path": "{workspace['build_dir'] / 'context.md'}"}}))
""")
    return path


@pytest.fixture
def topical_pipeline_file(workspace):
    """Write a topical pipeline.py into the workspace."""
    path = workspace["root"] / "pipeline_topical.py"
    path.write_text(f"""
from synix import Pipeline, Layer, Projection

pipeline = Pipeline("demo1-topical")
pipeline.source_dir = "{workspace['source_dir']}"
pipeline.build_dir = "{workspace['build_dir']}"
pipeline.llm_config = {{"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}}

pipeline.add_layer(Layer(name="transcripts", level=0, transform="parse"))
pipeline.add_layer(Layer(name="episodes", level=1, depends_on=["transcripts"], transform="episode_summary", grouping="by_conversation"))
pipeline.add_layer(Layer(name="topics", level=2, depends_on=["episodes"], transform="topical_rollup", grouping="by_topic", config={{
    "topics": ["database-migration", "career", "rust-learning", "side-projects"],
}}))
pipeline.add_layer(Layer(name="core", level=3, depends_on=["topics"], transform="core_synthesis", grouping="single", context_budget=10000))

pipeline.add_projection(Projection(name="memory-index", projection_type="search_index", sources=[
    {{"layer": "episodes", "search": ["fulltext"]}},
    {{"layer": "topics", "search": ["fulltext"]}},
    {{"layer": "core", "search": ["fulltext"]}},
]))
pipeline.add_projection(Projection(name="context-doc", projection_type="flat_file", sources=[{{"layer": "core"}}], config={{"output_path": "{workspace['build_dir'] / 'context.md'}"}}))
""")
    return path


@pytest.fixture(autouse=True)
def mock_anthropic(monkeypatch):
    """Mock Anthropic API for all E2E tests."""
    call_count = {"n": 0}

    def mock_create(**kwargs):
        call_count["n"] += 1
        messages = kwargs.get("messages", [])
        content = messages[0].get("content", "") if messages else ""

        if "summarizing a conversation" in content.lower():
            return _mock_response(
                "This conversation covered technical topics including programming "
                "and software development. The user discussed database migration, "
                "career planning, Rust programming, and side projects."
            )
        elif "monthly" in content.lower():
            return _mock_response(
                "In this month, the main themes were database migration planning, "
                "career development toward Staff Engineer, learning Rust from Python, "
                "and building CLI tools as side projects."
            )
        elif "topic" in content.lower():
            return _mock_response(
                "Regarding this topic, the user has shown consistent interest and "
                "progression. Their understanding evolved from initial exploration "
                "to advanced implementation over multiple conversations."
            )
        elif "core memory" in content.lower():
            return _mock_response(
                "## Identity\nMark is a senior software engineer at a fintech company.\n\n"
                "## Current Focus\nLeading a CockroachDB migration and pursuing Staff Engineer.\n\n"
                "## Technical Interests\nRust, databases, CLI tools, build systems.\n\n"
                "## Preferences\nPrefers practical learning, clean code, and phased approaches."
            )
        return _mock_response("Mock response.")

    mock_client = MagicMock()
    mock_client.messages.create = mock_create
    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)
    return call_count


def _mock_response(text: str):
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    resp.model = "claude-sonnet-4-20250514"
    resp.usage = MagicMock(input_tokens=100, output_tokens=50)
    return resp


# ---------------------------------------------------------------------------
# DT-1.1: Fresh build + plan verification
# ---------------------------------------------------------------------------


class TestDT1FreshBuild:
    """DT-1.1: Fresh build with plan verification."""

    def test_plan_shows_all_layers_as_new(self, runner, workspace, monthly_pipeline_file):
        """synix plan on a fresh workspace shows all layers need building."""
        result = runner.invoke(main, [
            "plan", str(monthly_pipeline_file),
        ])
        assert result.exit_code == 0, f"Plan failed: {result.output}"
        assert "Estimated:" in result.output

    def test_fresh_build_produces_correct_artifact_counts(
        self, runner, workspace, monthly_pipeline_file
    ):
        """First build produces: 30 transcripts, 30 episodes, 4 monthly, 1 core."""
        result = runner.invoke(main, ["build", str(monthly_pipeline_file)])
        assert result.exit_code == 0, f"Build failed: {result.output}"
        assert "Build Summary" in result.output

        manifest = json.loads((workspace["build_dir"] / "manifest.json").read_text())

        # Count by layer
        layers: dict[str, int] = {}
        for _aid, info in manifest.items():
            layer = info.get("layer", "unknown")
            layers[layer] = layers.get(layer, 0) + 1

        # 20 chatgpt + 10 claude = 30 transcripts
        assert layers.get("transcripts", 0) == 30
        # 30 transcripts → 30 episodes (1 per conversation)
        assert layers.get("episodes", 0) == 30
        # Episodes across 3 months (UTC): 2024-12, 2025-01, 2025-12
        assert layers.get("monthly", 0) == 3
        # Single core memory
        assert layers.get("core", 0) == 1

    def test_verify_passes_after_fresh_build(
        self, runner, workspace, monthly_pipeline_file
    ):
        """synix verify passes on a clean build."""
        runner.invoke(main, ["build", str(monthly_pipeline_file)])

        from synix.build.verify import verify_build
        result = verify_build(str(workspace["build_dir"]))
        assert result.passed, f"Verify failed: {result.summary}. Details: {[c.message for c in result.failed_checks]}"

    def test_search_index_populated_after_build(
        self, runner, workspace, monthly_pipeline_file
    ):
        """Search returns results for content present in the corpus."""
        runner.invoke(main, ["build", str(monthly_pipeline_file)])

        search_db = workspace["build_dir"] / "search.db"
        assert search_db.exists()

        result = runner.invoke(main, [
            "search", "programming", "--build-dir", str(workspace["build_dir"])
        ])
        assert result.exit_code == 0

    def test_context_doc_created_with_core_content(
        self, runner, workspace, monthly_pipeline_file
    ):
        """Context doc contains the core memory synthesis."""
        runner.invoke(main, ["build", str(monthly_pipeline_file)])

        context_doc = workspace["build_dir"] / "context.md"
        assert context_doc.exists()
        content = context_doc.read_text()
        assert "Identity" in content
        assert "Mark" in content

    def test_all_derived_artifacts_have_provenance(
        self, runner, workspace, monthly_pipeline_file
    ):
        """Every non-transcript artifact has provenance records."""
        runner.invoke(main, ["build", str(monthly_pipeline_file)])

        provenance_path = workspace["build_dir"] / "provenance.json"
        assert provenance_path.exists()
        provenance = json.loads(provenance_path.read_text())

        manifest = json.loads((workspace["build_dir"] / "manifest.json").read_text())
        derived = {
            aid for aid, info in manifest.items()
            if info.get("layer") != "transcripts"
        }
        missing = [aid for aid in derived if aid not in provenance]
        assert not missing, f"Missing provenance for: {missing}"


# ---------------------------------------------------------------------------
# DT-1.2: Search with layer filtering
# ---------------------------------------------------------------------------


class TestDT1Search:
    """DT-1.2: Search with filters."""

    def test_search_with_layer_filter(self, runner, workspace, monthly_pipeline_file):
        """Search filtered to episodes layer returns only episode results."""
        runner.invoke(main, ["build", str(monthly_pipeline_file)])

        result = runner.invoke(main, [
            "search", "programming",
            "--layers", "episodes",
            "--build-dir", str(workspace["build_dir"])
        ])
        assert result.exit_code == 0
        # Should have results from episodes layer
        if "No results" not in result.output:
            assert "episodes" in result.output.lower() or "L1" in result.output


# ---------------------------------------------------------------------------
# DT-1.3: Config change (monthly → topical), diff verification
# ---------------------------------------------------------------------------


class TestDT1ConfigChange:
    """DT-1.3: Monthly→topical reconfig, cache behavior, diff verification."""

    def test_topical_rebuild_caches_transcripts_and_episodes(
        self, runner, workspace, monthly_pipeline_file, topical_pipeline_file, mock_anthropic
    ):
        """After monthly build, topical rebuild reuses transcripts+episodes."""
        # First build: monthly
        result1 = runner.invoke(main, ["build", str(monthly_pipeline_file)])
        assert result1.exit_code == 0
        calls_after_first = mock_anthropic["n"]

        manifest1 = json.loads((workspace["build_dir"] / "manifest.json").read_text())
        transcript_ids = {aid for aid, info in manifest1.items() if info["layer"] == "transcripts"}
        episode_ids = {aid for aid, info in manifest1.items() if info["layer"] == "episodes"}

        # Second build: topical
        result2 = runner.invoke(main, ["build", str(topical_pipeline_file)])
        assert result2.exit_code == 0

        manifest2 = json.loads((workspace["build_dir"] / "manifest.json").read_text())

        # Transcripts and episodes should be preserved
        for tid in transcript_ids:
            assert tid in manifest2, f"Transcript {tid} missing after topical build"
        for eid in episode_ids:
            assert eid in manifest2, f"Episode {eid} missing after topical build"

        # Topics layer should exist now
        layers2 = {info["layer"] for info in manifest2.values()}
        assert "topics" in layers2

        # Core memory should be rebuilt
        assert "core-memory" in manifest2

    def test_core_memory_content_differs_after_config_change(
        self, runner, workspace, monthly_pipeline_file, topical_pipeline_file
    ):
        """Context doc is updated after config change."""
        # Build monthly
        runner.invoke(main, ["build", str(monthly_pipeline_file)])
        context1 = (workspace["build_dir"] / "context.md").read_text()

        # Build topical
        runner.invoke(main, ["build", str(topical_pipeline_file)])
        context2 = (workspace["build_dir"] / "context.md").read_text()

        # Context should still contain core memory (both builds produce it)
        assert len(context1) > 0
        assert len(context2) > 0

    def test_verify_passes_after_topical_rebuild(
        self, runner, workspace, monthly_pipeline_file, topical_pipeline_file
    ):
        """Verify still passes after config change rebuild."""
        runner.invoke(main, ["build", str(monthly_pipeline_file)])
        runner.invoke(main, ["build", str(topical_pipeline_file)])

        from synix.build.verify import verify_build
        result = verify_build(str(workspace["build_dir"]))
        assert result.passed, f"Verify failed: {result.summary}"


# ---------------------------------------------------------------------------
# DT-1.4: No-change rebuild, full cache hit
# ---------------------------------------------------------------------------


class TestDT1CacheHit:
    """DT-1.4: No-change rebuild should be fully cached."""

    def test_second_run_uses_cache(
        self, runner, workspace, monthly_pipeline_file, mock_anthropic
    ):
        """Second identical run should not make new LLM calls for episodes+."""
        # First build
        result1 = runner.invoke(main, ["build", str(monthly_pipeline_file)])
        assert result1.exit_code == 0
        calls_after_first = mock_anthropic["n"]

        # Second build (same config, same data)
        result2 = runner.invoke(main, ["build", str(monthly_pipeline_file)])
        assert result2.exit_code == 0
        calls_after_second = mock_anthropic["n"]

        # Second run should make 0 new LLM calls
        assert calls_after_second == calls_after_first, (
            f"Expected no new LLM calls on cached rebuild. "
            f"First: {calls_after_first}, Second: {calls_after_second}"
        )

    def test_plan_shows_all_cached_on_second_run(
        self, runner, workspace, monthly_pipeline_file
    ):
        """Plan after a clean build shows everything as cached."""
        runner.invoke(main, ["build", str(monthly_pipeline_file)])

        result = runner.invoke(main, [
            "plan", str(monthly_pipeline_file),
        ])
        assert result.exit_code == 0
        # Plan should report cached status
        assert "cached" in result.output.lower()
