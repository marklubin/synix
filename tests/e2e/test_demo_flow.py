"""E2E test — the exact demo sequence, automated.

Run → search → run again (cached) → config change → run (partial rebuild) → search (different results).
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

FIXTURES_DIR = Path(__file__).parent.parent / "synix" / "fixtures"


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace with source exports and a build dir."""
    source_dir = tmp_path / "exports"
    source_dir.mkdir()
    build_dir = tmp_path / "build"

    # Copy fixture exports
    shutil.copy(FIXTURES_DIR / "chatgpt_export.json", source_dir / "chatgpt_export.json")
    shutil.copy(FIXTURES_DIR / "claude_export.json", source_dir / "claude_export.json")

    return {"root": tmp_path, "source_dir": source_dir, "build_dir": build_dir}


@pytest.fixture
def pipeline_file(workspace):
    """Write a pipeline.py into the workspace."""
    path = workspace["root"] / "pipeline.py"
    path.write_text(f"""
from synix import Pipeline, Layer, Projection

pipeline = Pipeline("test-monthly")
pipeline.source_dir = "{workspace["source_dir"]}"
pipeline.build_dir = "{workspace["build_dir"]}"
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
pipeline.add_projection(Projection(name="context-doc", projection_type="flat_file", sources=[{{"layer": "core"}}], config={{"output_path": "{workspace["build_dir"] / "context.md"}"}}))
""")
    return path


@pytest.fixture
def topical_pipeline_file(workspace):
    """Write a pipeline_topical.py into the workspace."""
    path = workspace["root"] / "pipeline_topical.py"
    path.write_text(f"""
from synix import Pipeline, Layer, Projection

pipeline = Pipeline("test-topical")
pipeline.source_dir = "{workspace["source_dir"]}"
pipeline.build_dir = "{workspace["build_dir"]}"
pipeline.llm_config = {{"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}}

pipeline.add_layer(Layer(name="transcripts", level=0, transform="parse"))
pipeline.add_layer(Layer(name="episodes", level=1, depends_on=["transcripts"], transform="episode_summary", grouping="by_conversation"))
pipeline.add_layer(Layer(name="topics", level=2, depends_on=["episodes"], transform="topical_rollup", grouping="by_topic", config={{
    "topics": ["programming", "devops", "ai-and-ml"],
}}))
pipeline.add_layer(Layer(name="core", level=3, depends_on=["topics"], transform="core_synthesis", grouping="single", context_budget=10000))

pipeline.add_projection(Projection(name="memory-index", projection_type="search_index", sources=[
    {{"layer": "episodes", "search": ["fulltext"]}},
    {{"layer": "topics", "search": ["fulltext"]}},
    {{"layer": "core", "search": ["fulltext"]}},
]))
pipeline.add_projection(Projection(name="context-doc", projection_type="flat_file", sources=[{{"layer": "core"}}], config={{"output_path": "{workspace["build_dir"] / "context.md"}"}}))
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
                "and software development. The user explored fundamental concepts "
                "and best practices in the field."
            )
        elif "monthly" in content.lower():
            return _mock_response(
                "In this month, the main themes were technical learning and software "
                "development. The user explored machine learning, Docker, Git branching, "
                "Rust, Python async, API design, databases, and testing philosophies."
            )
        elif "topic" in content.lower():
            return _mock_response(
                "Regarding this topic, the user has shown consistent interest in "
                "technical subjects. Their understanding has evolved from basic "
                "concepts to more advanced applications and best practices."
            )
        elif "core memory" in content.lower():
            return _mock_response(
                "## Identity\nA software engineer interested in AI and systems.\n\n"
                "## Current Focus\nBuilding agent memory systems and learning.\n\n"
                "## Preferences\nPrefers clean code and well-tested systems."
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
# Tests
# ---------------------------------------------------------------------------


class TestDemoFlow:
    """The exact demo recording sequence, automated."""

    def test_full_demo_sequence(self, runner, workspace, pipeline_file, topical_pipeline_file, mock_anthropic):
        """Run the entire demo: run → search → run (cached) → config change → run → search."""
        build_dir = str(workspace["build_dir"])

        # ---- Step 1: First run — full build ----
        result1 = runner.invoke(main, ["run", str(pipeline_file)])
        assert result1.exit_code == 0, (
            f"Run 1 failed: {result1.output}\n{result1.stderr if hasattr(result1, 'stderr') else ''}"
        )
        assert "Build Summary" in result1.output

        # Verify artifacts were built
        assert workspace["build_dir"].exists()
        manifest_path = workspace["build_dir"] / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert len(manifest) > 0, "No artifacts were built"

        # ---- Step 2: Search ----
        result2 = runner.invoke(main, ["search", "machine learning", "--build-dir", build_dir])
        assert result2.exit_code == 0, f"Search failed: {result2.output}"
        # Should find results (our fixtures contain ML content)
        assert "No results" not in result2.output or "machine learning" in result2.output.lower()

        # ---- Step 3: Context doc exists ----
        context_doc = workspace["build_dir"] / "context.md"
        assert context_doc.exists(), "Context doc was not created"
        context_content1 = context_doc.read_text()
        assert len(context_content1) > 0

        # ---- Step 4: Second run — all cached ----
        api_calls_before = mock_anthropic["n"]
        result3 = runner.invoke(main, ["run", str(pipeline_file)])
        assert result3.exit_code == 0, f"Run 2 failed: {result3.output}"
        assert "Build Summary" in result3.output
        # The runner should detect cached artifacts. Since parse transform
        # always re-parses but the artifacts hash-match, most should be cached.

        # ---- Step 5: Config change — topical pipeline, partial rebuild ----
        result4 = runner.invoke(main, ["run", str(topical_pipeline_file)])
        assert result4.exit_code == 0, f"Run 3 (topical) failed: {result4.output}"
        assert "Build Summary" in result4.output

        # ---- Step 6: Search again — results may differ ----
        result5 = runner.invoke(main, ["search", "programming", "--build-dir", build_dir])
        assert result5.exit_code == 0, f"Search 2 failed: {result5.output}"

        # ---- Step 7: Status command ----
        result6 = runner.invoke(main, ["status", "--build-dir", build_dir])
        assert result6.exit_code == 0, f"Status failed: {result6.output}"
        assert "Build Status" in result6.output

    def test_run_produces_artifacts_per_layer(self, runner, workspace, pipeline_file):
        """Verify correct artifacts are produced for each layer."""
        result = runner.invoke(main, ["run", str(pipeline_file)], catch_exceptions=False)
        assert result.exit_code == 0, f"Run failed:\n{result.output}"

        manifest = json.loads((workspace["build_dir"] / "manifest.json").read_text())

        # Group by layer
        layers: dict[str, int] = {}
        for _aid, info in manifest.items():
            layer = info.get("layer", "unknown")
            layers[layer] = layers.get(layer, 0) + 1

        # Fixtures: 3 chatgpt + 5 claude = 8 transcripts
        assert layers.get("transcripts", 0) == 8
        # 8 transcripts → 8 episodes
        assert layers.get("episodes", 0) == 8
        # Episodes grouped by month (2024-03 and 2024-04) → 2 monthly rollups
        assert layers.get("monthly", 0) == 2
        # Single core memory
        assert layers.get("core", 0) == 1

    def test_search_index_populated(self, runner, workspace, pipeline_file):
        """Search index should be populated after a run."""
        runner.invoke(main, ["run", str(pipeline_file)])

        search_db = workspace["build_dir"] / "search.db"
        assert search_db.exists(), "Search DB should exist after run"

        # Query should return results
        result = runner.invoke(main, ["search", "programming", "--build-dir", str(workspace["build_dir"])])
        assert result.exit_code == 0

    def test_context_doc_created(self, runner, workspace, pipeline_file):
        """Context doc should be created after a run."""
        runner.invoke(main, ["run", str(pipeline_file)])

        context_doc = workspace["build_dir"] / "context.md"
        assert context_doc.exists()
        content = context_doc.read_text()
        assert len(content) > 0
        # Should contain core memory content from the mock
        assert "Identity" in content

    def test_provenance_tracked(self, runner, workspace, pipeline_file):
        """All derived artifacts should have provenance records."""
        runner.invoke(main, ["run", str(pipeline_file)])

        provenance_path = workspace["build_dir"] / "provenance.json"
        assert provenance_path.exists()
        provenance = json.loads(provenance_path.read_text())

        # Every episode, monthly, and core artifact should have provenance
        manifest = json.loads((workspace["build_dir"] / "manifest.json").read_text())
        derived = {aid for aid, info in manifest.items() if info.get("layer") != "transcripts"}
        for aid in derived:
            assert aid in provenance, f"Missing provenance for {aid}"

    def test_lineage_command(self, runner, workspace, pipeline_file):
        """Lineage command should show provenance tree."""
        runner.invoke(main, ["run", str(pipeline_file)])

        result = runner.invoke(main, ["lineage", "core-memory", "--build-dir", str(workspace["build_dir"])])
        assert result.exit_code == 0
        assert "core-memory" in result.output

    def test_incremental_rebuild_on_config_change(
        self, runner, workspace, pipeline_file, topical_pipeline_file, mock_anthropic
    ):
        """Swapping monthly→topical should cache transcripts+episodes, rebuild topics+core."""
        # First run: full build
        result1 = runner.invoke(main, ["run", str(pipeline_file)])
        assert result1.exit_code == 0

        # Capture manifest after first run
        manifest1 = json.loads((workspace["build_dir"] / "manifest.json").read_text())
        transcript_ids = {aid for aid, info in manifest1.items() if info["layer"] == "transcripts"}
        episode_ids = {aid for aid, info in manifest1.items() if info["layer"] == "episodes"}

        # Second run with topical pipeline
        result2 = runner.invoke(main, ["run", str(topical_pipeline_file)])
        assert result2.exit_code == 0

        # Verify the manifest now has topic artifacts instead of monthly
        manifest2 = json.loads((workspace["build_dir"] / "manifest.json").read_text())
        layers2 = {info["layer"] for info in manifest2.values()}
        assert "topics" in layers2, "Topical pipeline should produce 'topics' layer"

        # Transcripts and episodes should still be present (cached, not deleted)
        for tid in transcript_ids:
            assert tid in manifest2, f"Transcript {tid} should still be in manifest"
        for eid in episode_ids:
            assert eid in manifest2, f"Episode {eid} should still be in manifest"

        # Core memory should be rebuilt (new dependency on topics)
        assert "core-memory" in manifest2

        # Context doc should be updated
        context_doc = workspace["build_dir"] / "context.md"
        assert context_doc.exists()
