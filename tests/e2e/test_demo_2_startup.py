"""Demo 2: Startup Financial Pipeline — E2E tests.

Tests the financial advisor pipeline using the demo_2_startup corpus (50 conversations, 10 customers).
Exercises: build, plan, search, verify, cache behavior.
Tests requiring parallel pipeline paths or model-per-step are marked as skipped.
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

CORPUS_DIR = Path(__file__).parent.parent / "fixtures" / "corpus" / "demo_2_startup"


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace with demo_2 corpus and a build dir."""
    source_dir = tmp_path / "exports"
    source_dir.mkdir()
    build_dir = tmp_path / "build"

    # Copy demo_2 corpus
    shutil.copy(CORPUS_DIR / "chatgpt_export.json", source_dir / "chatgpt_export.json")

    return {"root": tmp_path, "source_dir": source_dir, "build_dir": build_dir}


@pytest.fixture
def financial_pipeline_file(workspace):
    """Write a financial profile pipeline into the workspace."""
    path = workspace["root"] / "pipeline_financial.py"
    path.write_text(f"""
from synix import Pipeline, Layer, Projection

pipeline = Pipeline("demo2-financial")
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
def financial_pipeline_v2_file(workspace):
    """Write a modified financial pipeline (prompt change at level 2)."""
    path = workspace["root"] / "pipeline_financial_v2.py"
    path.write_text(f"""
from synix import Pipeline, Layer, Projection

pipeline = Pipeline("demo2-financial-v2")
pipeline.source_dir = "{workspace["source_dir"]}"
pipeline.build_dir = "{workspace["build_dir"]}"
pipeline.llm_config = {{"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}}

pipeline.add_layer(Layer(name="transcripts", level=0, transform="parse"))
pipeline.add_layer(Layer(name="episodes", level=1, depends_on=["transcripts"], transform="episode_summary", grouping="by_conversation"))
pipeline.add_layer(Layer(name="topics", level=2, depends_on=["episodes"], transform="topical_rollup", grouping="by_topic", config={{
    "topics": ["investment-strategy", "risk-management", "retirement-planning", "debt-reduction"],
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
                "This conversation covered financial advisory topics including "
                "investment portfolio management, risk assessment, and retirement "
                "planning. The client discussed their goals and concerns."
            )
        elif "monthly" in content.lower():
            return _mock_response(
                "This month's financial advisory sessions focused on portfolio "
                "diversification, market volatility response, and long-term "
                "retirement strategy adjustments across multiple clients."
            )
        elif "topic" in content.lower():
            return _mock_response(
                "Regarding this financial topic, clients showed varied risk "
                "tolerances and investment strategies. Common themes included "
                "concern about market downturns and interest in passive index funds."
            )
        elif "core memory" in content.lower():
            return _mock_response(
                "## Client Base Overview\nThis financial advisory practice "
                "serves 10 clients with diverse financial goals.\n\n"
                "## Key Patterns\nClients generally prefer conservative "
                "to moderate risk profiles.\n\n"
                "## Active Concerns\nMarket volatility and retirement "
                "readiness are top concerns across the client base."
            )
        return _mock_response("Mock financial advisory response.")

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
# DT-2.1: Fresh build — Financial profile pipeline
# ---------------------------------------------------------------------------


class TestDT2FreshBuild:
    """DT-2.1: Fresh build with 50-conversation financial corpus."""

    def test_plan_shows_all_layers_as_new(self, runner, workspace, financial_pipeline_file):
        """synix plan on a fresh workspace shows all layers need building."""
        result = runner.invoke(
            main,
            [
                "plan",
                str(financial_pipeline_file),
            ],
        )
        assert result.exit_code == 0, f"Plan failed: {result.output}"
        assert "Estimated:" in result.output

    def test_fresh_build_produces_correct_artifact_counts(self, runner, workspace, financial_pipeline_file):
        """First build produces: 50 transcripts, 50 episodes, N monthly, 1 core."""
        result = runner.invoke(main, ["build", str(financial_pipeline_file)])
        assert result.exit_code == 0, f"Build failed: {result.output}"
        assert "Build Summary" in result.output

        manifest = json.loads((workspace["build_dir"] / "manifest.json").read_text())

        # Count by layer
        layers: dict[str, int] = {}
        for _aid, info in manifest.items():
            layer = info.get("layer", "unknown")
            layers[layer] = layers.get(layer, 0) + 1

        # 50 chatgpt conversations = 50 transcripts
        assert layers.get("transcripts", 0) == 50
        # 50 transcripts → 50 episodes (1 per conversation)
        assert layers.get("episodes", 0) == 50
        # Monthly rollups — at least 1
        assert layers.get("monthly", 0) >= 1
        # Single core memory
        assert layers.get("core", 0) == 1

    def test_verify_passes_after_fresh_build(self, runner, workspace, financial_pipeline_file):
        """synix verify passes on a clean build."""
        runner.invoke(main, ["build", str(financial_pipeline_file)])

        from synix.build.verify import verify_build

        result = verify_build(str(workspace["build_dir"]))
        assert result.passed, f"Verify failed: {result.summary}. Details: {[c.message for c in result.failed_checks]}"

    def test_search_index_populated_after_build(self, runner, workspace, financial_pipeline_file):
        """Search returns results for financial content."""
        runner.invoke(main, ["build", str(financial_pipeline_file)])

        search_db = workspace["build_dir"] / "search.db"
        assert search_db.exists()

        result = runner.invoke(main, ["search", "investment", "--build-dir", str(workspace["build_dir"])])
        assert result.exit_code == 0

    def test_context_doc_created(self, runner, workspace, financial_pipeline_file):
        """Context doc contains the core memory synthesis."""
        runner.invoke(main, ["build", str(financial_pipeline_file)])

        context_doc = workspace["build_dir"] / "context.md"
        assert context_doc.exists()
        content = context_doc.read_text()
        assert len(content) > 0

    def test_all_derived_artifacts_have_provenance(self, runner, workspace, financial_pipeline_file):
        """Every non-transcript artifact has provenance records."""
        runner.invoke(main, ["build", str(financial_pipeline_file)])

        provenance_path = workspace["build_dir"] / "provenance.json"
        assert provenance_path.exists()
        provenance = json.loads(provenance_path.read_text())

        manifest = json.loads((workspace["build_dir"] / "manifest.json").read_text())
        derived = {aid for aid, info in manifest.items() if info.get("layer") != "transcripts"}
        missing = [aid for aid in derived if aid not in provenance]
        assert not missing, f"Missing provenance for: {missing}"


# ---------------------------------------------------------------------------
# DT-2.2: Prompt/config change — preserve extractions
# ---------------------------------------------------------------------------


class TestDT2ConfigChange:
    """DT-2.2: Config change preserves episodes, rebuilds upper layers."""

    def test_config_change_caches_transcripts_and_episodes(
        self, runner, workspace, financial_pipeline_file, financial_pipeline_v2_file, mock_anthropic
    ):
        """After monthly build, topical rebuild reuses transcripts+episodes."""
        # First build: monthly
        result1 = runner.invoke(main, ["build", str(financial_pipeline_file)])
        assert result1.exit_code == 0
        calls_after_first = mock_anthropic["n"]

        manifest1 = json.loads((workspace["build_dir"] / "manifest.json").read_text())
        transcript_ids = {aid for aid, info in manifest1.items() if info["layer"] == "transcripts"}
        episode_ids = {aid for aid, info in manifest1.items() if info["layer"] == "episodes"}

        # Second build: topical
        result2 = runner.invoke(main, ["build", str(financial_pipeline_v2_file)])
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

    def test_verify_passes_after_config_change(
        self, runner, workspace, financial_pipeline_file, financial_pipeline_v2_file
    ):
        """Verify still passes after config change rebuild."""
        runner.invoke(main, ["build", str(financial_pipeline_file)])
        runner.invoke(main, ["build", str(financial_pipeline_v2_file)])

        from synix.build.verify import verify_build

        result = verify_build(str(workspace["build_dir"]))
        assert result.passed, f"Verify failed: {result.summary}"


# ---------------------------------------------------------------------------
# DT-2.3–2.6: Parallel pipeline paths (requires unbuilt features)
# ---------------------------------------------------------------------------


class TestDT2ParallelPaths:
    """DT-2.3–2.6: Parallel pipeline paths — two pipelines sharing source_dir/build_dir.

    Path A uses monthly rollups (financial_pipeline_file).
    Path B uses topical rollups (financial_pipeline_v2_file).
    Both share the same transcript and episode layers but diverge at level 2+.
    """

    def test_parallel_paths_source_sharing(
        self, runner, workspace, financial_pipeline_file, financial_pipeline_v2_file
    ):
        """Both pipeline paths share the same source (transcript + episode) artifacts."""
        # Build path A (monthly rollups)
        result_a = runner.invoke(main, ["build", str(financial_pipeline_file)])
        assert result_a.exit_code == 0, f"Path A build failed: {result_a.output}"

        manifest_a = json.loads((workspace["build_dir"] / "manifest.json").read_text())
        transcript_ids_a = {aid for aid, info in manifest_a.items() if info["layer"] == "transcripts"}
        episode_ids_a = {aid for aid, info in manifest_a.items() if info["layer"] == "episodes"}
        assert len(transcript_ids_a) == 50, "Path A should have 50 transcripts"
        assert len(episode_ids_a) == 50, "Path A should have 50 episodes"

        # Build path B (topical rollups) — shares same build_dir and source_dir
        result_b = runner.invoke(main, ["build", str(financial_pipeline_v2_file)])
        assert result_b.exit_code == 0, f"Path B build failed: {result_b.output}"

        manifest_b = json.loads((workspace["build_dir"] / "manifest.json").read_text())
        transcript_ids_b = {aid for aid, info in manifest_b.items() if info["layer"] == "transcripts"}
        episode_ids_b = {aid for aid, info in manifest_b.items() if info["layer"] == "episodes"}

        # All transcript and episode IDs from path A must still exist in path B's manifest
        for tid in transcript_ids_a:
            assert tid in manifest_b, f"Transcript {tid} from path A missing after path B build"
        for eid in episode_ids_a:
            assert eid in manifest_b, f"Episode {eid} from path A missing after path B build"

        # Path B should have "topics" layer artifacts
        layers_b = {info["layer"] for info in manifest_b.values()}
        assert "topics" in layers_b, "Path B should have 'topics' layer"

        # Source layers (transcripts + episodes) are identical across paths
        assert transcript_ids_a == transcript_ids_b, "Transcript IDs should be identical across paths"
        assert episode_ids_a == episode_ids_b, "Episode IDs should be identical across paths"

    def test_independent_search_indexes(self, runner, workspace, financial_pipeline_file, financial_pipeline_v2_file):
        """Each pipeline config produces a search index reflecting its own layer structure."""
        # Build path A (monthly rollups) and search
        result_a = runner.invoke(main, ["build", str(financial_pipeline_file)])
        assert result_a.exit_code == 0, f"Path A build failed: {result_a.output}"

        from synix.build.provenance import ProvenanceTracker
        from synix.search.indexer import SearchIndexProjection

        # Query path A search index — should contain "monthly" layer results.
        # Use "portfolio" which appears in the monthly mock response:
        # "portfolio diversification, market volatility response..."
        proj_a = SearchIndexProjection(workspace["build_dir"])
        prov_a = ProvenanceTracker(workspace["build_dir"])
        results_a = proj_a.query("portfolio", provenance_tracker=prov_a)
        proj_a.close()

        layers_in_results_a = {r.layer_name for r in results_a}
        assert "monthly" in layers_in_results_a, (
            f"Path A search results should include 'monthly' layer, got: {layers_in_results_a}"
        )
        assert "topics" not in layers_in_results_a, "Path A search results should NOT include 'topics' layer"

        # Build path B (topical rollups) — rebuilds search index with topics instead of monthly
        result_b = runner.invoke(main, ["build", str(financial_pipeline_v2_file)])
        assert result_b.exit_code == 0, f"Path B build failed: {result_b.output}"

        # Query path B search index — should contain "topics" layer results, not "monthly".
        # Use "tolerances" which appears in the topical mock response:
        # "clients showed varied risk tolerances and investment strategies"
        proj_b = SearchIndexProjection(workspace["build_dir"])
        prov_b = ProvenanceTracker(workspace["build_dir"])
        results_b = proj_b.query("tolerances", provenance_tracker=prov_b)
        proj_b.close()

        layers_in_results_b = {r.layer_name for r in results_b}
        assert "topics" in layers_in_results_b, (
            f"Path B search results should include 'topics' layer, got: {layers_in_results_b}"
        )
        assert "monthly" not in layers_in_results_b, "Path B search results should NOT include 'monthly' layer"

    def test_cross_path_artifact_diffing(self, runner, workspace, financial_pipeline_file, financial_pipeline_v2_file):
        """Core memory artifacts differ between monthly and topical pipeline paths."""
        from synix.build.artifacts import ArtifactStore
        from synix.build.diff import diff_artifact

        # Build path A (monthly rollups) — save core memory content
        result_a = runner.invoke(main, ["build", str(financial_pipeline_file)])
        assert result_a.exit_code == 0, f"Path A build failed: {result_a.output}"

        store_a = ArtifactStore(workspace["build_dir"])
        core_artifacts_a = store_a.list_artifacts("core")
        assert len(core_artifacts_a) == 1, "Path A should have exactly 1 core artifact"
        core_a = core_artifacts_a[0]
        core_a_content = core_a.content

        # Also snapshot the context.md
        context_a = (workspace["build_dir"] / "context.md").read_text()

        # Build path B (topical rollups) — core memory is rebuilt with different inputs
        result_b = runner.invoke(main, ["build", str(financial_pipeline_v2_file)])
        assert result_b.exit_code == 0, f"Path B build failed: {result_b.output}"

        store_b = ArtifactStore(workspace["build_dir"])
        core_artifacts_b = store_b.list_artifacts("core")
        assert len(core_artifacts_b) == 1, "Path B should have exactly 1 core artifact"
        core_b = core_artifacts_b[0]

        # The core artifacts should differ because they were synthesized from
        # different level-2 inputs (monthly rollups vs topical rollups).
        # Even with mocked LLM, the input_hashes differ, so they get rebuilt.
        artifact_diff = diff_artifact(core_a, core_b)

        # Core was rebuilt, so at minimum input_hashes differ (different parents)
        assert core_a.input_hashes != core_b.input_hashes, (
            "Core input_hashes should differ between monthly and topical paths"
        )

        # Context doc should also reflect the rebuild (content may or may not differ
        # with mocked LLM, but the file was rewritten)
        context_b = (workspace["build_dir"] / "context.md").read_text()
        assert len(context_b) > 0, "Path B context.md should have content"

    def test_incremental_update_both_paths(
        self, runner, workspace, financial_pipeline_file, financial_pipeline_v2_file, mock_anthropic
    ):
        """Both pipeline paths produce the same base artifacts, different upper layers."""
        # Build path A (monthly)
        result_a = runner.invoke(main, ["build", str(financial_pipeline_file)])
        assert result_a.exit_code == 0, f"Path A build failed: {result_a.output}"

        manifest_a = json.loads((workspace["build_dir"] / "manifest.json").read_text())
        layers_a: dict[str, int] = {}
        for _aid, info in manifest_a.items():
            layer = info.get("layer", "unknown")
            layers_a[layer] = layers_a.get(layer, 0) + 1

        # Path A: 50 transcripts, 50 episodes, N monthly, 1 core
        assert layers_a.get("transcripts", 0) == 50
        assert layers_a.get("episodes", 0) == 50
        assert layers_a.get("monthly", 0) >= 1
        assert layers_a.get("core", 0) == 1

        calls_after_a = mock_anthropic["n"]

        # Build path B (topical) — reuses transcripts + episodes
        result_b = runner.invoke(main, ["build", str(financial_pipeline_v2_file)])
        assert result_b.exit_code == 0, f"Path B build failed: {result_b.output}"

        calls_after_b = mock_anthropic["n"]

        manifest_b = json.loads((workspace["build_dir"] / "manifest.json").read_text())
        layers_b: dict[str, int] = {}
        for _aid, info in manifest_b.items():
            layer = info.get("layer", "unknown")
            layers_b[layer] = layers_b.get(layer, 0) + 1

        # Path B: same 50 transcripts, same 50 episodes, topics instead of monthly, 1 core
        assert layers_b.get("transcripts", 0) == 50
        assert layers_b.get("episodes", 0) == 50
        assert layers_b.get("topics", 0) >= 1, "Path B should have topics layer artifacts"
        assert layers_b.get("core", 0) == 1

        # Path B should make fewer LLM calls than path A because transcripts + episodes
        # are cached (only topics + core need LLM calls)
        new_calls_b = calls_after_b - calls_after_a
        # Path A made LLM calls for: episodes (50) + monthly (N) + core (1) = 50+
        # Path B should only call LLM for: topics (4) + core (1) = 5
        assert new_calls_b < calls_after_a, (
            f"Path B should make fewer LLM calls than path A. Path A: {calls_after_a}, Path B new calls: {new_calls_b}"
        )


# ---------------------------------------------------------------------------
# DT-2 Cache: No-change rebuild
# ---------------------------------------------------------------------------


class TestDT2CacheHit:
    """No-change rebuild should be fully cached."""

    def test_second_run_uses_cache(self, runner, workspace, financial_pipeline_file, mock_anthropic):
        """Second identical run should not make new LLM calls."""
        # First build
        result1 = runner.invoke(main, ["build", str(financial_pipeline_file)])
        assert result1.exit_code == 0
        calls_after_first = mock_anthropic["n"]

        # Second build (same config, same data)
        result2 = runner.invoke(main, ["build", str(financial_pipeline_file)])
        assert result2.exit_code == 0
        calls_after_second = mock_anthropic["n"]

        # Second run should make 0 new LLM calls
        assert calls_after_second == calls_after_first, (
            f"Expected no new LLM calls on cached rebuild. First: {calls_after_first}, Second: {calls_after_second}"
        )

    def test_plan_shows_all_cached_on_second_run(self, runner, workspace, financial_pipeline_file):
        """Plan after a clean build shows everything as cached."""
        runner.invoke(main, ["build", str(financial_pipeline_file)])

        result = runner.invoke(
            main,
            [
                "plan",
                str(financial_pipeline_file),
            ],
        )
        assert result.exit_code == 0
        assert "cached" in result.output.lower()
