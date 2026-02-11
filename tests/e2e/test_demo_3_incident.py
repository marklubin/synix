"""Demo 3: Incident Response Pipeline — E2E tests.

Tests the incident response pipeline using the demo_3_incident corpus (100 support
conversations across 20 customers). Exercises: build, plan, verify, search, cache
behavior, and the merge transform (cross-customer merge contamination and fix flow).
"""

from __future__ import annotations

import hashlib
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

CORPUS_DIR = Path(__file__).parent.parent / "fixtures" / "corpus" / "demo_3_incident"


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace with demo_3 corpus and a build dir."""
    source_dir = tmp_path / "exports"
    source_dir.mkdir()
    build_dir = tmp_path / "build"

    # Copy demo_3 corpus (only chatgpt_export.json exists for this corpus)
    shutil.copy(CORPUS_DIR / "chatgpt_export.json", source_dir / "chatgpt_export.json")

    return {"root": tmp_path, "source_dir": source_dir, "build_dir": build_dir}


@pytest.fixture
def incident_pipeline_file(workspace):
    """Write an incident pipeline.py into the workspace."""
    path = workspace["root"] / "pipeline_incident.py"
    path.write_text(f"""
from synix import Pipeline, Layer, Projection

pipeline = Pipeline("demo3-incident")
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


@pytest.fixture(autouse=True)
def mock_anthropic(monkeypatch):
    """Mock Anthropic API for all E2E tests."""
    call_count = {"n": 0}

    def mock_create(**kwargs):
        call_count["n"] += 1
        messages = kwargs.get("messages", [])
        content = messages[0].get("content", "") if messages else ""

        if "summarizing a conversation" in content.lower():
            # Produce a deterministic but unique summary per conversation.
            # Use a short hash of the prompt to differentiate episodes so that
            # the merge transform and provenance tracking work correctly with
            # different content hashes per episode.
            prompt_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
            return _mock_response(
                f"This support conversation ({prompt_hash}) involved a customer "
                "reaching out about a platform issue. The agent assisted with "
                "troubleshooting, covering billing inquiries, shipping status, "
                "account configuration, and API integration. The issue was "
                "resolved with step-by-step guidance."
            )
        elif "monthly" in content.lower():
            return _mock_response(
                "This month saw a high volume of support tickets covering billing "
                "disputes, shipping delays, account access issues, and API integration "
                "problems. Recurring themes include PCI compliance questions, webhook "
                "configuration difficulties, and multi-currency support requests. "
                "Several customers reported issues with inventory sync and tax "
                "configuration across different states."
            )
        elif "core memory" in content.lower():
            return _mock_response(
                "## Support Overview\n"
                "This is a customer support knowledge base covering 20 customers.\n\n"
                "## Common Issues\n"
                "Billing disputes, shipping delays, account access, API integration.\n\n"
                "## Recurring Themes\n"
                "PCI compliance, webhook configuration, multi-currency, inventory sync.\n\n"
                "## Resolution Patterns\n"
                "Most issues resolved via settings adjustments or documentation referral."
            )
        return _mock_response("Mock support response.")

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
# DT-3.1: Fresh build of the 100-conversation corpus
# ---------------------------------------------------------------------------


class TestDT3FreshBuild:
    """DT-3.1: Fresh build with plan verification on incident corpus."""

    def test_plan_shows_all_layers_as_new(self, runner, workspace, incident_pipeline_file):
        """synix plan on a fresh workspace shows all layers need building."""
        result = runner.invoke(
            main,
            [
                "plan",
                str(incident_pipeline_file),
            ],
        )
        assert result.exit_code == 0, f"Plan failed: {result.output}"
        assert "Estimated:" in result.output

    def test_fresh_build_produces_correct_artifact_counts(self, runner, workspace, incident_pipeline_file):
        """First build produces: 100 transcripts, 100 episodes, 2 monthly rollups, 1 core."""
        result = runner.invoke(main, ["build", str(incident_pipeline_file)])
        assert result.exit_code == 0, f"Build failed: {result.output}"
        assert "Build Summary" in result.output

        manifest = json.loads((workspace["build_dir"] / "manifest.json").read_text())

        # Count by layer
        layers: dict[str, int] = {}
        for _aid, info in manifest.items():
            layer = info.get("layer", "unknown")
            layers[layer] = layers.get(layer, 0) + 1

        # 100 chatgpt conversations = 100 transcripts
        assert layers.get("transcripts", 0) == 100
        # 100 transcripts -> 100 episodes (1 per conversation)
        assert layers.get("episodes", 0) == 100
        # Conversations span 2025-11 and 2025-12 = at least 1 monthly rollup
        assert layers.get("monthly", 0) > 0
        # Single core memory
        assert layers.get("core", 0) == 1

    def test_verify_passes_after_fresh_build(self, runner, workspace, incident_pipeline_file):
        """synix verify passes on a clean build."""
        runner.invoke(main, ["build", str(incident_pipeline_file)])

        from synix.build.verify import verify_build

        result = verify_build(str(workspace["build_dir"]))
        assert result.passed, f"Verify failed: {result.summary}. Details: {[c.message for c in result.failed_checks]}"

    def test_search_returns_results(self, runner, workspace, incident_pipeline_file):
        """Search returns results for support-domain content after build."""
        runner.invoke(main, ["build", str(incident_pipeline_file)])

        search_db = workspace["build_dir"] / "search.db"
        assert search_db.exists()

        result = runner.invoke(main, ["search", "billing", "--build-dir", str(workspace["build_dir"])])
        assert result.exit_code == 0

    def test_all_derived_artifacts_have_provenance(self, runner, workspace, incident_pipeline_file):
        """Every non-transcript artifact has provenance records."""
        runner.invoke(main, ["build", str(incident_pipeline_file)])

        provenance_path = workspace["build_dir"] / "provenance.json"
        assert provenance_path.exists()
        provenance = json.loads(provenance_path.read_text())

        manifest = json.loads((workspace["build_dir"] / "manifest.json").read_text())
        derived = {aid for aid, info in manifest.items() if info.get("layer") != "transcripts"}
        missing = [aid for aid in derived if aid not in provenance]
        assert not missing, f"Missing provenance for: {missing}"


# ---------------------------------------------------------------------------
# DT-3.2: Cache behavior — identical re-run is fully cached
# ---------------------------------------------------------------------------


class TestDT3CacheBehavior:
    """DT-3.2: No-change rebuild should be fully cached."""

    def test_second_run_fully_cached(self, runner, workspace, incident_pipeline_file, mock_anthropic):
        """Second identical run should not make new LLM calls."""
        # First build
        result1 = runner.invoke(main, ["build", str(incident_pipeline_file)])
        assert result1.exit_code == 0
        calls_after_first = mock_anthropic["n"]

        # Second build (same config, same data)
        result2 = runner.invoke(main, ["build", str(incident_pipeline_file)])
        assert result2.exit_code == 0
        calls_after_second = mock_anthropic["n"]

        # Second run should make 0 new LLM calls
        assert calls_after_second == calls_after_first, (
            f"Expected no new LLM calls on cached rebuild. First: {calls_after_first}, Second: {calls_after_second}"
        )


# ---------------------------------------------------------------------------
# DT-3.3: Merge transform tests — cross-customer merge contamination & fix
# ---------------------------------------------------------------------------


@pytest.fixture
def merge_pipeline_file(workspace):
    """Write a merge pipeline with LOW threshold and NO constraints.

    This intentionally causes cross-customer contamination: all episodes have
    identical mock content, so the merge transform (threshold=0.3) merges
    episodes across different customers into the same artifact.
    """
    path = workspace["root"] / "pipeline_merge.py"
    path.write_text(f"""
from synix import Pipeline, Layer, Projection

pipeline = Pipeline("demo3-merge-bad")
pipeline.source_dir = "{workspace["source_dir"]}"
pipeline.build_dir = "{workspace["build_dir"]}"
pipeline.llm_config = {{"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}}

pipeline.add_layer(Layer(name="transcripts", level=0, transform="parse"))
pipeline.add_layer(Layer(name="episodes", level=1, depends_on=["transcripts"], transform="episode_summary", grouping="by_conversation"))
pipeline.add_layer(Layer(name="merged", level=2, depends_on=["episodes"], transform="merge", config={{"similarity_threshold": 0.3}}))
pipeline.add_layer(Layer(name="core", level=3, depends_on=["merged"], transform="core_synthesis", grouping="single", context_budget=10000))

pipeline.add_projection(Projection(name="memory-index", projection_type="search_index", sources=[
    {{"layer": "episodes", "search": ["fulltext"]}},
    {{"layer": "merged", "search": ["fulltext"]}},
    {{"layer": "core", "search": ["fulltext"]}},
]))
pipeline.add_projection(Projection(name="context-doc", projection_type="flat_file", sources=[{{"layer": "core"}}], config={{"output_path": "{workspace["build_dir"] / "context.md"}"}}))
""")
    return path


@pytest.fixture
def merge_pipeline_fixed_file(workspace):
    """Write a merge pipeline with constraint preventing cross-customer merges.

    Same as merge_pipeline_file but adds a constraint that prevents merging
    records with different customer_id values.
    """
    path = workspace["root"] / "pipeline_merge_fixed.py"
    path.write_text(f"""
from synix import Pipeline, Layer, Projection

pipeline = Pipeline("demo3-merge-fixed")
pipeline.source_dir = "{workspace["source_dir"]}"
pipeline.build_dir = "{workspace["build_dir"]}"
pipeline.llm_config = {{"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}}

pipeline.add_layer(Layer(name="transcripts", level=0, transform="parse"))
pipeline.add_layer(Layer(name="episodes", level=1, depends_on=["transcripts"], transform="episode_summary", grouping="by_conversation"))
pipeline.add_layer(Layer(name="merged", level=2, depends_on=["episodes"], transform="merge", config={{"similarity_threshold": 0.3, "constraints": ["NEVER merge records with different customer_id"]}}))
pipeline.add_layer(Layer(name="core", level=3, depends_on=["merged"], transform="core_synthesis", grouping="single", context_budget=10000))

pipeline.add_projection(Projection(name="memory-index", projection_type="search_index", sources=[
    {{"layer": "episodes", "search": ["fulltext"]}},
    {{"layer": "merged", "search": ["fulltext"]}},
    {{"layer": "core", "search": ["fulltext"]}},
]))
pipeline.add_projection(Projection(name="context-doc", projection_type="flat_file", sources=[{{"layer": "core"}}], config={{"output_path": "{workspace["build_dir"] / "context.md"}"}}))
""")
    return path


class TestDT3MergeTransform:
    """DT-3.3: Merge transform tests for cross-customer contamination scenario.

    The scenario: a merge layer with low threshold and no constraints accidentally
    combines episode data across customers. The tests verify that provenance
    tracking detects the contamination, that verify catches it, and that fixing
    the config triggers an incremental rebuild of only the affected layers.
    """

    def test_build_with_merge_layer(self, runner, workspace, merge_pipeline_file):
        """Build pipeline with a merge layer — produces merge artifacts with cross-customer data."""
        result = runner.invoke(main, ["build", str(merge_pipeline_file)])
        assert result.exit_code == 0, f"Build failed: {result.output}"

        manifest = json.loads((workspace["build_dir"] / "manifest.json").read_text())

        # Count by layer
        layers: dict[str, int] = {}
        for _aid, info in manifest.items():
            layer = info.get("layer", "unknown")
            layers[layer] = layers.get(layer, 0) + 1

        assert layers.get("transcripts", 0) == 100
        assert layers.get("episodes", 0) == 100
        assert layers.get("merged", 0) >= 1, "Expected at least one merge artifact"
        assert layers.get("core", 0) == 1

        # Verify merge artifacts exist (IDs start with "merge-" or type is "merge")
        merge_aids = [
            aid for aid, info in manifest.items() if aid.startswith("merge-") or info.get("artifact_type") == "merge"
        ]
        assert len(merge_aids) >= 1, "No merge artifacts found in manifest"

        # Verify at least one merge artifact contains content from multiple customers
        from synix.build.artifacts import ArtifactStore

        store = ArtifactStore(workspace["build_dir"])
        found_multi_customer = False
        for merge_id in merge_aids:
            art = store.load_artifact(merge_id)
            if art is None:
                continue
            source_customers = art.metadata.get("source_customer_ids", [])
            if len(source_customers) > 1:
                found_multi_customer = True
                break
        assert found_multi_customer, "Expected at least one merge artifact with data from multiple customers"

    def test_provenance_trace_reveals_bad_merge(self, runner, workspace, merge_pipeline_file):
        """Provenance chain from a merged artifact reveals cross-customer contamination."""
        runner.invoke(main, ["build", str(merge_pipeline_file)])

        from synix.build.artifacts import ArtifactStore
        from synix.build.provenance import ProvenanceTracker

        store = ArtifactStore(workspace["build_dir"])
        tracker = ProvenanceTracker(workspace["build_dir"])
        manifest = json.loads((workspace["build_dir"] / "manifest.json").read_text())

        # Find a merge artifact with multiple customers
        merge_aids = [aid for aid, info in manifest.items() if aid.startswith("merge-")]
        assert merge_aids, "No merge artifacts found"

        # Get provenance parents for a multi-customer merge artifact
        contaminated_merge_id = None
        for merge_id in merge_aids:
            art = store.load_artifact(merge_id)
            if art and len(art.metadata.get("source_customer_ids", [])) > 1:
                contaminated_merge_id = merge_id
                break

        assert contaminated_merge_id is not None, "No merge artifact found with cross-customer contamination"

        # Get the provenance parents of this merge artifact
        parent_ids = tracker.get_parents(contaminated_merge_id)
        assert len(parent_ids) > 1, f"Expected merge artifact to have multiple parents, got {len(parent_ids)}"

        # Check that parent artifacts map to different customer_ids
        customer_ids_in_parents: set[str] = set()
        for parent_id in parent_ids:
            parent = store.load_artifact(parent_id)
            if parent and parent.metadata.get("customer_id"):
                customer_ids_in_parents.add(parent.metadata["customer_id"])

        assert len(customer_ids_in_parents) > 1, (
            f"Expected parents to span multiple customers, got: {customer_ids_in_parents}"
        )

    def test_verify_merge_integrity_check(self, runner, workspace, merge_pipeline_file):
        """Verify detects integrity violations in cross-customer merged artifacts."""
        runner.invoke(main, ["build", str(merge_pipeline_file)])

        from synix.build.verify import verify_build

        result = verify_build(str(workspace["build_dir"]), checks=["merge_integrity"])
        assert not result.passed, (
            f"merge_integrity check should FAIL because merge artifacts span multiple customers. Got: {result.summary}"
        )

        # The failed check should mention cross-customer data
        failed = result.failed_checks
        assert len(failed) == 1
        assert failed[0].name == "merge_integrity"
        assert "cross-customer" in failed[0].message.lower() or "customer" in failed[0].message.lower()

    def test_fix_config_incremental_rebuild(
        self, runner, workspace, merge_pipeline_file, merge_pipeline_fixed_file, mock_anthropic
    ):
        """Fixing the merge config triggers rebuild of only merge+downstream layers."""
        # First build: bad merge pipeline
        result1 = runner.invoke(main, ["build", str(merge_pipeline_file)])
        assert result1.exit_code == 0, f"First build failed: {result1.output}"

        manifest_before = json.loads((workspace["build_dir"] / "manifest.json").read_text())
        transcript_count_before = sum(1 for info in manifest_before.values() if info.get("layer") == "transcripts")
        episode_count_before = sum(1 for info in manifest_before.values() if info.get("layer") == "episodes")
        calls_after_first = mock_anthropic["n"]

        # Second build: fixed merge pipeline (adds constraint)
        result2 = runner.invoke(main, ["build", str(merge_pipeline_fixed_file)])
        assert result2.exit_code == 0, f"Second build failed: {result2.output}"

        manifest_after = json.loads((workspace["build_dir"] / "manifest.json").read_text())
        transcript_count_after = sum(1 for info in manifest_after.values() if info.get("layer") == "transcripts")
        episode_count_after = sum(1 for info in manifest_after.values() if info.get("layer") == "episodes")

        # Transcripts and episodes should be cached (same counts)
        assert transcript_count_after == transcript_count_before, (
            f"Transcript count changed: {transcript_count_before} -> {transcript_count_after}"
        )
        assert episode_count_after == episode_count_before, (
            f"Episode count changed: {episode_count_before} -> {episode_count_after}"
        )

        # Merge layer should be rebuilt (constraint changed -> different cache key)
        merge_count_after = sum(1 for info in manifest_after.values() if info.get("layer") == "merged")
        assert merge_count_after >= 1, "Expected merge artifacts after fix"

        # Core should also be rebuilt (dependency on merge changed)
        assert sum(1 for info in manifest_after.values() if info.get("layer") == "core") == 1

        # The fixed pipeline should produce no cross-customer merges
        from synix.build.verify import verify_build

        verify_result = verify_build(str(workspace["build_dir"]), checks=["merge_integrity"])
        assert verify_result.passed, (
            f"Fixed pipeline should pass merge_integrity. Got: {verify_result.summary}. "
            f"Details: {[c.message for c in verify_result.failed_checks]}"
        )

    def test_post_fix_no_collateral_damage(self, runner, workspace, merge_pipeline_file, merge_pipeline_fixed_file):
        """After fix, transcript and episode artifact IDs are preserved — no collateral damage."""
        # First build: bad merge pipeline
        runner.invoke(main, ["build", str(merge_pipeline_file)])

        manifest_before = json.loads((workspace["build_dir"] / "manifest.json").read_text())
        transcript_ids_before = {aid for aid, info in manifest_before.items() if info.get("layer") == "transcripts"}
        episode_ids_before = {aid for aid, info in manifest_before.items() if info.get("layer") == "episodes"}

        # Second build: fixed merge pipeline
        runner.invoke(main, ["build", str(merge_pipeline_fixed_file)])

        manifest_after = json.loads((workspace["build_dir"] / "manifest.json").read_text())
        transcript_ids_after = {aid for aid, info in manifest_after.items() if info.get("layer") == "transcripts"}
        episode_ids_after = {aid for aid, info in manifest_after.items() if info.get("layer") == "episodes"}

        # All original transcript IDs should still be present
        missing_transcripts = transcript_ids_before - transcript_ids_after
        assert not missing_transcripts, (
            f"Fix caused collateral damage: {len(missing_transcripts)} transcript(s) lost: "
            f"{sorted(list(missing_transcripts))[:5]}"
        )

        # All original episode IDs should still be present
        missing_episodes = episode_ids_before - episode_ids_after
        assert not missing_episodes, (
            f"Fix caused collateral damage: {len(missing_episodes)} episode(s) lost: "
            f"{sorted(list(missing_episodes))[:5]}"
        )

    def test_incident_timeline_reconstruction(self, runner, workspace, merge_pipeline_file):
        """Reconstruct incident timeline: full chain from merge -> episodes -> transcripts."""
        runner.invoke(main, ["build", str(merge_pipeline_file)])

        from synix.build.artifacts import ArtifactStore
        from synix.build.provenance import ProvenanceTracker

        store = ArtifactStore(workspace["build_dir"])
        tracker = ProvenanceTracker(workspace["build_dir"])
        manifest = json.loads((workspace["build_dir"] / "manifest.json").read_text())

        # Find a merge artifact that spans customers
        merge_aids = [aid for aid, info in manifest.items() if aid.startswith("merge-")]
        assert merge_aids, "No merge artifacts found"

        contaminated_id = None
        for merge_id in merge_aids:
            art = store.load_artifact(merge_id)
            if art and len(art.metadata.get("source_customer_ids", [])) > 1:
                contaminated_id = merge_id
                break

        assert contaminated_id is not None, "No cross-customer merge artifact found"

        # Walk the full provenance chain
        chain = tracker.get_chain(contaminated_id)
        assert len(chain) >= 1, "Provenance chain should not be empty"

        # The chain should reach transcript-level artifacts
        chain_artifact_ids = {rec.artifact_id for rec in chain}
        # Also collect all parent IDs referenced in the chain
        all_referenced_ids: set[str] = set()
        for rec in chain:
            all_referenced_ids.add(rec.artifact_id)
            all_referenced_ids.update(rec.parent_artifact_ids)

        # Verify that some of the referenced IDs are transcript-level artifacts
        transcript_ids_in_chain = {
            aid for aid in all_referenced_ids if aid in manifest and manifest[aid].get("layer") == "transcripts"
        }
        assert transcript_ids_in_chain, (
            f"Provenance chain should trace back to transcript-level artifacts. "
            f"Chain artifact IDs: {sorted(chain_artifact_ids)}"
        )
