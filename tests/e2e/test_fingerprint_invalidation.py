"""E2E tests for fingerprint-based cache invalidation.

Tests the full flow: build → verify fingerprints → modify → detect invalidation.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from synix.build.fingerprint import Fingerprint
from synix.build.refs import synix_dir_for_build_dir
from synix.build.snapshot_view import SnapshotArtifactCache
from synix.cli import main


def _tamper_snapshot_artifacts(build_dir, mutate_fn):
    """Walk snapshot objects, apply mutate_fn, and rebuild the snapshot chain.

    mutate_fn(art_obj) -> bool: return True if modified.

    Because the object store is content-addressed, modifying an artifact changes
    its oid.  We must re-store the modified artifact as a new object, update the
    manifest entry to point at the new oid, re-store the manifest, re-store the
    snapshot, and advance the HEAD ref — otherwise ``put_json`` will skip
    writing (path already exists) and the tampered content is lost.
    """
    from synix.build.object_store import ObjectStore
    from synix.build.refs import RefStore

    synix_dir = synix_dir_for_build_dir(build_dir)
    obj_store = ObjectStore(synix_dir)
    ref_store = RefStore(synix_dir)

    head_oid = ref_store.read_ref("refs/heads/main")
    snapshot_obj = obj_store.get_json(head_oid)
    manifest_obj = obj_store.get_json(snapshot_obj["manifest_oid"])

    modified = False
    for entry in manifest_obj["artifacts"]:
        art_obj = obj_store.get_json(entry["oid"])
        if mutate_fn(art_obj):
            # Store the modified artifact as a new content-addressed object
            new_oid = obj_store.put_json(art_obj)
            entry["oid"] = new_oid
            modified = True

    if modified:
        # Re-store the manifest with updated artifact oids
        new_manifest_oid = obj_store.put_json(manifest_obj)
        snapshot_obj["manifest_oid"] = new_manifest_oid
        # Re-store the snapshot with updated manifest oid
        new_snapshot_oid = obj_store.put_json(snapshot_obj)
        # Advance HEAD ref
        ref_store.write_ref("refs/heads/main", new_snapshot_oid)

    return modified


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
from synix import Pipeline, Source, FlatFile
from synix.ext import EpisodeSummary, MonthlyRollup, CoreSynthesis

pipeline = Pipeline("test-fp")
pipeline.source_dir = "{workspace["source_dir"]}"
pipeline.build_dir = "{workspace["build_dir"]}"
pipeline.llm_config = {{"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}}

transcripts = Source("transcripts")
episodes = EpisodeSummary("episodes", depends_on=[transcripts])
monthly = MonthlyRollup("monthly", depends_on=[episodes])
core = CoreSynthesis("core", depends_on=[monthly], context_budget=10000)

pipeline.add(transcripts, episodes, monthly, core)
pipeline.add(FlatFile("context-doc", sources=[core], output_path="{workspace["build_dir"] / "context.md"}"))
""")
    return path


@pytest.fixture(autouse=True)
def mock_anthropic(monkeypatch):
    """Mock Anthropic API for all tests."""

    def mock_create(**kwargs):
        messages = kwargs.get("messages", [])
        content = messages[0].get("content", "") if messages else ""

        if "summarizing a conversation" in content.lower():
            return _mock_response("This conversation covered technical topics.")
        elif "monthly" in content.lower():
            return _mock_response("Monthly themes: technical learning and development.")
        elif "core memory" in content.lower():
            return _mock_response("## Identity\nA software engineer.\n\n## Focus\nMemory systems.")
        return _mock_response("Mock response.")

    mock_client = MagicMock()
    mock_client.messages.create = mock_create
    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)


def _mock_response(text: str):
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    resp.model = "claude-sonnet-4-20250514"
    resp.usage = MagicMock(input_tokens=100, output_tokens=50)
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFingerprintStoredOnBuild:
    """Verify fingerprints are stored in artifact metadata after build."""

    def test_fingerprint_stored_on_build(self, runner, workspace, pipeline_file):
        """Build stores both build_fingerprint and transform_fingerprint in metadata."""
        result = runner.invoke(main, ["build", str(pipeline_file)])
        assert result.exit_code == 0, f"Build failed: {result.output}"

        store = SnapshotArtifactCache(synix_dir_for_build_dir(workspace["build_dir"]))
        manifest = store.iter_entries()
        # Check derived artifacts (not transcripts — level 0 doesn't use fingerprints)
        derived = {aid: info for aid, info in manifest.items() if info.get("layer") not in ("transcripts",)}
        assert len(derived) > 0, "No derived artifacts found"

        for aid in derived:
            art = store.load_artifact(aid)
            assert art is not None, f"Could not load artifact {aid}"
            metadata = art.metadata

            assert "build_fingerprint" in metadata, f"Missing build_fingerprint for {aid}"
            build_fp = Fingerprint.from_dict(metadata["build_fingerprint"])
            assert build_fp is not None
            assert build_fp.scheme == "synix:build:v1"
            assert "transform" in build_fp.components
            assert "inputs" in build_fp.components

            assert "transform_fingerprint" in metadata, f"Missing transform_fingerprint for {aid}"
            transform_fp = Fingerprint.from_dict(metadata["transform_fingerprint"])
            assert transform_fp is not None
            assert transform_fp.scheme == "synix:transform:v2"
            assert "source" in transform_fp.components


class TestUpgradeForcesRebuild:
    """Simulate pre-upgrade artifacts without fingerprints."""

    def test_upgrade_forces_rebuild_without_fingerprint(self, runner, workspace, pipeline_file):
        """Artifacts without fingerprints trigger one-time rebuild."""
        # First build — populates fingerprints
        result1 = runner.invoke(main, ["build", str(pipeline_file)])
        assert result1.exit_code == 0

        # Strip fingerprints from all stored artifacts (simulate pre-upgrade state)
        build_dir = workspace["build_dir"]

        def strip_fingerprints(art_obj):
            meta = art_obj.get("metadata", {})
            layer = meta.get("layer_name", "")
            if layer == "transcripts":
                return False
            changed = False
            if "build_fingerprint" in meta:
                del meta["build_fingerprint"]
                changed = True
            if "transform_fingerprint" in meta:
                del meta["transform_fingerprint"]
                changed = True
            return changed

        _tamper_snapshot_artifacts(build_dir, strip_fingerprints)

        # Second build — should rebuild (no stored fingerprint)
        result2 = runner.invoke(main, ["build", str(pipeline_file)])
        assert result2.exit_code == 0

        # Verify fingerprints are now restored
        store = SnapshotArtifactCache(synix_dir_for_build_dir(build_dir))
        manifest = store.iter_entries()
        for aid, info in manifest.items():
            if info.get("layer") == "transcripts":
                continue
            art = store.load_artifact(aid)
            assert art is not None
            assert "build_fingerprint" in art.metadata, f"Fingerprint not restored for {aid}"

        # Third build — should be fully cached now
        result3 = runner.invoke(main, ["build", str(pipeline_file)])
        assert result3.exit_code == 0


class TestExplainCacheOutput:
    """Test --explain-cache flag on plan command."""

    def test_explain_cache_json(self, runner, workspace, pipeline_file):
        """Plan with --json includes fingerprint data in steps."""
        # Build first so there's cache state
        runner.invoke(main, ["build", str(pipeline_file)])

        # Run plan with --json
        result = runner.invoke(main, ["plan", str(pipeline_file), "--json"])
        assert result.exit_code == 0, f"Plan failed: {result.output}"

        plan_data = json.loads(result.output)
        steps = plan_data.get("steps", [])
        assert len(steps) > 0

        # LLM layers (level > 0) should have fingerprint field
        llm_steps = [s for s in steps if s["level"] > 0]
        for step in llm_steps:
            assert step.get("fingerprint") is not None, f"Missing fingerprint for {step['name']}"
            fp = step["fingerprint"]
            assert "scheme" in fp
            assert "digest" in fp
            assert "components" in fp

    def test_explain_cache_display(self, runner, workspace, pipeline_file):
        """Plan with --explain-cache shows inline cache breakdown."""
        runner.invoke(main, ["build", str(pipeline_file)])
        result = runner.invoke(main, ["plan", str(pipeline_file), "--explain-cache"])
        assert result.exit_code == 0, f"Plan failed: {result.output}"
        assert "cache: all components match" in result.output


class TestFingerprintSchemeMismatch:
    """Test that scheme mismatch triggers rebuild."""

    def test_scheme_mismatch_triggers_rebuild(self, runner, workspace, pipeline_file):
        """Manually editing fingerprint scheme forces rebuild."""
        # Build first
        result1 = runner.invoke(main, ["build", str(pipeline_file)])
        assert result1.exit_code == 0

        # Tamper with the fingerprint scheme on a derived artifact in the snapshot
        build_dir = workspace["build_dir"]
        tampered = [False]

        def tamper_episode_scheme(art_obj):
            if tampered[0]:
                return False
            if not art_obj.get("label", "").startswith("ep-"):
                return False
            bf = art_obj.get("metadata", {}).get("build_fingerprint")
            if bf:
                bf["scheme"] = "synix:build:v0"  # Fake old scheme
                tampered[0] = True
                return True
            return False

        _tamper_snapshot_artifacts(build_dir, tamper_episode_scheme)
        assert tampered[0], "No episode artifact found to tamper with"

        # Second build — tampered artifact should be rebuilt
        result2 = runner.invoke(main, ["build", str(pipeline_file)])
        assert result2.exit_code == 0
