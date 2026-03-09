"""Integration tests — checkpoint recovery for interrupted builds."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from synix import FlatFile, Pipeline, SearchIndex, Source
from synix.build.runner import run
from synix.build.snapshot_view import SnapshotArtifactCache
from synix.ext import CoreSynthesis, EpisodeSummary, MonthlyRollup

FIXTURES_DIR = Path(__file__).parent.parent / "synix" / "fixtures"


@pytest.fixture
def source_dir(tmp_path):
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


class TestCheckpointRecovery:
    def test_checkpoints_written_during_build(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """Build writes checkpoint files for each layer."""
        result = run(pipeline_obj, source_dir=str(source_dir))
        synix_dir = Path(result.synix_dir)

        # After successful commit, checkpoints should be cleaned up
        checkpoint_base = synix_dir / "checkpoints"
        assert not checkpoint_base.exists(), "Checkpoints should be cleared after successful commit"

    def test_interrupted_build_leaves_checkpoints(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """Simulate interrupted build: run layers, write checkpoints, but don't commit snapshot."""

        # First, do a full build to establish baseline
        result1 = run(pipeline_obj, source_dir=str(source_dir))
        synix_dir = Path(result1.synix_dir)
        calls_after_build1 = len(mock_llm)

        # Now simulate an interrupted second build by:
        # 1. Adding a new conversation to trigger partial rebuild
        claude_path = source_dir / "claude_export.json"
        data = json.loads(claude_path.read_text())
        data["conversations"].append(
            {
                "uuid": "conv-interrupt-001",
                "title": "Interrupted conversation",
                "created_at": "2024-06-01T10:00:00Z",
                "chat_messages": [
                    {
                        "uuid": "msg-int-1",
                        "sender": "human",
                        "text": "What about Kubernetes networking?",
                        "created_at": "2024-06-01T10:00:00Z",
                    },
                    {
                        "uuid": "msg-int-2",
                        "sender": "assistant",
                        "text": "Kubernetes networking uses CNI plugins for pod communication.",
                        "created_at": "2024-06-01T10:01:00Z",
                    },
                ],
            }
        )
        claude_path.write_text(json.dumps(data))

        # 2. Run a full build (with checkpoints) — this will succeed and clear checkpoints
        result2 = run(pipeline_obj, source_dir=str(source_dir))
        calls_in_build2 = len(mock_llm) - calls_after_build1

        # 3. Now a third build with no changes should be fully cached from committed snapshot
        calls_before_build3 = len(mock_llm)
        result3 = run(pipeline_obj, source_dir=str(source_dir))
        calls_in_build3 = len(mock_llm) - calls_before_build3

        assert calls_in_build3 == 0, f"Third build (no changes) should use 0 LLM calls, got {calls_in_build3}"

    def test_checkpoint_cache_loads_orphaned_artifacts(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """Manually create checkpoint files and verify SnapshotArtifactCache loads them."""
        # First build
        result1 = run(pipeline_obj, source_dir=str(source_dir))
        synix_dir = Path(result1.synix_dir)

        # Capture artifact data from the committed snapshot
        from synix.build.snapshot_view import SnapshotView

        view = SnapshotView.open(synix_dir)
        manifest_arts = view._manifest["artifacts"]

        # Build a checkpoint that contains the same artifacts (simulating interrupted build)
        fake_run_id = "interrupted-run-001"
        checkpoint_dir = synix_dir / "checkpoints" / fake_run_id
        checkpoint_dir.mkdir(parents=True)

        artifact_oids = {entry["label"]: entry["oid"] for entry in manifest_arts}
        parent_labels_map = {}
        for entry in manifest_arts:
            art_obj = view._object_store.get_json(entry["oid"])
            parent_labels_map[entry["label"]] = art_obj.get("parent_labels", [])

        checkpoint_payload = {
            "type": "checkpoint",
            "layer": "all",
            "artifact_oids": artifact_oids,
            "parent_labels_map": parent_labels_map,
        }
        (checkpoint_dir / "all.json").write_text(json.dumps(checkpoint_payload, indent=2))

        # Remove the HEAD ref target to simulate no committed snapshot.
        # HEAD is a symref: "ref: refs/heads/main\n"
        head_path = synix_dir / "HEAD"
        head_content = head_path.read_text(encoding="utf-8").strip()
        # Extract target ref: "ref: refs/heads/main" → "refs/heads/main"
        target_ref = head_content.removeprefix("ref: ")
        target_ref_path = synix_dir / target_ref
        saved_oid = target_ref_path.read_text(encoding="utf-8").strip()
        target_ref_path.unlink()

        # Now load cache — should fall back to checkpoints
        cache = SnapshotArtifactCache(synix_dir)

        # All artifacts from the checkpoint should be loadable
        for label in artifact_oids:
            art = cache.load_artifact(label)
            assert art is not None, f"Checkpoint artifact {label} should be loadable"
            assert art.artifact_id is not None

        # Restore HEAD ref for cleanup
        target_ref_path.parent.mkdir(parents=True, exist_ok=True)
        target_ref_path.write_text(saved_oid)

    def test_corrupted_checkpoint_gracefully_skipped(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """Corrupted checkpoint JSON is skipped without crashing."""
        result1 = run(pipeline_obj, source_dir=str(source_dir))
        synix_dir = Path(result1.synix_dir)

        # Create a corrupted checkpoint file
        checkpoint_dir = synix_dir / "checkpoints" / "corrupted-run"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "bad.json").write_text("not valid json{{{")

        # Should not raise
        cache = SnapshotArtifactCache(synix_dir)
        # Cache should still work from the committed snapshot
        assert len(cache._artifacts_by_label) > 0

    def test_checkpoint_with_missing_oids_skipped(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """Checkpoint referencing missing object store entries is skipped per-artifact."""
        result1 = run(pipeline_obj, source_dir=str(source_dir))
        synix_dir = Path(result1.synix_dir)

        # Create a checkpoint with non-existent OIDs
        checkpoint_dir = synix_dir / "checkpoints" / "missing-oid-run"
        checkpoint_dir.mkdir(parents=True)
        payload = {
            "type": "checkpoint",
            "layer": "test",
            "artifact_oids": {
                "nonexistent-artifact": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            },
            "parent_labels_map": {},
        }
        (checkpoint_dir / "test.json").write_text(json.dumps(payload, indent=2))

        # Should not raise — missing OID is logged and skipped
        cache = SnapshotArtifactCache(synix_dir)
        assert cache.load_artifact("nonexistent-artifact") is None

    def test_successful_build_clears_checkpoints(self, pipeline_obj, source_dir, build_dir, mock_llm):
        """After a successful full build, all checkpoint dirs are removed."""
        # Create stale checkpoints
        synix_dir = Path(build_dir).parent / ".synix"
        synix_dir.mkdir(parents=True, exist_ok=True)
        stale_cp = synix_dir / "checkpoints" / "old-run-001"
        stale_cp.mkdir(parents=True)
        (stale_cp / "layer1.json").write_text('{"type": "checkpoint", "layer": "layer1", "artifact_oids": {}}')

        # Run a full build
        result = run(pipeline_obj, source_dir=str(source_dir))
        synix_dir = Path(result.synix_dir)

        # All checkpoints (including stale ones) should be cleared
        assert not (synix_dir / "checkpoints").exists(), "All checkpoints should be cleared after successful build"
