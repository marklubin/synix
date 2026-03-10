"""E2E tests for snapshot-era diff (Bug 11).

Verifies that diff_artifact_by_label can find previous versions
by walking refs/runs instead of relying on legacy build/versions/ dirs.
"""

from __future__ import annotations

import pytest

from synix import Artifact
from synix.build.diff import diff_artifact_by_label
from tests.helpers.snapshot_factory import create_test_snapshot


@pytest.fixture
def workspace(tmp_path):
    """Workspace with source dir and build dir paths."""
    source_dir = tmp_path / "exports"
    source_dir.mkdir()
    build_dir = tmp_path / "build"
    return {"root": tmp_path, "source_dir": source_dir, "build_dir": build_dir}


class TestDiffArtifactSingleRun:
    def test_diff_artifact_single_run_returns_none(self, tmp_path):
        """With only one run, diff_artifact_by_label without --old-build-dir returns None."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        art = Artifact(
            label="ep-001",
            artifact_type="episode",
            content="Episode content v1",
            metadata={"layer_name": "episodes", "layer_level": 1},
        )
        create_test_snapshot(build_dir, {"episodes": [art]}, synix_subdir=".synix")

        # Only one run => no previous version to diff against
        result = diff_artifact_by_label(str(build_dir), "ep-001")
        assert result is None


class TestDiffArtifactAcrossRuns:
    def test_diff_artifact_across_runs(self, tmp_path):
        """Build twice with different content, diff should show changes."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        synix_dir = build_dir / ".synix"

        # First run: create snapshot with v1 content
        art_v1 = Artifact(
            label="ep-001",
            artifact_type="episode",
            content="Episode content version 1",
            metadata={"layer_name": "episodes", "layer_level": 1},
        )
        create_test_snapshot(build_dir, {"episodes": [art_v1]}, synix_subdir=".synix")

        # Now we need to create a second run with different content.
        # create_test_snapshot writes to refs/heads/main. We need to:
        # 1. Record the current HEAD oid as a run ref
        # 2. Create a new snapshot with different content
        import hashlib
        from datetime import UTC, datetime

        from synix.build.object_store import SCHEMA_VERSION, ObjectStore
        from synix.build.refs import RefStore

        ref_store = RefStore(synix_dir)
        object_store = ObjectStore(synix_dir)

        # Record the first snapshot as a run ref
        first_oid = ref_store.read_ref("HEAD")
        assert first_oid is not None
        ref_store.write_ref("refs/runs/run-001", first_oid)

        # Create second snapshot with different content
        art_v2 = Artifact(
            label="ep-001",
            artifact_type="episode",
            content="Episode content version 2 -- CHANGED",
            metadata={"layer_name": "episodes", "layer_level": 1},
        )
        content_v2 = art_v2.content
        content_oid = object_store.put_bytes(content_v2.encode("utf-8"))
        content_hash = f"sha256:{hashlib.sha256(content_v2.encode('utf-8')).hexdigest()}"
        art_v2.artifact_id = content_hash

        artifact_payload = {
            "type": "artifact",
            "schema_version": SCHEMA_VERSION,
            "label": art_v2.label,
            "artifact_type": art_v2.artifact_type,
            "artifact_id": art_v2.artifact_id,
            "content_oid": content_oid,
            "input_ids": [],
            "metadata": dict(art_v2.metadata),
            "parent_labels": [],
        }
        artifact_oid = object_store.put_json(artifact_payload)

        manifest_payload = {
            "type": "manifest",
            "schema_version": SCHEMA_VERSION,
            "pipeline_name": "test-pipeline",
            "pipeline_fingerprint": "sha256:test-v2",
            "artifacts": [{"label": "ep-001", "oid": artifact_oid}],
            "projections": {},
        }
        manifest_oid = object_store.put_json(manifest_payload)

        snapshot_payload = {
            "type": "snapshot",
            "schema_version": SCHEMA_VERSION,
            "manifest_oid": manifest_oid,
            "parent_snapshot_oids": [first_oid],
            "created_at": datetime.now(UTC).isoformat(),
            "pipeline_name": "test-pipeline",
            "run_id": "test-run-002",
        }
        snapshot_oid = object_store.put_json(snapshot_payload)

        # Update HEAD and create second run ref
        ref_store.write_ref("refs/heads/main", snapshot_oid)
        ref_store.write_ref("refs/runs/run-002", snapshot_oid)

        # Now diff should find the difference between run-002 (HEAD) and run-001
        result = diff_artifact_by_label(str(build_dir), "ep-001")
        assert result is not None, "Expected diff result across two runs"
        assert result.has_changes, "Expected changes between v1 and v2"
        assert "version 1" in result.content_diff
        assert "CHANGED" in result.content_diff

    def test_diff_artifact_not_in_previous_run(self, tmp_path):
        """Artifact that only exists in HEAD (not in previous run) returns None."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        synix_dir = build_dir / ".synix"

        # First run: only has transcript
        t1 = Artifact(
            label="t-001",
            artifact_type="transcript",
            content="Transcript v1",
            metadata={"layer_name": "transcripts", "layer_level": 0},
        )
        create_test_snapshot(build_dir, {"transcripts": [t1]}, synix_subdir=".synix")

        import hashlib
        from datetime import UTC, datetime

        from synix.build.object_store import SCHEMA_VERSION, ObjectStore
        from synix.build.refs import RefStore

        ref_store = RefStore(synix_dir)
        object_store = ObjectStore(synix_dir)

        first_oid = ref_store.read_ref("HEAD")
        ref_store.write_ref("refs/runs/run-001", first_oid)

        # Second run: has transcript + new episode
        ep1 = Artifact(
            label="ep-001",
            artifact_type="episode",
            content="New episode",
            metadata={"layer_name": "episodes", "layer_level": 1},
        )
        # Build second snapshot with both t-001 and ep-001
        entries = []
        for art in [t1, ep1]:
            c_oid = object_store.put_bytes(art.content.encode("utf-8"))
            c_hash = f"sha256:{hashlib.sha256(art.content.encode('utf-8')).hexdigest()}"
            if not art.artifact_id:
                art.artifact_id = c_hash
            payload = {
                "type": "artifact",
                "schema_version": SCHEMA_VERSION,
                "label": art.label,
                "artifact_type": art.artifact_type,
                "artifact_id": art.artifact_id,
                "content_oid": c_oid,
                "input_ids": [],
                "metadata": dict(art.metadata),
                "parent_labels": [],
            }
            a_oid = object_store.put_json(payload)
            entries.append({"label": art.label, "oid": a_oid})

        manifest_payload = {
            "type": "manifest",
            "schema_version": SCHEMA_VERSION,
            "pipeline_name": "test-pipeline",
            "pipeline_fingerprint": "sha256:test-v2",
            "artifacts": sorted(entries, key=lambda e: e["label"]),
            "projections": {},
        }
        manifest_oid = object_store.put_json(manifest_payload)
        snapshot_payload = {
            "type": "snapshot",
            "schema_version": SCHEMA_VERSION,
            "manifest_oid": manifest_oid,
            "parent_snapshot_oids": [first_oid],
            "created_at": datetime.now(UTC).isoformat(),
            "pipeline_name": "test-pipeline",
            "run_id": "test-run-002",
        }
        snapshot_oid = object_store.put_json(snapshot_payload)
        ref_store.write_ref("refs/heads/main", snapshot_oid)
        ref_store.write_ref("refs/runs/run-002", snapshot_oid)

        # ep-001 doesn't exist in run-001, so diff should return None
        result = diff_artifact_by_label(str(build_dir), "ep-001")
        assert result is None, "Expected None when artifact not in previous run"
