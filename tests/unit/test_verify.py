"""Unit tests for synix verify command."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from synix import Artifact
from synix.build.verify import verify_build
from synix.cli import main
from synix.search.indexer import SearchIndex
from tests.helpers.snapshot_factory import create_test_snapshot


@pytest.fixture
def populated_build(tmp_path):
    """Create a build directory with a .synix snapshot."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    t1 = Artifact(
        label="t-001",
        artifact_type="transcript",
        content="User: Hello\n\nAssistant: Hi there!",
        metadata={"source": "chatgpt", "date": "2025-01-15"},
    )

    ep1 = Artifact(
        label="ep-001",
        artifact_type="episode",
        content="This conversation was about greetings.",
        input_ids=[t1.artifact_id],
        prompt_id="episode_summary_v1",
        model_config={"model": "test"},
        metadata={"source_conversation_id": "001", "layer_level": 1},
    )

    core = Artifact(
        label="core-memory",
        artifact_type="core_memory",
        content="Mark is a software engineer.",
        input_ids=[ep1.artifact_id],
        prompt_id="core_memory_v1",
        model_config={"model": "test"},
        metadata={"layer_level": 3},
    )

    create_test_snapshot(
        build_dir,
        {"transcripts": [t1], "episodes": [ep1], "core": [core]},
        parent_labels_map={
            "ep-001": ["t-001"],
            "core-memory": ["ep-001"],
        },
    )

    return build_dir


class TestVerifyBuild:
    def test_all_checks_pass(self, populated_build):
        result = verify_build(populated_build)
        assert result.passed
        assert len(result.checks) == 8
        assert all(c.passed for c in result.checks)

    def test_missing_build_dir(self, tmp_path):
        result = verify_build(tmp_path / "nonexistent")
        assert not result.passed
        failed = [c for c in result.checks if not c.passed]
        assert any(c.name == "build_exists" for c in failed)

    def test_missing_manifest(self, tmp_path):
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        result = verify_build(build_dir)
        assert not result.passed

    def test_missing_artifact_file(self, populated_build):
        """Delete an artifact object from the snapshot store."""
        from synix.build.object_store import ObjectStore
        from synix.build.refs import RefStore, synix_dir_for_build_dir

        synix_dir = synix_dir_for_build_dir(populated_build)
        obj_store = ObjectStore(synix_dir)
        ref_store = RefStore(synix_dir)

        head_oid = ref_store.read_ref("refs/heads/main")
        snapshot_obj = obj_store.get_json(head_oid)
        manifest_obj = obj_store.get_json(snapshot_obj["manifest_oid"])

        # Delete the content blob for the first artifact
        target_entry = manifest_obj["artifacts"][0]
        art_obj = obj_store.get_json(target_entry["oid"])
        content_oid = art_obj["content_oid"]
        content_path = synix_dir / "objects" / content_oid[:2] / content_oid[2:]
        content_path.unlink()

        result = verify_build(populated_build, checks=["artifacts_exist"])
        assert not result.passed

    def test_missing_provenance(self, populated_build):
        """Remove parent_labels for a non-root artifact in the snapshot."""
        from synix.build.object_store import ObjectStore
        from synix.build.refs import RefStore, synix_dir_for_build_dir

        synix_dir = synix_dir_for_build_dir(populated_build)
        obj_store = ObjectStore(synix_dir)
        ref_store = RefStore(synix_dir)

        head_oid = ref_store.read_ref("refs/heads/main")
        snapshot_obj = obj_store.get_json(head_oid)
        manifest_obj = obj_store.get_json(snapshot_obj["manifest_oid"])

        # Find ep-001 and remove its parent_labels
        for entry in manifest_obj["artifacts"]:
            art_obj = obj_store.get_json(entry["oid"])
            if art_obj["label"] == "ep-001":
                art_obj["parent_labels"] = []
                oid = entry["oid"]
                obj_path = synix_dir / "objects" / oid[:2] / oid[2:]
                obj_path.write_text(json.dumps(art_obj))
                break

        result = verify_build(populated_build, checks=["provenance_complete"])
        assert not result.passed
        check = result.checks[0]
        assert "1" in check.message  # 1 artifact missing provenance

    def test_content_hash_mismatch(self, populated_build):
        """Tamper with artifact_id in snapshot object store."""
        from synix.build.object_store import ObjectStore
        from synix.build.refs import RefStore, synix_dir_for_build_dir

        synix_dir = synix_dir_for_build_dir(populated_build)
        obj_store = ObjectStore(synix_dir)
        ref_store = RefStore(synix_dir)

        # Walk the snapshot -> manifest -> artifact objects to find one to tamper
        head_oid = ref_store.read_ref("refs/heads/main")
        snapshot_obj = obj_store.get_json(head_oid)
        manifest_obj = obj_store.get_json(snapshot_obj["manifest_oid"])

        target_entry = manifest_obj["artifacts"][0]
        art_obj = obj_store.get_json(target_entry["oid"])

        # Tamper: set artifact_id to a wrong hash
        art_obj["artifact_id"] = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
        # Overwrite the object file in place (bypassing content-addressing)
        oid = target_entry["oid"]
        obj_path = synix_dir / "objects" / oid[:2] / oid[2:]
        obj_path.write_text(json.dumps(art_obj))

        result = verify_build(populated_build, checks=["content_hashes"])
        assert not result.passed

    def test_orphaned_artifact(self, populated_build):
        """With content-addressed store, orphan check always passes."""
        result = verify_build(populated_build, checks=["no_orphans"])
        assert result.passed
        assert "no orphans" in result.checks[0].message.lower()

    def test_specific_checks(self, populated_build):
        """Run only specific checks."""
        result = verify_build(populated_build, checks=["build_exists", "manifest_valid"])
        assert len(result.checks) == 2
        assert all(c.passed for c in result.checks)

    def test_unknown_check(self, populated_build):
        result = verify_build(populated_build, checks=["nonexistent_check"])
        assert not result.passed
        assert "Unknown check" in result.checks[0].message

    def test_synix_search_uses_custom_output_path(self, populated_build):
        custom_db = populated_build / "outputs" / "memory.db"
        custom_db.parent.mkdir()

        artifact = Artifact(
            label="search-001",
            artifact_type="episode",
            content="Machine learning verification artifact",
            metadata={"layer_name": "episodes"},
        )
        index = SearchIndex(custom_db)
        index.create()
        index.insert(artifact, "episodes", 1)
        index.close()

        (populated_build / ".projection_cache.json").write_text(
            json.dumps(
                {
                    "search": {
                        "projection_type": "synix_search",
                        "db_path": "outputs/memory.db",
                    }
                }
            )
        )

        result = verify_build(populated_build, checks=["synix_search"])
        assert result.passed
        assert result.checks[0].message == "Synix search 'search' has 1 entries"

    def test_to_dict(self, populated_build):
        result = verify_build(populated_build)
        d = result.to_dict()
        assert d["passed"] is True
        assert "checks" in d
        assert len(d["checks"]) == 8
        for check in d["checks"]:
            assert "fix_hint" in check

    def test_summary(self, populated_build):
        result = verify_build(populated_build)
        assert "8 checks passed" in result.summary


class TestVerifyCLI:
    def test_verify_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["verify", "--help"])
        assert result.exit_code == 0
        assert "--check" in result.output

    def test_verify_passing(self, populated_build):
        runner = CliRunner()
        result = runner.invoke(main, ["verify", "--build-dir", str(populated_build)])
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_verify_json_output(self, populated_build):
        runner = CliRunner()
        result = runner.invoke(main, ["verify", "--build-dir", str(populated_build), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["passed"] is True

    def test_verify_failing(self, tmp_path):
        runner = CliRunner()
        build_dir = tmp_path / "empty_build"
        build_dir.mkdir()
        result = runner.invoke(main, ["verify", "--build-dir", str(build_dir)])
        assert result.exit_code != 0
        assert "FAIL" in result.output
