"""E2E test: clean command with release targets."""

from __future__ import annotations

from click.testing import CliRunner

from synix.build.refs import RefStore
from synix.build.release_engine import execute_release, get_release
from synix.cli.main import main
from synix.core.models import Artifact
from tests.helpers.snapshot_factory import create_test_snapshot


def _make_artifact(label, content, atype="episode", layer_name="episodes"):
    return Artifact(
        label=label,
        artifact_type=atype,
        content=content,
        metadata={"layer_name": layer_name, "layer_level": 1 if layer_name == "episodes" else 2},
    )


def _setup_snapshot_and_release(tmp_path, release_name="local"):
    """Create snapshot with artifacts and projections, then release."""
    ep1 = _make_artifact("ep-1", "Episode one about testing and clean commands.")
    ep2 = _make_artifact("ep-2", "Episode two about release lifecycle management.")
    core = _make_artifact(
        "core-1",
        "Core memory: user builds agent memory systems.",
        atype="core",
        layer_name="core",
    )

    projections = {
        "search": {
            "adapter": "synix_search",
            "input_artifacts": ["ep-1", "ep-2"],
            "config": {"modes": ["fulltext"]},
            "config_fingerprint": "sha256:test",
            "precomputed_oid": None,
        },
        "context-doc": {
            "adapter": "flat_file",
            "input_artifacts": ["core-1"],
            "config": {"output_path": "context.md"},
            "config_fingerprint": "sha256:test2",
            "precomputed_oid": None,
        },
    }

    synix_dir = create_test_snapshot(
        tmp_path,
        {"episodes": [ep1, ep2], "core": [core]},
        projections=projections,
    )

    receipt = execute_release(synix_dir, release_name=release_name)
    return synix_dir, receipt


class TestCleanRemovesReleaseAndWork:
    def test_clean_removes_releases_and_work(self, tmp_path):
        """Build, release, clean, verify .synix/releases/ and .synix/work/ removed but objects survives."""
        synix_dir, _ = _setup_snapshot_and_release(tmp_path)

        # Create a work directory to simulate build-time artifacts
        work_dir = synix_dir / "work"
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "temp_search.db").write_text("temp")

        # Verify pre-conditions
        assert (synix_dir / "releases" / "local").exists()
        assert (synix_dir / "objects").exists()

        runner = CliRunner()
        # Build dir must resolve to the right synix dir
        # Since synix_dir is at tmp_path/.synix, build_dir convention is tmp_path/build
        build_dir = tmp_path / "build"
        result = runner.invoke(main, ["clean", "--synix-dir", str(synix_dir), "--yes"])
        assert result.exit_code == 0, result.output

        # Releases and work should be removed
        assert not (synix_dir / "releases").exists()
        assert not (synix_dir / "work").exists()

        # Objects and refs should survive (they are the immutable snapshot store)
        assert (synix_dir / "objects").exists()
        assert (synix_dir / "refs").exists()


class TestCleanSpecificRelease:
    def test_clean_specific_release(self, tmp_path):
        """Build, release as 'v1' and 'v2', clean --release v1, verify only v1 removed."""
        synix_dir, _ = _setup_snapshot_and_release(tmp_path, release_name="v1")
        execute_release(synix_dir, release_name="v2")

        # Verify both exist
        assert (synix_dir / "releases" / "v1" / "receipt.json").exists()
        assert (synix_dir / "releases" / "v2" / "receipt.json").exists()

        runner = CliRunner()
        result = runner.invoke(main, ["clean", "--synix-dir", str(synix_dir), "--release", "v1", "--yes"])
        assert result.exit_code == 0, result.output

        # v1 should be removed, v2 should survive
        assert not (synix_dir / "releases" / "v1").exists()
        assert (synix_dir / "releases" / "v2" / "receipt.json").exists()


class TestCleanPreservesSnapshotHistory:
    def test_clean_preserves_snapshot_refs(self, tmp_path):
        """Build, release, clean, verify snapshot refs still resolve."""
        synix_dir, _ = _setup_snapshot_and_release(tmp_path)

        ref_store = RefStore(synix_dir)
        head_before = ref_store.read_ref("HEAD")
        assert head_before is not None

        runner = CliRunner()
        result = runner.invoke(main, ["clean", "--synix-dir", str(synix_dir), "--yes"])
        assert result.exit_code == 0, result.output

        # HEAD should still resolve after clean
        head_after = ref_store.read_ref("HEAD")
        assert head_after == head_before

        # synix list should still work (reads from snapshot, not releases)
        list_result = runner.invoke(main, ["list", "--synix-dir", str(synix_dir)])
        assert list_result.exit_code == 0, list_result.output
        # Should find artifacts since the object store and refs survived
        assert "ep-1" in list_result.output or "episodes" in list_result.output


class TestCleanThenReleaseAgain:
    def test_clean_then_release_recreates(self, tmp_path):
        """Build, release, clean, release again — should recreate from surviving snapshot."""
        synix_dir, receipt_v1 = _setup_snapshot_and_release(tmp_path)

        runner = CliRunner()
        runner.invoke(main, ["clean", "--synix-dir", str(synix_dir), "--yes"])

        # Releases dir should be gone
        assert not (synix_dir / "releases").exists()

        # Release again from HEAD — should succeed since objects/refs survived
        release_result = runner.invoke(main, ["release", "HEAD", "--to", "local", "--synix-dir", str(synix_dir)])
        assert release_result.exit_code == 0, release_result.output
        assert "Released" in release_result.output

        # Verify the new release is functional
        new_receipt = get_release(synix_dir, "local")
        assert new_receipt is not None
        assert new_receipt.snapshot_oid == receipt_v1.snapshot_oid
        assert (synix_dir / "releases" / "local" / "search.db").exists()
        assert (synix_dir / "releases" / "local" / "context.md").exists()
