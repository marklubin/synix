"""Unit tests for synix verify command."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from synix import Artifact
from synix.build.artifacts import ArtifactStore
from synix.build.provenance import ProvenanceTracker
from synix.build.verify import verify_build
from synix.cli import main


@pytest.fixture
def populated_build(tmp_path):
    """Create a build directory with artifacts and provenance."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    store = ArtifactStore(build_dir)
    provenance = ProvenanceTracker(build_dir)

    # Create transcript (level 0 — no provenance needed)
    t1 = Artifact(
        artifact_id="t-001",
        artifact_type="transcript",
        content="User: Hello\n\nAssistant: Hi there!",
        metadata={"source": "chatgpt", "date": "2025-01-15"},
    )
    store.save_artifact(t1, "transcripts", 0)

    # Create episode (level 1 — needs provenance)
    ep1 = Artifact(
        artifact_id="ep-001",
        artifact_type="episode",
        content="This conversation was about greetings.",
        input_hashes=[t1.content_hash],
        prompt_id="episode_summary_v1",
        model_config={"model": "test"},
        metadata={"source_conversation_id": "001"},
    )
    store.save_artifact(ep1, "episodes", 1)
    provenance.record("ep-001", parent_ids=["t-001"], prompt_id="episode_summary_v1")

    # Create core memory (level 3)
    core = Artifact(
        artifact_id="core-memory",
        artifact_type="core_memory",
        content="Mark is a software engineer.",
        input_hashes=[ep1.content_hash],
        prompt_id="core_memory_v1",
        model_config={"model": "test"},
        metadata={},
    )
    store.save_artifact(core, "core", 3)
    provenance.record("core-memory", parent_ids=["ep-001"], prompt_id="core_memory_v1")

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
        """Remove an artifact file but keep manifest entry."""
        # Find and delete an artifact file
        for layer_dir in populated_build.iterdir():
            if layer_dir.is_dir() and layer_dir.name.startswith("layer"):
                for f in layer_dir.glob("*.json"):
                    f.unlink()
                    break
                break

        result = verify_build(populated_build, checks=["artifacts_exist"])
        assert not result.passed

    def test_missing_provenance(self, populated_build):
        """Remove provenance for a non-root artifact."""
        prov_path = populated_build / "provenance.json"
        prov_data = json.loads(prov_path.read_text())
        del prov_data["ep-001"]
        prov_path.write_text(json.dumps(prov_data))

        result = verify_build(populated_build, checks=["provenance_complete"])
        assert not result.passed
        check = result.checks[0]
        assert "1" in check.message  # 1 artifact missing provenance

    def test_content_hash_mismatch(self, populated_build):
        """Tamper with artifact content without updating hash."""
        manifest = json.loads((populated_build / "manifest.json").read_text())
        for aid, entry in manifest.items():
            art_path = populated_build / entry["path"]
            data = json.loads(art_path.read_text())
            data["content"] = "TAMPERED CONTENT"
            # Don't update content_hash — this should fail verification
            art_path.write_text(json.dumps(data))
            break

        result = verify_build(populated_build, checks=["content_hashes"])
        assert not result.passed

    def test_orphaned_artifact(self, populated_build):
        """Create an artifact file not in the manifest."""
        layer_dir = populated_build / "layer0-transcripts"
        orphan = layer_dir / "orphan-999.json"
        orphan.write_text(json.dumps({"artifact_id": "orphan-999", "content": "stale"}))

        result = verify_build(populated_build, checks=["no_orphans"])
        assert not result.passed

    def test_specific_checks(self, populated_build):
        """Run only specific checks."""
        result = verify_build(populated_build, checks=["build_exists", "manifest_valid"])
        assert len(result.checks) == 2
        assert all(c.passed for c in result.checks)

    def test_unknown_check(self, populated_build):
        result = verify_build(populated_build, checks=["nonexistent_check"])
        assert not result.passed
        assert "Unknown check" in result.checks[0].message

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
        result = runner.invoke(main, [
            "verify", "--build-dir", str(populated_build), "--json"
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["passed"] is True

    def test_verify_failing(self, tmp_path):
        runner = CliRunner()
        build_dir = tmp_path / "empty_build"
        build_dir.mkdir()
        result = runner.invoke(main, [
            "verify", "--build-dir", str(build_dir)
        ])
        assert result.exit_code != 0
        assert "FAIL" in result.output
