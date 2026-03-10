"""E2E tests for refs CLI commands."""

from __future__ import annotations

import json

from click.testing import CliRunner

from synix.cli.main import main
from synix.core.models import Artifact
from tests.helpers.snapshot_factory import create_test_snapshot


def _make_artifact(label, content):
    return Artifact(
        label=label,
        artifact_type="episode",
        content=content,
        metadata={"layer_name": "episodes", "layer_level": 0},
    )


class TestRefsListE2E:
    def test_refs_list_shows_heads(self, tmp_path):
        synix_dir = create_test_snapshot(tmp_path, {"episodes": [_make_artifact("ep-1", "Content")]})
        runner = CliRunner()
        result = runner.invoke(main, ["refs", "list", "--synix-dir", str(synix_dir)])
        assert result.exit_code == 0, result.output
        assert "refs/heads/main" in result.output

    def test_refs_list_json(self, tmp_path):
        synix_dir = create_test_snapshot(tmp_path, {"episodes": [_make_artifact("ep-1", "Content")]})
        runner = CliRunner()
        result = runner.invoke(main, ["refs", "list", "--synix-dir", str(synix_dir), "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert any(r["ref"] == "refs/heads/main" for r in data["refs"])

    def test_refs_list_empty(self, tmp_path):
        synix_dir = tmp_path / ".synix"
        synix_dir.mkdir()
        runner = CliRunner()
        result = runner.invoke(main, ["refs", "list", "--synix-dir", str(synix_dir)])
        assert result.exit_code == 0, result.output

    def test_refs_list_no_store(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["refs", "list", "--synix-dir", str(tmp_path / "nonexistent")])
        assert result.exit_code == 0, result.output
        assert "No snapshot store found" in result.output


    def test_refs_list_shows_plan_refs(self, tmp_path):
        """refs list includes refs/plans/ prefix."""
        synix_dir = create_test_snapshot(tmp_path, {"episodes": [_make_artifact("ep-1", "Content")]})

        # Manually create a plan ref
        plan_ref_dir = synix_dir / "refs" / "plans"
        plan_ref_dir.mkdir(parents=True, exist_ok=True)

        # Read existing HEAD oid to use as a valid snapshot ref
        head_oid = (synix_dir / "refs" / "heads" / "main").read_text().strip()
        (plan_ref_dir / "latest").write_text(head_oid)

        runner = CliRunner()
        result = runner.invoke(main, ["refs", "list", "--synix-dir", str(synix_dir)])
        assert result.exit_code == 0, result.output
        assert "refs/plans/latest" in result.output

    def test_refs_list_json_includes_plan_refs(self, tmp_path):
        """refs list --json includes plan refs in output."""
        synix_dir = create_test_snapshot(tmp_path, {"episodes": [_make_artifact("ep-1", "Content")]})

        plan_ref_dir = synix_dir / "refs" / "plans"
        plan_ref_dir.mkdir(parents=True, exist_ok=True)
        head_oid = (synix_dir / "refs" / "heads" / "main").read_text().strip()
        (plan_ref_dir / "latest").write_text(head_oid)

        runner = CliRunner()
        result = runner.invoke(main, ["refs", "list", "--synix-dir", str(synix_dir), "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert any(r["ref"] == "refs/plans/latest" for r in data["refs"])


class TestRefsShowE2E:
    def test_refs_show_head(self, tmp_path):
        synix_dir = create_test_snapshot(tmp_path, {"episodes": [_make_artifact("ep-1", "Content")]})
        runner = CliRunner()
        result = runner.invoke(main, ["refs", "show", "HEAD", "--synix-dir", str(synix_dir)])
        assert result.exit_code == 0, result.output
        assert "Pipeline" in result.output
        assert "test-pipeline" in result.output

    def test_refs_show_json(self, tmp_path):
        synix_dir = create_test_snapshot(tmp_path, {"episodes": [_make_artifact("ep-1", "Content")]})
        runner = CliRunner()
        result = runner.invoke(main, ["refs", "show", "HEAD", "--synix-dir", str(synix_dir), "--json"])
        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "oid" in data
        assert data["snapshot"]["type"] == "snapshot"

    def test_refs_show_by_full_ref_name(self, tmp_path):
        synix_dir = create_test_snapshot(tmp_path, {"episodes": [_make_artifact("ep-1", "Content")]})
        runner = CliRunner()
        result = runner.invoke(main, ["refs", "show", "refs/heads/main", "--synix-dir", str(synix_dir)])
        assert result.exit_code == 0, result.output
        assert "snapshot" in result.output.lower() or "Pipeline" in result.output

    def test_refs_show_nonexistent(self, tmp_path):
        synix_dir = create_test_snapshot(tmp_path, {"episodes": [_make_artifact("ep-1", "Content")]})
        runner = CliRunner()
        result = runner.invoke(main, ["refs", "show", "refs/heads/nonexistent", "--synix-dir", str(synix_dir)])
        assert result.exit_code != 0

    def test_refs_show_no_store(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["refs", "show", "HEAD", "--synix-dir", str(tmp_path / "nonexistent")])
        assert result.exit_code != 0
