"""E2E tests for consistent error handling when an invalid ref is given."""

from __future__ import annotations

from click.testing import CliRunner

from synix.cli.main import main
from synix.core.models import Artifact
from tests.helpers.snapshot_factory import create_test_snapshot


def _make_artifact(label: str, content: str) -> Artifact:
    return Artifact(
        label=label,
        artifact_type="episode",
        content=content,
        metadata={"layer_name": "episodes", "layer_level": 0},
    )


class TestInvalidRefConsistency:
    """All commands that accept --ref should exit non-zero with a clear message
    when given an invalid/nonexistent ref."""

    def test_list_invalid_ref_exits_nonzero(self, tmp_path):
        """synix list --ref refs/heads/bogus should fail with a clear error."""
        synix_dir = create_test_snapshot(tmp_path, {"episodes": [_make_artifact("ep-1", "Content")]})
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["list", "--ref", "refs/heads/bogus", "--synix-dir", str(synix_dir)],
        )
        assert result.exit_code != 0, f"Expected non-zero exit code, got {result.exit_code}\n{result.output}"
        assert "Cannot open snapshot" in result.output

    def test_search_invalid_ref_exits_nonzero(self, tmp_path):
        """synix search --ref refs/heads/bogus should fail with a clear error."""
        synix_dir = create_test_snapshot(tmp_path, {"episodes": [_make_artifact("ep-1", "Content")]})
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["search", "query", "--ref", "refs/heads/bogus", "--synix-dir", str(synix_dir)],
        )
        assert result.exit_code != 0, f"Expected non-zero exit code, got {result.exit_code}\n{result.output}"

    def test_show_invalid_ref_exits_nonzero(self, tmp_path):
        """synix show --ref refs/heads/bogus should fail with a clear error (regression)."""
        synix_dir = create_test_snapshot(tmp_path, {"episodes": [_make_artifact("ep-1", "Content")]})
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["show", "ep-1", "--ref", "refs/heads/bogus", "--synix-dir", str(synix_dir)],
        )
        assert result.exit_code != 0, f"Expected non-zero exit code, got {result.exit_code}\n{result.output}"
        assert "Cannot open snapshot" in result.output

    def test_lineage_invalid_ref_exits_nonzero(self, tmp_path):
        """synix lineage --ref refs/heads/bogus should fail (regression)."""
        synix_dir = create_test_snapshot(tmp_path, {"episodes": [_make_artifact("ep-1", "Content")]})
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["lineage", "ep-1", "--ref", "refs/heads/bogus", "--synix-dir", str(synix_dir)],
        )
        assert result.exit_code != 0, f"Expected non-zero exit code, got {result.exit_code}\n{result.output}"
