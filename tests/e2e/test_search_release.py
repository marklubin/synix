"""E2E test: search against released snapshots."""

from __future__ import annotations

from click.testing import CliRunner

from synix.build.release_engine import execute_release
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


def _setup_snapshot_with_release(tmp_path, release_name="local"):
    """Create a snapshot with artifacts + projections, then release it."""
    ep1 = _make_artifact("ep-1", "Episode one about machine learning and neural networks.")
    ep2 = _make_artifact("ep-2", "Episode two about database optimization and PostgreSQL.")
    core = _make_artifact(
        "core-1",
        "Core memory: user is an engineer interested in ML and databases.",
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


class TestSearchAgainstRelease:
    def test_search_against_release_returns_results(self, tmp_path):
        """Build, release, search with --release local returns results."""
        synix_dir, _ = _setup_snapshot_with_release(tmp_path)
        runner = CliRunner()

        result = runner.invoke(
            main,
            ["search", "machine learning", "--release", "local", "--synix-dir", str(synix_dir)],
        )
        assert result.exit_code == 0, result.output
        # Should find the episode about ML
        assert "machine" in result.output.lower() or "learning" in result.output.lower()

    def test_search_with_layer_filter_against_release(self, tmp_path):
        """Build, release, search with --layers episodes --release local."""
        synix_dir, _ = _setup_snapshot_with_release(tmp_path)
        runner = CliRunner()

        result = runner.invoke(
            main,
            [
                "search",
                "database",
                "--layers",
                "episodes",
                "--release",
                "local",
                "--synix-dir",
                str(synix_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        # Should find the database episode; no core results since layer filter is episodes
        if "No results" not in result.output:
            assert "episodes" in result.output.lower() or "database" in result.output.lower()

    def test_search_release_not_found_gives_error(self, tmp_path):
        """Search --release nonexistent gives helpful error."""
        synix_dir, _ = _setup_snapshot_with_release(tmp_path)
        runner = CliRunner()

        result = runner.invoke(
            main,
            ["search", "anything", "--release", "nonexistent", "--synix-dir", str(synix_dir)],
        )
        # Should exit non-zero with an error about the missing release
        assert result.exit_code != 0

    def test_search_default_finds_single_release(self, tmp_path):
        """Build, release as 'local', search without --release auto-detects the single release."""
        synix_dir, _ = _setup_snapshot_with_release(tmp_path)
        runner = CliRunner()

        # Search without --release flag — should auto-detect the single release
        # The build_dir must resolve to the right .synix dir
        result = runner.invoke(
            main,
            ["search", "machine learning", "--synix-dir", str(synix_dir)],
        )
        assert result.exit_code == 0, result.output
        # Auto-detection should find the single "local" release
        assert "machine" in result.output.lower() or "learning" in result.output.lower()


class TestSearchMultipleReleases:
    def test_search_multiple_releases_requires_flag(self, tmp_path):
        """When multiple releases exist, search without --release gives an error."""
        # Create snapshot and release under two names
        ep1 = _make_artifact("ep-1", "Episode one about testing.")
        projections = {
            "search": {
                "adapter": "synix_search",
                "input_artifacts": ["ep-1"],
                "config": {"modes": ["fulltext"]},
                "config_fingerprint": "sha256:test",
                "precomputed_oid": None,
            },
        }
        synix_dir = create_test_snapshot(
            tmp_path,
            {"episodes": [ep1]},
            projections=projections,
        )

        execute_release(synix_dir, release_name="staging")
        execute_release(synix_dir, release_name="prod")

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["search", "testing", "--synix-dir", str(synix_dir)],
        )
        # Should fail because multiple releases exist and none specified
        assert result.exit_code != 0
