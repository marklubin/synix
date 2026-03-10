"""CLI tests for the clean command."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from synix.cli.main import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def synix_dir(tmp_path):
    """Create a .synix directory with releases and work subdirectories."""
    sd = tmp_path / ".synix"
    sd.mkdir()
    releases = sd / "releases"
    releases.mkdir()
    local_release = releases / "local"
    local_release.mkdir()
    (local_release / "search.db").write_text("fake")
    work = sd / "work"
    work.mkdir()
    (work / "scratch.tmp").write_text("temp")
    return sd


class TestCleanHelp:
    def test_help(self, runner):
        result = runner.invoke(main, ["clean", "--help"])
        assert result.exit_code == 0
        assert "--build-dir" in result.output
        assert "--yes" in result.output or "-y" in result.output

    def test_help_description(self, runner):
        result = runner.invoke(main, ["clean", "--help"])
        assert "release" in result.output.lower()


class TestClean:
    def test_no_synix_dir(self, runner, tmp_path):
        """Clean when .synix dir doesn't exist — nothing to do."""
        result = runner.invoke(
            main,
            ["clean", "--build-dir", str(tmp_path / "nobuild"), "-y"],
        )
        assert result.exit_code == 0
        assert "Nothing to clean" in result.output

    def test_clean_removes_releases_and_work(self, runner, tmp_path, synix_dir):
        """clean -y removes releases and work directories under .synix."""
        assert (synix_dir / "releases").exists()
        assert (synix_dir / "work").exists()

        # --build-dir points to tmp_path/build so .synix is sibling at tmp_path/.synix
        build_dir = tmp_path / "build"
        result = runner.invoke(
            main,
            ["clean", "--synix-dir", str(synix_dir), "-y"],
        )
        assert result.exit_code == 0
        assert "Cleaned" in result.output
        assert not (synix_dir / "releases").exists()
        assert not (synix_dir / "work").exists()

    def test_clean_specific_release(self, runner, tmp_path, synix_dir):
        """clean --release NAME removes only that release."""
        # Add a second release to verify only one is removed
        prod = synix_dir / "releases" / "prod"
        prod.mkdir()
        (prod / "search.db").write_text("fake")

        result = runner.invoke(
            main,
            ["clean", "--synix-dir", str(synix_dir), "--release", "local", "-y"],
        )
        assert result.exit_code == 0
        assert "Cleaned" in result.output
        assert not (synix_dir / "releases" / "local").exists()
        # prod release should still exist
        assert (synix_dir / "releases" / "prod").exists()

    def test_clean_confirmation_abort(self, runner, tmp_path, synix_dir):
        """Without -y, answering 'n' aborts."""
        result = runner.invoke(
            main,
            ["clean", "--synix-dir", str(synix_dir)],
            input="n\n",
        )
        assert result.exit_code == 0
        assert "Aborted" in result.output
        assert (synix_dir / "releases").exists()

    def test_clean_confirmation_proceed(self, runner, tmp_path, synix_dir):
        """Without -y, answering 'y' proceeds."""
        result = runner.invoke(
            main,
            ["clean", "--synix-dir", str(synix_dir)],
            input="y\n",
        )
        assert result.exit_code == 0
        assert "Cleaned" in result.output
        assert not (synix_dir / "releases").exists()

    def test_clean_idempotent(self, runner, tmp_path, synix_dir):
        """Cleaning twice — second time says nothing to clean."""
        runner.invoke(
            main,
            ["clean", "--synix-dir", str(synix_dir), "-y"],
        )
        assert not (synix_dir / "releases").exists()
        assert not (synix_dir / "work").exists()

        result = runner.invoke(
            main,
            ["clean", "--synix-dir", str(synix_dir), "-y"],
        )
        assert result.exit_code == 0
        assert "Nothing to clean" in result.output

    def test_clean_legacy_build_dir(self, runner, tmp_path, synix_dir):
        """Clean also removes legacy build/ directory if it exists."""
        legacy_build = tmp_path / "build"
        legacy_build.mkdir()
        (legacy_build / "manifest.json").write_text("{}")

        result = runner.invoke(
            main,
            ["clean", "--build-dir", str(legacy_build), "--synix-dir", str(synix_dir), "-y"],
        )
        assert result.exit_code == 0
        assert "Cleaned" in result.output
        assert not legacy_build.exists()

    def test_clean_specific_release_preserves_ref(self, runner, tmp_path, synix_dir):
        """clean --release NAME removes payloads but preserves the release ref."""
        # Create a release ref file
        ref_dir = synix_dir / "refs" / "releases"
        ref_dir.mkdir(parents=True)
        (ref_dir / "local").write_text("fake-oid")

        result = runner.invoke(
            main,
            ["clean", "--synix-dir", str(synix_dir), "--release", "local", "-y"],
        )
        assert result.exit_code == 0
        assert "Cleaned" in result.output
        assert not (synix_dir / "releases" / "local").exists()
        # Ref is preserved — it still points at a valid snapshot
        assert (ref_dir / "local").exists()

    def test_clean_all_releases_preserves_release_refs(self, runner, tmp_path, synix_dir):
        """clean -y removes all release payloads but preserves release refs."""
        # Create release ref files
        ref_dir = synix_dir / "refs" / "releases"
        ref_dir.mkdir(parents=True)
        (ref_dir / "local").write_text("fake-oid-1")
        (ref_dir / "prod").write_text("fake-oid-2")

        # Add prod release payload
        prod = synix_dir / "releases" / "prod"
        prod.mkdir()
        (prod / "search.db").write_text("fake")

        result = runner.invoke(
            main,
            ["clean", "--synix-dir", str(synix_dir), "-y"],
        )
        assert result.exit_code == 0
        assert "Cleaned" in result.output
        assert not (synix_dir / "releases").exists()
        # Refs are preserved — they still point at valid snapshots
        assert ref_dir.exists()
        assert (ref_dir / "local").exists()
        assert (ref_dir / "prod").exists()
