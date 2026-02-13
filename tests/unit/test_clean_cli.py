"""CLI tests for the clean command."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from synix.cli.main import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def build_dir(tmp_path):
    d = tmp_path / "build"
    d.mkdir()
    return d


class TestCleanHelp:
    def test_help(self, runner):
        result = runner.invoke(main, ["clean", "--help"])
        assert result.exit_code == 0
        assert "BUILD_DIR" in result.output
        assert "--yes" in result.output

    def test_help_description(self, runner):
        result = runner.invoke(main, ["clean", "--help"])
        assert "Remove all build artifacts" in result.output


class TestClean:
    def test_no_build_dir(self, runner, tmp_path):
        """Clean when build dir doesn't exist — nothing to do."""
        result = runner.invoke(main, ["clean", str(tmp_path / "nobuild"), "-y"])
        assert result.exit_code == 0
        assert "Nothing to clean" in result.output

    def test_clean_removes_build_dir(self, runner, build_dir):
        """clean -y removes the build directory."""
        (build_dir / "manifest.json").write_text("{}")
        (build_dir / "provenance.json").write_text("{}")
        layer_dir = build_dir / "layer0-transcripts"
        layer_dir.mkdir()
        (layer_dir / "t-1.json").write_text("{}")

        assert build_dir.exists()
        result = runner.invoke(main, ["clean", str(build_dir), "-y"])
        assert result.exit_code == 0
        assert "Cleaned" in result.output
        assert not build_dir.exists()

    def test_clean_confirmation_abort(self, runner, build_dir):
        """Without -y, answering 'n' aborts."""
        (build_dir / "manifest.json").write_text("{}")
        result = runner.invoke(main, ["clean", str(build_dir)], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output
        assert build_dir.exists()

    def test_clean_confirmation_proceed(self, runner, build_dir):
        """Without -y, answering 'y' proceeds."""
        (build_dir / "manifest.json").write_text("{}")
        result = runner.invoke(main, ["clean", str(build_dir)], input="y\n")
        assert result.exit_code == 0
        assert "Cleaned" in result.output
        assert not build_dir.exists()

    def test_clean_idempotent(self, runner, build_dir):
        """Cleaning twice — second time says nothing to clean."""
        (build_dir / "manifest.json").write_text("{}")
        runner.invoke(main, ["clean", str(build_dir), "-y"])
        assert not build_dir.exists()

        result = runner.invoke(main, ["clean", str(build_dir), "-y"])
        assert result.exit_code == 0
        assert "Nothing to clean" in result.output
