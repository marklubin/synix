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


@pytest.fixture
def pipeline_file(tmp_path, build_dir):
    """Create a minimal pipeline file pointing at build_dir."""
    f = tmp_path / "pipeline.py"
    f.write_text(
        f"from synix.core.models import Pipeline, Layer\n"
        f'pipeline = Pipeline("test")\n'
        f'pipeline.build_dir = "{build_dir}"\n'
        f'pipeline.source_dir = "{tmp_path}"\n'
        f'pipeline.add_layer(Layer(name="transcripts", level=0, transform="parse"))\n'
    )
    return f


class TestCleanHelp:
    def test_help(self, runner):
        result = runner.invoke(main, ["clean", "--help"])
        assert result.exit_code == 0
        assert "PIPELINE_PATH" in result.output
        assert "--yes" in result.output
        assert "--build-dir" in result.output

    def test_help_description(self, runner):
        result = runner.invoke(main, ["clean", "--help"])
        assert "Remove all build artifacts" in result.output


class TestClean:
    def test_nonexistent_pipeline_errors(self, runner):
        result = runner.invoke(main, ["clean", "nonexistent.py"])
        assert result.exit_code != 0

    def test_no_build_dir(self, runner, tmp_path):
        """Clean when build dir doesn't exist — nothing to do."""
        f = tmp_path / "pipeline.py"
        f.write_text(
            f"from synix.core.models import Pipeline, Layer\n"
            f'pipeline = Pipeline("test")\n'
            f'pipeline.build_dir = "{tmp_path / "nobuild"}"\n'
            f'pipeline.source_dir = "{tmp_path}"\n'
            f'pipeline.add_layer(Layer(name="t", level=0, transform="parse"))\n'
        )
        result = runner.invoke(main, ["clean", str(f), "-y"])
        assert result.exit_code == 0
        assert "Nothing to clean" in result.output

    def test_clean_removes_build_dir(self, runner, pipeline_file, build_dir):
        """clean -y removes the build directory."""
        # Put some files in build_dir so it's non-empty
        (build_dir / "manifest.json").write_text("{}")
        (build_dir / "provenance.json").write_text("{}")
        layer_dir = build_dir / "layer0-transcripts"
        layer_dir.mkdir()
        (layer_dir / "t-1.json").write_text("{}")

        assert build_dir.exists()
        result = runner.invoke(main, ["clean", str(pipeline_file), "-y"])
        assert result.exit_code == 0
        assert "Cleaned" in result.output
        assert not build_dir.exists()

    def test_clean_confirmation_abort(self, runner, pipeline_file, build_dir):
        """Without -y, answering 'n' aborts."""
        (build_dir / "manifest.json").write_text("{}")
        result = runner.invoke(main, ["clean", str(pipeline_file)], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output
        assert build_dir.exists()

    def test_clean_confirmation_proceed(self, runner, pipeline_file, build_dir):
        """Without -y, answering 'y' proceeds."""
        (build_dir / "manifest.json").write_text("{}")
        result = runner.invoke(main, ["clean", str(pipeline_file)], input="y\n")
        assert result.exit_code == 0
        assert "Cleaned" in result.output
        assert not build_dir.exists()

    def test_clean_with_build_dir_override(self, runner, pipeline_file, tmp_path):
        """--build-dir overrides the pipeline's build_dir."""
        alt_build = tmp_path / "alt_build"
        alt_build.mkdir()
        (alt_build / "search.db").write_text("")

        result = runner.invoke(main, [
            "clean", str(pipeline_file),
            "--build-dir", str(alt_build),
            "-y",
        ])
        assert result.exit_code == 0
        assert "Cleaned" in result.output
        assert not alt_build.exists()

    def test_clean_idempotent(self, runner, pipeline_file, build_dir):
        """Cleaning twice — second time says nothing to clean."""
        (build_dir / "manifest.json").write_text("{}")
        runner.invoke(main, ["clean", str(pipeline_file), "-y"])
        assert not build_dir.exists()

        result = runner.invoke(main, ["clean", str(pipeline_file), "-y"])
        assert result.exit_code == 0
        assert "Nothing to clean" in result.output
