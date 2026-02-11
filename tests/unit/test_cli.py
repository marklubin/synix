"""Unit tests for Synix CLI commands."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from synix.cli import main


@pytest.fixture
def runner():
    return CliRunner()


def test_run_command_exists(runner):
    """synix run --help succeeds."""
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "PIPELINE_PATH" in result.output


def test_search_command_exists(runner):
    """synix search --help succeeds."""
    result = runner.invoke(main, ["search", "--help"])
    assert result.exit_code == 0
    assert "QUERY" in result.output


def test_lineage_command_exists(runner):
    """synix lineage --help succeeds."""
    result = runner.invoke(main, ["lineage", "--help"])
    assert result.exit_code == 0
    assert "ARTIFACT_ID" in result.output


def test_status_command_exists(runner):
    """synix status --help succeeds."""
    result = runner.invoke(main, ["status", "--help"])
    assert result.exit_code == 0


def test_run_missing_pipeline_errors(runner, tmp_path):
    """synix run with nonexistent file gives clear error."""
    result = runner.invoke(main, ["run", str(tmp_path / "nonexistent.py")])
    assert result.exit_code != 0


def test_search_no_index_errors(runner, tmp_path):
    """synix search before any run gives clear error."""
    result = runner.invoke(main, ["search", "test query", "--build-dir", str(tmp_path)])
    assert result.exit_code != 0 or "No search index" in result.output


def test_status_no_build_dir_errors(runner, tmp_path):
    """synix status with nonexistent build dir gives clear error."""
    result = runner.invoke(main, ["status", "--build-dir", str(tmp_path / "nonexistent")])
    assert result.exit_code != 0 or "No build directory" in result.output


def test_lineage_no_provenance(runner, tmp_path):
    """synix lineage with no provenance data gives clear error."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    result = runner.invoke(main, ["lineage", "fake-id", "--build-dir", str(build_dir)])
    assert result.exit_code != 0 or "No provenance" in result.output


def test_status_empty_build_dir(runner, tmp_path):
    """synix status with empty build dir shows empty table."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    result = runner.invoke(main, ["status", "--build-dir", str(build_dir)])
    assert result.exit_code == 0
    assert "Build Status" in result.output
    assert "not built yet" in result.output


def test_main_help(runner):
    """synix --help shows group help."""
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "build system for agent memory" in result.output.lower()


def test_run_help_shows_options(runner):
    """synix run --help shows all options."""
    result = runner.invoke(main, ["run", "--help"])
    assert result.exit_code == 0
    assert "--source-dir" in result.output
    assert "--build-dir" in result.output


def test_search_help_shows_options(runner):
    """synix search --help shows all options."""
    result = runner.invoke(main, ["search", "--help"])
    assert result.exit_code == 0
    assert "--layers" in result.output
    assert "--build-dir" in result.output
    assert "--limit" in result.output


def test_build_command_exists(runner):
    """synix build --help succeeds (renamed from 'run')."""
    result = runner.invoke(main, ["build", "--help"])
    assert result.exit_code == 0
    assert "PIPELINE_PATH" in result.output


def test_list_command_exists(runner):
    """synix list --help succeeds."""
    result = runner.invoke(main, ["list", "--help"])
    assert result.exit_code == 0
    assert "LAYER" in result.output


def test_show_command_exists(runner):
    """synix show --help succeeds."""
    result = runner.invoke(main, ["show", "--help"])
    assert result.exit_code == 0
    assert "ARTIFACT_ID" in result.output


def test_list_empty_build_dir(runner, tmp_path):
    """synix list with empty build dir shows no artifacts."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    result = runner.invoke(main, ["list", "--build-dir", str(build_dir)])
    assert result.exit_code == 0
    assert "No artifacts" in result.output


def test_show_nonexistent_artifact(runner, tmp_path):
    """synix show with nonexistent artifact gives clear error."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    result = runner.invoke(main, ["show", "fake-id", "--build-dir", str(build_dir)])
    assert result.exit_code != 0 or "not found" in result.output


def test_list_with_layer_filter(runner, tmp_path):
    """synix list with a nonexistent layer shows no results."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    result = runner.invoke(main, ["list", "nonexistent", "--build-dir", str(build_dir)])
    assert result.exit_code == 0
    assert "No artifacts" in result.output


def test_verify_command_exists(runner):
    """synix verify --help succeeds."""
    result = runner.invoke(main, ["verify", "--help"])
    assert result.exit_code == 0
    assert "--build-dir" in result.output


def test_verify_pipeline_option_exists(runner):
    """synix verify --help shows --pipeline option."""
    result = runner.invoke(main, ["verify", "--help"])
    assert result.exit_code == 0
    assert "--pipeline" in result.output


def test_verify_with_nonexistent_pipeline(runner, tmp_path):
    """synix verify --pipeline with nonexistent file errors."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    result = runner.invoke(main, [
        "verify", "--build-dir", str(build_dir),
        "--pipeline", str(tmp_path / "nonexistent.py"),
    ])
    # Click validates exists=True before the command runs
    assert result.exit_code != 0


def test_info_command_exists(runner):
    """synix info --help succeeds."""
    result = runner.invoke(main, ["info", "--help"])
    assert result.exit_code == 0
    assert "system information" in result.output.lower() or "configuration" in result.output.lower()


def test_info_command_runs(runner, tmp_path, monkeypatch):
    """synix info runs without crashing (no pipeline or build dir)."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(main, ["info"])
    assert result.exit_code == 0
    assert "build system for agent memory" in result.output.lower()
    assert "Version" in result.output
    assert "Python" in result.output
    assert "Platform" in result.output
    assert "No pipeline.py" in result.output


def test_info_shows_logo(runner, tmp_path, monkeypatch):
    """synix info displays the ASCII art logo."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(main, ["info"])
    assert result.exit_code == 0
    # The logo contains these distinctive Unicode box-drawing characters
    assert "███" in result.output


def test_build_does_not_import_search():
    """The build module must not directly import search — uses projection registry instead.

    FR-2.3: Build pipeline should not depend on the search module. The search
    projection registers itself via @register_projection and the runner looks
    it up via get_projection(). Direct imports from synix.search in synix.build
    would violate this decoupling.
    """
    import sys

    # Clear any cached imports of the build modules
    build_modules = [k for k in sys.modules if k.startswith("synix.build")]
    search_modules = [k for k in sys.modules if k.startswith("synix.search")]
    saved = {}
    for mod in build_modules + search_modules:
        saved[mod] = sys.modules.pop(mod)

    try:
        # Read source files directly to check for search imports
        from pathlib import Path
        build_dir = Path(__file__).parent.parent.parent / "src" / "synix" / "build"
        for py_file in build_dir.glob("*.py"):
            source = py_file.read_text()
            lines = source.split("\n")
            for i, line in enumerate(lines, 1):
                # Skip comments
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # Check for direct imports from synix.search
                if "from synix.search" in stripped or "import synix.search" in stripped:
                    # Allow the lazy import in llm_transforms (topical rollup needs SearchIndex at runtime)
                    if "llm_transforms" in py_file.name and "SearchIndex" in stripped:
                        continue
                    raise AssertionError(
                        f"Build module {py_file.name}:{i} directly imports search: {stripped!r}"
                    )
    finally:
        # Restore modules
        sys.modules.update(saved)
