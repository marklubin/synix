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
