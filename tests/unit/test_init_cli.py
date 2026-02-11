"""Tests for synix init command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from synix.cli import main


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# Unit tests — CLI behaviour
# ---------------------------------------------------------------------------

def test_init_help(runner):
    """synix init --help succeeds."""
    result = runner.invoke(main, ["init", "--help"])
    assert result.exit_code == 0
    assert "PROJECT_NAME" in result.output


def test_init_creates_project(runner, tmp_path, monkeypatch):
    """synix init creates the expected directory structure."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(main, ["init", "my-project"])
    assert result.exit_code == 0

    project = tmp_path / "my-project"
    assert project.is_dir()
    assert (project / "pipeline.py").is_file()
    assert (project / "README.md").is_file()
    assert (project / "sources").is_dir()
    assert (project / "sources" / "alice.md").is_file()
    assert (project / "sources" / "bob.md").is_file()
    assert (project / "sources" / "carol.md").is_file()


def test_init_source_content(runner, tmp_path, monkeypatch):
    """Source files contain expected content."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(main, ["init", "test-proj"])

    alice = (tmp_path / "test-proj" / "sources" / "alice.md").read_text()
    assert "Alice" in alice
    assert "hiking" in alice.lower()

    bob = (tmp_path / "test-proj" / "sources" / "bob.md").read_text()
    assert "Bob" in bob

    carol = (tmp_path / "test-proj" / "sources" / "carol.md").read_text()
    assert "Carol" in carol


def test_init_existing_dir_errors(runner, tmp_path, monkeypatch):
    """synix init errors if directory already exists."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "existing").mkdir()
    result = runner.invoke(main, ["init", "existing"])
    assert result.exit_code != 0
    assert "already exists" in result.output


def test_init_readme_content(runner, tmp_path, monkeypatch):
    """README.md contains the expected quick-start instruction."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(main, ["init", "proj"])
    readme = (tmp_path / "proj" / "README.md").read_text()
    assert "synix build" in readme
    assert "synix validate" in readme
    assert "synix search" in readme


def test_init_pipeline_loadable(runner, tmp_path, monkeypatch):
    """The generated pipeline.py can be loaded by Synix."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(main, ["init", "load-test"])

    from synix.build.pipeline import load_pipeline

    pipeline = load_pipeline(str(tmp_path / "load-test" / "pipeline.py"))
    assert pipeline.name == "my-first-pipeline"
    assert len(pipeline.layers) == 2
    assert pipeline.layers[0].name == "bios"
    assert pipeline.layers[1].name == "team_profile"
    assert len(pipeline.projections) == 1
    assert len(pipeline.validators) == 1


def test_init_output_message(runner, tmp_path, monkeypatch):
    """synix init prints helpful next-steps."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(main, ["init", "my-project"])
    assert "cd my-project" in result.output
    assert "synix build pipeline.py" in result.output


# ---------------------------------------------------------------------------
# Integration test — init + build + validate + search (mocked LLM)
# ---------------------------------------------------------------------------

def test_init_build_validate_search_e2e(runner, tmp_path, monkeypatch):
    """Full flow: init → build → validate → search with mocked LLM."""
    monkeypatch.chdir(tmp_path)

    # 1. Init
    result = runner.invoke(main, ["init", "e2e-test"])
    assert result.exit_code == 0

    project_dir = tmp_path / "e2e-test"
    pipeline_path = str(project_dir / "pipeline.py")

    # 2. Build with mocked LLM
    monkeypatch.chdir(project_dir)

    mock_response = MagicMock()
    mock_response.content = (
        "This small team spans Portland, Austin, and Chicago. "
        "Alice Chen is a backend engineer who enjoys hiking, "
        "Bob Martinez is a product designer who teaches pottery, "
        "and Carol Okafor is a data scientist and STEM mentor."
    )
    mock_response.input_tokens = 100
    mock_response.output_tokens = 50

    with patch("synix.build.llm_client.LLMClient.complete", return_value=mock_response):
        result = runner.invoke(main, [
            "build", pipeline_path,
            "--source-dir", str(project_dir / "sources"),
            "--build-dir", str(project_dir / "build"),
        ])
        assert result.exit_code == 0, f"Build failed: {result.output}"

    # 3. Validate — should pass (mock response is under 2000 chars)
    result = runner.invoke(main, [
        "validate", pipeline_path,
        "--build-dir", str(project_dir / "build"),
    ])
    assert result.exit_code == 0, f"Validate failed: {result.output}"

    # 4. Search — should find hiking
    result = runner.invoke(main, [
        "search", "hiking",
        "--build-dir", str(project_dir / "build"),
    ])
    assert result.exit_code == 0, f"Search failed: {result.output}"
    assert "hiking" in result.output.lower()


def test_init_validate_fails_on_long_content(runner, tmp_path, monkeypatch):
    """Validate catches content that exceeds max_chars."""
    monkeypatch.chdir(tmp_path)

    # Init
    runner.invoke(main, ["init", "long-test"])
    project_dir = tmp_path / "long-test"
    pipeline_path = str(project_dir / "pipeline.py")
    monkeypatch.chdir(project_dir)

    # Build with an LLM response that exceeds 2000 chars
    mock_response = MagicMock()
    mock_response.content = "x" * 2500
    mock_response.input_tokens = 100
    mock_response.output_tokens = 500

    with patch("synix.build.llm_client.LLMClient.complete", return_value=mock_response):
        result = runner.invoke(main, [
            "build", pipeline_path,
            "--source-dir", str(project_dir / "sources"),
            "--build-dir", str(project_dir / "build"),
        ])
        assert result.exit_code == 0

    # Validate should fail
    result = runner.invoke(main, [
        "validate", pipeline_path,
        "--build-dir", str(project_dir / "build"),
    ])
    assert result.exit_code != 0
    assert "max_length" in result.output.lower() or "2500" in result.output
