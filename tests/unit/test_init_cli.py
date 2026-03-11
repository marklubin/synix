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
    assert "--template" in result.output
    assert "--list" in result.output


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
    assert (project / "sources" / "session-example.md").is_file()


def test_init_source_content(runner, tmp_path, monkeypatch):
    """Source files contain expected content."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(main, ["init", "test-proj"])

    session = (tmp_path / "test-proj" / "sources" / "session-example.md").read_text()
    assert "Session" in session
    assert "User:" in session or "**User:**" in session


def test_init_env_example(runner, tmp_path, monkeypatch):
    """synix init includes .env.example with supported API key fields."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(main, ["init", "env-test"])
    env_example = (tmp_path / "env-test" / ".env.example").read_text()
    assert "ANTHROPIC_API_KEY" in env_example
    assert "OPENAI_API_KEY" in env_example
    assert "DEEPSEEK_API_KEY" in env_example


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
    assert "synix search" in readme


def test_init_pipeline_loadable(runner, tmp_path, monkeypatch):
    """The generated pipeline.py can be loaded by Synix."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(main, ["init", "load-test"])

    from synix.build.pipeline import load_pipeline

    pipeline = load_pipeline(str(tmp_path / "load-test" / "pipeline.py"))
    assert pipeline.name == "agent-memory"
    layer_names = [l.name for l in pipeline.layers]
    assert "transcripts" in layer_names
    assert "episodes" in layer_names
    assert "monthly" in layer_names
    assert "core" in layer_names
    assert len(pipeline.projections) >= 1
    assert len(pipeline.validators) == 0


def test_init_pipeline_dag_structure(runner, tmp_path, monkeypatch):
    """The pipeline DAG has correct dependencies and levels."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(main, ["init", "dag-test"])

    from synix.build.pipeline import load_pipeline

    pipeline = load_pipeline(str(tmp_path / "dag-test" / "pipeline.py"))
    by_name = {l.name: l for l in pipeline.layers}

    # Single root source
    assert by_name["transcripts"]._level == 0
    assert by_name["transcripts"].depends_on == []

    # 1:1 episode summaries
    assert by_name["episodes"]._level == 1
    assert [d.name for d in by_name["episodes"].depends_on] == ["transcripts"]

    # Monthly rollup
    assert by_name["monthly"]._level == 2
    assert [d.name for d in by_name["monthly"].depends_on] == ["episodes"]

    # Core synthesis
    assert by_name["core"]._level == 3
    assert [d.name for d in by_name["core"].depends_on] == ["monthly"]


def test_init_output_message(runner, tmp_path, monkeypatch):
    """synix init prints helpful next-steps including release workflow."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(main, ["init", "my-project"])
    assert "cd my-project" in result.output
    assert ".env.example" in result.output
    assert "synix build" in result.output
    assert "synix release HEAD --to local" in result.output
    assert "synix search" in result.output
    assert "--release local" in result.output


# ---------------------------------------------------------------------------
# Integration test — init + build + validate + search (mocked LLM)
# ---------------------------------------------------------------------------


def _make_mock_response(text):
    """Create a mock LLM response with given text."""
    resp = MagicMock()
    resp.content = text
    resp.input_tokens = 100
    resp.output_tokens = 50
    return resp


def test_init_build_validate_search_e2e(runner, tmp_path, monkeypatch):
    """Full flow: init -> build -> release -> search with mocked LLM."""
    monkeypatch.chdir(tmp_path)

    # 1. Init
    result = runner.invoke(main, ["init", "e2e-test"])
    assert result.exit_code == 0

    project_dir = tmp_path / "e2e-test"
    pipeline_path = str(project_dir / "pipeline.py")
    monkeypatch.chdir(project_dir)

    # 2. Build with mocked LLM — returns different content per call
    call_count = 0
    mock_responses = [
        # 1x episode summary (one per source, via EpisodeSummary)
        _make_mock_response(
            "This session covered CI/CD pipeline setup for a FastAPI + PostgreSQL "
            "microservice. Key decisions: rolling updates, Alembic migrations with "
            "timeout safeguards, staging environment with manual approval gate, "
            "and GitHub Actions secrets for credential management."
        ),
        # 1x monthly rollup (via MonthlyRollup)
        _make_mock_response(
            "March 2026: Focus on infrastructure and deployment. The team "
            "established CI/CD practices with a staging-to-production pipeline "
            "including migration safeguards and secrets management."
        ),
        # 1x core synthesis (via CoreSynthesis)
        _make_mock_response(
            "Core knowledge: Engineering team of 3 uses FastAPI + PostgreSQL. "
            "Deployment follows a staged pipeline with Alembic migrations, "
            "manual approval gates, and GitHub Actions for CI/CD. Security "
            "approach: environment variables for secrets, no secrets in images."
        ),
    ]

    def mock_complete(**kwargs):
        nonlocal call_count
        resp = mock_responses[min(call_count, len(mock_responses) - 1)]
        call_count += 1
        return resp

    with patch("synix.build.llm_client.LLMClient.complete", side_effect=mock_complete):
        result = runner.invoke(
            main,
            [
                "build",
                pipeline_path,
                "--source-dir",
                str(project_dir / "sources"),
                "--build-dir",
                str(project_dir / "build"),
            ],
        )
        assert result.exit_code == 0, f"Build failed: {result.output}"

    # Verify all 3 LLM calls: 1 episode + 1 monthly rollup + 1 core synthesis
    assert call_count == 3

    # 2b. Release — materialize projections (search.db, context.md) into a release target
    result = runner.invoke(
        main,
        [
            "release",
            "HEAD",
            "--to",
            "local",
            "--build-dir",
            str(project_dir / "build"),
        ],
    )
    assert result.exit_code == 0, f"Release failed: {result.output}"

    # 3. Search — should find deployment (from session transcript and episode)
    #    Search uses the released search.db in .synix/releases/local/
    result = runner.invoke(
        main,
        [
            "search",
            "deployment",
            "--build-dir",
            str(project_dir / "build"),
            "--release",
            "local",
        ],
    )
    assert result.exit_code == 0, f"Search failed: {result.output}"
    assert "deployment" in result.output.lower()


def test_init_pipeline_has_no_validators(runner, tmp_path, monkeypatch):
    """Init scaffold pipeline declares no validators (experimental feature)."""
    monkeypatch.chdir(tmp_path)

    runner.invoke(main, ["init", "valid-test"])
    project_dir = tmp_path / "valid-test"
    pipeline_path = str(project_dir / "pipeline.py")

    from synix.build.pipeline import load_pipeline

    pipeline = load_pipeline(pipeline_path)
    assert pipeline.validators == []
    assert pipeline.fixers == []
