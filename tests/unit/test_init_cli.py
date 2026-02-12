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
    assert (project / "sources" / "bios").is_dir()
    assert (project / "sources" / "bios" / "alice.md").is_file()
    assert (project / "sources" / "bios" / "bob.md").is_file()
    assert (project / "sources" / "bios" / "carol.md").is_file()
    assert (project / "sources" / "brief").is_dir()
    assert (project / "sources" / "brief" / "project_brief.md").is_file()


def test_init_source_content(runner, tmp_path, monkeypatch):
    """Source files contain expected content."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(main, ["init", "test-proj"])

    alice = (tmp_path / "test-proj" / "sources" / "bios" / "alice.md").read_text()
    assert "Alice" in alice
    assert "hiking" in alice.lower()

    bob = (tmp_path / "test-proj" / "sources" / "bios" / "bob.md").read_text()
    assert "Bob" in bob

    carol = (tmp_path / "test-proj" / "sources" / "bios" / "carol.md").read_text()
    assert "Carol" in carol

    brief = (tmp_path / "test-proj" / "sources" / "brief" / "project_brief.md").read_text()
    assert "dashboard" in brief.lower()


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
    assert "synix validate" in readme
    assert "synix search" in readme


def test_init_pipeline_loadable(runner, tmp_path, monkeypatch):
    """The generated pipeline.py can be loaded by Synix."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(main, ["init", "load-test"])

    from synix.build.pipeline import load_pipeline

    pipeline = load_pipeline(str(tmp_path / "load-test" / "pipeline.py"))
    assert pipeline.name == "my-first-pipeline"
    assert len(pipeline.layers) == 5
    layer_names = [l.name for l in pipeline.layers]
    assert "bios" in layer_names
    assert "project_brief" in layer_names
    assert "work_styles" in layer_names
    assert "team_dynamics" in layer_names
    assert "final_report" in layer_names
    assert len(pipeline.projections) == 1
    assert len(pipeline.validators) == 1


def test_init_pipeline_dag_structure(runner, tmp_path, monkeypatch):
    """The pipeline DAG has correct dependencies and levels."""
    monkeypatch.chdir(tmp_path)
    runner.invoke(main, ["init", "dag-test"])

    from synix.build.pipeline import load_pipeline

    pipeline = load_pipeline(str(tmp_path / "dag-test" / "pipeline.py"))
    by_name = {l.name: l for l in pipeline.layers}

    # Two independent roots
    assert by_name["bios"].level == 0
    assert by_name["bios"].depends_on == []
    assert by_name["project_brief"].level == 0
    assert by_name["project_brief"].depends_on == []

    # 1:1 work style per bio
    assert by_name["work_styles"].level == 1
    assert by_name["work_styles"].depends_on == ["bios"]

    # Many:1 rollup
    assert by_name["team_dynamics"].level == 2
    assert by_name["team_dynamics"].depends_on == ["work_styles"]

    # Multi-source synthesis
    assert by_name["final_report"].level == 3
    assert set(by_name["final_report"].depends_on) == {"team_dynamics", "project_brief"}


def test_init_output_message(runner, tmp_path, monkeypatch):
    """synix init prints helpful next-steps."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(main, ["init", "my-project"])
    assert "cd my-project" in result.output
    assert ".env.example" in result.output
    assert "synix build" in result.output


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
    """Full flow: init → build → validate → search with mocked LLM."""
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
        # 3x work_style (one per bio)
        _make_mock_response(
            "Alice is a systematic thinker who thrives on technical depth. "
            "She naturally takes the architect role and brings hiking "
            "discipline to her engineering work."
        ),
        _make_mock_response(
            "Bob is a user-focused collaborator who bridges design and "
            "engineering. He brings startup hustle and accessibility expertise."
        ),
        _make_mock_response(
            "Carol is a rigorous analyst who leads with data. "
            "Her academic background makes her the team's methodologist."
        ),
        # 1x team_dynamics
        _make_mock_response(
            "This team combines deep backend expertise, user-centered design, "
            "and data science rigor. Alice and Carol share analytical thinking "
            "while Bob bridges technical work with user needs. The main risk "
            "is geographic distribution across three time zones."
        ),
        # 1x final_report
        _make_mock_response(
            "Alice should own the backend sensor ingestion pipeline given her "
            "distributed systems expertise. Bob should lead the dashboard UI "
            "and ensure WCAG compliance. Carol should build the data pipeline "
            "and aggregation layer. The team's skills map well to the project "
            "but they'll need strong async communication across time zones."
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

    # Verify all 5 LLM calls were made (3 work_style + 1 dynamics + 1 report)
    assert call_count == 5

    # 3. Validate — should pass (mock responses are under 5000 chars)
    result = runner.invoke(
        main,
        [
            "validate",
            pipeline_path,
            "--build-dir",
            str(project_dir / "build"),
        ],
    )
    assert result.exit_code == 0, f"Validate failed: {result.output}"

    # 4. Search — should find hiking (from bios and work_style artifacts)
    result = runner.invoke(
        main,
        [
            "search",
            "hiking",
            "--build-dir",
            str(project_dir / "build"),
        ],
    )
    assert result.exit_code == 0, f"Search failed: {result.output}"
    assert "hiking" in result.output.lower()


def test_init_validate_fails_on_long_content(runner, tmp_path, monkeypatch):
    """Validate catches final_report content that exceeds max_chars."""
    monkeypatch.chdir(tmp_path)

    # Init
    runner.invoke(main, ["init", "long-test"])
    project_dir = tmp_path / "long-test"
    pipeline_path = str(project_dir / "pipeline.py")
    monkeypatch.chdir(project_dir)

    # Build — work_styles and dynamics are short, final_report is too long
    call_count = 0
    short = _make_mock_response("Short mock response.")
    long_resp = _make_mock_response("x" * 6000)

    def mock_complete(**kwargs):
        nonlocal call_count
        call_count += 1
        # Calls 1-4 are work_styles (3) + dynamics (1), call 5 is final_report
        if call_count <= 4:
            return short
        return long_resp

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
        assert result.exit_code == 0

    # Validate should fail on the oversized final_report
    result = runner.invoke(
        main,
        [
            "validate",
            pipeline_path,
            "--build-dir",
            str(project_dir / "build"),
        ],
    )
    assert result.exit_code != 0
    assert "max_length" in result.output.lower() or "6000" in result.output
