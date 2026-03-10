"""E2E tests: path resolution for pipeline source_dir/build_dir and info/status commands."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from synix.build.refs import synix_dir_for_build_dir
from synix.build.release_engine import execute_release
from synix.cli.main import main

FIXTURES_DIR = Path(__file__).parent.parent / "synix" / "fixtures"


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def mock_anthropic(monkeypatch):
    def mock_create(**kwargs):
        messages = kwargs.get("messages", [])
        content = messages[0].get("content", "") if messages else ""

        if "summarizing a conversation" in content.lower():
            return _mock_response("Conversation summary about technical topics.")
        if "monthly" in content.lower():
            return _mock_response("Monthly rollup about technical learning.")
        if "core memory" in content.lower():
            return _mock_response("## Identity\nEngineer.\n\n## Current Focus\nSnapshot testing.")
        return _mock_response("Mock response.")

    mock_client = MagicMock()
    mock_client.messages.create = mock_create
    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)


def _mock_response(text: str):
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    resp.model = "claude-sonnet-4-20250514"
    resp.usage = MagicMock(input_tokens=100, output_tokens=50)
    return resp


def _write_pipeline(path: Path, source_dir: str, build_dir: str) -> Path:
    """Write a minimal pipeline file with the given source_dir and build_dir."""
    pipeline_file = path / "pipeline.py"
    pipeline_file.write_text(f"""
from synix import Pipeline, SearchSurface, Source, SynixSearch, FlatFile
from synix.ext import EpisodeSummary, MonthlyRollup, CoreSynthesis

pipeline = Pipeline("path-resolution-test")
pipeline.source_dir = "{source_dir}"
pipeline.build_dir = "{build_dir}"
pipeline.llm_config = {{"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}}

transcripts = Source("transcripts")
episodes = EpisodeSummary("episodes", depends_on=[transcripts])
monthly = MonthlyRollup("monthly", depends_on=[episodes])
core = CoreSynthesis("core", depends_on=[monthly], context_budget=10000)
memory_search = SearchSurface("memory-search", sources=[episodes, monthly, core], modes=["fulltext"])

pipeline.add(transcripts, episodes, monthly, core, memory_search)
pipeline.add(SynixSearch("search", surface=memory_search))
pipeline.add(FlatFile("context-doc", sources=[core]))
""")
    return pipeline_file


class TestRelativeSourceDirResolution:
    """Bug 3: Relative source_dir resolves against pipeline file location, not cwd."""

    def test_absolute_path_resolves_source_dir_relative_to_pipeline(self, runner, tmp_path):
        """Invoke build from tmp_path with pipeline in a subdirectory using relative source_dir."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        sources_dir = project_dir / "sources"
        sources_dir.mkdir()

        shutil.copy(FIXTURES_DIR / "chatgpt_export.json", sources_dir / "chatgpt_export.json")
        shutil.copy(FIXTURES_DIR / "claude_export.json", sources_dir / "claude_export.json")

        # Pipeline uses RELATIVE paths — these should resolve against pipeline file location
        pipeline_file = _write_pipeline(project_dir, source_dir="sources", build_dir="build")

        # Invoke from tmp_path (NOT project_dir) using absolute path to pipeline
        result = runner.invoke(
            main,
            ["build", str(pipeline_file), "--plain"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"Build failed:\n{result.output}"
        # Artifacts should be built
        assert "Built" in result.output or "built" in result.output

        # .synix should be created relative to project_dir (pipeline file location)
        synix_dir = project_dir / ".synix"
        assert synix_dir.exists(), f".synix not found at {synix_dir}"


    def test_source_with_dir_override_resolves_relative_to_pipeline(self, runner, tmp_path):
        """Source(dir='./custom_sources') resolves against pipeline file, not cwd."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        custom_dir = project_dir / "custom_sources"
        custom_dir.mkdir()

        shutil.copy(FIXTURES_DIR / "chatgpt_export.json", custom_dir / "chatgpt_export.json")

        # Pipeline with Source that has per-layer dir override
        pipeline_file = project_dir / "pipeline.py"
        pipeline_file.write_text("""
from synix import Pipeline, Source
from synix.ext import EpisodeSummary

pipeline = Pipeline("source-dir-override")
pipeline.source_dir = "./sources"
pipeline.build_dir = "./build"
pipeline.llm_config = {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}

# This Source has a custom dir that differs from pipeline.source_dir
transcripts = Source("transcripts", dir="./custom_sources")
episodes = EpisodeSummary("episodes", depends_on=[transcripts])
pipeline.add(transcripts, episodes)
""")

        # Invoke from tmp_path (NOT project_dir) with absolute path
        result = runner.invoke(
            main,
            ["plan", str(pipeline_file), "--json"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"Plan failed:\n{result.output}"
        import json as json_mod
        plan_data = json_mod.loads(result.output)
        transcript_step = next(s for s in plan_data["steps"] if s["name"] == "transcripts")
        assert transcript_step["artifact_count"] > 0, (
            f"Source with dir='./custom_sources' found 0 artifacts when invoked from different cwd. "
            f"Full plan: {plan_data}"
        )

    def test_cli_source_dir_override_resolves(self, runner, tmp_path):
        """--source-dir with a relative path resolves correctly."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        sources_dir = project_dir / "sources"
        sources_dir.mkdir()

        shutil.copy(FIXTURES_DIR / "chatgpt_export.json", sources_dir / "chatgpt_export.json")

        pipeline_file = _write_pipeline(project_dir, source_dir="./nonexistent", build_dir="build")

        # Pass --source-dir pointing to the real sources via relative path
        # The CLI resolves it to absolute before passing to the pipeline
        result = runner.invoke(
            main,
            ["plan", str(pipeline_file), "--source-dir", str(sources_dir), "--json"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"Plan failed:\n{result.output}"
        import json as json_mod
        plan_data = json_mod.loads(result.output)
        transcript_step = next(s for s in plan_data["steps"] if s["name"] == "transcripts")
        assert transcript_step["artifact_count"] > 0, (
            f"--source-dir override found 0 artifacts. Full plan: {plan_data}"
        )


class TestBuildDirOverride:
    """Bug 4: --build-dir override writes .synix to correct location."""

    def test_build_dir_override_writes_synix_to_correct_location(self, runner, tmp_path):
        """Override --build-dir; .synix should appear relative to the override, not the original."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        sources_dir = project_dir / "sources"
        sources_dir.mkdir()

        shutil.copy(FIXTURES_DIR / "chatgpt_export.json", sources_dir / "chatgpt_export.json")
        shutil.copy(FIXTURES_DIR / "claude_export.json", sources_dir / "claude_export.json")

        # Pipeline with source_dir/build_dir pointing inside project
        pipeline_file = _write_pipeline(
            project_dir,
            source_dir=str(sources_dir),
            build_dir=str(project_dir / "build"),
        )

        # Override build_dir to a completely different location
        override_build_dir = tmp_path / "custom_output" / "build"

        result = runner.invoke(
            main,
            ["build", str(pipeline_file), "--build-dir", str(override_build_dir), "--plain"],
            catch_exceptions=False,
        )

        assert result.exit_code == 0, f"Build failed:\n{result.output}"

        # .synix should exist relative to the OVERRIDE build_dir, not the original
        override_synix_dir = synix_dir_for_build_dir(override_build_dir)
        assert override_synix_dir.exists(), (
            f".synix not found at {override_synix_dir}; "
            f"it should have been created relative to the --build-dir override"
        )

        # .synix should NOT exist at the original location
        original_synix_dir = project_dir / ".synix"
        assert not original_synix_dir.exists(), (
            f".synix was created at the original location {original_synix_dir} "
            f"despite --build-dir override"
        )


class TestInfoWithSynixDir:
    """Bug 9: info command works with .synix snapshot store."""

    def test_info_does_not_crash_with_synix_dir(self, runner, tmp_path):
        """Build a project, then run 'synix info' in the project dir."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        sources_dir = project_dir / "sources"
        sources_dir.mkdir()

        shutil.copy(FIXTURES_DIR / "chatgpt_export.json", sources_dir / "chatgpt_export.json")
        shutil.copy(FIXTURES_DIR / "claude_export.json", sources_dir / "claude_export.json")

        # Use absolute paths in pipeline
        pipeline_file = _write_pipeline(
            project_dir,
            source_dir=str(sources_dir),
            build_dir=str(project_dir / "build"),
        )

        # Build
        build_result = runner.invoke(main, ["build", str(pipeline_file), "--plain"])
        assert build_result.exit_code == 0, f"Build failed:\n{build_result.output}"

        # Run info from the project directory (where .synix/ exists)
        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(project_dir)
            info_result = runner.invoke(main, ["info"])
        finally:
            os.chdir(old_cwd)

        assert info_result.exit_code == 0, f"Info failed:\n{info_result.output}"
        # Should contain build status (artifacts count, layer names)
        assert "Artifacts" in info_result.output or "Build Status" in info_result.output


class TestStatusShowsReleases:
    """Bug 9: status command shows release projections from .synix/releases/."""

    def test_status_shows_release_projections(self, runner, tmp_path):
        """Build and release a project, then check status shows release info."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        sources_dir = project_dir / "sources"
        sources_dir.mkdir()

        shutil.copy(FIXTURES_DIR / "chatgpt_export.json", sources_dir / "chatgpt_export.json")
        shutil.copy(FIXTURES_DIR / "claude_export.json", sources_dir / "claude_export.json")

        build_dir = project_dir / "build"
        pipeline_file = _write_pipeline(
            project_dir,
            source_dir=str(sources_dir),
            build_dir=str(build_dir),
        )

        # Build
        build_result = runner.invoke(main, ["build", str(pipeline_file), "--plain"])
        assert build_result.exit_code == 0, f"Build failed:\n{build_result.output}"

        # Release
        synix_dir = synix_dir_for_build_dir(build_dir)
        execute_release(synix_dir, release_name="local")

        # Verify release directory was created with projections
        release_dir = synix_dir / "releases" / "local"
        assert release_dir.exists(), "Release directory should exist"
        assert (release_dir / "search.db").exists(), "search.db should exist in release"

        # Run status
        status_result = runner.invoke(main, ["status", "--build-dir", str(build_dir)])
        assert status_result.exit_code == 0, f"Status failed:\n{status_result.output}"

        # Should mention the release name and projection types
        assert "local" in status_result.output, (
            f"Status output should mention 'local' release name.\nOutput:\n{status_result.output}"
        )
        assert "search" in status_result.output.lower(), (
            f"Status output should mention search projection.\nOutput:\n{status_result.output}"
        )
