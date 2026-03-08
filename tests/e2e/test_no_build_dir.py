"""E2E test: verify build/ is no longer created, .synix/ is the single source of truth."""

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


@pytest.fixture
def workspace(tmp_path):
    source_dir = tmp_path / "exports"
    source_dir.mkdir()
    build_dir = tmp_path / "build"

    shutil.copy(FIXTURES_DIR / "chatgpt_export.json", source_dir / "chatgpt_export.json")
    shutil.copy(FIXTURES_DIR / "claude_export.json", source_dir / "claude_export.json")

    return {"root": tmp_path, "source_dir": source_dir, "build_dir": build_dir}


@pytest.fixture
def pipeline_file(workspace):
    path = workspace["root"] / "pipeline.py"
    path.write_text(f"""
from synix import Pipeline, SearchSurface, Source, SynixSearch, FlatFile
from synix.ext import EpisodeSummary, MonthlyRollup, CoreSynthesis

pipeline = Pipeline("no-build-dir-test")
pipeline.source_dir = "{workspace["source_dir"]}"
pipeline.build_dir = "{workspace["build_dir"]}"
pipeline.llm_config = {{"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}}

transcripts = Source("transcripts")
episodes = EpisodeSummary("episodes", depends_on=[transcripts])
monthly = MonthlyRollup("monthly", depends_on=[episodes])
core = CoreSynthesis("core", depends_on=[monthly], context_budget=10000)
memory_search = SearchSurface("memory-search", sources=[episodes, monthly, core], modes=["fulltext"])

pipeline.add(transcripts, episodes, monthly, core, memory_search)
pipeline.add(SynixSearch("search", surface=memory_search))
pipeline.add(FlatFile("context-doc", sources=[core], output_path="{workspace["build_dir"] / "context.md"}"))
""")
    return path


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


class TestBuildCreatesOnlySynixDir:
    def test_build_creates_only_synix_dir(self, runner, workspace, pipeline_file):
        """Build pipeline, verify NO build/ directory exists, only .synix/."""
        result = runner.invoke(main, ["run", str(pipeline_file), "--plain"])
        assert result.exit_code == 0, result.output

        build_dir = workspace["build_dir"]
        synix_dir = workspace["root"] / ".synix"

        # .synix should exist
        assert synix_dir.exists(), ".synix directory should be created"

        # build/ should NOT be created (Phase 12+: build/ is no longer written to)
        assert not build_dir.exists(), f"build/ directory should not exist after Phase 12, but found: {build_dir}"


class TestSynixDirStructure:
    def test_synix_dir_contains_expected_structure(self, runner, workspace, pipeline_file):
        """Build pipeline, verify .synix/ has objects/, refs/, and work/ directories."""
        result = runner.invoke(main, ["run", str(pipeline_file), "--plain"])
        assert result.exit_code == 0, result.output

        synix_dir = workspace["root"] / ".synix"
        assert (synix_dir / "objects").exists(), "objects/ should exist in .synix/"
        assert (synix_dir / "refs").exists(), "refs/ should exist in .synix/"
        # work/ is created during build for search surfaces
        assert (synix_dir / "work").exists(), "work/ should exist in .synix/"

    def test_release_creates_under_synix_releases(self, runner, workspace, pipeline_file):
        """Build + release, verify .synix/releases/local/ has search.db and context.md."""
        result = runner.invoke(main, ["run", str(pipeline_file), "--plain"])
        assert result.exit_code == 0, result.output

        synix_dir = synix_dir_for_build_dir(workspace["build_dir"])
        execute_release(synix_dir, release_name="local")

        release_dir = synix_dir / "releases" / "local"
        assert release_dir.exists(), "releases/local/ should be created"
        assert (release_dir / "search.db").exists(), "search.db should exist in release"
        assert (release_dir / "context.md").exists(), "context.md should exist in release"
        assert (release_dir / "receipt.json").exists(), "receipt.json should exist in release"


class TestListWithoutBuildDir:
    def test_list_works_without_build_dir(self, runner, workspace, pipeline_file):
        """Build, verify synix list works using --synix-dir (no build/ needed)."""
        result = runner.invoke(main, ["run", str(pipeline_file), "--plain"])
        assert result.exit_code == 0, result.output

        synix_dir = workspace["root"] / ".synix"

        # List using --synix-dir directly (bypasses build_dir resolution)
        list_result = runner.invoke(main, ["list", "--synix-dir", str(synix_dir)])
        assert list_result.exit_code == 0, list_result.output
        # Should show artifacts grouped by layer
        assert "transcripts" in list_result.output.lower() or "episodes" in list_result.output.lower()


class TestShowWithoutBuildDir:
    def test_show_works_without_build_dir(self, runner, workspace, pipeline_file):
        """Build, verify synix show works using --synix-dir (no build/ needed)."""
        result = runner.invoke(main, ["run", str(pipeline_file), "--plain"])
        assert result.exit_code == 0, result.output

        synix_dir = workspace["root"] / ".synix"

        # Show a specific artifact — core-memory is always produced
        show_result = runner.invoke(main, ["show", "core-memory", "--synix-dir", str(synix_dir)])
        assert show_result.exit_code == 0, show_result.output
        # Should display core memory content (Layer: core visible in metadata header)
        assert "core" in show_result.output.lower()


class TestSnapshotRefsWithoutBuildDir:
    def test_runs_list_works_via_synix_dir(self, runner, workspace, pipeline_file):
        """Build twice, verify runs list works with --synix-dir."""
        # First build
        result1 = runner.invoke(main, ["run", str(pipeline_file), "--plain"])
        assert result1.exit_code == 0, result1.output

        # Second build (fully cached, same pipeline)
        result2 = runner.invoke(main, ["run", str(pipeline_file), "--plain"])
        assert result2.exit_code == 0, result2.output

        synix_dir = workspace["root"] / ".synix"

        runs_result = runner.invoke(main, ["runs", "list", "--synix-dir", str(synix_dir)])
        assert runs_result.exit_code == 0, runs_result.output
        assert "Run Artifact Snapshots" in runs_result.output or "Run ID" in runs_result.output
