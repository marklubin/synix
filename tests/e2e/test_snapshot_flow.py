"""E2E tests for the snapshot-aware CLI workflow."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from synix.build.snapshots import list_runs
from synix.cli import main

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
from synix import Pipeline, Source, SearchIndex, FlatFile
from synix.transforms import EpisodeSummary, MonthlyRollup, CoreSynthesis

pipeline = Pipeline("snapshot-cli")
pipeline.source_dir = "{workspace["source_dir"]}"
pipeline.build_dir = "{workspace["build_dir"]}"
pipeline.llm_config = {{"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}}

transcripts = Source("transcripts")
episodes = EpisodeSummary("episodes", depends_on=[transcripts])
monthly = MonthlyRollup("monthly", depends_on=[episodes])
core = CoreSynthesis("core", depends_on=[monthly], context_budget=10000)

pipeline.add(transcripts, episodes, monthly, core)
pipeline.add(SearchIndex("memory-index", sources=[episodes, monthly, core], search=["fulltext"]))
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


class TestSnapshotFlow:
    def test_build_outputs_snapshot_and_runs_list_shows_history(self, runner, workspace, pipeline_file):
        build_dir = str(workspace["build_dir"])

        first = runner.invoke(main, ["run", str(pipeline_file), "--plain"])
        assert first.exit_code == 0, first.output
        assert "Snapshot:" in first.output
        assert "Run Ref:" in first.output

        first_run_ref = next(line.split("Run Ref:", 1)[1].strip() for line in first.output.splitlines() if "Run Ref:" in line)

        second = runner.invoke(main, ["run", str(pipeline_file), "--plain"])
        assert second.exit_code == 0, second.output
        assert "Snapshot:" in second.output
        assert "Run Ref:" in second.output

        second_run_ref = next(
            line.split("Run Ref:", 1)[1].strip() for line in second.output.splitlines() if "Run Ref:" in line
        )
        assert second_run_ref != first_run_ref

        runs_list = runner.invoke(main, ["runs", "list", "--build-dir", build_dir], terminal_width=160)
        assert runs_list.exit_code == 0, runs_list.output
        assert "Run Snapshots" in runs_list.output
        assert "Run ID" in runs_list.output
        assert "Ref" in runs_list.output

        runs_json = runner.invoke(main, ["runs", "list", "--build-dir", build_dir, "--json"])
        assert runs_json.exit_code == 0, runs_json.output
        payload = json.loads(runs_json.output)
        assert payload["schema_version"] == 1
        assert {run_info["ref"] for run_info in payload["runs"]} == {first_run_ref, second_run_ref}

        recorded_runs = list_runs(build_dir)
        assert {run_info["ref"] for run_info in recorded_runs} == {first_run_ref, second_run_ref}

    def test_clean_removes_build_surface_but_preserves_snapshot_history(self, runner, workspace, pipeline_file):
        build_dir = str(workspace["build_dir"])
        synix_dir = workspace["root"] / ".synix"

        built = runner.invoke(main, ["run", str(pipeline_file), "--plain"])
        assert built.exit_code == 0, built.output
        assert synix_dir.exists()
        assert workspace["build_dir"].exists()

        cleaned = runner.invoke(main, ["clean", build_dir, "--yes"])
        assert cleaned.exit_code == 0, cleaned.output
        assert not workspace["build_dir"].exists()
        assert synix_dir.exists()

        runs_json = runner.invoke(main, ["runs", "list", "--build-dir", build_dir, "--json"])
        assert runs_json.exit_code == 0, runs_json.output
        payload = json.loads(runs_json.output)
        assert payload["schema_version"] == 1
        assert len(payload["runs"]) == 1
        assert payload["runs"][0]["pipeline_name"] == "snapshot-cli"
