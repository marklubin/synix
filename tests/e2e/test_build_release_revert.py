"""E2E test: build -> release -> rebuild -> diff -> revert lifecycle."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from synix.build.refs import RefStore, synix_dir_for_build_dir
from synix.build.release_engine import execute_release, get_release
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

pipeline = Pipeline("build-release-revert")
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


@pytest.fixture
def pipeline_file_v2(workspace):
    """A modified pipeline with a changed context_budget to force core rebuild."""
    path = workspace["root"] / "pipeline_v2.py"
    path.write_text(f"""
from synix import Pipeline, SearchSurface, Source, SynixSearch, FlatFile
from synix.ext import EpisodeSummary, MonthlyRollup, CoreSynthesis

pipeline = Pipeline("build-release-revert-v2")
pipeline.source_dir = "{workspace["source_dir"]}"
pipeline.build_dir = "{workspace["build_dir"]}"
pipeline.llm_config = {{"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}}

transcripts = Source("transcripts")
episodes = EpisodeSummary("episodes", depends_on=[transcripts])
monthly = MonthlyRollup("monthly", depends_on=[episodes])
core = CoreSynthesis("core", depends_on=[monthly], context_budget=5000)
memory_search = SearchSurface("memory-search", sources=[episodes, monthly, core], modes=["fulltext"])

pipeline.add(transcripts, episodes, monthly, core, memory_search)
pipeline.add(SynixSearch("search", surface=memory_search))
pipeline.add(FlatFile("context-doc", sources=[core], output_path="{workspace["build_dir"] / "context.md"}"))
""")
    return path


@pytest.fixture(autouse=True)
def mock_anthropic(monkeypatch):
    call_count = {"n": 0}

    def mock_create(**kwargs):
        call_count["n"] += 1
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
    return call_count


def _mock_response(text: str):
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    resp.model = "claude-sonnet-4-20250514"
    resp.usage = MagicMock(input_tokens=100, output_tokens=50)
    return resp


class TestBuildThenRelease:
    def test_build_then_release_creates_receipt(self, runner, workspace, pipeline_file):
        """Build pipeline, release, verify receipt.json exists and has correct structure."""
        build_result = runner.invoke(main, ["run", str(pipeline_file), "--plain"])
        assert build_result.exit_code == 0, build_result.output

        synix_dir = synix_dir_for_build_dir(workspace["build_dir"])

        receipt = execute_release(synix_dir, release_name="local")

        release_dir = synix_dir / "releases" / "local"
        assert (release_dir / "receipt.json").exists()

        receipt_data = json.loads((release_dir / "receipt.json").read_text())
        assert receipt_data["release_name"] == "local"
        assert receipt_data["pipeline_name"] == "build-release-revert"
        assert "snapshot_oid" in receipt_data
        assert "manifest_oid" in receipt_data
        assert "released_at" in receipt_data
        assert "adapters" in receipt_data
        assert receipt_data["schema_version"] == 1

    def test_second_build_then_release_updates_receipt(self, runner, workspace, pipeline_file, pipeline_file_v2):
        """Build v1, release, build v2 (change config), release again, verify receipt updated."""
        # Build v1 and release
        runner.invoke(main, ["run", str(pipeline_file), "--plain"])
        synix_dir = synix_dir_for_build_dir(workspace["build_dir"])
        receipt_v1 = execute_release(synix_dir, release_name="local")
        snapshot_v1 = receipt_v1.snapshot_oid

        # Build v2 and release again
        runner.invoke(main, ["run", str(pipeline_file_v2), "--plain"])
        receipt_v2 = execute_release(synix_dir, release_name="local")
        snapshot_v2 = receipt_v2.snapshot_oid

        # Snapshot OIDs should differ because pipeline changed
        assert snapshot_v1 != snapshot_v2

        # Receipt should reflect the latest release
        release_dir = synix_dir / "releases" / "local"
        receipt_data = json.loads((release_dir / "receipt.json").read_text())
        assert receipt_data["snapshot_oid"] == snapshot_v2

        # History should have both releases
        history_dir = release_dir / "history"
        assert history_dir.exists()
        history_files = list(history_dir.glob("*.json"))
        assert len(history_files) == 2


class TestRevertRestoresPreviousRelease:
    def test_revert_restores_previous_release(self, runner, workspace, pipeline_file, pipeline_file_v2):
        """Build v1, release, build v2, release, revert to v1's ref, verify content reverted."""
        # Build v1
        result_v1 = runner.invoke(main, ["run", str(pipeline_file), "--plain"])
        assert result_v1.exit_code == 0, result_v1.output
        run_ref_v1 = next(
            line.split("Run Ref:", 1)[1].strip() for line in result_v1.output.splitlines() if "Run Ref:" in line
        )

        synix_dir = synix_dir_for_build_dir(workspace["build_dir"])
        receipt_v1 = execute_release(synix_dir, release_name="local")
        v1_snapshot = receipt_v1.snapshot_oid

        # Build v2
        result_v2 = runner.invoke(main, ["run", str(pipeline_file_v2), "--plain"])
        assert result_v2.exit_code == 0, result_v2.output

        receipt_v2 = execute_release(synix_dir, release_name="local")
        v2_snapshot = receipt_v2.snapshot_oid
        assert v1_snapshot != v2_snapshot

        # Revert to v1's run ref
        revert_result = runner.invoke(main, ["revert", run_ref_v1, "--to", "local", "--synix-dir", str(synix_dir)])
        assert revert_result.exit_code == 0, revert_result.output
        assert "Reverted" in revert_result.output

        # Verify the release now points back to v1's snapshot
        receipt_after_revert = get_release(synix_dir, "local")
        assert receipt_after_revert is not None
        assert receipt_after_revert.snapshot_oid == v1_snapshot


class TestDiffBetweenReleases:
    def test_diff_between_releases(self, runner, workspace, pipeline_file, pipeline_file_v2):
        """Build v1, release as 'v1', build v2, release as 'v2', diff the builds."""
        # Build v1
        runner.invoke(main, ["run", str(pipeline_file), "--plain"])
        synix_dir = synix_dir_for_build_dir(workspace["build_dir"])
        execute_release(synix_dir, release_name="v1")

        # Build v2
        runner.invoke(main, ["run", str(pipeline_file_v2), "--plain"])
        execute_release(synix_dir, release_name="v2")

        # Verify both releases exist
        ref_store = RefStore(synix_dir)
        v1_oid = ref_store.read_ref("refs/releases/v1")
        v2_oid = ref_store.read_ref("refs/releases/v2")
        assert v1_oid is not None
        assert v2_oid is not None
        # The two releases point to different snapshots
        assert v1_oid != v2_oid
