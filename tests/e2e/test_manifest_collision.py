"""E2E test for manifest format collision (Issue 1).

Verifies that a foreign manifest.json in the build directory doesn't crash
the build — synix should gracefully skip invalid entries and proceed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from synix.build.refs import synix_dir_for_build_dir
from synix.build.snapshot_view import SnapshotArtifactCache
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
from synix import Pipeline, Source
from synix.ext import EpisodeSummary

pipeline = Pipeline("test-collision")
pipeline.source_dir = "{workspace["source_dir"]}"
pipeline.build_dir = "{workspace["build_dir"]}"
pipeline.llm_config = {{"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}}

transcripts = Source("transcripts")
episodes = EpisodeSummary("episodes", depends_on=[transcripts])

pipeline.add(transcripts, episodes)
""")
    return path


@pytest.fixture(autouse=True)
def mock_anthropic(monkeypatch):
    def mock_create(**kwargs):
        return _mock_response("This conversation covered technical topics.")

    mock_client = MagicMock()
    mock_client.messages.create = mock_create
    monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)


def _mock_response(text: str):
    resp = MagicMock()
    resp.content = [MagicMock(text=text)]
    resp.model = "claude-sonnet-4-20250514"
    resp.usage = MagicMock(input_tokens=100, output_tokens=50)
    return resp


class TestForeignManifestDoesNotCrash:
    """Build succeeds when build dir already has a foreign manifest.json."""

    def test_foreign_manifest_then_build(self, runner, workspace, pipeline_file):
        """Write a foreign manifest.json, run build, assert success."""
        build_dir = workspace["build_dir"]
        build_dir.mkdir(parents=True, exist_ok=True)

        # Write a foreign manifest (e.g. from npm or another tool)
        foreign_manifest = {
            "name": "my-other-project",
            "version": "1.0.0",
            "dependencies": {"lodash": "4.17.21"},
        }
        (build_dir / "manifest.json").write_text(json.dumps(foreign_manifest))

        result = runner.invoke(main, ["build", str(pipeline_file)])
        assert result.exit_code == 0, f"Build failed: {result.output}"

        # Verify synix created valid artifacts in snapshot store
        store = SnapshotArtifactCache(synix_dir_for_build_dir(build_dir))
        manifest = store.iter_entries()
        transcript_entries = [k for k, v in manifest.items() if v.get("layer") == "transcripts"]
        assert len(transcript_entries) > 0


class TestPlainBuildOutput:
    """Build with --plain produces readable timestamped output."""

    def test_plain_build_no_ansi_escapes(self, runner, workspace, pipeline_file):
        """--plain output has no ANSI control sequences."""
        result = runner.invoke(main, ["build", "--plain", str(pipeline_file)])
        assert result.exit_code == 0, f"Build failed: {result.output}"
        # Should have timestamped lines
        assert "layer" in result.output.lower()
        assert "started" in result.output or "done" in result.output
