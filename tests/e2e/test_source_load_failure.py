"""E2E tests for source load failure handling (Bug 5).

Verifies that:
- synix build exits nonzero when a source fails to load
- synix plan --json reports error status for failed sources
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from synix.cli.main import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def bad_source_pipeline(tmp_path):
    """Pipeline file with a custom Source subclass that raises on load()."""
    build_dir = tmp_path / "build"

    pipeline_file = tmp_path / "pipeline.py"
    pipeline_file.write_text(f"""\
from synix import Pipeline, Source


class FailingSource(Source):
    \"\"\"Source that always raises on load().\"\"\"

    def load(self, config):
        raise RuntimeError("source directory is corrupt")


pipeline = Pipeline("bad-source-test")
pipeline.source_dir = {str(tmp_path / "sources")!r}
pipeline.build_dir = {str(build_dir)!r}
pipeline.llm_config = {{"model": "test", "temperature": 0.3}}

transcripts = FailingSource("transcripts")
pipeline.add(transcripts)
""")
    return pipeline_file, tmp_path


class TestBuildWithBadSourceExitsNonzero:
    def test_build_with_bad_source_exits_nonzero(self, runner, bad_source_pipeline):
        """Build with a source that fails to load should exit nonzero."""
        pipeline_file, tmp_path = bad_source_pipeline

        result = runner.invoke(main, ["build", str(pipeline_file), "--plain"])
        assert result.exit_code != 0, (
            f"Expected nonzero exit code for build with bad source, got 0.\n"
            f"Output: {result.output}"
        )

    def test_build_with_bad_source_no_snapshot_committed(self, runner, bad_source_pipeline):
        """Build with a source that fails should not commit a snapshot."""
        pipeline_file, tmp_path = bad_source_pipeline

        runner.invoke(main, ["build", str(pipeline_file), "--plain"])

        # No .synix/refs/heads/main should exist
        synix_dir = tmp_path / ".synix"
        heads_main = synix_dir / "refs" / "heads" / "main"
        assert not heads_main.exists(), (
            f"Expected no snapshot committed, but refs/heads/main exists at {heads_main}"
        )


class TestPlanWithBadSourceShowsErrorStatus:
    def test_plan_with_bad_source_shows_error_status(self, runner, bad_source_pipeline):
        """Plan with a source that fails should report error status in JSON."""
        pipeline_file, tmp_path = bad_source_pipeline

        result = runner.invoke(main, ["plan", str(pipeline_file), "--json"])
        assert result.exit_code == 0, (
            f"Plan should succeed (report errors, not crash).\nOutput: {result.output}"
        )

        parsed = json.loads(result.output)
        steps = parsed["steps"]
        assert len(steps) >= 1

        source_step = steps[0]
        assert source_step["name"] == "transcripts"
        assert source_step["status"] == "error"
        assert "source load failed" in source_step["reason"]
        assert source_step["artifact_count"] == 0
