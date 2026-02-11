"""CLI smoke tests for the fix command."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from synix.cli.main import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def build_dir(tmp_path):
    d = tmp_path / "build"
    d.mkdir()
    return d


@pytest.fixture
def simple_pipeline(tmp_path, build_dir):
    """Create a minimal pipeline file with PII validator (no fixer)."""
    pipeline_file = tmp_path / "pipeline.py"
    pipeline_file.write_text(f"""
from synix.core.models import Pipeline, Layer, ValidatorDecl

pipeline = Pipeline("test")
pipeline.build_dir = "{build_dir}"
pipeline.source_dir = "{tmp_path / "exports"}"
pipeline.add_layer(Layer(name="transcripts", level=0, transform="parse"))
pipeline.add_validator(ValidatorDecl(
    name="pii",
    config={{"layers": ["episodes"], "severity": "warning"}},
))
""")
    (tmp_path / "exports").mkdir(exist_ok=True)
    return pipeline_file


@pytest.fixture
def pipeline_with_fixer(tmp_path, build_dir):
    """Create a pipeline with both validator and fixer."""
    pipeline_file = tmp_path / "pipeline.py"
    pipeline_file.write_text(f"""
from synix.core.models import Pipeline, Layer, ValidatorDecl, FixerDecl

pipeline = Pipeline("test")
pipeline.build_dir = "{build_dir}"
pipeline.source_dir = "{tmp_path / "exports"}"
pipeline.add_layer(Layer(name="transcripts", level=0, transform="parse"))
pipeline.add_validator(ValidatorDecl(
    name="pii",
    config={{"layers": ["episodes"], "severity": "warning"}},
))
pipeline.add_fixer(FixerDecl(
    name="semantic_enrichment",
    config={{}},
))
""")
    (tmp_path / "exports").mkdir(exist_ok=True)
    return pipeline_file


class TestFixHelp:
    def test_fix_help(self, runner):
        result = runner.invoke(main, ["fix", "--help"])
        assert result.exit_code == 0
        assert "PIPELINE_PATH" in result.output
        assert "--build-dir" in result.output
        assert "--json" in result.output
        assert "--dry-run" in result.output

    def test_fix_help_description(self, runner):
        result = runner.invoke(main, ["fix", "--help"])
        assert "Fix violations" in result.output


class TestFixCommand:
    def test_fix_no_queue(self, runner, pipeline_with_fixer, build_dir):
        """No violations_state.json → 'No active violations'."""
        result = runner.invoke(
            main,
            [
                "fix",
                str(pipeline_with_fixer),
            ],
        )
        assert result.exit_code == 0
        assert "No active violations" in result.output

    def test_fix_stale_violations_expired(self, runner, simple_pipeline, build_dir):
        """Violation with hash X, artifact rebuilt with hash Y → auto-expired."""
        from synix.build.artifacts import ArtifactStore
        from synix.build.validators import Violation, ViolationQueue
        from synix.core.models import Artifact

        store = ArtifactStore(build_dir)

        # Save an artifact with original content
        art = Artifact(
            artifact_id="ep-1",
            artifact_type="episode",
            content="SSN: 123-45-6789",
            metadata={"layer_name": "episodes"},
        )
        store.save_artifact(art, "episodes", 1)
        original_hash = store.get_content_hash("ep-1")

        # Persist a violation against the original hash
        queue = ViolationQueue.load(build_dir)
        queue.upsert(
            Violation(
                violation_type="pii",
                severity="warning",
                message="PII detected (ssn)",
                artifact_id="ep-1",
                field="content",
                metadata={"content_hash": original_hash},
                violation_id="test-vid-1",
            )
        )
        queue.save_state()

        # Verify it's active before rebuild
        assert len(queue.active()) == 1

        # "Rebuild" the artifact with different content
        art2 = Artifact(
            artifact_id="ep-1",
            artifact_type="episode",
            content="No PII here, all clean.",
            metadata={"layer_name": "episodes"},
        )
        store.save_artifact(art2, "episodes", 1)

        # Now run fix — violation should be auto-expired
        result = runner.invoke(
            main,
            [
                "fix",
                str(simple_pipeline),
            ],
        )
        assert result.exit_code == 0
        assert "No active violations" in result.output

        # Verify the violation was marked expired in state
        queue2 = ViolationQueue.load(build_dir)
        expired = [e for e in queue2._state.values() if e.get("status") == "expired"]
        assert len(expired) == 1

    def test_fix_active_violations_no_fixer(self, runner, simple_pipeline, build_dir):
        """Active violations but no fixer → 'No fixers declared'."""
        from synix.build.artifacts import ArtifactStore
        from synix.build.validators import Violation, ViolationQueue
        from synix.core.models import Artifact

        store = ArtifactStore(build_dir)
        art = Artifact(
            artifact_id="ep-1",
            artifact_type="episode",
            content="SSN: 123-45-6789",
            metadata={"layer_name": "episodes"},
        )
        store.save_artifact(art, "episodes", 1)
        content_hash = store.get_content_hash("ep-1")

        queue = ViolationQueue.load(build_dir)
        queue.upsert(
            Violation(
                violation_type="pii",
                severity="warning",
                message="PII detected",
                artifact_id="ep-1",
                field="content",
                metadata={"content_hash": content_hash},
                violation_id="test-vid-2",
            )
        )
        queue.save_state()

        result = runner.invoke(
            main,
            [
                "fix",
                str(simple_pipeline),
            ],
        )
        assert result.exit_code == 0
        assert "No fixers" in result.output

    def test_fix_dry_run(self, runner, pipeline_with_fixer, build_dir):
        """--dry-run flag is accepted."""
        result = runner.invoke(
            main,
            [
                "fix",
                str(pipeline_with_fixer),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0

    def test_fix_nonexistent_pipeline_errors(self, runner):
        result = runner.invoke(main, ["fix", "nonexistent.py"])
        assert result.exit_code != 0

    def test_fix_nonexistent_build_dir_errors(self, runner, simple_pipeline, tmp_path):
        result = runner.invoke(
            main,
            [
                "fix",
                str(simple_pipeline),
                "--build-dir",
                str(tmp_path / "no_such_dir"),
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "Build directory" in result.output
