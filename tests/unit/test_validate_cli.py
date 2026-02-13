"""CLI smoke tests for the validate command."""

from __future__ import annotations

import json

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
    """Create a minimal pipeline file with PII validator."""
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


class TestValidateHelp:
    def test_validate_help(self, runner):
        result = runner.invoke(main, ["validate", "--help"])
        assert result.exit_code == 0
        assert "PIPELINE_PATH" in result.output
        assert "--build-dir" in result.output
        assert "--json" in result.output

    def test_validate_help_description(self, runner):
        result = runner.invoke(main, ["validate", "--help"])
        assert "Validate built artifacts" in result.output

    def test_validate_no_fix_flag(self, runner):
        """--fix is no longer accepted by validate."""
        result = runner.invoke(main, ["validate", "--fix"])
        assert result.exit_code != 0
        assert "No such option" in result.output or "no such option" in result.output

    def test_validate_no_dry_run_flag(self, runner):
        """--dry-run is no longer accepted by validate."""
        result = runner.invoke(main, ["validate", "--dry-run"])
        assert result.exit_code != 0
        assert "No such option" in result.output or "no such option" in result.output


class TestValidateCommand:
    def test_nonexistent_pipeline_errors(self, runner):
        result = runner.invoke(main, ["validate", "nonexistent.py"])
        assert result.exit_code != 0

    def test_nonexistent_build_dir_errors(self, runner, simple_pipeline, tmp_path):
        result = runner.invoke(
            main,
            [
                "validate",
                str(simple_pipeline),
                "--build-dir",
                str(tmp_path / "no_such_dir"),
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "Build directory" in result.output

    def test_validate_no_violations(self, runner, simple_pipeline, build_dir):
        """Empty build dir → no violations."""
        result = runner.invoke(main, ["validate", str(simple_pipeline)])
        assert result.exit_code == 0
        assert "passed" in result.output.lower() or "No violations" in result.output

    def test_validate_with_pii(self, runner, simple_pipeline, build_dir):
        """PII in artifacts → violations reported."""
        from synix.build.artifacts import ArtifactStore
        from synix.core.models import Artifact

        store = ArtifactStore(build_dir)
        art = Artifact(
            label="ep-1",
            artifact_type="episode",
            content="Email me at user@example.com for details.",
            metadata={"layer_name": "episodes"},
        )
        store.save_artifact(art, "episodes", 1)

        result = runner.invoke(main, ["validate", str(simple_pipeline)])
        assert "pii" in result.output.lower() or "PII" in result.output

    def test_validate_json_output(self, runner, simple_pipeline, build_dir):
        """--json produces parseable JSON."""
        result = runner.invoke(
            main,
            [
                "validate",
                str(simple_pipeline),
                "--json",
            ],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "passed" in data
        assert "validators_run" in data
        assert "violations" in data

    def test_validate_json_with_violations(self, runner, simple_pipeline, build_dir):
        """--json includes violation_ids."""
        from synix.build.artifacts import ArtifactStore
        from synix.core.models import Artifact

        store = ArtifactStore(build_dir)
        art = Artifact(
            label="ep-1",
            artifact_type="episode",
            content="SSN: 123-45-6789",
            metadata={"layer_name": "episodes"},
        )
        store.save_artifact(art, "episodes", 1)

        result = runner.invoke(
            main,
            [
                "validate",
                str(simple_pipeline),
                "--json",
            ],
        )
        data = json.loads(result.output)
        assert len(data["violations"]) >= 1
        assert "violation_id" in data["violations"][0]

    def test_validate_writes_state_file(self, runner, simple_pipeline, build_dir):
        """Validate creates violations_state.json."""
        from synix.build.artifacts import ArtifactStore
        from synix.core.models import Artifact

        store = ArtifactStore(build_dir)
        art = Artifact(
            label="ep-1",
            artifact_type="episode",
            content="Credit card: 4111-1111-1111-1111",
            metadata={"layer_name": "episodes"},
        )
        store.save_artifact(art, "episodes", 1)

        runner.invoke(main, ["validate", str(simple_pipeline)])
        assert (build_dir / "violations_state.json").exists()

    def test_validate_writes_audit_log(self, runner, simple_pipeline, build_dir):
        """Validate appends to violations.jsonl."""
        from synix.build.artifacts import ArtifactStore
        from synix.core.models import Artifact

        store = ArtifactStore(build_dir)
        art = Artifact(
            label="ep-1",
            artifact_type="episode",
            content="Phone: (555) 123-4567",
            metadata={"layer_name": "episodes"},
        )
        store.save_artifact(art, "episodes", 1)

        runner.invoke(main, ["validate", str(simple_pipeline)])
        assert (build_dir / "violations.jsonl").exists()
