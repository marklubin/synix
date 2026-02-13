"""E2E tests for fingerprint-based cache invalidation.

Tests the full flow: build → verify fingerprints → modify → detect invalidation.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from synix.build.fingerprint import Fingerprint
from synix.cli import main

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent.parent / "synix" / "fixtures"


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def workspace(tmp_path):
    """Create a workspace with source exports and a build dir."""
    source_dir = tmp_path / "exports"
    source_dir.mkdir()
    build_dir = tmp_path / "build"

    # Copy fixture exports
    shutil.copy(FIXTURES_DIR / "chatgpt_export.json", source_dir / "chatgpt_export.json")
    shutil.copy(FIXTURES_DIR / "claude_export.json", source_dir / "claude_export.json")

    return {"root": tmp_path, "source_dir": source_dir, "build_dir": build_dir}


@pytest.fixture
def pipeline_file(workspace):
    """Write a pipeline.py into the workspace."""
    path = workspace["root"] / "pipeline.py"
    path.write_text(f"""
from synix import Pipeline, Layer, Projection

pipeline = Pipeline("test-fp")
pipeline.source_dir = "{workspace["source_dir"]}"
pipeline.build_dir = "{workspace["build_dir"]}"
pipeline.llm_config = {{"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}}

pipeline.add_layer(Layer(name="transcripts", level=0, transform="parse"))
pipeline.add_layer(Layer(name="episodes", level=1, depends_on=["transcripts"], transform="episode_summary", grouping="by_conversation"))
pipeline.add_layer(Layer(name="monthly", level=2, depends_on=["episodes"], transform="monthly_rollup", grouping="by_month"))
pipeline.add_layer(Layer(name="core", level=3, depends_on=["monthly"], transform="core_synthesis", grouping="single", context_budget=10000))

pipeline.add_projection(Projection(name="context-doc", projection_type="flat_file", sources=[{{"layer": "core"}}], config={{"output_path": "{workspace["build_dir"] / "context.md"}"}}))
""")
    return path


@pytest.fixture(autouse=True)
def mock_anthropic(monkeypatch):
    """Mock Anthropic API for all tests."""

    def mock_create(**kwargs):
        messages = kwargs.get("messages", [])
        content = messages[0].get("content", "") if messages else ""

        if "summarizing a conversation" in content.lower():
            return _mock_response("This conversation covered technical topics.")
        elif "monthly" in content.lower():
            return _mock_response("Monthly themes: technical learning and development.")
        elif "core memory" in content.lower():
            return _mock_response("## Identity\nA software engineer.\n\n## Focus\nMemory systems.")
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFingerprintStoredOnBuild:
    """Verify fingerprints are stored in artifact metadata after build."""

    def test_fingerprint_stored_on_build(self, runner, workspace, pipeline_file):
        """Build stores both build_fingerprint and transform_fingerprint in metadata."""
        from synix.build.artifacts import ArtifactStore

        result = runner.invoke(main, ["build", str(pipeline_file)])
        assert result.exit_code == 0, f"Build failed: {result.output}"

        store = ArtifactStore(workspace["build_dir"])
        manifest = json.loads((workspace["build_dir"] / "manifest.json").read_text())
        # Check derived artifacts (not transcripts — level 0 doesn't use fingerprints)
        derived = {aid: info for aid, info in manifest.items() if info.get("layer") not in ("transcripts",)}
        assert len(derived) > 0, "No derived artifacts found"

        for aid in derived:
            art = store.load_artifact(aid)
            assert art is not None, f"Could not load artifact {aid}"
            metadata = art.metadata

            assert "build_fingerprint" in metadata, f"Missing build_fingerprint for {aid}"
            build_fp = Fingerprint.from_dict(metadata["build_fingerprint"])
            assert build_fp is not None
            assert build_fp.scheme == "synix:build:v1"
            assert "transform" in build_fp.components
            assert "inputs" in build_fp.components

            assert "transform_fingerprint" in metadata, f"Missing transform_fingerprint for {aid}"
            transform_fp = Fingerprint.from_dict(metadata["transform_fingerprint"])
            assert transform_fp is not None
            assert transform_fp.scheme == "synix:transform:v1"
            assert "source" in transform_fp.components


class TestUpgradeForcesRebuild:
    """Simulate pre-upgrade artifacts without fingerprints."""

    def test_upgrade_forces_rebuild_without_fingerprint(self, runner, workspace, pipeline_file):
        """Artifacts without fingerprints trigger one-time rebuild."""
        from synix.build.artifacts import ArtifactStore

        # First build — populates fingerprints
        result1 = runner.invoke(main, ["build", str(pipeline_file)])
        assert result1.exit_code == 0

        # Strip fingerprints from all stored artifacts (simulate pre-upgrade state)
        build_dir = workspace["build_dir"]
        manifest = json.loads((build_dir / "manifest.json").read_text())
        for aid, info in manifest.items():
            if info.get("layer") == "transcripts":
                continue
            art_path = build_dir / info["path"]
            if art_path.exists():
                data = json.loads(art_path.read_text())
                data.get("metadata", {}).pop("build_fingerprint", None)
                data.get("metadata", {}).pop("transform_fingerprint", None)
                art_path.write_text(json.dumps(data, indent=2))

        # Second build — should rebuild (no stored fingerprint)
        result2 = runner.invoke(main, ["build", str(pipeline_file)])
        assert result2.exit_code == 0

        # Verify fingerprints are now restored
        store = ArtifactStore(build_dir)
        manifest = json.loads((build_dir / "manifest.json").read_text())
        for aid, info in manifest.items():
            if info.get("layer") == "transcripts":
                continue
            art = store.load_artifact(aid)
            assert art is not None
            assert "build_fingerprint" in art.metadata, f"Fingerprint not restored for {aid}"

        # Third build — should be fully cached now
        result3 = runner.invoke(main, ["build", str(pipeline_file)])
        assert result3.exit_code == 0


class TestExplainCacheOutput:
    """Test --explain-cache flag on plan command."""

    def test_explain_cache_json(self, runner, workspace, pipeline_file):
        """Plan with --json includes fingerprint data in steps."""
        # Build first so there's cache state
        runner.invoke(main, ["build", str(pipeline_file)])

        # Run plan with --json
        result = runner.invoke(main, ["plan", str(pipeline_file), "--json"])
        assert result.exit_code == 0, f"Plan failed: {result.output}"

        plan_data = json.loads(result.output)
        steps = plan_data.get("steps", [])
        assert len(steps) > 0

        # LLM layers (level > 0) should have fingerprint field
        llm_steps = [s for s in steps if s["level"] > 0]
        for step in llm_steps:
            assert step.get("fingerprint") is not None, f"Missing fingerprint for {step['name']}"
            fp = step["fingerprint"]
            assert "scheme" in fp
            assert "digest" in fp
            assert "components" in fp

    def test_explain_cache_display(self, runner, workspace, pipeline_file):
        """Plan with --explain-cache shows inline cache breakdown."""
        runner.invoke(main, ["build", str(pipeline_file)])
        result = runner.invoke(main, ["plan", str(pipeline_file), "--explain-cache"])
        assert result.exit_code == 0, f"Plan failed: {result.output}"
        assert "cache: all components match" in result.output


class TestFingerprintSchemeMismatch:
    """Test that scheme mismatch triggers rebuild."""

    def test_scheme_mismatch_triggers_rebuild(self, runner, workspace, pipeline_file):
        """Manually editing fingerprint scheme forces rebuild."""
        # Build first
        result1 = runner.invoke(main, ["build", str(pipeline_file)])
        assert result1.exit_code == 0

        # Tamper with the fingerprint scheme on a derived artifact
        build_dir = workspace["build_dir"]
        manifest = json.loads((build_dir / "manifest.json").read_text())
        tampered = False
        for aid, info in manifest.items():
            if not aid.startswith("ep-"):
                continue
            art_path = build_dir / info["path"]
            if art_path.exists():
                data = json.loads(art_path.read_text())
                bf = data.get("metadata", {}).get("build_fingerprint")
                if bf:
                    bf["scheme"] = "synix:build:v0"  # Fake old scheme
                    art_path.write_text(json.dumps(data, indent=2))
                    tampered = True
                    break

        assert tampered, "No episode artifact found to tamper with"

        # Second build — tampered artifact should be rebuilt
        result2 = runner.invoke(main, ["build", str(pipeline_file)])
        assert result2.exit_code == 0
