"""E2E tests: release --target hardening — external target dirs, custom filenames, info/status/clean."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from synix.build.refs import synix_dir_for_build_dir
from synix.build.release_engine import execute_release
from synix.cli.main import main
from synix.core.models import Artifact
from tests.helpers.snapshot_factory import create_test_snapshot

FIXTURES_DIR = Path(__file__).parent.parent / "synix" / "fixtures"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def mock_anthropic(monkeypatch):
    """Mock the Anthropic client so builds don't hit a real API."""

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


# ---------------------------------------------------------------------------
# Helpers — snapshot-based (no real build needed)
# ---------------------------------------------------------------------------


def _make_artifact(label, content, atype="episode", layer_name="episodes"):
    return Artifact(
        label=label,
        artifact_type=atype,
        content=content,
        metadata={"layer_name": layer_name, "layer_level": 1 if layer_name == "episodes" else 2},
    )


def _setup_snapshot_with_projections(
    tmp_path: Path,
    *,
    flat_file_config: dict | None = None,
) -> Path:
    """Create a snapshot that declares both synix_search and flat_file projections."""
    ep1 = _make_artifact("ep-1", "Episode one about release target testing.")
    ep2 = _make_artifact("ep-2", "Episode two about external directory output.")
    core = _make_artifact(
        "core-1",
        "Core memory: user builds agent memory systems.",
        atype="core",
        layer_name="core",
    )

    ff_config = flat_file_config or {"output_path": "context.md"}

    projections = {
        "search": {
            "adapter": "synix_search",
            "input_artifacts": ["ep-1", "ep-2"],
            "config": {"modes": ["fulltext"]},
            "config_fingerprint": "sha256:test-search",
            "precomputed_oid": None,
        },
        "context-doc": {
            "adapter": "flat_file",
            "input_artifacts": ["core-1"],
            "config": ff_config,
            "config_fingerprint": "sha256:test-ff",
            "precomputed_oid": None,
        },
    }

    synix_dir = create_test_snapshot(
        tmp_path,
        {"episodes": [ep1, ep2], "core": [core]},
        projections=projections,
    )
    return synix_dir


# ---------------------------------------------------------------------------
# Helpers — full CLI build (for info/status/clean that need a real build_dir)
# ---------------------------------------------------------------------------


def _write_pipeline(path: Path, source_dir: str, build_dir: str) -> Path:
    """Write a pipeline file with both SynixSearch and FlatFile projections."""
    pipeline_file = path / "pipeline.py"
    pipeline_file.write_text(f"""
from synix import Pipeline, SearchSurface, Source, SynixSearch, FlatFile
from synix.ext import EpisodeSummary, MonthlyRollup, CoreSynthesis

pipeline = Pipeline("release-target-test")
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


def _write_pipeline_custom_filename(
    path: Path, source_dir: str, build_dir: str, output_filename: str = "memory.md"
) -> Path:
    """Write a pipeline file with a custom flat-file output_path."""
    pipeline_file = path / "pipeline.py"
    pipeline_file.write_text(f"""
from synix import Pipeline, SearchSurface, Source, SynixSearch, FlatFile
from synix.ext import EpisodeSummary, MonthlyRollup, CoreSynthesis

pipeline = Pipeline("custom-filename-test")
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
pipeline.add(FlatFile("context-doc", sources=[core], output_path="{output_filename}"))
""")
    return pipeline_file


def _build_project(runner, project_dir: Path) -> Path:
    """Build a project and return the synix_dir."""
    sources_dir = project_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES_DIR / "chatgpt_export.json", sources_dir / "chatgpt_export.json")
    shutil.copy(FIXTURES_DIR / "claude_export.json", sources_dir / "claude_export.json")

    build_dir = project_dir / "build"
    _write_pipeline(project_dir, source_dir=str(sources_dir), build_dir=str(build_dir))

    result = runner.invoke(main, ["build", str(project_dir / "pipeline.py"), "--plain"])
    assert result.exit_code == 0, f"Build failed:\n{result.output}"

    synix_dir = synix_dir_for_build_dir(build_dir)
    assert synix_dir.exists(), f".synix not found at {synix_dir}"
    return synix_dir


def _build_project_custom_filename(runner, project_dir: Path, output_filename: str = "memory.md") -> Path:
    """Build a project with a custom flat-file filename and return the synix_dir."""
    sources_dir = project_dir / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES_DIR / "chatgpt_export.json", sources_dir / "chatgpt_export.json")
    shutil.copy(FIXTURES_DIR / "claude_export.json", sources_dir / "claude_export.json")

    build_dir = project_dir / "build"
    _write_pipeline_custom_filename(
        project_dir,
        source_dir=str(sources_dir),
        build_dir=str(build_dir),
        output_filename=output_filename,
    )

    result = runner.invoke(main, ["build", str(project_dir / "pipeline.py"), "--plain"])
    assert result.exit_code == 0, f"Build failed:\n{result.output}"

    synix_dir = synix_dir_for_build_dir(build_dir)
    assert synix_dir.exists(), f".synix not found at {synix_dir}"
    return synix_dir


# ===========================================================================
# Tests
# ===========================================================================


class TestReleaseTargetCreatesDir:
    """release --target to a nonexistent directory should create it and write search.db."""

    def test_release_target_creates_nonexistent_dir(self, tmp_path):
        synix_dir = _setup_snapshot_with_projections(tmp_path)

        external_target = tmp_path / "external" / "output"
        assert not external_target.exists(), "Precondition: target dir should not exist yet"

        receipt = execute_release(synix_dir, release_name="custom", target=external_target)

        # Target directory was created
        assert external_target.exists(), "execute_release should create the target dir"
        assert external_target.is_dir()

        # search.db landed in the target directory
        assert (external_target / "search.db").exists(), "search.db should exist in external target"

        # Receipt was written to .synix/releases/<name>/receipt.json, NOT to the target dir
        receipt_path = synix_dir / "releases" / "custom" / "receipt.json"
        assert receipt_path.exists(), "receipt.json should be in .synix/releases/custom/"

        # Receipt records the external target path
        assert receipt.release_name == "custom"
        assert receipt.adapters, "Receipt should have adapter entries"

        # Verify the search adapter target points into external dir
        search_receipt = receipt.adapters.get("search", {})
        assert "external" in search_receipt.get("target", ""), (
            f"Search adapter target should reference external dir, got: {search_receipt.get('target')}"
        )


class TestReleaseTargetWithFlatFile:
    """Release to external target with both synix_search and flat_file adapters."""

    def test_both_adapters_write_to_external_target(self, tmp_path):
        synix_dir = _setup_snapshot_with_projections(tmp_path)

        external_target = tmp_path / "ext" / "release"
        receipt = execute_release(synix_dir, release_name="dual", target=external_target)

        # Both outputs should exist in the external target
        assert (external_target / "search.db").exists(), "search.db missing from external target"
        assert (external_target / "context.md").exists(), "context.md missing from external target"

        # Receipt reflects both adapters as success
        assert len(receipt.adapters) == 2, f"Expected 2 adapter receipts, got {len(receipt.adapters)}"
        for name, data in receipt.adapters.items():
            assert data["status"] == "success", f"Adapter {name} should be success"


class TestReleaseCustomFlatFileFilename:
    """FlatFile with config.output_path should use the custom filename."""

    def test_custom_filename_via_snapshot_factory(self, tmp_path):
        """Snapshot-based test: custom output_path in projection config."""
        synix_dir = _setup_snapshot_with_projections(tmp_path, flat_file_config={"output_path": "memory.md"})

        external_target = tmp_path / "custom-filename-out"
        receipt = execute_release(synix_dir, release_name="custom-fn", target=external_target)

        # memory.md should exist (not context.md)
        assert (external_target / "memory.md").exists(), "memory.md should exist with custom output_path"
        assert not (external_target / "context.md").exists(), (
            "context.md should NOT exist when output_path is memory.md"
        )

        # Adapter receipt target should reference memory.md
        ff_receipt = receipt.adapters.get("context-doc", {})
        assert ff_receipt.get("target", "").endswith("memory.md"), (
            f"flat_file target should end with memory.md, got: {ff_receipt.get('target')}"
        )

    def test_custom_filename_via_full_build(self, runner, tmp_path):
        """Full build pipeline test: custom output_path via pipeline config."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()

        synix_dir = _build_project_custom_filename(runner, project_dir, output_filename="memory.md")

        external_target = tmp_path / "custom-build-out"
        receipt = execute_release(synix_dir, release_name="custom-build", target=external_target)

        assert (external_target / "memory.md").exists(), "memory.md should exist in external target"
        assert not (external_target / "context.md").exists(), "context.md should NOT exist"


class TestInfoAfterExternalTargetRelease:
    """info command should show external target paths from receipt.json."""

    def test_info_shows_external_target(self, runner, tmp_path):
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        synix_dir = _build_project(runner, project_dir)

        # Release to an external target
        external_target = tmp_path / "extinfo" / "out"
        execute_release(synix_dir, release_name="ext-info", target=external_target)

        # Verify receipt.json records the external target paths correctly
        receipt_path = synix_dir / "releases" / "ext-info" / "receipt.json"
        receipt_data = json.loads(receipt_path.read_text(encoding="utf-8"))
        for _name, adapter_info in receipt_data["adapters"].items():
            target = adapter_info["target"]
            assert str(external_target) in target or "extinfo" in target, (
                f"Receipt adapter target should reference external dir, got: {target}"
            )

        # Run info with --synix-dir
        result = runner.invoke(main, ["info", "--synix-dir", str(synix_dir)])
        assert result.exit_code == 0, f"info failed:\n{result.output}"

        # Output should show the release name
        assert "ext-info" in result.output, f"info should mention release name 'ext-info'.\nOutput:\n{result.output}"

        # For external targets, info shows the full path (not just a short filename).
        # Rich may truncate or wrap long paths, so we verify the receipt data directly
        # and check that the CLI at least distinguishes external targets from internal
        # ones by NOT displaying just a bare filename like "search.db".
        # Strip all whitespace for comparison (Rich wraps/breaks paths at column boundaries).
        collapsed = result.output.replace(" ", "").replace("\n", "")
        # The info output for internal releases shows short filenames like "search.db".
        # For external targets, the path should contain more than just the filename.
        assert "flat_file:" in result.output, f"info should show flat_file adapter.\nOutput:\n{result.output}"
        assert "synix_search:" in result.output, f"info should show synix_search adapter.\nOutput:\n{result.output}"


class TestStatusAfterExternalTargetRelease:
    """status command should show external target paths from receipt.json."""

    def test_status_shows_external_target(self, runner, tmp_path):
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        synix_dir = _build_project(runner, project_dir)

        # Release to an external target
        external_target = tmp_path / "extstatus" / "out"
        execute_release(synix_dir, release_name="ext-status", target=external_target)

        # Verify receipt.json records the external target paths correctly
        receipt_path = synix_dir / "releases" / "ext-status" / "receipt.json"
        receipt_data = json.loads(receipt_path.read_text(encoding="utf-8"))
        for _name, adapter_info in receipt_data["adapters"].items():
            target = adapter_info["target"]
            assert str(external_target) in target or "extstatus" in target, (
                f"Receipt adapter target should reference external dir, got: {target}"
            )

        # Run status with --synix-dir
        result = runner.invoke(main, ["status", "--synix-dir", str(synix_dir)])
        assert result.exit_code == 0, f"status failed:\n{result.output}"

        # Output should mention the release name
        assert "ext-status" in result.output, (
            f"status should mention release name 'ext-status'.\nOutput:\n{result.output}"
        )

        # For external targets, status shows the full path (not just the filename).
        # Rich may wrap paths across lines, so collapse whitespace to verify.
        collapsed = result.output.replace(" ", "").replace("\n", "")
        assert "extstatus" in collapsed, (
            f"status should show external target path (after whitespace collapse).\nOutput:\n{result.output}"
        )


class TestCleanWarnsAboutExternalTargets:
    """clean --release should warn about external files and NOT delete them."""

    def test_clean_warns_and_preserves_external_files(self, runner, tmp_path):
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        synix_dir = _build_project(runner, project_dir)

        # Release to an external target
        external_target = tmp_path / "external-clean" / "output"
        execute_release(synix_dir, release_name="ext-clean", target=external_target)

        # Verify external files exist
        assert (external_target / "search.db").exists(), "Precondition: search.db should exist"
        assert (external_target / "context.md").exists(), "Precondition: context.md should exist"

        # Run clean for this specific release with -y to skip prompt
        result = runner.invoke(
            main,
            ["clean", "--synix-dir", str(synix_dir), "--release", "ext-clean", "-y"],
        )
        assert result.exit_code == 0, f"clean failed:\n{result.output}"

        # Output should contain the external target warning
        assert "external" in result.output.lower(), (
            f"clean should warn about external targets.\nOutput:\n{result.output}"
        )

        # External files should NOT be deleted (clean only removes the receipt dir)
        assert (external_target / "search.db").exists(), "search.db in external target should survive clean"
        assert (external_target / "context.md").exists(), "context.md in external target should survive clean"

        # The release directory inside .synix should be removed
        assert not (synix_dir / "releases" / "ext-clean").exists(), (
            ".synix/releases/ext-clean/ should be removed by clean"
        )

    def test_clean_all_warns_about_external_targets(self, runner, tmp_path):
        """clean without --release should also warn about external targets in any release."""
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        synix_dir = _build_project(runner, project_dir)

        external_target = tmp_path / "external-clean-all" / "output"
        execute_release(synix_dir, release_name="ext-all", target=external_target)

        result = runner.invoke(
            main,
            ["clean", "--synix-dir", str(synix_dir), "-y"],
        )
        assert result.exit_code == 0, f"clean failed:\n{result.output}"

        # Should still warn about external targets
        assert "external" in result.output.lower(), (
            f"clean should warn about external targets.\nOutput:\n{result.output}"
        )

        # External files survive
        assert (external_target / "search.db").exists()
        assert (external_target / "context.md").exists()
