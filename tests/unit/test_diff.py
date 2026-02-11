"""Unit tests for synix diff command."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from synix import Artifact
from synix.build.artifacts import ArtifactStore
from synix.build.diff import diff_artifact, diff_artifact_by_id, diff_builds
from synix.cli import main


@pytest.fixture
def two_builds(tmp_path):
    """Create two build directories with overlapping artifacts."""
    old_dir = tmp_path / "build_old"
    new_dir = tmp_path / "build_new"

    old_store = ArtifactStore(old_dir)
    new_store = ArtifactStore(new_dir)

    # Shared artifact with same content (no change)
    t1 = Artifact(
        artifact_id="t-001",
        artifact_type="transcript",
        content="User: Hello\n\nAssistant: Hi!",
        metadata={"source": "chatgpt", "date": "2025-01-15"},
    )
    old_store.save_artifact(t1, "transcripts", 0)
    new_store.save_artifact(t1, "transcripts", 0)

    # Shared artifact with changed content
    ep_old = Artifact(
        artifact_id="ep-001",
        artifact_type="episode",
        content="Original episode summary about greetings.",
        prompt_id="v1",
        metadata={"source_conversation_id": "001"},
    )
    ep_new = Artifact(
        artifact_id="ep-001",
        artifact_type="episode",
        content="Updated episode summary about greetings and farewells.",
        prompt_id="v2",
        metadata={"source_conversation_id": "001", "updated": True},
    )
    old_store.save_artifact(ep_old, "episodes", 1)
    new_store.save_artifact(ep_new, "episodes", 1)

    # Artifact only in old (removed)
    removed = Artifact(
        artifact_id="ep-002",
        artifact_type="episode",
        content="This episode was removed.",
        metadata={},
    )
    old_store.save_artifact(removed, "episodes", 1)

    # Artifact only in new (added)
    added = Artifact(
        artifact_id="ep-003",
        artifact_type="episode",
        content="This episode was added.",
        metadata={},
    )
    new_store.save_artifact(added, "episodes", 1)

    return old_dir, new_dir


class TestDiffArtifact:
    def test_no_changes(self):
        art = Artifact(
            artifact_id="test",
            artifact_type="episode",
            content="Same content",
            prompt_id="v1",
            metadata={"key": "value"},
        )
        result = diff_artifact(art, art)
        assert not result.has_changes
        assert result.content_diff == ""
        assert result.metadata_diff == {}

    def test_content_change(self):
        old = Artifact(artifact_id="test", artifact_type="episode", content="Old text")
        new = Artifact(artifact_id="test", artifact_type="episode", content="New text")
        result = diff_artifact(old, new)
        assert result.has_changes
        assert "Old text" in result.content_diff
        assert "New text" in result.content_diff

    def test_metadata_change(self):
        old = Artifact(
            artifact_id="test",
            artifact_type="episode",
            content="Same",
            metadata={"key": "old_value"},
        )
        new = Artifact(
            artifact_id="test",
            artifact_type="episode",
            content="Same",
            metadata={"key": "new_value"},
        )
        result = diff_artifact(old, new)
        assert result.has_changes
        assert "key" in result.metadata_diff
        assert result.metadata_diff["key"]["old"] == "old_value"
        assert result.metadata_diff["key"]["new"] == "new_value"

    def test_prompt_change(self):
        old = Artifact(
            artifact_id="test",
            artifact_type="episode",
            content="Same",
            prompt_id="v1",
        )
        new = Artifact(
            artifact_id="test",
            artifact_type="episode",
            content="Same",
            prompt_id="v2",
        )
        result = diff_artifact(old, new)
        assert result.has_changes
        assert result.old_prompt_id == "v1"
        assert result.new_prompt_id == "v2"


class TestDiffBuilds:
    def test_full_diff(self, two_builds):
        old_dir, new_dir = two_builds
        result = diff_builds(old_dir, new_dir)
        assert result.has_changes
        assert "ep-003" in result.added
        assert "ep-002" in result.removed
        assert any(d.artifact_id == "ep-001" for d in result.diffs)

    def test_no_changes(self, tmp_path):
        """Same build compared to itself."""
        store = ArtifactStore(tmp_path / "build")
        art = Artifact(artifact_id="t-001", artifact_type="transcript", content="Hello")
        store.save_artifact(art, "transcripts", 0)

        result = diff_builds(tmp_path / "build", tmp_path / "build")
        assert not result.has_changes

    def test_layer_filter(self, two_builds):
        old_dir, new_dir = two_builds
        # Filter to transcripts only — should show no changes
        result = diff_builds(old_dir, new_dir, layer="transcripts")
        assert not result.has_changes

        # Filter to episodes — should show changes
        result = diff_builds(old_dir, new_dir, layer="episodes")
        assert result.has_changes


class TestDiffArtifactById:
    def test_cross_build_diff(self, two_builds):
        old_dir, new_dir = two_builds
        result = diff_artifact_by_id(str(new_dir), "ep-001", previous_build_dir=str(old_dir))
        assert result is not None
        assert result.has_changes
        assert "Updated" in result.content_diff

    def test_missing_artifact(self, two_builds):
        _, new_dir = two_builds
        result = diff_artifact_by_id(str(new_dir), "nonexistent")
        assert result is None

    def test_no_previous_build(self, two_builds):
        _, new_dir = two_builds
        # No previous build dir and no version history
        result = diff_artifact_by_id(str(new_dir), "ep-001")
        assert result is None


class TestDiffCLI:
    def test_diff_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["diff", "--help"])
        assert result.exit_code == 0
        assert "--build-dir" in result.output
        assert "--old-build-dir" in result.output

    def test_diff_single_artifact(self, two_builds):
        old_dir, new_dir = two_builds
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "diff",
                "ep-001",
                "--build-dir",
                str(new_dir),
                "--old-build-dir",
                str(old_dir),
            ],
        )
        assert result.exit_code == 0

    def test_diff_builds_cli(self, two_builds):
        old_dir, new_dir = two_builds
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "diff",
                "--build-dir",
                str(new_dir),
                "--old-build-dir",
                str(old_dir),
            ],
        )
        assert result.exit_code == 0
        assert "added" in result.output or "removed" in result.output or "modified" in result.output

    def test_diff_no_old_build_dir_errors(self, two_builds):
        _, new_dir = two_builds
        runner = CliRunner()
        result = runner.invoke(main, ["diff", "--build-dir", str(new_dir)])
        assert result.exit_code != 0
