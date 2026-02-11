"""Tests for flat file projection."""

from __future__ import annotations

from synix import Artifact
from synix.projections.flat_file import FlatFileProjection


class TestFlatFileProjection:
    def test_materialize_creates_file(self, tmp_build_dir):
        """Output file exists at specified path."""
        output_path = tmp_build_dir / "context.md"
        projection = FlatFileProjection()

        artifact = Artifact(
            artifact_id="core-memory",
            artifact_type="core_memory",
            content="## Identity\nMark is a software engineer.",
        )

        projection.materialize([artifact], {"output_path": str(output_path)})
        assert output_path.exists()

    def test_content_matches_core(self, tmp_build_dir):
        """File content matches core memory artifact."""
        output_path = tmp_build_dir / "context.md"
        projection = FlatFileProjection()

        content = "## Identity\nMark is a software engineer.\n\n## Current Focus\nBuilding Synix."
        artifact = Artifact(
            artifact_id="core-memory",
            artifact_type="core_memory",
            content=content,
        )

        projection.materialize([artifact], {"output_path": str(output_path)})
        assert output_path.read_text() == content

    def test_markdown_formatting(self, tmp_build_dir):
        """Output is valid markdown with multiple artifacts joined."""
        output_path = tmp_build_dir / "context.md"
        projection = FlatFileProjection()

        artifacts = [
            Artifact(artifact_id="core-1", artifact_type="core_memory", content="## Section One\nFirst part."),
            Artifact(artifact_id="core-2", artifact_type="core_memory", content="## Section Two\nSecond part."),
        ]

        projection.materialize(artifacts, {"output_path": str(output_path)})
        result = output_path.read_text()
        assert "## Section One" in result
        assert "## Section Two" in result
        # Sections are separated by double newline
        assert "\n\n" in result
