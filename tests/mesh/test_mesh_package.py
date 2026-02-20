"""Tests for mesh artifact bundling (package)."""

from __future__ import annotations

import tarfile
from pathlib import Path

from synix.mesh.package import create_bundle, extract_bundle


def _populate_build_dir(build_dir: Path) -> None:
    """Create a realistic build directory with various files."""
    (build_dir / "manifest.json").write_text('{"version": 1}')
    (build_dir / "context.md").write_text("# Context")
    (build_dir / "search.db").write_bytes(b"\x00" * 16)
    layer_dir = build_dir / "layer2-episodes"
    layer_dir.mkdir()
    (layer_dir / "ep-001.json").write_text('{"label": "ep-001"}')
    (layer_dir / "ep-002.json").write_text('{"label": "ep-002"}')
    logs_dir = build_dir / "logs"
    logs_dir.mkdir()
    (logs_dir / "build.log").write_text("log content")


class TestCreateBundle:
    def test_include_patterns(self, tmp_path):
        """Only files matching include patterns are bundled."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        _populate_build_dir(build_dir)

        tarball = create_bundle(
            build_dir,
            include=["manifest.json", "context.md"],
            exclude=[],
        )

        assert tarball.exists()
        with tarfile.open(tarball, "r:gz") as tar:
            names = tar.getnames()
        assert "manifest.json" in names
        assert "context.md" in names
        assert "search.db" not in names

    def test_exclude_overrides_include(self, tmp_path):
        """Exclude patterns take precedence over include patterns."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        _populate_build_dir(build_dir)

        tarball = create_bundle(
            build_dir,
            include=["manifest.json", "context.md", "search.db"],
            exclude=["search.db"],
        )

        with tarfile.open(tarball, "r:gz") as tar:
            names = tar.getnames()
        assert "manifest.json" in names
        assert "search.db" not in names

    def test_glob_patterns(self, tmp_path):
        """Glob-style include patterns match nested files."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        _populate_build_dir(build_dir)

        tarball = create_bundle(
            build_dir,
            include=["layer2-*/**"],
            exclude=[],
        )

        with tarfile.open(tarball, "r:gz") as tar:
            names = tar.getnames()
        assert "layer2-episodes/ep-001.json" in names
        assert "layer2-episodes/ep-002.json" in names
        assert "manifest.json" not in names

    def test_empty_bundle_no_matching_files(self, tmp_path):
        """An empty (but valid) tarball is created when no files match."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        _populate_build_dir(build_dir)

        tarball = create_bundle(
            build_dir,
            include=["nonexistent-pattern-*"],
            exclude=[],
        )

        assert tarball.exists()
        with tarfile.open(tarball, "r:gz") as tar:
            assert tar.getnames() == []

    def test_tarball_location(self, tmp_path):
        """Tarball is created in build_dir's parent directory."""
        build_dir = tmp_path / "my-build"
        build_dir.mkdir()
        (build_dir / "f.txt").write_text("data")

        tarball = create_bundle(build_dir, include=["f.txt"], exclude=[])
        assert tarball.parent == tmp_path
        assert tarball.name == "my-build.tar.gz"

    def test_exclude_logs_glob(self, tmp_path):
        """Exclude with glob pattern filters out nested directories."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        _populate_build_dir(build_dir)

        tarball = create_bundle(
            build_dir,
            include=["manifest.json", "logs/**"],
            exclude=["logs/**"],
        )

        with tarfile.open(tarball, "r:gz") as tar:
            names = tar.getnames()
        assert "manifest.json" in names
        assert "logs/build.log" not in names


class TestExtractBundle:
    def test_roundtrip(self, tmp_path):
        """Create then extract — files are preserved identically."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        _populate_build_dir(build_dir)

        tarball = create_bundle(
            build_dir,
            include=["manifest.json", "context.md", "layer2-*/**"],
            exclude=["search.db"],
        )

        dest = tmp_path / "extracted"
        extract_bundle(tarball, dest)

        assert (dest / "manifest.json").read_text() == '{"version": 1}'
        assert (dest / "context.md").read_text() == "# Context"
        assert (dest / "layer2-episodes" / "ep-001.json").exists()
        assert not (dest / "search.db").exists()

    def test_creates_dest_if_missing(self, tmp_path):
        """Extraction creates the destination directory if it doesn't exist."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        (build_dir / "a.txt").write_text("content")

        tarball = create_bundle(build_dir, include=["a.txt"], exclude=[])

        dest = tmp_path / "nested" / "deep" / "output"
        assert not dest.exists()
        extract_bundle(tarball, dest)
        assert (dest / "a.txt").read_text() == "content"

    def test_extract_empty_bundle(self, tmp_path):
        """Extracting an empty tarball succeeds without error."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        tarball = create_bundle(build_dir, include=["nope"], exclude=[])

        dest = tmp_path / "output"
        extract_bundle(tarball, dest)
        assert dest.is_dir()
        # No files beyond the directory itself
        assert list(dest.iterdir()) == []
