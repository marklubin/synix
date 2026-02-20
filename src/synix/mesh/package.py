"""Artifact bundling — create and extract tar.gz bundles with glob-based include/exclude."""

from __future__ import annotations

import logging
import sys
import tarfile
from fnmatch import fnmatch
from pathlib import Path

logger = logging.getLogger(__name__)


def _matches_any(rel_path: str, patterns: list[str]) -> bool:
    """Check if *rel_path* matches any of the given fnmatch patterns."""
    return any(fnmatch(rel_path, pat) for pat in patterns)


def create_bundle(build_dir: Path, include: list[str], exclude: list[str]) -> Path:
    """Create a tar.gz bundle from *build_dir*.

    Walks *build_dir*, includes files matching any *include* pattern,
    skips those matching any *exclude* pattern.  Patterns are matched
    against paths relative to *build_dir* using ``fnmatch``.

    The tarball is written to ``build_dir.parent / "{build_dir.name}.tar.gz"``.
    Returns the path to the created tarball.  If no files match the filters,
    a valid (empty) tarball is still created.
    """
    tarball_path = build_dir.parent / f"{build_dir.name}.tar.gz"
    file_count = 0

    with tarfile.open(tarball_path, "w:gz") as tar:
        for path in sorted(build_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(build_dir).as_posix()
            if not _matches_any(rel, include):
                continue
            if _matches_any(rel, exclude):
                continue
            tar.add(path, arcname=rel)
            file_count += 1

    logger.info("Created bundle %s (%d files)", tarball_path.name, file_count)
    return tarball_path


def _validate_tar_members(tar: tarfile.TarFile, dest: Path) -> list[tarfile.TarInfo]:
    """Validate tar members for path traversal attacks (pre-3.12 safety)."""
    safe_members = []
    for member in tar.getmembers():
        member_path = (dest / member.name).resolve()
        if not str(member_path).startswith(str(dest.resolve())):
            logger.warning("Skipping tar member with path traversal: %s", member.name)
            continue
        if member.issym() or member.islnk():
            logger.warning("Skipping symlink/hardlink in tar: %s", member.name)
            continue
        safe_members.append(member)
    return safe_members


def extract_bundle(tarball: Path, dest: Path) -> None:
    """Extract a tar.gz bundle to *dest* directory.

    Creates *dest* if it does not exist.  Uses ``filter='data'`` for
    security on Python 3.12+; validates members manually on earlier versions.
    """
    dest.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tarball, "r:gz") as tar:
        if sys.version_info >= (3, 12):
            tar.extractall(path=dest, filter="data")
        else:
            safe = _validate_tar_members(tar, dest)
            tar.extractall(path=dest, members=safe)

    logger.info("Extracted bundle %s to %s", tarball.name, dest)
