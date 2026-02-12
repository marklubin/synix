"""Synix error types and utilities."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write(path: Path, content: str) -> None:
    """Write content to a file atomically using temp file + rename.

    Writes to a temporary file in the same directory, fsyncs it,
    then atomically replaces the target path.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, content.encode())
        os.fsync(fd)
        os.close(fd)
        os.replace(tmp, str(path))
    except BaseException:
        os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class SynixError(Exception):
    """Base exception for Synix."""

    pass


class PipelineError(SynixError):
    """Error in pipeline configuration or execution."""

    pass


class ArtifactError(SynixError):
    """Error in artifact storage or retrieval."""

    pass


class TransformError(SynixError):
    """Error during transform execution."""

    pass
