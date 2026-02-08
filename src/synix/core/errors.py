"""Synix error types."""

from __future__ import annotations


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
