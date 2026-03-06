"""Synix — A build system for agent memory."""

__version__ = "0.15.0"

from synix.core.models import (  # noqa: F401
    Artifact,
    FlatFile,
    Layer,
    Pipeline,
    ProvenanceRecord,
    SearchIndex,
    SearchSurface,
    Source,
    SynixSearch,
    Transform,
    TransformContext,
)
from synix.core.search_handles import (  # noqa: F401
    SearchSurfaceError,
    SearchSurfaceHandle,
    SearchSurfaceLookupError,
    SearchSurfaceUnavailableError,
)
