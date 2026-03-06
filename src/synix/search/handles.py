"""Backward compatibility shim for the public search handle types."""

from synix.core.search_handles import (  # noqa: F401
    SearchSurfaceError,
    SearchSurfaceHandle,
    SearchSurfaceLookupError,
    SearchSurfaceUnavailableError,
    resolve_search_surface_handle,
)
