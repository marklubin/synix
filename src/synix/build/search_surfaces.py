"""Shared helpers for build-time search surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from synix.core.models import Pipeline, SearchIndex, SearchSurface, Transform
from synix.core.search_handles import SearchSurfaceHandle


def surface_local_path(build_dir: Path, surface: SearchSurface) -> Path:
    """Return the local compatibility path for a search surface."""
    return build_dir / "surfaces" / f"{surface.name}.db"


def search_surface_handles(build_dir: Path, surfaces: list[SearchSurface]) -> dict[str, SearchSurfaceHandle]:
    """Build typed search-surface handles for transform runtime context."""
    return {
        surface.name: SearchSurfaceHandle(
            name=surface.name,
            db_path=str(surface_local_path(build_dir, surface)),
            modes=tuple(surface.modes),
            sources=tuple(source.name for source in surface.sources),
        )
        for surface in surfaces
    }


def transform_runtime_search_updates(
    layer: Transform,
    *,
    build_dir: Path,
    projections: list,
) -> dict[str, Any]:
    """Build runtime-only search capability injection for a transform."""
    runtime_updates: dict[str, Any] = {}
    search_uses = [surface for surface in layer.uses if isinstance(surface, SearchSurface)]
    if search_uses:
        handles = search_surface_handles(build_dir, search_uses)
        runtime_updates["search_surfaces"] = handles
        if len(search_uses) == 1:
            default_handle = handles[search_uses[0].name]
            runtime_updates["search_surface"] = default_handle
            runtime_updates["search_db_path"] = default_handle.db_path
    elif any(isinstance(proj, SearchIndex) for proj in projections):
        # Legacy compatibility path for transforms that still rely on the
        # global search projection convention.
        runtime_updates["search_db_path"] = str(build_dir / "search.db")
    return runtime_updates


def search_surface_ready(surface: SearchSurface, available_layer_names: set[str]) -> bool:
    """Return True when all source layers for a surface are available."""
    return all(source.name in available_layer_names for source in surface.sources)


def validate_search_surface_uses(pipeline: Pipeline) -> None:
    """Validate build-time search surface usage after levels are computed."""
    registered_surfaces = {
        layer.name: layer
        for layer in [*pipeline.surfaces, *pipeline.projections]
        if isinstance(layer, SearchSurface)
    }

    for layer in pipeline.layers:
        if not isinstance(layer, Transform):
            continue

        for used in layer.uses:
            if not isinstance(used, SearchSurface):
                raise TypeError(
                    f"Transform '{layer.name}' declares unsupported use '{getattr(used, 'name', used)}'; "
                    "only SearchSurface-compatible capabilities are supported in uses=[...]."
                )

            registered = registered_surfaces.get(used.name)
            if registered is None:
                raise ValueError(
                    f"Transform '{layer.name}' uses search surface '{used.name}' but that surface "
                    "was not added to the pipeline."
                )

            if registered.usage_signature() != used.usage_signature():
                raise ValueError(
                    f"Transform '{layer.name}' uses search surface '{used.name}' with a declaration "
                    "that does not match the surface added to the pipeline."
                )

            blocking_sources = sorted(source.name for source in registered.sources if source._level >= layer._level)
            if blocking_sources:
                names = ", ".join(blocking_sources)
                raise ValueError(
                    f"Transform '{layer.name}' uses search surface '{registered.name}' before all of its "
                    f"source layers are built: {names}."
                )
