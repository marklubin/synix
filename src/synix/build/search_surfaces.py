"""Shared helpers for build-time search surfaces."""

from __future__ import annotations

from pathlib import Path

from synix.core.models import Pipeline, SearchSurface, Transform


def surface_local_path(build_dir: Path, surface: SearchSurface) -> Path:
    """Return the local compatibility path for a search surface."""
    return build_dir / "surfaces" / f"{surface.name}.db"


def search_surface_handles(build_dir: Path, surfaces: list[SearchSurface]) -> dict[str, dict]:
    """Build lightweight search-surface handles for transform config injection."""
    return {
        surface.name: {
            "name": surface.name,
            "kind": "search_surface",
            "db_path": str(surface_local_path(build_dir, surface)),
            "modes": list(surface.modes),
            "sources": [source.name for source in surface.sources],
        }
        for surface in surfaces
    }


def search_surface_ready(surface: SearchSurface, available_layer_names: set[str]) -> bool:
    """Return True when all source layers for a surface are available."""
    return all(source.name in available_layer_names for source in surface.sources)


def validate_search_surface_uses(pipeline: Pipeline) -> None:
    """Validate build-time search surface usage after levels are computed."""
    registered_surfaces = {
        layer
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

            if used not in registered_surfaces:
                raise ValueError(
                    f"Transform '{layer.name}' uses search surface '{used.name}' but that surface "
                    "was not added to the pipeline."
                )

            blocking_sources = sorted(source.name for source in used.sources if source._level >= layer._level)
            if blocking_sources:
                names = ", ".join(blocking_sources)
                raise ValueError(
                    f"Transform '{layer.name}' uses search surface '{used.name}' before all of its "
                    f"source layers are built: {names}."
                )
