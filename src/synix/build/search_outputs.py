"""Helpers for resolving local search output artifacts from build metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECTION_CACHE_FILE = ".projection_cache.json"
SEARCH_OUTPUT_TYPES = {"synix_search", "search_index"}


@dataclass(frozen=True)
class SearchOutputSpec:
    """A materialized local search output recorded in a build."""

    name: str
    projection_type: str
    db_path: Path


class SearchOutputResolutionError(ValueError):
    """Raised when a build contains ambiguous or invalid search output metadata."""


def _load_projection_cache(build_dir: Path) -> dict:
    cache_path = build_dir / PROJECTION_CACHE_FILE
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _normalized_output_path(build_path: Path, db_path_raw: str) -> Path:
    path = Path(db_path_raw)
    if not path.is_absolute():
        path = build_path / path

    resolved_build = build_path.resolve(strict=False)
    resolved_path = path.resolve(strict=False)
    if not resolved_path.is_relative_to(resolved_build):
        raise SearchOutputResolutionError(
            f"Search output path '{db_path_raw}' escapes the build directory '{build_path}'."
        )
    return resolved_path


def list_search_outputs(build_dir: str | Path) -> list[SearchOutputSpec]:
    """List local search outputs recorded for a build.

    Falls back to the legacy ``build/search.db`` location when no projection
    metadata is available.
    """
    build_path = Path(build_dir)
    cache = _load_projection_cache(build_path)
    outputs: list[SearchOutputSpec] = []

    for name, entry in sorted(cache.items()):
        projection_type = entry.get("projection_type")
        db_path_raw = entry.get("db_path")
        if projection_type not in SEARCH_OUTPUT_TYPES or not db_path_raw:
            continue
        db_path = _normalized_output_path(build_path, db_path_raw)
        outputs.append(SearchOutputSpec(name=name, projection_type=projection_type, db_path=db_path))

    if outputs:
        return outputs

    legacy_db = (build_path / "search.db").resolve(strict=False)
    if legacy_db.exists():
        return [SearchOutputSpec(name="search", projection_type="search_index", db_path=legacy_db)]

    return []


def resolve_search_output(build_dir: str | Path, *, projection_name: str | None = None) -> SearchOutputSpec | None:
    """Resolve the search output the CLI should query by default."""
    outputs = list_search_outputs(build_dir)
    if not outputs:
        return None

    if projection_name is not None:
        for output in outputs:
            if output.name == projection_name:
                return output
        available = ", ".join(output.name for output in outputs)
        raise SearchOutputResolutionError(
            f"Unknown search projection '{projection_name}'. Available: {available}."
        )

    if len(outputs) == 1:
        return outputs[0]

    for preferred_type, preferred_name in (
        ("synix_search", "search"),
        ("search_index", "search"),
    ):
        for output in outputs:
            if output.projection_type == preferred_type and output.name == preferred_name:
                return output

    synix_outputs = [output for output in outputs if output.projection_type == "synix_search"]
    if len(synix_outputs) == 1:
        return synix_outputs[0]

    available = ", ".join(output.name for output in outputs)
    raise SearchOutputResolutionError(
        f"Multiple local search outputs found ({available}). Re-run with --projection <name>."
    )
