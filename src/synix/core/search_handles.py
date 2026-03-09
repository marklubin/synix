"""Public runtime handles for build-time search surfaces."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from synix.core.models import Layer, Transform
    from synix.search.results import SearchResult


class SearchSurfaceError(RuntimeError):
    """Base class for search-surface runtime resolution failures."""


class SearchSurfaceLookupError(SearchSurfaceError):
    """Raised when a transform requests an undeclared or ambiguous surface."""


class SearchSurfaceUnavailableError(SearchSurfaceError):
    """Raised when a declared surface cannot be opened for querying."""


@dataclass(frozen=True)
class SearchSurfaceHandle(Mapping[str, Any]):
    """Typed runtime handle for a declared build-time search surface.

    The handle is also mapping-compatible so legacy transform code that still
    treats ``search_surface`` like a dict keeps working during migration.
    """

    name: str
    db_path: str
    modes: tuple[str, ...] = ("fulltext",)
    sources: tuple[str, ...] = ()
    kind: str = "search_surface"

    @classmethod
    def from_value(cls, value: SearchSurfaceHandle | Mapping[str, Any]) -> SearchSurfaceHandle:
        """Coerce a mapping-compatible handle into the public typed handle."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"Expected search surface handle mapping, got {type(value)!r}")

        name = str(value.get("name", ""))
        if not name:
            raise ValueError("Search surface handle is missing required field 'name'")

        db_path = str(value.get("db_path", ""))
        if not db_path:
            raise ValueError(f"Search surface '{name}' is missing required field 'db_path'")

        raw_modes = value.get("modes") or value.get("search") or ["fulltext"]
        raw_sources = value.get("sources") or []
        kind = str(value.get("kind", "search_surface"))
        return cls(
            name=name,
            db_path=db_path,
            modes=tuple(str(mode) for mode in raw_modes),
            sources=tuple(str(source) for source in raw_sources),
            kind=kind,
        )

    @property
    def path(self) -> Path:
        """Filesystem path for the local search realization."""
        return Path(self.db_path)

    def to_dict(self) -> dict[str, Any]:
        """Return a legacy dict representation."""
        return {
            "name": self.name,
            "kind": self.kind,
            "db_path": self.db_path,
            "modes": list(self.modes),
            "sources": list(self.sources),
        }

    def is_available(self) -> bool:
        """Return True when the local search realization can be queried."""
        if not self.path.exists():
            return False

        from synix.search.indexer import SearchIndex

        index = None
        try:
            index = SearchIndex(self.path)
            return index.has_table("search_index")
        except sqlite3.Error:
            return False
        finally:
            if index is not None:
                index.close()

    def query(
        self,
        q: str,
        *,
        layers: list[str] | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Query the underlying local search realization."""
        from synix.search.indexer import SearchIndex

        if not self.path.exists():
            raise SearchSurfaceUnavailableError(f"Search surface '{self.name}' is not available at {self.path}.")

        index = SearchIndex(self.path)
        try:
            if not index.has_table("search_index"):
                raise SearchSurfaceUnavailableError(
                    f"Search surface '{self.name}' at {self.path} is missing the search_index table."
                )
            results = index.query(q, layers=layers)
            if limit is not None:
                results = results[:limit]
            return results
        except sqlite3.Error as exc:
            raise SearchSurfaceUnavailableError(
                f"Search surface '{self.name}' could not be opened from {self.path}."
            ) from exc
        finally:
            index.close()

    def search(
        self,
        q: str,
        *,
        mode: str = "fulltext",
        layers: list[str] | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Search the surface.

        Today Synix only materializes the local compatibility index as keyword
        search, so ``fulltext`` and ``keyword`` are equivalent here.
        """
        if mode not in {"fulltext", "keyword"}:
            raise NotImplementedError(f"Search surface '{self.name}' does not support runtime mode '{mode}' yet.")
        return self.query(q, layers=layers, limit=limit)

    def __getitem__(self, key: str) -> Any:
        if key == "name":
            return self.name
        if key == "kind":
            return self.kind
        if key == "db_path":
            return self.db_path
        if key == "modes":
            return list(self.modes)
        if key == "sources":
            return list(self.sources)
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(("name", "kind", "db_path", "modes", "sources"))

    def __len__(self) -> int:
        return 5


def resolve_search_surface_handle(
    context: Mapping[str, Any] | None,
    *,
    surface: str | Layer | None = None,
    transform: Transform | None = None,
    required: bool = False,
) -> SearchSurfaceHandle | None:
    """Resolve a declared search surface handle from runtime context."""
    data = context or {}
    handles = _coerce_search_surface_handles(data.get("search_surfaces"))

    name: str | None = None
    if surface is not None:
        name = getattr(surface, "name", surface)
    elif transform is not None:
        declared = [getattr(layer, "name", "") for layer in getattr(transform, "uses", [])]
        declared = [declared_name for declared_name in declared if declared_name]
        if len(declared) == 1:
            name = declared[0]
        elif len(declared) > 1:
            raise SearchSurfaceLookupError(
                f"Transform '{transform.name}' declares multiple search surfaces; pass surface='<name>' explicitly."
            )

    handle: SearchSurfaceHandle | None = None
    if name is not None:
        handle = handles.get(name)
        if handle is None:
            direct = data.get("search_surface")
            if direct is not None:
                direct_handle = SearchSurfaceHandle.from_value(direct)
                if direct_handle.name == name:
                    handle = direct_handle
    else:
        direct = data.get("search_surface")
        if direct is not None:
            handle = SearchSurfaceHandle.from_value(direct)
        elif len(handles) == 1:
            handle = next(iter(handles.values()))

    if handle is None:
        if required:
            if name is None and transform is not None and not getattr(transform, "uses", []):
                raise SearchSurfaceLookupError(
                    f"Transform '{transform.name}' did not declare a search surface in uses=[...]."
                )
            target = name or "<unspecified>"
            raise SearchSurfaceLookupError(f"Search surface '{target}' is not available in this transform context.")
        return None

    if handle.is_available():
        return handle

    if required:
        raise SearchSurfaceUnavailableError(
            f"Search surface '{handle.name}' is declared but not available at {handle.path}."
        )
    return None


def _coerce_search_surface_handles(raw: Any) -> dict[str, SearchSurfaceHandle]:
    """Normalize ``search_surfaces`` mappings into typed handles."""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"Expected search_surfaces mapping, got {type(raw)!r}")
    return {str(name): SearchSurfaceHandle.from_value(handle) for name, handle in raw.items()}
