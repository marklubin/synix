"""SQLite FTS5 projection — materializes artifacts into searchable index."""

from __future__ import annotations

from pathlib import Path

from synix import Artifact
from synix.artifacts.provenance import ProvenanceTracker
from synix.projections.base import BaseProjection
from synix.search.index import SearchIndex
from synix.search.results import SearchResult


class SearchIndexProjection(BaseProjection):
    """Materializes artifacts into a SQLite FTS5 search index."""

    def __init__(self, build_dir: str | Path):
        self.build_dir = Path(build_dir)
        self.db_path = self.build_dir / "search.db"
        self._index: SearchIndex | None = None

    def _get_index(self) -> SearchIndex:
        if self._index is None:
            self._index = SearchIndex(self.db_path)
        return self._index

    def materialize(self, artifacts: list[Artifact], config: dict) -> None:
        """Populate FTS5 from artifacts across specified layers.

        Config should contain 'sources' — a list of dicts with 'layer' and 'level' keys.
        """
        index = self._get_index()
        index.create()

        sources = config.get("sources", [])
        layer_levels = {s["layer"]: s.get("level", 0) for s in sources}
        source_layers = set(layer_levels.keys())

        for artifact in artifacts:
            layer_name = artifact.metadata.get("layer_name", "")
            if layer_name in source_layers:
                index.insert(artifact, layer_name, layer_levels[layer_name])

    def query(
        self,
        q: str,
        layers: list[str] | None = None,
        provenance_tracker: ProvenanceTracker | None = None,
    ) -> list[SearchResult]:
        """Query the search index with optional layer filtering and provenance."""
        index = self._get_index()
        return index.query(q, layers=layers, provenance_tracker=provenance_tracker)

    def close(self) -> None:
        """Close the underlying search index."""
        if self._index is not None:
            self._index.close()
            self._index = None
