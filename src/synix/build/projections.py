"""Projection base and implementations — materialize artifacts into output surfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from synix.core.models import Artifact


class BaseProjection(ABC):
    """Abstract base for projections that materialize artifacts into output surfaces."""

    @abstractmethod
    def materialize(self, artifacts: list[Artifact], config: dict) -> None:
        """Materialize artifacts into the projection's output format."""
        ...


class FlatFileProjection(BaseProjection):
    """Renders core memory artifact as a markdown context document."""

    def materialize(self, artifacts: list[Artifact], config: dict) -> None:
        """Render artifacts to a markdown file."""
        output_path = Path(config.get("output_path", "./build/context.md"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        parts: list[str] = []
        for artifact in artifacts:
            parts.append(artifact.content)

        content = "\n\n".join(parts)
        output_path.write_text(content)


def get_projection(name: str, *args, **kwargs):
    """Get an instantiated projection by type name.

    Only search_index is registered dynamically (via search module).
    flat_file is handled directly by the runner via FlatFileProjection.
    """
    # Lazy import to avoid build->search circular dependency
    if name == "search_index":
        from synix.search.indexer import SearchIndexProjection

        return SearchIndexProjection(*args, **kwargs)
    raise ValueError(f"Unknown projection type: {name}. Available: ['search_index', 'flat_file']")
