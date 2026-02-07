"""Search result formatting with provenance."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """A single search result with provenance chain."""

    content: str
    artifact_id: str
    layer_name: str
    layer_level: int
    score: float
    provenance_chain: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
