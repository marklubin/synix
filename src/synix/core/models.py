"""Core data models for Synix."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Artifact:
    """Immutable, versioned build output."""

    artifact_id: str
    artifact_type: str  # "transcript", "episode", "rollup", "core_memory", "search_index"
    content: str
    content_hash: str = ""
    input_hashes: list[str] = field(default_factory=list)
    prompt_id: str | None = None
    model_config: dict | None = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.content_hash and self.content:
            self.content_hash = f"sha256:{hashlib.sha256(self.content.encode()).hexdigest()}"


@dataclass
class ProvenanceRecord:
    """Lineage record for an artifact."""

    artifact_id: str
    parent_artifact_ids: list[str] = field(default_factory=list)
    prompt_id: str | None = None
    model_config: dict | None = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Layer:
    """A named level in the memory hierarchy."""

    name: str
    level: int
    transform: str
    depends_on: list[str] = field(default_factory=list)
    grouping: str | None = None  # "by_conversation", "by_month", "by_topic", "single"
    config: dict = field(default_factory=dict)
    context_budget: int | None = None


@dataclass
class Projection:
    """Materializes build artifacts into a usable output surface."""

    name: str
    projection_type: str  # "search_index", "flat_file"
    sources: list[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)


@dataclass
class Pipeline:
    """The full declared memory architecture."""

    name: str
    layers: list[Layer] = field(default_factory=list)
    projections: list[Projection] = field(default_factory=list)
    source_dir: str = "./exports"
    build_dir: str = "./build"
    llm_config: dict = field(default_factory=dict)

    def add_layer(self, layer: Layer) -> None:
        self.layers.append(layer)

    def add_projection(self, projection: Projection) -> None:
        self.projections.append(projection)
