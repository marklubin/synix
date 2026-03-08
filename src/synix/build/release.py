"""Release closure — fully-resolved artifact bundle built from a snapshot."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from synix.build.object_store import ObjectStore
from synix.build.refs import RefStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedArtifact:
    """A fully-resolved artifact with content and provenance."""

    label: str
    artifact_type: str
    content: str
    artifact_id: str
    layer_name: str
    layer_level: int
    provenance_chain: list[str]  # BFS walk from parent_labels
    metadata: dict


@dataclass(frozen=True)
class ProjectionDeclaration:
    """A structured projection declaration from the manifest."""

    name: str
    adapter: str
    input_artifacts: list[str]  # labels
    config: dict
    config_fingerprint: str
    precomputed_oid: str | None = None


def _walk_provenance(label: str, artifact_objects: dict[str, dict]) -> list[str]:
    """BFS walk of parent_labels to build a provenance chain."""
    visited: set[str] = set()
    queue: deque[str] = deque([label])
    chain: list[str] = []
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        chain.append(current)
        art = artifact_objects.get(current)
        if art:
            for parent in art.get("parent_labels", []):
                if parent not in visited:
                    queue.append(parent)
    return chain


@dataclass
class ReleaseClosure:
    """Fully-resolved artifact bundle built from a snapshot.

    This is the universal input that every adapter receives. It contains
    all artifacts with their content resolved and provenance chains walked,
    plus all projection declarations from the manifest.
    """

    snapshot_oid: str
    manifest_oid: str
    pipeline_name: str
    created_at: str
    artifacts: dict[str, ResolvedArtifact]  # label -> resolved
    projections: dict[str, ProjectionDeclaration]

    @classmethod
    def from_snapshot(cls, synix_dir: str | Path, snapshot_oid: str) -> ReleaseClosure:
        """Build a release closure from a snapshot OID.

        Loads the snapshot -> manifest -> all artifact objects + content blobs,
        walks parent_labels for provenance, and parses projection declarations.
        """
        synix_path = Path(synix_dir)
        object_store = ObjectStore(synix_path)

        # Load and verify snapshot
        snapshot = object_store.get_json(snapshot_oid)
        if snapshot.get("type") != "snapshot":
            msg = f"object {snapshot_oid} has type {snapshot.get('type')!r}, expected 'snapshot'"
            raise ValueError(msg)

        # Load manifest
        manifest_oid = snapshot["manifest_oid"]
        manifest = object_store.get_json(manifest_oid)

        # Load all artifact objects first (needed for provenance walking)
        artifact_objects: dict[str, dict] = {}
        for entry in manifest["artifacts"]:
            label = entry["label"]
            art_obj = object_store.get_json(entry["oid"])
            artifact_objects[label] = art_obj

        # Build resolved artifacts with provenance
        resolved_artifacts: dict[str, ResolvedArtifact] = {}
        for label, art_obj in artifact_objects.items():
            content = object_store.get_bytes(art_obj["content_oid"]).decode("utf-8")
            provenance_chain = _walk_provenance(label, artifact_objects)
            metadata = art_obj.get("metadata", {})

            resolved_artifacts[label] = ResolvedArtifact(
                label=label,
                artifact_type=art_obj["artifact_type"],
                content=content,
                artifact_id=art_obj["artifact_id"],
                layer_name=metadata.get("layer_name", ""),
                layer_level=metadata.get("layer_level", 0),
                provenance_chain=provenance_chain,
                metadata=metadata,
            )

        # Parse projection declarations
        projections: dict[str, ProjectionDeclaration] = {}
        for proj_name, proj_entry in manifest.get("projections", {}).items():
            projections[proj_name] = ProjectionDeclaration(
                name=proj_name,
                adapter=proj_entry["adapter"],
                input_artifacts=proj_entry["input_artifacts"],
                config=proj_entry["config"],
                config_fingerprint=proj_entry["config_fingerprint"],
                precomputed_oid=proj_entry.get("precomputed_oid"),
            )

        return cls(
            snapshot_oid=snapshot_oid,
            manifest_oid=manifest_oid,
            pipeline_name=snapshot["pipeline_name"],
            created_at=snapshot["created_at"],
            artifacts=resolved_artifacts,
            projections=projections,
        )

    @classmethod
    def from_ref(cls, synix_dir: str | Path, ref: str = "HEAD") -> ReleaseClosure:
        """Build a release closure by resolving a ref first."""
        synix_path = Path(synix_dir)
        ref_store = RefStore(synix_path)

        snapshot_oid = ref_store.read_ref(ref)
        if snapshot_oid is None:
            msg = f"ref {ref!r} does not resolve to a snapshot"
            raise ValueError(msg)

        return cls.from_snapshot(synix_path, snapshot_oid)

    def artifacts_for_projection(self, projection_name: str) -> dict[str, ResolvedArtifact]:
        """Return only the artifacts referenced by a projection's input_artifacts."""
        projection = self.projections.get(projection_name)
        if projection is None:
            msg = f"projection {projection_name!r} not found in closure"
            raise KeyError(msg)

        return {label: self.artifacts[label] for label in projection.input_artifacts if label in self.artifacts}
