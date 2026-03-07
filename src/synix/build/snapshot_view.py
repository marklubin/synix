"""Ref-resolved read API over immutable snapshots."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from synix.build.object_store import ObjectStore
from synix.build.refs import RefStore
from synix.core.models import Artifact

logger = logging.getLogger(__name__)


class SnapshotView:
    """Ref-resolved read API over immutable snapshots."""

    def __init__(
        self,
        synix_dir: Path,
        object_store: ObjectStore,
        ref_store: RefStore,
        snapshot_oid: str,
        snapshot: dict,
        manifest_oid: str,
        manifest: dict,
    ):
        self._synix_dir = synix_dir
        self._object_store = object_store
        self._ref_store = ref_store
        self._snapshot_oid = snapshot_oid
        self._snapshot = snapshot
        self._manifest_oid = manifest_oid
        self._manifest = manifest
        self._artifact_oids: dict[str, str] = {entry["label"]: entry["oid"] for entry in manifest["artifacts"]}

    @classmethod
    def open(cls, synix_dir: str | Path, ref: str = "HEAD") -> SnapshotView:
        """Open a snapshot view by resolving a ref."""
        synix_path = Path(synix_dir)
        object_store = ObjectStore(synix_path)
        ref_store = RefStore(synix_path)

        snapshot_oid = ref_store.read_ref(ref)
        if snapshot_oid is None:
            raise ValueError(f"ref {ref!r} does not resolve to a snapshot")

        snapshot = object_store.get_json(snapshot_oid)
        if snapshot.get("type") != "snapshot":
            raise ValueError(f"ref {ref!r} resolves to {snapshot.get('type')!r}, expected 'snapshot'")

        manifest_oid = snapshot["manifest_oid"]
        manifest = object_store.get_json(manifest_oid)

        return cls(synix_path, object_store, ref_store, snapshot_oid, snapshot, manifest_oid, manifest)

    @property
    def snapshot_oid(self) -> str:
        return self._snapshot_oid

    @property
    def manifest_oid(self) -> str:
        return self._manifest_oid

    def list_artifacts(self) -> list[dict]:
        """Return manifest artifacts with full artifact objects loaded."""
        results = []
        for entry in self._manifest["artifacts"]:
            artifact_obj = self._object_store.get_json(entry["oid"])
            results.append(artifact_obj)
        return results

    def get_artifact(self, label: str) -> dict:
        """Load artifact object + content blob for a label."""
        oid = self._artifact_oids.get(label)
        if oid is None:
            raise KeyError(f"artifact {label!r} not found in snapshot")
        artifact_obj = self._object_store.get_json(oid)
        content = self._object_store.get_bytes(artifact_obj["content_oid"]).decode("utf-8")
        result = dict(artifact_obj)
        result["content"] = content
        return result

    def get_content(self, label: str) -> str:
        """Return raw content text for an artifact label."""
        oid = self._artifact_oids.get(label)
        if oid is None:
            raise KeyError(f"artifact {label!r} not found in snapshot")
        artifact_obj = self._object_store.get_json(oid)
        return self._object_store.get_bytes(artifact_obj["content_oid"]).decode("utf-8")

    def get_manifest(self) -> dict:
        """Return the full manifest dict."""
        return dict(self._manifest)

    def get_provenance(self, label: str) -> list[str]:
        """Walk parent_labels transitively, returning BFS-ordered labels."""
        visited: set[str] = set()
        queue: list[str] = [label]
        chain: list[str] = []

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            chain.append(current)

            oid = self._artifact_oids.get(current)
            if oid is None:
                continue
            artifact_obj = self._object_store.get_json(oid)
            for parent in artifact_obj.get("parent_labels", []):
                if parent not in visited:
                    queue.append(parent)

        return chain

    def resolve_prefix(self, prefix: str) -> str | None:
        """Git-like prefix resolution against artifact labels and artifact_ids."""
        hash_prefix = prefix.removeprefix("sha256:")

        # Exact match
        if prefix in self._artifact_oids:
            return prefix

        # Label prefix match
        label_matches = [lbl for lbl in self._artifact_oids if lbl.startswith(prefix)]
        if len(label_matches) == 1:
            return label_matches[0]
        if len(label_matches) > 1:
            labels = ", ".join(sorted(label_matches)[:5])
            raise ValueError(f"ambiguous prefix '{prefix}' matches {len(label_matches)} labels: {labels}")

        # Hash prefix match
        hash_matches = []
        for lbl, oid in self._artifact_oids.items():
            artifact_obj = self._object_store.get_json(oid)
            aid = artifact_obj.get("artifact_id", "").removeprefix("sha256:")
            if aid.startswith(hash_prefix):
                hash_matches.append(lbl)

        if len(hash_matches) == 1:
            return hash_matches[0]
        if len(hash_matches) > 1:
            labels = ", ".join(sorted(hash_matches)[:5])
            raise ValueError(f"ambiguous hash prefix '{prefix}' matches {len(hash_matches)} artifacts: {labels}")

        return None


class SnapshotArtifactCache:
    """Read-only artifact cache backed by the previous snapshot.

    Provides the same read interface as ArtifactStore (load_artifact,
    list_artifacts, get_artifact_id, iter_entries, resolve_prefix) plus
    provenance methods (get_parents, get_chain). Used as a drop-in
    replacement for ArtifactStore in the runner and planner.
    """

    def __init__(self, synix_dir: Path):
        self._artifacts_by_label: dict[str, Artifact] = {}
        self._artifacts_by_layer: dict[str, list[Artifact]] = {}
        self._parent_labels_map: dict[str, list[str]] = {}
        self._manifest_entries: dict[str, dict] = {}
        self._view: SnapshotView | None = None

        try:
            view = SnapshotView.open(synix_dir)
            self._view = view
            for entry in view._manifest["artifacts"]:
                label = entry["label"]
                oid = entry["oid"]
                art_obj = view._object_store.get_json(oid)
                content = view._object_store.get_bytes(art_obj["content_oid"]).decode("utf-8")

                created_at_str = art_obj.get("metadata", {}).get("created_at")
                created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now()

                artifact = Artifact(
                    label=art_obj["label"],
                    artifact_type=art_obj["artifact_type"],
                    artifact_id=art_obj["artifact_id"],
                    input_ids=art_obj.get("input_ids", []),
                    prompt_id=art_obj.get("prompt_id"),
                    model_config=art_obj.get("model_config"),
                    created_at=created_at,
                    content=content,
                    metadata=art_obj.get("metadata", {}),
                )

                layer_name = art_obj.get("metadata", {}).get("layer_name", "")
                layer_level = art_obj.get("metadata", {}).get("layer_level", 0)

                self._artifacts_by_label[label] = artifact
                self._artifacts_by_layer.setdefault(layer_name, []).append(artifact)
                self._parent_labels_map[label] = art_obj.get("parent_labels", [])
                self._manifest_entries[label] = {
                    "path": "",
                    "artifact_id": art_obj["artifact_id"],
                    "layer": layer_name,
                    "level": layer_level,
                }
        except (ValueError, FileNotFoundError, KeyError):
            logger.debug("No previous snapshot found in %s — starting with empty cache", synix_dir)

    def load_artifact(self, label: str) -> Artifact | None:
        """Load an artifact by label. Returns None if not found."""
        return self._artifacts_by_label.get(label)

    def list_artifacts(self, layer: str) -> list[Artifact]:
        """Return all artifacts for a given layer name."""
        return list(self._artifacts_by_layer.get(layer, []))

    def get_artifact_id(self, label: str) -> str | None:
        """Quick artifact ID (hash) lookup without loading full artifact."""
        art = self._artifacts_by_label.get(label)
        return art.artifact_id if art else None

    def iter_entries(self) -> dict[str, dict]:
        """Return a shallow copy of the manifest entries keyed by label."""
        return dict(self._manifest_entries)

    def resolve_prefix(self, prefix: str) -> str | None:
        """Resolve a prefix to a full label (git-like semantics)."""
        hash_prefix = prefix.removeprefix("sha256:")

        if prefix in self._manifest_entries:
            return prefix

        label_matches = [lbl for lbl in self._manifest_entries if lbl.startswith(prefix)]
        if len(label_matches) == 1:
            return label_matches[0]
        if len(label_matches) > 1:
            labels = ", ".join(sorted(label_matches)[:5])
            raise ValueError(f"ambiguous prefix '{prefix}' matches {len(label_matches)} labels: {labels}")

        hash_matches = [
            lbl
            for lbl, entry in self._manifest_entries.items()
            if entry["artifact_id"].removeprefix("sha256:").startswith(hash_prefix)
        ]
        if len(hash_matches) == 1:
            return hash_matches[0]
        if len(hash_matches) > 1:
            labels = ", ".join(sorted(hash_matches)[:5])
            raise ValueError(f"ambiguous hash prefix '{prefix}' matches {len(hash_matches)} artifacts: {labels}")

        return None

    def get_parents(self, label: str) -> list[str]:
        """Return direct parent labels for an artifact."""
        return list(self._parent_labels_map.get(label, []))

    def get_chain(self, label: str) -> list[str]:
        """Walk parent_labels transitively, returning BFS-ordered labels."""
        visited: set[str] = set()
        queue: list[str] = [label]
        chain: list[str] = []

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            chain.append(current)

            for parent in self._parent_labels_map.get(current, []):
                if parent not in visited:
                    queue.append(parent)

        return chain

    def update_from_build(self, layer_artifacts: dict[str, list]) -> None:
        """Merge in-memory build results into the cache.

        After a pipeline run completes, the cache is still a snapshot of the
        *previous* build.  This method overlays the current build's artifacts so
        that downstream consumers (validators, fixers) see the freshly built
        data without requiring a full snapshot commit + reload cycle.
        """
        for layer_name, artifacts in layer_artifacts.items():
            self._artifacts_by_layer[layer_name] = list(artifacts)
            for art in artifacts:
                self._artifacts_by_label[art.label] = art
                level = art.metadata.get("layer_level", 0)
                self._manifest_entries[art.label] = {
                    "path": "",
                    "artifact_id": art.artifact_id,
                    "layer": layer_name,
                    "level": level,
                }
