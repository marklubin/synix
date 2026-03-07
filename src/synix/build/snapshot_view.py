"""Ref-resolved read API over immutable snapshots."""

from __future__ import annotations

from pathlib import Path

from synix.build.object_store import ObjectStore
from synix.build.refs import RefStore


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
