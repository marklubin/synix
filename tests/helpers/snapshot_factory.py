"""Test helper — create .synix snapshot state from Artifact objects.

Usage in tests:

    from tests.helpers.snapshot_factory import create_test_snapshot

    synix_dir = create_test_snapshot(tmp_path, {
        "transcripts": [artifact_a, artifact_b],
        "episodes": [artifact_c],
    })
    store = SnapshotArtifactCache(synix_dir)
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from synix.build.object_store import SCHEMA_VERSION, ObjectStore
from synix.build.refs import RefStore
from synix.core.models import Artifact


def create_test_snapshot(
    base_dir: Path,
    layer_artifacts: dict[str, list[Artifact]] | None = None,
    *,
    synix_subdir: str = ".synix",
    parent_labels_map: dict[str, list[str]] | None = None,
    projections: dict[str, dict] | None = None,
) -> Path:
    """Create a minimal .synix snapshot from a layer -> artifacts mapping.

    Args:
        base_dir: Parent directory (e.g. tmp_path). The .synix dir is
            created as ``base_dir / synix_subdir``.
        layer_artifacts: Mapping of layer_name -> list[Artifact].
            If None, an empty snapshot is created.
        synix_subdir: Name of the synix directory (default ".synix").
        parent_labels_map: Optional mapping of artifact label -> parent labels.
        projections: Optional structured projection declarations for the manifest.

    Returns:
        Path to the .synix directory, suitable for SnapshotArtifactCache().
    """
    synix_dir = base_dir / synix_subdir
    object_store = ObjectStore(synix_dir)
    ref_store = RefStore(synix_dir)

    layer_artifacts = layer_artifacts or {}
    parent_labels_map = parent_labels_map or {}

    artifact_entries: list[dict] = []

    for layer_name, artifacts in layer_artifacts.items():
        for artifact in artifacts:
            # Store content blob
            content = artifact.content or ""
            content_oid = object_store.put_bytes(content.encode("utf-8"))
            content_hash = f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"

            # Ensure artifact_id is set
            if not artifact.artifact_id:
                artifact.artifact_id = content_hash

            # Build metadata
            metadata = dict(artifact.metadata)
            metadata.setdefault("layer_name", layer_name)
            metadata.setdefault("layer_level", artifact.metadata.get("layer_level", 0))
            if artifact.prompt_id:
                metadata["prompt_id"] = artifact.prompt_id
            if artifact.model_config:
                metadata["model_config"] = artifact.model_config
            if artifact.created_at:
                ts = artifact.created_at
                metadata.setdefault("created_at", ts.isoformat() if hasattr(ts, "isoformat") else str(ts))

            parents = parent_labels_map.get(artifact.label, [])

            artifact_payload = {
                "type": "artifact",
                "schema_version": SCHEMA_VERSION,
                "label": artifact.label,
                "artifact_type": artifact.artifact_type,
                "artifact_id": artifact.artifact_id,
                "content_oid": content_oid,
                "input_ids": list(artifact.input_ids),
                "metadata": metadata,
                "parent_labels": parents,
            }
            artifact_oid = object_store.put_json(artifact_payload)
            artifact_entries.append({"label": artifact.label, "oid": artifact_oid})

    # Build manifest
    manifest_payload = {
        "type": "manifest",
        "schema_version": SCHEMA_VERSION,
        "pipeline_name": "test-pipeline",
        "pipeline_fingerprint": "sha256:test",
        "artifacts": sorted(artifact_entries, key=lambda e: e["label"]),
        "projections": projections if projections is not None else {},
    }
    manifest_oid = object_store.put_json(manifest_payload)

    # Build snapshot
    snapshot_payload = {
        "type": "snapshot",
        "schema_version": SCHEMA_VERSION,
        "manifest_oid": manifest_oid,
        "parent_snapshot_oids": [],
        "created_at": datetime.now(UTC).isoformat(),
        "pipeline_name": "test-pipeline",
        "run_id": "test-run-001",
    }
    snapshot_oid = object_store.put_json(snapshot_payload)

    # Set refs
    ref_store.ensure_head()
    ref_store.write_ref("refs/heads/main", snapshot_oid)

    return synix_dir
