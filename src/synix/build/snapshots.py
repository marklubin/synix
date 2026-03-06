"""Snapshot creation and lookup for Synix builds."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synix.build.artifacts import ArtifactStore
from synix.build.object_store import ObjectStore
from synix.build.provenance import ProvenanceTracker
from synix.build.refs import DEFAULT_HEAD_REF, RefStore, synix_dir_for_build_dir
from synix.core.models import FlatFile, Pipeline, SearchIndex


def _pipeline_fingerprint(pipeline: Pipeline) -> str:
    payload = {
        "name": pipeline.name,
        "source_dir": pipeline.source_dir,
        "llm_config": pipeline.llm_config,
        "layers": [
            {
                "name": layer.name,
                "class": type(layer).__name__,
                "depends_on": [dep.name for dep in layer.depends_on],
                "config": layer.config,
            }
            for layer in pipeline.layers
        ],
        "projections": [
            {
                "name": proj.name,
                "class": type(proj).__name__,
                "sources": [src.name for src in proj.sources],
                "config": proj.config,
            }
            for proj in pipeline.projections
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    import hashlib

    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _blob_object(data: bytes) -> dict[str, Any]:
    return {
        "type": "blob",
        "encoding": "base64",
        "bytes": base64.b64encode(data).decode("ascii"),
    }


def commit_build_snapshot(pipeline: Pipeline, build_dir: str | Path, run_id: str) -> dict[str, str]:
    """Create a snapshot from the current build outputs and advance refs."""
    build_path = Path(build_dir)
    synix_dir = synix_dir_for_build_dir(build_path)
    synix_dir.mkdir(parents=True, exist_ok=True)

    ref_store = RefStore(synix_dir)
    object_store = ObjectStore(synix_dir)
    store = ArtifactStore(build_path)
    provenance = ProvenanceTracker(build_path)

    head_ref = ref_store.ensure_head(DEFAULT_HEAD_REF)
    parent_snapshot_oid = ref_store.read_ref("HEAD")

    artifact_oids: dict[str, str] = {}
    for label in sorted(store._manifest):
        artifact = store.load_artifact(label)
        if artifact is None:
            continue
        content_oid = object_store.put_json(_blob_object(artifact.content.encode("utf-8")))
        artifact_payload = {
            "type": "artifact",
            "label": artifact.label,
            "artifact_type": artifact.artifact_type,
            "artifact_id": artifact.artifact_id,
            "content_oid": content_oid,
            "input_ids": list(artifact.input_ids),
            "prompt_id": artifact.prompt_id,
            "model_config": artifact.model_config,
            "metadata": artifact.metadata,
            "created_at": artifact.created_at.isoformat(),
            "parent_labels": provenance.get_parents(artifact.label),
        }
        artifact_oids[label] = object_store.put_json(artifact_payload)

    projection_oids: dict[str, str] = {}
    for projection in pipeline.projections:
        if isinstance(projection, SearchIndex):
            projection_path = build_path / "search.db"
            if not projection_path.exists():
                continue
            state_oid = object_store.put_json(_blob_object(projection_path.read_bytes()))
            projection_payload = {
                "type": "projection",
                "name": projection.name,
                "projection_type": "search_index",
                "input_oids": [artifact_oids[label] for label in artifact_oids],
                "build_state_oid": state_oid,
                "adapter": "search_index",
                "release_mode": "full",
                "metadata": {
                    "db_filename": "search.db",
                },
            }
            projection_oids[projection.name] = object_store.put_json(projection_payload)
        elif isinstance(projection, FlatFile):
            projection_path = Path(projection.output_path)
            if not projection_path.exists():
                continue
            state_oid = object_store.put_json(_blob_object(projection_path.read_bytes()))
            projection_payload = {
                "type": "projection",
                "name": projection.name,
                "projection_type": "flat_file",
                "input_oids": [artifact_oids[label] for label in artifact_oids],
                "build_state_oid": state_oid,
                "adapter": "flat_file",
                "release_mode": "copy",
                "metadata": {
                    "output_path": str(projection.output_path),
                },
            }
            projection_oids[projection.name] = object_store.put_json(projection_payload)

    manifest_payload = {
        "type": "manifest",
        "pipeline_name": pipeline.name,
        "pipeline_fingerprint": _pipeline_fingerprint(pipeline),
        "artifacts": artifact_oids,
        "projections": projection_oids,
    }
    manifest_oid = object_store.put_json(manifest_payload)

    snapshot_payload = {
        "type": "snapshot",
        "manifest_oid": manifest_oid,
        "parent_snapshot_oids": [parent_snapshot_oid] if parent_snapshot_oid else [],
        "created_at": datetime.now(UTC).isoformat(),
        "pipeline_name": pipeline.name,
        "run_id": run_id,
    }
    snapshot_oid = object_store.put_json(snapshot_payload)

    run_ref = f"refs/runs/{run_id}"
    ref_store.write_ref(run_ref, snapshot_oid)
    ref_store.write_ref(head_ref, snapshot_oid)

    return {
        "snapshot_oid": snapshot_oid,
        "manifest_oid": manifest_oid,
        "head_ref": head_ref,
        "run_ref": run_ref,
        "synix_dir": str(synix_dir),
    }


def list_runs(build_dir: str | Path) -> list[dict[str, str]]:
    """List recorded run refs for a build dir."""
    build_path = Path(build_dir)
    synix_dir = synix_dir_for_build_dir(build_path)
    if not synix_dir.exists():
        return []
    object_store = ObjectStore(synix_dir)
    ref_store = RefStore(synix_dir)
    runs: list[dict[str, str]] = []
    for ref_name, oid in ref_store.iter_refs("refs/runs"):
        snapshot = object_store.get_json(oid)
        runs.append(
            {
                "ref": ref_name,
                "snapshot_oid": oid,
                "run_id": snapshot.get("run_id", ""),
                "created_at": snapshot.get("created_at", ""),
                "pipeline_name": snapshot.get("pipeline_name", ""),
            }
        )
    return runs
