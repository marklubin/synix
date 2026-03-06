"""Snapshot creation and lookup for Synix builds."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synix.build.artifacts import ArtifactStore
from synix.build.object_store import SCHEMA_VERSION, ObjectStore
from synix.build.provenance import ProvenanceTracker
from synix.build.refs import DEFAULT_HEAD_REF, RefStore, synix_dir_for_build_dir
from synix.core.models import FlatFile, Pipeline, SearchIndex


def _object(object_type: str, **fields: Any) -> dict[str, Any]:
    payload = {
        "type": object_type,
        "schema_version": SCHEMA_VERSION,
    }
    payload.update(fields)
    return payload


def _sanitize_llm_config(config: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in config.items():
        lower = key.lower()
        if any(token in lower for token in ("key", "token", "secret", "password")):
            continue
        redacted[key] = value
    return redacted


def _pipeline_fingerprint(pipeline: Pipeline) -> str:
    payload = {
        "name": pipeline.name,
        "llm_config": _sanitize_llm_config(pipeline.llm_config),
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
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _store_blob_bytes(object_store: ObjectStore, data: bytes) -> str:
    content_oid = object_store.put_bytes(data)
    return object_store.put_json(
        _object(
            "blob",
            content_oid=content_oid,
            size_bytes=len(data),
        )
    )


def _store_blob_text(object_store: ObjectStore, text: str) -> str:
    content_oid, size_bytes = object_store.put_text(text)
    return object_store.put_json(
        _object(
            "blob",
            content_oid=content_oid,
            size_bytes=size_bytes,
        )
    )


def _store_blob_file(object_store: ObjectStore, path: str | Path) -> str:
    content_oid, size_bytes = object_store.put_file(path)
    return object_store.put_json(
        _object(
            "blob",
            content_oid=content_oid,
            size_bytes=size_bytes,
        )
    )


def _projection_input_oids(
    source_layers: list[Any],
    layer_artifact_oids: dict[str, list[str]],
) -> list[str]:
    input_oids: list[str] = []
    for source in source_layers:
        input_oids.extend(layer_artifact_oids.get(source.name, []))
    return input_oids


@contextmanager
def _snapshot_lock(synix_dir: Path):
    lock_path = synix_dir / ".lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import fcntl
    except ImportError:
        fcntl = None

    with lock_path.open("a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def commit_build_snapshot(pipeline: Pipeline, build_dir: str | Path, run_id: str) -> dict[str, str]:
    """Create a snapshot from the current build outputs and advance refs."""
    build_path = Path(build_dir).resolve()
    synix_dir = synix_dir_for_build_dir(build_path, configured_synix_dir=pipeline.synix_dir)
    synix_dir.mkdir(parents=True, exist_ok=True)

    with _snapshot_lock(synix_dir):
        ref_store = RefStore(synix_dir)
        object_store = ObjectStore(synix_dir)
        store = ArtifactStore(build_path)
        provenance = ProvenanceTracker(build_path)

        head_ref = ref_store.ensure_head(DEFAULT_HEAD_REF)
        parent_snapshot_oid = ref_store.read_ref("HEAD")

        artifact_oids: dict[str, str] = {}
        layer_artifact_oids: dict[str, list[str]] = {}
        for label, entry in sorted(store.iter_entries().items()):
            artifact = store.load_artifact(label)
            if artifact is None:
                continue

            content_oid = _store_blob_text(object_store, artifact.content)
            artifact_payload = _object(
                "artifact",
                label=artifact.label,
                artifact_type=artifact.artifact_type,
                artifact_id=artifact.artifact_id,
                content_oid=content_oid,
                input_ids=list(artifact.input_ids),
                prompt_id=artifact.prompt_id,
                model_config=artifact.model_config,
                metadata=artifact.metadata,
                created_at=artifact.created_at.isoformat(),
                parent_labels=provenance.get_parents(artifact.label),
            )
            artifact_oid = object_store.put_json(artifact_payload)
            artifact_oids[label] = artifact_oid
            layer_artifact_oids.setdefault(entry["layer"], []).append(artifact_oid)

        projection_oids: dict[str, str] = {}
        for projection in pipeline.projections:
            if isinstance(projection, SearchIndex):
                projection_path = build_path / "search.db"
                if not projection_path.exists():
                    continue
                state_oid = _store_blob_file(object_store, projection_path)
                projection_payload = _object(
                    "projection",
                    name=projection.name,
                    projection_type="search_index",
                    input_oids=_projection_input_oids(projection.sources, layer_artifact_oids),
                    build_state_oid=state_oid,
                    adapter="search_index",
                    release_mode="full",
                    metadata={"db_filename": "search.db"},
                )
                projection_oids[projection.name] = object_store.put_json(projection_payload)
            elif isinstance(projection, FlatFile):
                projection_path = Path(projection.output_path).resolve()
                if not projection_path.exists():
                    continue
                try:
                    relative_output_path = projection_path.relative_to(build_path)
                except ValueError as exc:
                    msg = (
                        f"FlatFile projection {projection.name!r} must write inside the build directory "
                        f"to be snapshotted: {projection.output_path}"
                    )
                    raise ValueError(msg) from exc

                state_oid = _store_blob_file(object_store, projection_path)
                projection_payload = _object(
                    "projection",
                    name=projection.name,
                    projection_type="flat_file",
                    input_oids=_projection_input_oids(projection.sources, layer_artifact_oids),
                    build_state_oid=state_oid,
                    adapter="flat_file",
                    release_mode="copy",
                    metadata={"output_path": relative_output_path.as_posix()},
                )
                projection_oids[projection.name] = object_store.put_json(projection_payload)

        manifest_payload = _object(
            "manifest",
            pipeline_name=pipeline.name,
            pipeline_fingerprint=_pipeline_fingerprint(pipeline),
            artifacts=artifact_oids,
            projections=projection_oids,
        )
        manifest_oid = object_store.put_json(manifest_payload)

        snapshot_payload = _object(
            "snapshot",
            manifest_oid=manifest_oid,
            parent_snapshot_oids=[parent_snapshot_oid] if parent_snapshot_oid else [],
            created_at=datetime.now(UTC).isoformat(),
            pipeline_name=pipeline.name,
            run_id=run_id,
        )
        snapshot_oid = object_store.put_json(snapshot_payload)

        run_ref = f"refs/runs/{run_id}"
        # Write the immutable run ref first, then advance HEAD to the new tip.
        ref_store.write_ref(run_ref, snapshot_oid)
        ref_store.write_ref(head_ref, snapshot_oid)

        return {
            "snapshot_oid": snapshot_oid,
            "manifest_oid": manifest_oid,
            "head_ref": head_ref,
            "run_ref": run_ref,
            "synix_dir": str(synix_dir),
        }


def list_runs(build_dir: str | Path, *, synix_dir: str | Path | None = None) -> list[dict[str, str]]:
    """List recorded run refs for a build dir."""
    build_path = Path(build_dir).resolve()
    resolved_synix_dir = synix_dir_for_build_dir(build_path, configured_synix_dir=synix_dir)
    if not resolved_synix_dir.exists():
        return []

    object_store = ObjectStore(resolved_synix_dir)
    ref_store = RefStore(resolved_synix_dir)
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
