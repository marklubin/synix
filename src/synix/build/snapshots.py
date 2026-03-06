"""Snapshot creation and lookup for Synix builds."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synix.build.object_store import SCHEMA_VERSION, ObjectStore
from synix.build.refs import DEFAULT_HEAD_REF, RefStore, synix_dir_for_build_dir
from synix.core.models import Artifact, FlatFile, Pipeline, SearchIndex


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


@dataclass
class BuildTransaction:
    """Canonical build state accumulated during a single pipeline run.

    Artifacts and projection state are captured as they are produced or reused
    during the build so final snapshot commit does not need to scrape the
    mutable compatibility release surface under ``build/``.
    """

    pipeline: Pipeline
    build_dir: Path
    synix_dir: Path
    run_id: str
    object_store: ObjectStore
    head_ref: str
    parent_snapshot_oid: str | None
    artifact_oids: dict[str, str] = field(default_factory=dict)
    layer_artifact_oids: dict[str, list[str]] = field(default_factory=dict)
    projection_oids: dict[str, str] = field(default_factory=dict)

    @classmethod
    def start(cls, pipeline: Pipeline, build_dir: str | Path, run_id: str) -> BuildTransaction:
        build_path = Path(build_dir).resolve()
        synix_dir = synix_dir_for_build_dir(build_path, configured_synix_dir=pipeline.synix_dir)
        synix_dir.mkdir(parents=True, exist_ok=True)

        ref_store = RefStore(synix_dir)
        head_ref = ref_store.ensure_head(DEFAULT_HEAD_REF)

        return cls(
            pipeline=pipeline,
            build_dir=build_path,
            synix_dir=synix_dir,
            run_id=run_id,
            object_store=ObjectStore(synix_dir),
            head_ref=head_ref,
            parent_snapshot_oid=ref_store.read_ref("HEAD"),
        )

    def record_artifact(
        self,
        artifact: Artifact,
        *,
        layer_name: str,
        layer_level: int,
        parent_labels: list[str],
    ) -> str:
        artifact.metadata.setdefault("layer_name", layer_name)
        artifact.metadata.setdefault("layer_level", layer_level)
        content_oid = _store_blob_text(self.object_store, artifact.content)
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
            parent_labels=parent_labels,
        )
        artifact_oid = self.object_store.put_json(artifact_payload)

        previous_oid = self.artifact_oids.get(artifact.label)
        if previous_oid is not None and previous_oid != artifact_oid:
            layer_oids = self.layer_artifact_oids.get(layer_name, [])
            if previous_oid in layer_oids:
                layer_oids.remove(previous_oid)

        self.artifact_oids[artifact.label] = artifact_oid
        layer_oids = self.layer_artifact_oids.setdefault(layer_name, [])
        if artifact_oid not in layer_oids:
            layer_oids.append(artifact_oid)

        return artifact_oid

    def record_projection(self, projection: Any) -> str | None:
        if isinstance(projection, SearchIndex):
            projection_path = self.build_dir / "search.db"
            if not projection_path.exists():
                return None

            state_oid = _store_blob_file(self.object_store, projection_path)
            projection_payload = _object(
                "projection",
                name=projection.name,
                projection_type="search_index",
                input_oids=_projection_input_oids(projection.sources, self.layer_artifact_oids),
                build_state_oid=state_oid,
                adapter="search_index",
                release_mode="full",
                metadata={"db_filename": "search.db"},
            )
        elif isinstance(projection, FlatFile):
            projection_path = Path(projection.output_path).resolve()
            if not projection_path.exists():
                return None
            try:
                relative_output_path = projection_path.relative_to(self.build_dir)
            except ValueError as exc:
                msg = (
                    f"FlatFile projection {projection.name!r} must write inside the build directory "
                    f"to be snapshotted: {projection.output_path}"
                )
                raise ValueError(msg) from exc

            state_oid = _store_blob_file(self.object_store, projection_path)
            projection_payload = _object(
                "projection",
                name=projection.name,
                projection_type="flat_file",
                input_oids=_projection_input_oids(projection.sources, self.layer_artifact_oids),
                build_state_oid=state_oid,
                adapter="flat_file",
                release_mode="copy",
                metadata={"output_path": relative_output_path.as_posix()},
            )
        else:
            return None

        projection_oid = self.object_store.put_json(projection_payload)
        self.projection_oids[projection.name] = projection_oid
        return projection_oid


def start_build_transaction(pipeline: Pipeline, build_dir: str | Path, run_id: str) -> BuildTransaction:
    """Create a build transaction that accumulates canonical snapshot state."""
    return BuildTransaction.start(pipeline, build_dir, run_id)


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


def commit_build_snapshot(transaction: BuildTransaction) -> dict[str, str]:
    """Commit a build transaction as an immutable snapshot and advance refs."""
    synix_dir = transaction.synix_dir
    synix_dir.mkdir(parents=True, exist_ok=True)

    with _snapshot_lock(synix_dir):
        ref_store = RefStore(synix_dir)
        current_head_ref = ref_store.ensure_head(DEFAULT_HEAD_REF)
        if current_head_ref != transaction.head_ref:
            msg = (
                "HEAD target changed during build "
                f"({transaction.head_ref!r} -> {current_head_ref!r}); rerun against the latest ref state"
            )
            raise RuntimeError(msg)

        current_head_oid = ref_store.read_ref("HEAD")
        if current_head_oid != transaction.parent_snapshot_oid:
            msg = "HEAD advanced during build; rerun against the latest snapshot"
            raise RuntimeError(msg)

        manifest_payload = _object(
            "manifest",
            pipeline_name=transaction.pipeline.name,
            pipeline_fingerprint=_pipeline_fingerprint(transaction.pipeline),
            artifacts=transaction.artifact_oids,
            projections=transaction.projection_oids,
        )
        manifest_oid = transaction.object_store.put_json(manifest_payload)

        snapshot_payload = _object(
            "snapshot",
            manifest_oid=manifest_oid,
            parent_snapshot_oids=[transaction.parent_snapshot_oid] if transaction.parent_snapshot_oid else [],
            created_at=datetime.now(UTC).isoformat(),
            pipeline_name=transaction.pipeline.name,
            run_id=transaction.run_id,
        )
        snapshot_oid = transaction.object_store.put_json(snapshot_payload)

        run_ref = f"refs/runs/{transaction.run_id}"
        # Write the immutable run ref first, then advance HEAD to the new tip.
        ref_store.write_ref(run_ref, snapshot_oid)
        ref_store.write_ref(transaction.head_ref, snapshot_oid)

        return {
            "snapshot_oid": snapshot_oid,
            "manifest_oid": manifest_oid,
            "head_ref": transaction.head_ref,
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
