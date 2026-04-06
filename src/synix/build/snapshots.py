"""Snapshot creation and lookup for Synix builds."""

from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from synix.build.object_store import SCHEMA_VERSION, ObjectStore
from synix.build.refs import DEFAULT_HEAD_REF, RefStore, synix_dir_for_build_dir
from synix.core.errors import atomic_write
from synix.core.models import Artifact, Pipeline


def _object(object_type: str, **fields: Any) -> dict[str, Any]:
    payload = {
        "type": object_type,
        "schema_version": SCHEMA_VERSION,
    }
    payload.update(fields)
    return payload


def _sanitize_llm_config(config: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    secret_keys = {
        "api_key",
        "apikey",
        "access_token",
        "auth_token",
        "refresh_token",
        "secret",
        "secret_key",
        "password",
        "token",
    }
    for key, value in config.items():
        lower = key.lower()
        if lower in secret_keys or lower.endswith(("_api_key", "_secret", "_password")):
            continue
        redacted[key] = _normalize_fingerprint_value(value)
    return redacted


def _normalize_fingerprint_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _normalize_fingerprint_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, set):
        normalized = [_normalize_fingerprint_value(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=True))
    if isinstance(value, (list, tuple)):
        return [_normalize_fingerprint_value(item) for item in value]
    if callable(value):
        qualname = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
        normalized = {
            "callable": f"{getattr(value, '__module__', 'builtins')}.{qualname}",
        }
        try:
            source = inspect.getsource(value)
        except (OSError, TypeError):
            source = None
        if source is not None:
            normalized["source_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
        return normalized
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        try:
            return isoformat()
        except TypeError:
            pass
    state = getattr(value, "__dict__", None)
    object_type = f"{type(value).__module__}.{type(value).__qualname__}"
    if isinstance(state, dict) and state:
        return {
            "object_type": object_type,
            "state": _normalize_fingerprint_value(state),
        }
    return {"object_type": object_type}


def _pipeline_fingerprint(pipeline: Pipeline) -> str:
    payload = {
        "name": pipeline.name,
        "llm_config": _sanitize_llm_config(pipeline.llm_config),
        "layers": [
            {
                "name": layer.name,
                "class": type(layer).__name__,
                "depends_on": [dep.name for dep in layer.depends_on],
                "config": _normalize_fingerprint_value(layer.config),
            }
            for layer in pipeline.layers
        ],
        "projections": [
            {
                "name": proj.name,
                "class": type(proj).__name__,
                "sources": [src.name for src in proj.sources],
                "config": _normalize_fingerprint_value(proj.config),
            }
            for proj in pipeline.projections
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _store_content_text(object_store: ObjectStore, text: str) -> str:
    content_oid, _size_bytes = object_store.put_text(text)
    return content_oid


def _artifact_text_content(artifact: Artifact) -> str:
    if not isinstance(artifact.content, str):
        msg = f"artifact {artifact.label!r} content must be a string, got {type(artifact.content).__name__}"
        raise TypeError(msg)
    return artifact.content


@dataclass
class BuildTransaction:
    """Canonical build state accumulated during a single pipeline run.

    Artifact state is captured as it is produced or reused during the build so
    final snapshot commit does not need to scrape the mutable compatibility
    release surface under ``build/``.

    Projection state capture is intentionally deferred to the explicit
    build/release adapter slice. Projections still materialize into ``build/``
    today as compatibility outputs, but they are not yet part of the canonical
    snapshot closure.
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
    parent_labels_map: dict[str, list[str]] = field(default_factory=dict)
    projection_oids: dict[str, str] = field(default_factory=dict)
    projection_declarations: dict[str, dict] = field(default_factory=dict)
    dlq_entries: list[dict[str, str]] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

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
        with self._lock:
            content = _artifact_text_content(artifact)
            snapshot_metadata = dict(artifact.metadata)
            snapshot_metadata.setdefault("layer_name", layer_name)
            snapshot_metadata.setdefault("layer_level", layer_level)
            content_hash = f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"
            if artifact.artifact_id and artifact.artifact_id != content_hash:
                msg = (
                    f"artifact {artifact.label!r} has artifact_id {artifact.artifact_id!r} "
                    f"that does not match its content hash {content_hash!r}"
                )
                raise ValueError(msg)
            if not artifact.artifact_id:
                artifact.artifact_id = content_hash
            content_oid = _store_content_text(self.object_store, content)
            artifact_payload = _object(
                "artifact",
                label=artifact.label,
                artifact_type=artifact.artifact_type,
                artifact_id=content_hash,
                content_oid=content_oid,
                input_ids=list(artifact.input_ids),
                prompt_id=artifact.prompt_id,
                agent_fingerprint=artifact.agent_fingerprint,
                model_config=artifact.model_config,
                metadata=snapshot_metadata,
                parent_labels=parent_labels,
            )
            artifact_oid = self.object_store.put_json(artifact_payload)

            previous_oid = self.artifact_oids.get(artifact.label)
            if previous_oid is not None and previous_oid != artifact_oid:
                layer_oids = self.layer_artifact_oids.get(layer_name, [])
                if previous_oid in layer_oids:
                    layer_oids.remove(previous_oid)

            self.artifact_oids[artifact.label] = artifact_oid
            self.parent_labels_map[artifact.label] = parent_labels
            layer_oids = self.layer_artifact_oids.setdefault(layer_name, [])
            if artifact_oid not in layer_oids:
                layer_oids.append(artifact_oid)

            return artifact_oid

    def record_projection(
        self,
        name: str,
        *,
        adapter: str,
        input_artifact_labels: list[str],
        config: dict,
        config_fingerprint: str,
        precomputed_oid: str | None = None,
    ) -> None:
        """Record a structured projection declaration for the manifest."""
        with self._lock:
            self.projection_declarations[name] = {
                "adapter": adapter,
                "input_artifacts": list(input_artifact_labels),
                "config": config,
                "config_fingerprint": config_fingerprint,
                "precomputed_oid": precomputed_oid,
            }

    def assert_complete(self, layer_artifacts: dict[str, list[Artifact]]) -> None:
        """Fail closed if the transaction missed artifacts present in the current build state."""
        expected_labels = {artifact.label for artifacts in layer_artifacts.values() for artifact in artifacts}
        recorded_labels = set(self.artifact_oids)

        missing = sorted(expected_labels.difference(recorded_labels))
        unexpected = sorted(recorded_labels.difference(expected_labels))
        if not missing and not unexpected:
            return

        details: list[str] = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:5]}")
        msg = "snapshot transaction closure mismatch: " + ", ".join(details)
        raise RuntimeError(msg)


def start_build_transaction(pipeline: Pipeline, build_dir: str | Path, run_id: str) -> BuildTransaction:
    """Create a build transaction that accumulates canonical snapshot state."""
    return BuildTransaction.start(pipeline, build_dir, run_id)


def _write_ref_update_journal(synix_dir: Path, journal_id: str, updates: dict[str, str]) -> Path:
    journal_dir = synix_dir / "ref_journal"
    journal_dir.mkdir(parents=True, exist_ok=True)
    journal_path = journal_dir / f"{journal_id}.json"
    atomic_write(
        journal_path,
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "type": "ref_update",
                "updates": updates,
            },
            sort_keys=True,
            indent=2,
        ),
    )
    return journal_path


def recover_pending_ref_updates(synix_dir: str | Path, *, ref_store: RefStore | None = None) -> None:
    """Apply any pending ref update journals left behind by interrupted commits."""
    resolved_synix_dir = Path(synix_dir)
    journal_dir = resolved_synix_dir / "ref_journal"
    if not journal_dir.exists():
        return

    store = ref_store or RefStore(resolved_synix_dir)
    for journal_path in sorted(journal_dir.glob("*.json")):
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
        updates = payload.get("updates")
        if not isinstance(updates, dict):
            msg = f"ref update journal {journal_path} is missing an 'updates' mapping"
            raise ValueError(msg)
        for ref_name, oid in updates.items():
            store.write_ref(ref_name, oid)
        journal_path.unlink()


@contextmanager
def _snapshot_lock(synix_dir: Path):
    lock_path = synix_dir / ".lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import fcntl
    except ImportError:
        fcntl = None
    try:
        import msvcrt
    except ImportError:
        msvcrt = None

    with lock_path.open("a+b") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:
            handle.seek(0)
            if handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def commit_build_snapshot(transaction: BuildTransaction) -> dict[str, str]:
    """Commit a build transaction as an immutable snapshot and advance refs."""
    synix_dir = transaction.synix_dir
    synix_dir.mkdir(parents=True, exist_ok=True)

    with _snapshot_lock(synix_dir):
        ref_store = RefStore(synix_dir)
        recover_pending_ref_updates(synix_dir, ref_store=ref_store)
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

        manifest_extra: dict[str, Any] = {}
        if transaction.dlq_entries:
            manifest_extra["dlq"] = transaction.dlq_entries

        manifest_payload = _object(
            "manifest",
            pipeline_name=transaction.pipeline.name,
            pipeline_fingerprint=_pipeline_fingerprint(transaction.pipeline),
            artifacts=[{"label": label, "oid": oid} for label, oid in sorted(transaction.artifact_oids.items())],
            projections=transaction.projection_declarations,
            **manifest_extra,
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
        pending_updates = {
            run_ref: snapshot_oid,
            transaction.head_ref: snapshot_oid,
        }
        journal_path = _write_ref_update_journal(synix_dir, transaction.run_id, pending_updates)
        refs_applied = False
        try:
            for ref_name, oid in pending_updates.items():
                ref_store.write_ref(ref_name, oid)
            refs_applied = True
        finally:
            if refs_applied and journal_path.exists():
                journal_path.unlink()

        return {
            "snapshot_oid": snapshot_oid,
            "manifest_oid": manifest_oid,
            "head_ref": transaction.head_ref,
            "run_ref": run_ref,
            "synix_dir": str(synix_dir),
        }


def write_layer_checkpoint(transaction: BuildTransaction, layer_name: str) -> None:
    """Write a checkpoint after a layer completes successfully.

    Checkpoints record which artifacts have been successfully built so far,
    enabling cache recovery after interrupted builds.
    """
    checkpoint_dir = transaction.synix_dir / "checkpoints" / transaction.run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with transaction._lock:
        payload = {
            "type": "checkpoint",
            "layer": layer_name,
            "artifact_oids": dict(transaction.artifact_oids),
            "parent_labels_map": {k: list(v) for k, v in transaction.parent_labels_map.items()},
        }
    atomic_write(
        checkpoint_dir / f"{layer_name}.json",
        json.dumps(payload, sort_keys=True, indent=2),
    )


def clear_checkpoints(synix_dir: Path) -> None:
    """Remove all checkpoint dirs after a successful snapshot commit."""
    checkpoint_base = synix_dir / "checkpoints"
    if checkpoint_base.exists():
        shutil.rmtree(checkpoint_base)


def list_runs(build_dir: str | Path, *, synix_dir: str | Path | None = None) -> list[dict[str, str]]:
    """List recorded run refs for a build dir."""
    build_path = Path(build_dir).resolve()
    resolved_synix_dir = synix_dir_for_build_dir(build_path, configured_synix_dir=synix_dir)
    if not resolved_synix_dir.exists():
        return []

    object_store = ObjectStore(resolved_synix_dir)
    ref_store = RefStore(resolved_synix_dir)
    recover_pending_ref_updates(resolved_synix_dir, ref_store=ref_store)
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
