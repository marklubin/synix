"""Immutable object storage for snapshots and related metadata."""

from __future__ import annotations

import codecs
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

_REQUIRED_FIELDS: dict[str, set[str]] = {
    "blob": {"type", "schema_version", "content_oid", "size_bytes"},
    "artifact": {
        "type",
        "schema_version",
        "label",
        "artifact_type",
        "artifact_id",
        "content_oid",
        "input_ids",
        "metadata",
    },
    "projection": {
        "type",
        "schema_version",
        "name",
        "projection_type",
        "input_oids",
        "build_state_oid",
        "adapter",
        "release_mode",
    },
    "manifest": {"type", "schema_version", "pipeline_name", "pipeline_fingerprint", "artifacts", "projections"},
    "snapshot": {
        "type",
        "schema_version",
        "manifest_oid",
        "parent_snapshot_oids",
        "created_at",
        "pipeline_name",
        "run_id",
    },
    "release_receipt": {
        "type",
        "schema_version",
        "resolved_snapshot_oid",
        "manifest_oid",
        "projection_oid",
        "adapter",
        "release_mode",
        "target",
        "created_at",
    },
}

_FIELD_TYPES: dict[str, dict[str, type | tuple[type, ...]]] = {
    "blob": {
        "content_oid": str,
        "size_bytes": int,
    },
    "artifact": {
        "label": str,
        "artifact_type": str,
        "artifact_id": str,
        "content_oid": str,
        "input_ids": list,
        "metadata": dict,
    },
    "projection": {
        "name": str,
        "projection_type": str,
        "input_oids": list,
        "build_state_oid": str,
        "adapter": str,
        "release_mode": str,
    },
    "manifest": {
        "pipeline_name": str,
        "pipeline_fingerprint": str,
        "artifacts": dict,
        "projections": dict,
    },
    "snapshot": {
        "manifest_oid": str,
        "parent_snapshot_oids": list,
        "created_at": str,
        "pipeline_name": str,
        "run_id": str,
    },
}


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        os.replace(tmp, str(path))
    except BaseException:
        os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _validate_object_payload(payload: dict[str, Any]) -> None:
    object_type = payload.get("type")
    if not isinstance(object_type, str) or not object_type:
        msg = "object payload must include a non-empty string 'type'"
        raise ValueError(msg)

    schema_version = payload.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        msg = f"object payload must include schema_version={SCHEMA_VERSION}"
        raise ValueError(msg)

    required = _REQUIRED_FIELDS.get(object_type)
    if required is None:
        msg = f"unsupported object type: {object_type!r}"
        raise ValueError(msg)

    missing = sorted(required.difference(payload))
    if missing:
        msg = f"object payload for type {object_type!r} is missing required fields: {', '.join(missing)}"
        raise ValueError(msg)

    for field_name, expected_type in _FIELD_TYPES.get(object_type, {}).items():
        value = payload.get(field_name)
        if not isinstance(value, expected_type):
            msg = (
                f"object payload for type {object_type!r} field {field_name!r} "
                f"must be of type {expected_type}, got {type(value)}"
            )
            raise ValueError(msg)


class ObjectStore:
    """Content-addressed storage rooted at .synix/objects."""

    def __init__(self, synix_dir: str | Path):
        self.synix_dir = Path(synix_dir)
        self.objects_dir = self.synix_dir / "objects"
        self.objects_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_oid(self, oid: str) -> Path:
        prefix = oid[:2]
        rest = oid[2:]
        return self.objects_dir / prefix / rest

    def put_bytes(self, data: bytes) -> str:
        """Store raw bytes and return the content-addressed oid."""
        oid = hashlib.sha256(data).hexdigest()
        path = self._path_for_oid(oid)
        if path.exists():
            return oid
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(path, data)
        return oid

    def put_file(self, path: str | Path) -> tuple[str, int]:
        """Store a file's bytes by content hash without reading the whole file into memory."""
        source_path = Path(path)
        digest = hashlib.sha256()
        size_bytes = 0
        with source_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size_bytes += len(chunk)
                digest.update(chunk)

        oid = digest.hexdigest()
        target_path = self._path_for_oid(oid)
        if target_path.exists():
            return oid, size_bytes

        target_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as tmp_handle, source_path.open("rb") as source_handle:
                shutil.copyfileobj(source_handle, tmp_handle, length=1024 * 1024)
                tmp_handle.flush()
                os.fsync(tmp_handle.fileno())
            os.replace(tmp, str(target_path))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

        return oid, size_bytes

    def put_text(self, text: str, *, encoding: str = "utf-8") -> tuple[str, int]:
        """Store text as encoded bytes without creating a second full-sized bytes copy."""
        digest = hashlib.sha256()
        size_bytes = 0
        encoder = codecs.getincrementalencoder(encoding)()

        fd, tmp = tempfile.mkstemp(dir=self.objects_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                for start in range(0, len(text), 64 * 1024):
                    chunk = text[start : start + 64 * 1024]
                    encoded = encoder.encode(chunk)
                    if not encoded:
                        continue
                    size_bytes += len(encoded)
                    digest.update(encoded)
                    handle.write(encoded)

                final_bytes = encoder.encode("", final=True)
                if final_bytes:
                    size_bytes += len(final_bytes)
                    digest.update(final_bytes)
                    handle.write(final_bytes)

                handle.flush()
                os.fsync(handle.fileno())

            oid = digest.hexdigest()
            path = self._path_for_oid(oid)
            if path.exists():
                os.unlink(tmp)
                return oid, size_bytes

            path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp, str(path))
            return oid, size_bytes
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def get_bytes(self, oid: str) -> bytes:
        """Load raw bytes by oid."""
        return self._path_for_oid(oid).read_bytes()

    def put_json(self, payload: dict[str, Any]) -> str:
        """Store canonical JSON and return the content-addressed oid."""
        _validate_object_payload(payload)
        encoded = _canonical_json_bytes(payload)
        oid = hashlib.sha256(encoded).hexdigest()
        path = self._path_for_oid(oid)
        if path.exists():
            return oid
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_bytes(path, encoded)
        return oid

    def get_json(self, oid: str) -> dict[str, Any]:
        """Load a structured JSON object by oid."""
        payload = json.loads(self._path_for_oid(oid).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            msg = f"object {oid} is not a JSON object"
            raise ValueError(msg)
        _validate_object_payload(payload)
        return payload
