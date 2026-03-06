"""Immutable object storage for snapshots and related metadata."""

from __future__ import annotations

import codecs
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_OID_RE = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_FIELDS: dict[str, set[str]] = {
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
        "artifacts": list,
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

_OPTIONAL_FIELD_TYPES: dict[str, dict[str, type | tuple[type, ...]]] = {
    "artifact": {
        "prompt_id": (str, type(None)),
        "model_config": (dict, type(None)),
        "created_at": str,
        "parent_labels": list,
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


def _validate_object_payload(payload: dict[str, Any], *, allow_older_schema: bool = False) -> None:
    object_type = payload.get("type")
    if not isinstance(object_type, str) or not object_type:
        msg = "object payload must include a non-empty string 'type'"
        raise ValueError(msg)

    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, int):
        msg = "object payload must include integer schema_version"
        raise ValueError(msg)
    if allow_older_schema:
        if schema_version > SCHEMA_VERSION:
            msg = f"object payload schema_version={schema_version} is newer than supported={SCHEMA_VERSION}"
            raise ValueError(msg)
    elif schema_version != SCHEMA_VERSION:
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

    for field_name, expected_type in _OPTIONAL_FIELD_TYPES.get(object_type, {}).items():
        if field_name not in payload:
            continue
        value = payload[field_name]
        if not isinstance(value, expected_type):
            msg = (
                f"object payload for type {object_type!r} field {field_name!r} "
                f"must be of type {expected_type}, got {type(value)}"
            )
            raise ValueError(msg)

    if object_type == "manifest":
        artifacts = payload["artifacts"]
        for idx, entry in enumerate(artifacts):
            if not isinstance(entry, dict):
                msg = f"manifest artifacts[{idx}] must be an object, got {type(entry)}"
                raise ValueError(msg)
            label = entry.get("label")
            oid = entry.get("oid")
            if not isinstance(label, str) or not label:
                msg = f"manifest artifacts[{idx}] must include non-empty string label"
                raise ValueError(msg)
            if not isinstance(oid, str) or not _OID_RE.fullmatch(oid):
                msg = f"manifest artifacts[{idx}] must include valid oid"
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
        """Store text as encoded bytes in one pass.

        Text is incrementally encoded while writing to a temporary file so the
        digest and persisted bytes are computed from the same byte stream.
        """
        if not isinstance(text, str):
            msg = f"text must be a string, got {type(text).__name__}"
            raise TypeError(msg)

        digest = hashlib.sha256()
        size_bytes = 0
        fd, tmp = tempfile.mkstemp(dir=self.objects_dir, suffix=".tmp")
        try:
            encoder = codecs.getincrementalencoder(encoding)()
            with os.fdopen(fd, "wb") as handle:
                for start in range(0, len(text), 64 * 1024):
                    chunk = text[start : start + 64 * 1024]
                    encoded = encoder.encode(chunk)
                    if encoded:
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
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return oid, size_bytes

    def get_bytes(self, oid: str) -> bytes:
        """Load raw bytes by oid."""
        path = self._path_for_oid(oid)
        try:
            return path.read_bytes()
        except OSError as exc:
            msg = f"failed to read object bytes for oid {oid} at {path}: {exc}"
            raise OSError(msg) from exc

    def put_json(self, payload: dict[str, Any]) -> str:
        """Store canonical JSON and return the content-addressed oid."""
        _validate_object_payload(payload, allow_older_schema=False)
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
        path = self._path_for_oid(oid)
        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError as exc:
            msg = f"failed to read object json for oid {oid} at {path}: {exc}"
            raise OSError(msg) from exc
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            msg = f"object {oid} at {path} is not valid JSON: {exc}"
            raise ValueError(msg) from exc
        if not isinstance(payload, dict):
            msg = f"object {oid} is not a JSON object"
            raise ValueError(msg)
        _validate_object_payload(payload, allow_older_schema=True)
        return payload
