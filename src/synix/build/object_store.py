"""Immutable object storage for snapshots and related metadata."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from synix.core.errors import atomic_write


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        os.write(fd, data)
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

    def get_bytes(self, oid: str) -> bytes:
        """Load raw bytes by oid."""
        return self._path_for_oid(oid).read_bytes()

    def put_json(self, payload: dict[str, Any]) -> str:
        """Store canonical JSON and return the content-addressed oid."""
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        oid = hashlib.sha256(encoded).hexdigest()
        path = self._path_for_oid(oid)
        if path.exists():
            return oid
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, encoded.decode("utf-8"))
        return oid

    def get_json(self, oid: str) -> dict[str, Any]:
        """Load a structured JSON object by oid."""
        return json.loads(self._path_for_oid(oid).read_text())
