"""Git-like refs and HEAD management for Synix snapshots."""

from __future__ import annotations

import re
from pathlib import Path

from synix.core.errors import atomic_write

HEAD_FILENAME = "HEAD"
DEFAULT_HEAD_REF = "refs/heads/main"
MAX_REF_DEPTH = 16

_REF_RE = re.compile(r"^refs(?:/[A-Za-z0-9._-]+)+$")
_OID_RE = re.compile(r"^[0-9a-f]{64}$")


def synix_dir_for_build_dir(build_dir: str | Path, *, configured_synix_dir: str | Path | None = None) -> Path:
    """Resolve the canonical .synix directory for a build.

    Write path precedence:
    - explicit configured_synix_dir
    - nested build-local store (`build/.synix`)

    Read-side helpers can still discover the legacy sibling layout by checking
    whether `build_dir.parent/.synix` already exists.
    """
    if configured_synix_dir is not None:
        return Path(configured_synix_dir).resolve()

    build_path = Path(build_dir).resolve()
    nested = build_path / ".synix"
    legacy = build_path.parent / ".synix"
    if nested.exists():
        return nested
    if legacy.exists():
        return legacy
    return nested


def _validate_ref_name(ref_name: str) -> None:
    if ref_name == "HEAD":
        return
    if not _REF_RE.fullmatch(ref_name):
        msg = f"invalid ref name: {ref_name!r}"
        raise ValueError(msg)


def _validate_oid(oid: str) -> None:
    if not _OID_RE.fullmatch(oid):
        msg = f"invalid oid: {oid!r}"
        raise ValueError(msg)


class RefStore:
    """Manage refs and HEAD under .synix."""

    def __init__(self, synix_dir: str | Path):
        self.synix_dir = Path(synix_dir)
        self.refs_dir = self.synix_dir / "refs"
        self.head_path = self.synix_dir / HEAD_FILENAME
        self.refs_dir.mkdir(parents=True, exist_ok=True)

    def ensure_head(self, default_ref: str = DEFAULT_HEAD_REF) -> str:
        """Create HEAD if needed and return its target ref."""
        _validate_ref_name(default_ref)
        if not self.head_path.exists():
            self.head_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(self.head_path, f"ref: {default_ref}\n")
        return self.read_head_target()

    def read_head_target(self) -> str:
        """Return the ref target that HEAD points to."""
        raw = self.head_path.read_text(encoding="utf-8").strip()
        if not raw.startswith("ref: "):
            msg = f"HEAD has invalid contents: {raw!r}"
            raise ValueError(msg)
        target = raw[5:]
        _validate_ref_name(target)
        return target

    def write_head(self, target_ref: str) -> None:
        """Update HEAD to a symbolic ref target."""
        _validate_ref_name(target_ref)
        atomic_write(self.head_path, f"ref: {target_ref}\n")

    def _ref_path(self, ref_name: str) -> Path:
        if ref_name == "HEAD":
            return self.head_path
        _validate_ref_name(ref_name)
        return self.synix_dir / ref_name

    def write_ref(self, ref_name: str, oid: str) -> None:
        """Update a direct ref to an oid."""
        _validate_oid(oid)
        path = self._ref_path(ref_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, f"{oid}\n")

    def read_ref(self, ref_name: str) -> str | None:
        """Read a direct or symbolic ref and return the resolved oid."""
        return self._read_ref(ref_name, seen=[])

    def _read_ref(self, ref_name: str, *, seen: list[str]) -> str | None:
        if len(seen) >= MAX_REF_DEPTH:
            msg = f"ref resolution exceeded max depth: {' -> '.join(seen + [ref_name])}"
            raise ValueError(msg)
        if ref_name in seen:
            msg = f"ref cycle detected: {' -> '.join(seen + [ref_name])}"
            raise ValueError(msg)

        if ref_name == "HEAD":
            if not self.head_path.exists():
                return None
            return self._read_ref(self.read_head_target(), seen=seen + [ref_name])

        path = self._ref_path(ref_name)
        if not path.exists():
            return None

        value = path.read_text(encoding="utf-8").strip()
        if value.startswith("ref: "):
            target = value[5:]
            _validate_ref_name(target)
            return self._read_ref(target, seen=seen + [ref_name])

        _validate_oid(value)
        return value or None

    def iter_refs(self, prefix: str) -> list[tuple[str, str]]:
        """List refs under a prefix along with their resolved oid values."""
        _validate_ref_name(prefix)
        root = self.synix_dir / prefix
        if not root.exists():
            return []
        refs: list[tuple[str, str]] = []
        for path in sorted(root.rglob("*")):
            if path.is_dir():
                continue
            rel = path.relative_to(self.synix_dir).as_posix()
            resolved = self.read_ref(rel)
            if resolved is not None:
                refs.append((rel, resolved))
        return refs
