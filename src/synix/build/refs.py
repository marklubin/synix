"""Git-like refs and HEAD management for Synix snapshots."""

from __future__ import annotations

from pathlib import Path

from synix.core.errors import atomic_write

HEAD_FILENAME = "HEAD"
DEFAULT_HEAD_REF = "refs/heads/main"


def synix_dir_for_build_dir(build_dir: str | Path) -> Path:
    """Resolve the canonical .synix directory from a build dir."""
    return Path(build_dir).resolve().parent / ".synix"


class RefStore:
    """Manage refs and HEAD under .synix."""

    def __init__(self, synix_dir: str | Path):
        self.synix_dir = Path(synix_dir)
        self.refs_dir = self.synix_dir / "refs"
        self.head_path = self.synix_dir / HEAD_FILENAME
        self.refs_dir.mkdir(parents=True, exist_ok=True)

    def ensure_head(self, default_ref: str = DEFAULT_HEAD_REF) -> str:
        """Create HEAD if needed and return its target ref."""
        if not self.head_path.exists():
            self.head_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(self.head_path, f"ref: {default_ref}\n")
        return self.read_head_target()

    def read_head_target(self) -> str:
        """Return the ref target that HEAD points to."""
        raw = self.head_path.read_text().strip()
        if not raw.startswith("ref: "):
            msg = f"HEAD has invalid contents: {raw!r}"
            raise ValueError(msg)
        return raw[5:]

    def write_head(self, target_ref: str) -> None:
        """Update HEAD to a symbolic ref target."""
        atomic_write(self.head_path, f"ref: {target_ref}\n")

    def _ref_path(self, ref_name: str) -> Path:
        if ref_name == "HEAD":
            return self.head_path
        return self.synix_dir / ref_name

    def write_ref(self, ref_name: str, oid: str) -> None:
        """Update a direct ref to an oid."""
        path = self._ref_path(ref_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, f"{oid}\n")

    def read_ref(self, ref_name: str) -> str | None:
        """Read a direct or symbolic ref and return the resolved oid."""
        if ref_name == "HEAD":
            if not self.head_path.exists():
                return None
            return self.read_ref(self.read_head_target())
        path = self._ref_path(ref_name)
        if not path.exists():
            return None
        value = path.read_text().strip()
        if value.startswith("ref: "):
            return self.read_ref(value[5:])
        return value or None

    def iter_refs(self, prefix: str) -> list[tuple[str, str]]:
        """List refs under a prefix along with their direct oid values."""
        root = self.synix_dir / prefix
        if not root.exists():
            return []
        refs: list[tuple[str, str]] = []
        for path in sorted(root.rglob("*")):
            if path.is_dir():
                continue
            rel = path.relative_to(self.synix_dir).as_posix()
            value = path.read_text().strip()
            refs.append((rel, value))
        return refs
