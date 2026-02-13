"""Semantic fingerprinting — self-describing, versioned hashes for cache invalidation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Fingerprint:
    """A self-describing, versioned hash for any entity in the build system.

    Each fingerprint records its scheme (how it was generated) and its
    components (what went into it), enabling explain-cache and future
    extensibility.
    """

    scheme: str  # e.g. "synix:transform:v1", "synix:build:v1"
    digest: str  # SHA256 hex (full)
    components: dict[str, str]  # component_name -> component_hash

    def matches(self, other: Fingerprint | None) -> bool:
        """Match requires same scheme AND same digest."""
        if other is None:
            return False
        return self.scheme == other.scheme and self.digest == other.digest

    def explain_diff(self, other: Fingerprint | None) -> list[str]:
        """Human-readable list of reasons these fingerprints differ."""
        if other is None:
            return ["no stored fingerprint"]
        if self.scheme != other.scheme:
            return [f"scheme changed ({other.scheme} -> {self.scheme})"]
        changed = []
        all_keys = sorted(set(self.components) | set(other.components))
        for k in all_keys:
            if self.components.get(k) != other.components.get(k):
                changed.append(k)
        return [f"{k} changed" for k in changed] or ["unknown"]

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON storage."""
        return {
            "scheme": self.scheme,
            "digest": self.digest,
            "components": dict(self.components),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Fingerprint | None:
        """Deserialize from a dict. Returns None if data is empty/missing."""
        if not data or "scheme" not in data:
            return None
        return cls(
            scheme=data["scheme"],
            digest=data["digest"],
            components=data.get("components", {}),
        )


def compute_digest(components: dict[str, str]) -> str:
    """Deterministic digest from sorted component hashes."""
    parts = "|".join(f"{k}={v}" for k, v in sorted(components.items()))
    return hashlib.sha256(parts.encode()).hexdigest()


def fingerprint_value(obj) -> str:
    """Deterministic SHA256 prefix for any common Python value.

    Python's built-in hash() is NOT suitable here because:
    - PYTHONHASHSEED randomizes hash() across sessions (not persistable)
    - Default __hash__ is identity-based (by-ref, not by-value)
    - Dicts and lists are unhashable
    - Produces platform-dependent int, not a portable digest

    This function serializes to a canonical string form, then SHA256s it.
    """
    if obj is None:
        raw = ""
    elif isinstance(obj, str):
        raw = obj.replace("\r\n", "\n").rstrip()
    elif isinstance(obj, dict):
        raw = json.dumps(obj, sort_keys=True, default=str)
    elif isinstance(obj, (list, tuple)):
        raw = "|".join(sorted(str(x) for x in obj))
    elif isinstance(obj, (int, float, bool)):
        raw = str(obj)
    else:
        raw = str(obj)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def compute_build_fingerprint(
    transform_fingerprint: Fingerprint,
    input_hashes: list[str],
) -> Fingerprint:
    """Combine transform identity with input hashes into a build fingerprint."""
    components = {
        "transform": transform_fingerprint.digest,
        "inputs": fingerprint_value(input_hashes),
    }
    return Fingerprint(
        scheme="synix:build:v1",
        digest=compute_digest(components),
        components=components,
    )


def compute_projection_fingerprint(
    artifact_hashes: list[str],
    config: dict | None = None,
) -> Fingerprint:
    """Compute a fingerprint for a projection from source artifact hashes and config."""
    components = {
        "sources": fingerprint_value(artifact_hashes),
    }
    if config:
        components["config"] = fingerprint_value(config)
    return Fingerprint(
        scheme="synix:projection:v1",
        digest=compute_digest(components),
        components=components,
    )
