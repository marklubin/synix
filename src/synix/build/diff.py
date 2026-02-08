"""Artifact diffing — compare artifacts between runs or versions."""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path

from synix.core.models import Artifact
from synix.build.artifacts import ArtifactStore


@dataclass
class ArtifactDiff:
    """Diff result for a single artifact."""

    artifact_id: str
    has_changes: bool
    content_diff: str  # unified diff of content
    metadata_diff: dict  # changed metadata keys
    old_hash: str | None = None
    new_hash: str | None = None
    old_prompt_id: str | None = None
    new_prompt_id: str | None = None


@dataclass
class DiffResult:
    """Result of comparing artifacts between two builds or versions."""

    diffs: list[ArtifactDiff] = field(default_factory=list)
    added: list[str] = field(default_factory=list)  # artifact IDs only in new
    removed: list[str] = field(default_factory=list)  # artifact IDs only in old

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or any(d.has_changes for d in self.diffs))


def diff_artifact(old: Artifact, new: Artifact) -> ArtifactDiff:
    """Compare two versions of the same artifact."""
    # Content diff
    old_lines = old.content.splitlines(keepends=True)
    new_lines = new.content.splitlines(keepends=True)
    content_diff = "".join(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{old.artifact_id}",
        tofile=f"b/{new.artifact_id}",
        lineterm="",
    ))

    # Metadata diff
    metadata_diff = {}
    all_keys = set(old.metadata.keys()) | set(new.metadata.keys())
    for key in all_keys:
        old_val = old.metadata.get(key)
        new_val = new.metadata.get(key)
        if old_val != new_val:
            metadata_diff[key] = {"old": old_val, "new": new_val}

    has_changes = bool(content_diff or metadata_diff
                       or old.content_hash != new.content_hash
                       or old.prompt_id != new.prompt_id)

    return ArtifactDiff(
        artifact_id=old.artifact_id,
        has_changes=has_changes,
        content_diff=content_diff,
        metadata_diff=metadata_diff,
        old_hash=old.content_hash,
        new_hash=new.content_hash,
        old_prompt_id=old.prompt_id,
        new_prompt_id=new.prompt_id,
    )


def diff_builds(old_build_dir: str | Path, new_build_dir: str | Path,
                layer: str | None = None) -> DiffResult:
    """Compare artifacts between two build directories."""
    old_store = ArtifactStore(old_build_dir)
    new_store = ArtifactStore(new_build_dir)

    old_ids = set(old_store._manifest.keys())
    new_ids = set(new_store._manifest.keys())

    # Filter by layer if specified
    if layer:
        old_ids = {aid for aid in old_ids
                   if old_store._manifest[aid].get("layer") == layer}
        new_ids = {aid for aid in new_ids
                   if new_store._manifest[aid].get("layer") == layer}

    result = DiffResult()
    result.added = sorted(new_ids - old_ids)
    result.removed = sorted(old_ids - new_ids)

    # Diff common artifacts
    for aid in sorted(old_ids & new_ids):
        old_art = old_store.load_artifact(aid)
        new_art = new_store.load_artifact(aid)
        if old_art and new_art:
            d = diff_artifact(old_art, new_art)
            if d.has_changes:
                result.diffs.append(d)

    return result


def diff_artifact_by_id(build_dir: str | Path, artifact_id: str,
                        previous_build_dir: str | Path | None = None) -> ArtifactDiff | None:
    """Diff a specific artifact against its previous version.

    If previous_build_dir is provided, compares across builds.
    Otherwise, checks for version history in the same build directory.
    """
    store = ArtifactStore(build_dir)
    new_art = store.load_artifact(artifact_id)
    if new_art is None:
        return None

    if previous_build_dir:
        old_store = ArtifactStore(previous_build_dir)
        old_art = old_store.load_artifact(artifact_id)
        if old_art is None:
            return None
        return diff_artifact(old_art, new_art)

    # No previous build dir — check for version history
    versions_dir = Path(build_dir) / "versions" / artifact_id
    if not versions_dir.exists():
        return None

    # Find the most recent previous version
    version_files = sorted(versions_dir.glob("*.json"), reverse=True)
    if not version_files:
        return None

    # Load the previous version
    data = json.loads(version_files[0].read_text())
    from datetime import datetime
    old_art = Artifact(
        artifact_id=data["artifact_id"],
        artifact_type=data["artifact_type"],
        content_hash=data["content_hash"],
        input_hashes=data.get("input_hashes", []),
        prompt_id=data.get("prompt_id"),
        model_config=data.get("model_config"),
        created_at=datetime.fromisoformat(data["created_at"]),
        content=data["content"],
        metadata=data.get("metadata", {}),
    )
    return diff_artifact(old_art, new_art)
