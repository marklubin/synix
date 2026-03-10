"""Artifact diffing — compare artifacts between runs or versions."""

from __future__ import annotations

import difflib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from synix.build.refs import RefStore, synix_dir_for_build_dir
from synix.build.snapshot_view import SnapshotArtifactCache, SnapshotView
from synix.core.models import Artifact

logger = logging.getLogger(__name__)


@dataclass
class ArtifactDiff:
    """Diff result for a single artifact."""

    label: str
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
    content_diff = "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=f"a/{old.label}",
            tofile=f"b/{new.label}",
            lineterm="",
        )
    )

    # Metadata diff
    metadata_diff = {}
    all_keys = set(old.metadata.keys()) | set(new.metadata.keys())
    for key in all_keys:
        old_val = old.metadata.get(key)
        new_val = new.metadata.get(key)
        if old_val != new_val:
            metadata_diff[key] = {"old": old_val, "new": new_val}

    has_changes = bool(
        content_diff or metadata_diff or old.artifact_id != new.artifact_id or old.prompt_id != new.prompt_id
    )

    return ArtifactDiff(
        label=old.label,
        has_changes=has_changes,
        content_diff=content_diff,
        metadata_diff=metadata_diff,
        old_hash=old.artifact_id,
        new_hash=new.artifact_id,
        old_prompt_id=old.prompt_id,
        new_prompt_id=new.prompt_id,
    )


def diff_builds(old_build_dir: str | Path, new_build_dir: str | Path, layer: str | None = None) -> DiffResult:
    """Compare artifacts between two build directories."""
    old_synix_dir = synix_dir_for_build_dir(Path(old_build_dir))
    new_synix_dir = synix_dir_for_build_dir(Path(new_build_dir))
    old_store = SnapshotArtifactCache(old_synix_dir)
    new_store = SnapshotArtifactCache(new_synix_dir)

    old_entries = old_store.iter_entries()
    new_entries = new_store.iter_entries()
    old_ids = set(old_entries.keys())
    new_ids = set(new_entries.keys())

    # Filter by layer if specified
    if layer:
        old_ids = {aid for aid in old_ids if old_entries[aid].get("layer") == layer}
        new_ids = {aid for aid in new_ids if new_entries[aid].get("layer") == layer}

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


def diff_artifact_by_label(
    build_dir: str | Path, label: str, previous_build_dir: str | Path | None = None
) -> ArtifactDiff | None:
    """Diff a specific artifact against its previous version.

    If previous_build_dir is provided, compares across builds.
    Otherwise, checks for version history in the same build directory.
    """
    synix_dir = synix_dir_for_build_dir(Path(build_dir))
    store = SnapshotArtifactCache(synix_dir)
    new_art = store.load_artifact(label)
    if new_art is None:
        return None

    if previous_build_dir:
        old_synix_dir = synix_dir_for_build_dir(Path(previous_build_dir))
        old_store = SnapshotArtifactCache(old_synix_dir)
        old_art = old_store.load_artifact(label)
        if old_art is None:
            return None
        return diff_artifact(old_art, new_art)

    # No previous build dir — try snapshot-era ref lookup.
    # Find the previous run ref whose OID differs from HEAD.
    try:
        ref_store = RefStore(synix_dir)
        head_oid = ref_store.read_ref("HEAD")
        if head_oid is not None:
            run_refs = ref_store.iter_refs("refs/runs")
            # Find runs whose OID differs from HEAD (i.e. previous runs)
            prev_ref: str | None = None
            # iter_refs returns sorted ascending; walk in reverse to find
            # the most recent run that is NOT the current HEAD.
            for ref_name, oid in reversed(run_refs):
                if oid != head_oid:
                    prev_ref = ref_name
                    break

            if prev_ref is not None:
                prev_view = SnapshotView.open(synix_dir, ref=prev_ref)
                try:
                    prev_data = prev_view.get_artifact(label)
                except KeyError:
                    prev_data = None

                if prev_data is not None:
                    from datetime import datetime

                    created_at_str = prev_data.get("metadata", {}).get("created_at")
                    created_at = (
                        datetime.fromisoformat(created_at_str) if created_at_str else datetime.now()
                    )
                    old_art = Artifact(
                        label=prev_data["label"],
                        artifact_type=prev_data["artifact_type"],
                        artifact_id=prev_data["artifact_id"],
                        input_ids=prev_data.get("input_ids", []),
                        prompt_id=prev_data.get("prompt_id"),
                        model_config=prev_data.get("model_config"),
                        created_at=created_at,
                        content=prev_data["content"],
                        metadata=prev_data.get("metadata", {}),
                    )
                    return diff_artifact(old_art, new_art)
    except (ValueError, FileNotFoundError, KeyError):
        logger.debug("Snapshot-era ref lookup failed for diff of %r", label, exc_info=True)

    # Legacy fallback: check for build/versions/<label> directories
    versions_dir = Path(build_dir) / "versions" / label
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
        label=data["label"],
        artifact_type=data["artifact_type"],
        artifact_id=data["artifact_id"],
        input_ids=data.get("input_ids", []),
        prompt_id=data.get("prompt_id"),
        model_config=data.get("model_config"),
        created_at=datetime.fromisoformat(data["created_at"]),
        content=data["content"],
        metadata=data.get("metadata", {}),
    )
    return diff_artifact(old_art, new_art)
