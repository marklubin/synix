"""Artifact diffing — compare artifacts between runs or versions."""

from __future__ import annotations

import difflib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from synix.build.refs import RefStore, synix_dir_for_build_dir
from synix.build.snapshot_view import SnapshotView
from synix.core.models import Artifact

logger = logging.getLogger(__name__)


def _artifact_from_view_data(data: dict) -> Artifact:
    """Build an Artifact from a SnapshotView artifact dict (with content loaded)."""
    created_at_str = data.get("metadata", {}).get("created_at")
    kwargs: dict = {
        "label": data["label"],
        "artifact_type": data["artifact_type"],
        "artifact_id": data["artifact_id"],
        "input_ids": data.get("input_ids", []),
        "prompt_id": data.get("prompt_id"),
        "agent_fingerprint": data.get("agent_fingerprint"),
        "model_config": data.get("model_config"),
        "content": data["content"],
        "metadata": data.get("metadata", {}),
    }
    if created_at_str:
        kwargs["created_at"] = datetime.fromisoformat(created_at_str)
    # If created_at is missing, let Artifact's default_factory handle it.
    return Artifact(**kwargs)


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
    old_agent_fingerprint: str | None = None
    new_agent_fingerprint: str | None = None


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
        content_diff
        or metadata_diff
        or old.artifact_id != new.artifact_id
        or old.prompt_id != new.prompt_id
        or old.agent_fingerprint != new.agent_fingerprint
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
        old_agent_fingerprint=old.agent_fingerprint,
        new_agent_fingerprint=new.agent_fingerprint,
    )


def diff_builds(old_build_dir: str | Path, new_build_dir: str | Path, layer: str | None = None) -> DiffResult:
    """Compare artifacts between two build directories.

    Uses SnapshotView directly (raises on broken refs) instead of
    SnapshotArtifactCache (which silently falls back to empty on errors).
    """
    old_synix_dir = synix_dir_for_build_dir(Path(old_build_dir))
    new_synix_dir = synix_dir_for_build_dir(Path(new_build_dir))

    old_view = SnapshotView.open(old_synix_dir)
    new_view = SnapshotView.open(new_synix_dir)

    # Build label→artifact_obj lookup from each view
    old_artifacts = {a["label"]: a for a in old_view.list_artifacts()}
    new_artifacts = {a["label"]: a for a in new_view.list_artifacts()}

    old_labels = set(old_artifacts.keys())
    new_labels = set(new_artifacts.keys())

    # Filter by layer if specified
    if layer:
        old_labels = {lbl for lbl in old_labels if old_artifacts[lbl].get("metadata", {}).get("layer_name") == layer}
        new_labels = {lbl for lbl in new_labels if new_artifacts[lbl].get("metadata", {}).get("layer_name") == layer}

    result = DiffResult()
    result.added = sorted(new_labels - old_labels)
    result.removed = sorted(old_labels - new_labels)

    # Diff common artifacts — load full content via get_artifact
    for lbl in sorted(old_labels & new_labels):
        old_data = old_view.get_artifact(lbl)
        new_data = new_view.get_artifact(lbl)
        old_art = _artifact_from_view_data(old_data)
        new_art = _artifact_from_view_data(new_data)
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
    view = SnapshotView.open(synix_dir)
    try:
        new_data = view.get_artifact(label)
    except KeyError:
        return None
    new_art = _artifact_from_view_data(new_data)

    if previous_build_dir:
        old_synix_dir = synix_dir_for_build_dir(Path(previous_build_dir))
        old_view = SnapshotView.open(old_synix_dir)
        try:
            old_data = old_view.get_artifact(label)
        except KeyError:
            return None
        old_art = _artifact_from_view_data(old_data)
        return diff_artifact(old_art, new_art)

    # No previous build dir — try snapshot-era ref lookup.
    # Find the previous run ref whose OID differs from HEAD.
    # Note: iter_refs returns refs sorted lexicographically ascending by name
    # (e.g., refs/runs/001, refs/runs/002). We walk in reverse to find the
    # most recent run that is NOT the current HEAD snapshot.
    try:
        ref_store = RefStore(synix_dir)
        head_oid = ref_store.read_ref("HEAD")
        if head_oid is not None:
            run_refs = ref_store.iter_refs("refs/runs")
            prev_ref: str | None = None
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
                    old_art = _artifact_from_view_data(prev_data)
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
    old_art = Artifact(
        label=data["label"],
        artifact_type=data["artifact_type"],
        artifact_id=data["artifact_id"],
        input_ids=data.get("input_ids", []),
        prompt_id=data.get("prompt_id"),
        agent_fingerprint=data.get("agent_fingerprint"),
        model_config=data.get("model_config"),
        created_at=datetime.fromisoformat(data["created_at"]),
        content=data["content"],
        metadata=data.get("metadata", {}),
    )
    return diff_artifact(old_art, new_art)
