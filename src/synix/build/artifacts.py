"""Artifact storage — save/load/query artifacts (filesystem-backed)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from synix.core.errors import atomic_write
from synix.core.models import Artifact


class ArtifactStore:
    """Filesystem-backed artifact storage with manifest tracking."""

    def __init__(self, build_dir: str | Path):
        self.build_dir = Path(build_dir)
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.build_dir / "manifest.json"
        self._manifest: dict[str, dict] = self._load_manifest()

    def _load_manifest(self) -> dict[str, dict]:
        if self._manifest_path.exists():
            return json.loads(self._manifest_path.read_text())
        return {}

    def _save_manifest(self) -> None:
        atomic_write(self._manifest_path, json.dumps(self._manifest, indent=2))

    def save_artifact(self, artifact: Artifact, layer_name: str, layer_level: int) -> None:
        """Save an artifact to the build directory."""
        # Ensure artifact ID (content hash) is computed
        if not artifact.artifact_id and artifact.content:
            artifact.artifact_id = f"sha256:{hashlib.sha256(artifact.content.encode()).hexdigest()}"

        # Create layer directory
        layer_dir = self.build_dir / f"layer{layer_level}-{layer_name}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        # Serialize artifact to JSON
        artifact_path = layer_dir / f"{artifact.label}.json"
        artifact_data = {
            "label": artifact.label,
            "artifact_type": artifact.artifact_type,
            "artifact_id": artifact.artifact_id,
            "input_ids": artifact.input_ids,
            "prompt_id": artifact.prompt_id,
            "model_config": artifact.model_config,
            "created_at": artifact.created_at.isoformat(),
            "content": artifact.content,
            "metadata": artifact.metadata,
        }
        atomic_write(artifact_path, json.dumps(artifact_data, indent=2))

        # Update manifest (keyed by label)
        rel_path = f"layer{layer_level}-{layer_name}/{artifact.label}.json"
        self._manifest[artifact.label] = {
            "path": rel_path,
            "artifact_id": artifact.artifact_id,
            "layer": layer_name,
            "level": layer_level,
        }
        self._save_manifest()

    def load_artifact(self, label: str) -> Artifact | None:
        """Load an artifact by label. Returns None if not found."""
        entry = self._manifest.get(label)
        if entry is None:
            return None

        artifact_path = self.build_dir / entry["path"]
        if not artifact_path.exists():
            return None

        data = json.loads(artifact_path.read_text())
        return Artifact(
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

    def list_artifacts(self, layer: str) -> list[Artifact]:
        """Return all artifacts for a given layer name."""
        artifacts = []
        for label, entry in self._manifest.items():
            if entry["layer"] == layer:
                artifact = self.load_artifact(label)
                if artifact is not None:
                    artifacts.append(artifact)
        return artifacts

    def get_artifact_id(self, label: str) -> str | None:
        """Quick artifact ID (hash) lookup from manifest without loading full artifact."""
        entry = self._manifest.get(label)
        if entry is None:
            return None
        return entry["artifact_id"]

    def resolve_prefix(self, prefix: str) -> str | None:
        """Resolve a prefix to a full label (git-like semantics).

        Matches against labels first, then artifact IDs (hashes).
        Returns the full label on unique match, None if no match.
        Raises ValueError on ambiguous match (multiple candidates).
        """
        # Strip sha256: prefix if user pasted a full hash
        hash_prefix = prefix.removeprefix("sha256:")

        # 1. Exact match on label
        if prefix in self._manifest:
            return prefix

        # 2. Prefix match on label
        label_matches = [lbl for lbl in self._manifest if lbl.startswith(prefix)]
        if len(label_matches) == 1:
            return label_matches[0]
        if len(label_matches) > 1:
            labels = ", ".join(sorted(label_matches)[:5])
            msg = f"ambiguous prefix '{prefix}' matches {len(label_matches)} labels: {labels}"
            raise ValueError(msg)

        # 3. Prefix match on artifact ID (hash) (with or without sha256: prefix)
        hash_matches = [
            lbl
            for lbl, entry in self._manifest.items()
            if entry["artifact_id"].removeprefix("sha256:").startswith(hash_prefix)
        ]
        if len(hash_matches) == 1:
            return hash_matches[0]
        if len(hash_matches) > 1:
            labels = ", ".join(sorted(hash_matches)[:5])
            msg = f"ambiguous hash prefix '{prefix}' matches {len(hash_matches)} artifacts: {labels}"
            raise ValueError(msg)

        return None
