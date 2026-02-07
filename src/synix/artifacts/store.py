"""Artifact storage — save/load/query artifacts (filesystem-backed)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from synix import Artifact


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
        self._manifest_path.write_text(json.dumps(self._manifest, indent=2))

    def save_artifact(self, artifact: Artifact, layer_name: str, layer_level: int) -> None:
        """Save an artifact to the build directory."""
        # Ensure content hash is computed
        if not artifact.content_hash and artifact.content:
            artifact.content_hash = f"sha256:{hashlib.sha256(artifact.content.encode()).hexdigest()}"

        # Create layer directory
        layer_dir = self.build_dir / f"layer{layer_level}-{layer_name}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        # Serialize artifact to JSON
        artifact_path = layer_dir / f"{artifact.artifact_id}.json"
        artifact_data = {
            "artifact_id": artifact.artifact_id,
            "artifact_type": artifact.artifact_type,
            "content_hash": artifact.content_hash,
            "input_hashes": artifact.input_hashes,
            "prompt_id": artifact.prompt_id,
            "model_config": artifact.model_config,
            "created_at": artifact.created_at.isoformat(),
            "content": artifact.content,
            "metadata": artifact.metadata,
        }
        artifact_path.write_text(json.dumps(artifact_data, indent=2))

        # Update manifest
        rel_path = f"layer{layer_level}-{layer_name}/{artifact.artifact_id}.json"
        self._manifest[artifact.artifact_id] = {
            "path": rel_path,
            "content_hash": artifact.content_hash,
            "layer": layer_name,
            "level": layer_level,
        }
        self._save_manifest()

    def load_artifact(self, artifact_id: str) -> Artifact | None:
        """Load an artifact by ID. Returns None if not found."""
        entry = self._manifest.get(artifact_id)
        if entry is None:
            return None

        artifact_path = self.build_dir / entry["path"]
        if not artifact_path.exists():
            return None

        data = json.loads(artifact_path.read_text())
        return Artifact(
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

    def list_artifacts(self, layer: str) -> list[Artifact]:
        """Return all artifacts for a given layer name."""
        artifacts = []
        for artifact_id, entry in self._manifest.items():
            if entry["layer"] == layer:
                artifact = self.load_artifact(artifact_id)
                if artifact is not None:
                    artifacts.append(artifact)
        return artifacts

    def get_content_hash(self, artifact_id: str) -> str | None:
        """Quick hash lookup from manifest without loading full artifact."""
        entry = self._manifest.get(artifact_id)
        if entry is None:
            return None
        return entry["content_hash"]
