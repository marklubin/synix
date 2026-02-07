"""Provenance tracking — record and query lineage chains."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from synix import ProvenanceRecord


class ProvenanceTracker:
    """Tracks artifact lineage backed by provenance.json."""

    def __init__(self, build_dir: str | Path):
        self.build_dir = Path(build_dir)
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self._provenance_path = self.build_dir / "provenance.json"
        self._records: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        if self._provenance_path.exists():
            return json.loads(self._provenance_path.read_text())
        return {}

    def _save(self) -> None:
        self._provenance_path.write_text(json.dumps(self._records, indent=2))

    def record(
        self,
        artifact_id: str,
        parent_ids: list[str],
        prompt_id: str | None = None,
        model_config: dict | None = None,
    ) -> None:
        """Record provenance for an artifact."""
        self._records[artifact_id] = {
            "artifact_id": artifact_id,
            "parent_artifact_ids": parent_ids,
            "prompt_id": prompt_id,
            "model_config": model_config,
            "created_at": datetime.now().isoformat(),
        }
        self._save()

    def get_parents(self, artifact_id: str) -> list[str]:
        """Return parent artifact IDs for this artifact."""
        rec = self._records.get(artifact_id)
        if rec is None:
            return []
        return rec["parent_artifact_ids"]

    def get_record(self, artifact_id: str) -> ProvenanceRecord | None:
        """Return the ProvenanceRecord for an artifact, or None."""
        rec = self._records.get(artifact_id)
        if rec is None:
            return None
        return ProvenanceRecord(
            artifact_id=rec["artifact_id"],
            parent_artifact_ids=rec["parent_artifact_ids"],
            prompt_id=rec.get("prompt_id"),
            model_config=rec.get("model_config"),
            created_at=datetime.fromisoformat(rec["created_at"]),
        )

    def get_chain(self, artifact_id: str) -> list[ProvenanceRecord]:
        """Recursive walk to roots — return full provenance chain (BFS)."""
        chain: list[ProvenanceRecord] = []
        visited: set[str] = set()
        queue: list[str] = [artifact_id]

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            record = self.get_record(current_id)
            if record is not None:
                chain.append(record)
                for parent_id in record.parent_artifact_ids:
                    if parent_id not in visited:
                        queue.append(parent_id)

        return chain
