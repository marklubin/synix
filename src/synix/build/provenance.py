"""Provenance tracking — record and query lineage chains."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from synix.core.errors import atomic_write
from synix.core.models import ProvenanceRecord


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
        atomic_write(self._provenance_path, json.dumps(self._records, indent=2))

    def record(
        self,
        label: str,
        parent_labels: list[str],
        prompt_id: str | None = None,
        model_config: dict | None = None,
    ) -> None:
        """Record provenance for an artifact."""
        self._records[label] = {
            "label": label,
            "parent_labels": parent_labels,
            "prompt_id": prompt_id,
            "model_config": model_config,
            "created_at": datetime.now().isoformat(),
        }
        self._save()

    def get_parents(self, label: str) -> list[str]:
        """Return parent labels for this artifact."""
        rec = self._records.get(label)
        if rec is None:
            return []
        return rec["parent_labels"]

    def get_record(self, label: str) -> ProvenanceRecord | None:
        """Return the ProvenanceRecord for an artifact, or None."""
        rec = self._records.get(label)
        if rec is None:
            return None
        return ProvenanceRecord(
            label=rec["label"],
            parent_labels=rec["parent_labels"],
            prompt_id=rec.get("prompt_id"),
            model_config=rec.get("model_config"),
            created_at=datetime.fromisoformat(rec["created_at"]),
        )

    def get_chain(self, label: str) -> list[ProvenanceRecord]:
        """Recursive walk to roots — return full provenance chain (BFS)."""
        chain: list[ProvenanceRecord] = []
        visited: set[str] = set()
        queue: list[str] = [label]

        while queue:
            current_label = queue.pop(0)
            if current_label in visited:
                continue
            visited.add(current_label)

            record = self.get_record(current_label)
            if record is not None:
                chain.append(record)
                for parent_label in record.parent_labels:
                    if parent_label not in visited:
                        queue.append(parent_label)

        return chain
