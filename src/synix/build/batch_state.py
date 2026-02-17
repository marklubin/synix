"""Batch build state — persistence for async OpenAI Batch API builds.

Manages build instances and their lifecycle:
  - Each batch-build creates a named build instance with its own ID
  - State tracks pending requests, batch submissions, results, and errors
  - All writes use atomic_write() for crash safety
  - Corrupted state files are quarantined (not silently dropped)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from synix.core.errors import atomic_write

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Build instance metadata
# ---------------------------------------------------------------------------


@dataclass
class BuildInstance:
    """Metadata for a single batch-build invocation."""

    build_id: str
    pipeline_hash: str
    status: str = "pending"  # pending, collecting, submitted, completed, completed_with_errors, failed
    created_at: float = field(default_factory=time.time)
    layers_completed: list[str] = field(default_factory=list)
    current_layer: str | None = None
    failed_requests: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> BuildInstance:
        return cls(
            build_id=data["build_id"],
            pipeline_hash=data["pipeline_hash"],
            status=data.get("status", "pending"),
            created_at=data.get("created_at", 0.0),
            layers_completed=data.get("layers_completed", []),
            current_layer=data.get("current_layer"),
            failed_requests=data.get("failed_requests", 0),
            error=data.get("error"),
        )


# ---------------------------------------------------------------------------
# Batch state (requests, batches, results)
# ---------------------------------------------------------------------------


class BatchState:
    """Persistent state for a batch-build instance.

    Directory layout:
        <build_dir>/builds/<build_id>/
            manifest.json    — BuildInstance metadata
            batch_state.json — requests, batches, results, errors

    State structure:
        pending:   {request_key: {layer, body, desc}}
        batch_map: {request_key: batch_id}
        batches:   {batch_id: {layer, keys, status}}
        results:   {request_key: {content, model, tokens}}
        errors:    {request_key: {code, message}}
    """

    def __init__(self, build_dir: Path, build_id: str):
        self.build_dir = Path(build_dir)
        self.build_id = build_id
        self._instance_dir = self.build_dir / "builds" / build_id
        self._manifest_path = self._instance_dir / "manifest.json"
        self._state_path = self._instance_dir / "batch_state.json"

        # In-memory state
        self._pending: dict[str, dict] = {}
        self._batch_map: dict[str, str] = {}
        self._batches: dict[str, dict] = {}
        self._results: dict[str, dict] = {}
        self._errors: dict[str, dict] = {}

        self._load_state()

    def _load_state(self) -> None:
        """Load batch state from disk. Corrupted JSON -> quarantine + raise."""
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            quarantine = self._state_path.with_suffix(f".corrupt.{int(time.time())}")
            try:
                self._state_path.rename(quarantine)
            except OSError:
                pass
            raise RuntimeError(
                f"Corrupted batch state at {self._state_path} "
                f"(quarantined to {quarantine.name}). "
                f"Use --reset-state to restart the current layer."
            ) from exc
        self._pending = data.get("pending", {})
        self._batch_map = data.get("batch_map", {})
        self._batches = data.get("batches", {})
        self._results = data.get("results", {})
        self._errors = data.get("errors", {})

    def save(self) -> None:
        """Persist batch state to disk atomically."""
        self._instance_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "pending": self._pending,
            "batch_map": self._batch_map,
            "batches": self._batches,
            "results": self._results,
            "errors": self._errors,
        }
        atomic_write(self._state_path, json.dumps(data, indent=2))

    # -- Manifest (BuildInstance) ------------------------------------------

    def save_manifest(self, instance: BuildInstance) -> None:
        """Persist build instance manifest."""
        self._instance_dir.mkdir(parents=True, exist_ok=True)
        atomic_write(self._manifest_path, json.dumps(instance.to_dict(), indent=2))

    def load_manifest(self) -> BuildInstance | None:
        """Load build instance manifest. Returns None if not found. Raises on corruption."""
        if not self._manifest_path.exists():
            return None
        try:
            data = json.loads(self._manifest_path.read_text())
            return BuildInstance.from_dict(data)
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            quarantine = self._manifest_path.with_suffix(f".corrupt.{int(time.time())}")
            try:
                self._manifest_path.rename(quarantine)
            except OSError:
                pass
            raise RuntimeError(
                f"Corrupted manifest at {self._manifest_path} (quarantined to {quarantine.name})"
            ) from exc

    # -- Request queue -----------------------------------------------------

    def queue_request(self, key: str, layer: str, body: dict, desc: str) -> None:
        """Queue a new LLM request for batch submission."""
        self._pending[key] = {"layer": layer, "body": body, "desc": desc}

    def get_pending(self, layer: str | None = None) -> dict[str, dict]:
        """Get pending requests, optionally filtered by layer."""
        if layer is None:
            return dict(self._pending)
        return {k: v for k, v in self._pending.items() if v.get("layer") == layer}

    def has_pending(self) -> bool:
        """Check if there are any pending requests."""
        return bool(self._pending)

    # -- Batch tracking ----------------------------------------------------

    def record_batch(self, batch_id: str, layer: str, keys: list[str], status: str = "submitted") -> None:
        """Record a submitted batch and map request keys to it."""
        self._batches[batch_id] = {"layer": layer, "keys": keys, "status": status}
        for key in keys:
            self._batch_map[key] = batch_id
            self._pending.pop(key, None)

    def get_batch_for_request(self, key: str) -> str | None:
        """Get the batch ID for a request key, if submitted."""
        return self._batch_map.get(key)

    def get_batch(self, batch_id: str) -> dict | None:
        """Get batch metadata by ID."""
        return self._batches.get(batch_id)

    def get_batches(self, layer: str | None = None) -> dict[str, dict]:
        """Get all batches, optionally filtered by layer."""
        if layer is None:
            return dict(self._batches)
        return {bid: b for bid, b in self._batches.items() if b.get("layer") == layer}

    def update_batch_status(self, batch_id: str, status: str) -> None:
        """Update a batch's status."""
        if batch_id in self._batches:
            self._batches[batch_id]["status"] = status

    # -- Results -----------------------------------------------------------

    def store_result(self, key: str, content: str, model: str, tokens: dict) -> None:
        """Store a completed result for a request key."""
        self._results[key] = {"content": content, "model": model, "tokens": tokens}

    def get_result(self, key: str) -> dict | None:
        """Get a stored result for a request key."""
        return self._results.get(key)

    def has_result(self, key: str) -> bool:
        """Check if a result exists for a request key."""
        return key in self._results

    # -- Errors ------------------------------------------------------------

    def store_error(self, key: str, code: str, message: str) -> None:
        """Store an error for a failed request."""
        self._errors[key] = {"code": code, "message": message}

    def get_error(self, key: str) -> dict | None:
        """Get a stored error for a request key."""
        return self._errors.get(key)

    def get_errors(self) -> dict[str, dict]:
        """Get all stored errors."""
        return dict(self._errors)

    # -- Static helpers ----------------------------------------------------

    @staticmethod
    def list_builds(build_dir: Path) -> list[BuildInstance]:
        """List all build instances in a build directory."""
        builds_dir = Path(build_dir) / "builds"
        if not builds_dir.exists():
            return []

        instances: list[BuildInstance] = []
        for entry in sorted(builds_dir.iterdir()):
            if not entry.is_dir():
                continue
            manifest_path = entry / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                data = json.loads(manifest_path.read_text())
                instances.append(BuildInstance.from_dict(data))
            except (json.JSONDecodeError, OSError, KeyError):
                logger.warning("Skipping corrupted build manifest: %s", manifest_path)
        return instances

    @staticmethod
    def compute_pipeline_hash(pipeline) -> str:
        """Compute a hash of the pipeline definition for change detection."""
        parts = [pipeline.name, pipeline.source_dir, pipeline.build_dir]
        for layer in pipeline.layers:
            parts.append(f"{layer.name}:{type(layer).__qualname__}")
            for dep in layer.depends_on:
                parts.append(f"  dep:{dep.name}")
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
