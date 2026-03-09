"""Release orchestration — resolve snapshot, dispatch adapters, write receipts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synix.build.adapters import AdapterReceipt, get_adapter
from synix.build.refs import RefStore
from synix.build.release import ReleaseClosure
from synix.core.errors import atomic_write

logger = logging.getLogger(__name__)

RECEIPT_SCHEMA_VERSION = 1


@dataclass
class ReleaseReceipt:
    """Full receipt for a release — covers all adapters."""

    schema_version: int = RECEIPT_SCHEMA_VERSION
    release_name: str = ""
    snapshot_oid: str = ""
    manifest_oid: str = ""
    pipeline_name: str = ""
    released_at: str = ""
    source_ref: str = ""
    adapters: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "release_name": self.release_name,
            "snapshot_oid": self.snapshot_oid,
            "manifest_oid": self.manifest_oid,
            "pipeline_name": self.pipeline_name,
            "released_at": self.released_at,
            "source_ref": self.source_ref,
            "adapters": self.adapters,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReleaseReceipt:
        return cls(
            schema_version=data.get("schema_version", RECEIPT_SCHEMA_VERSION),
            release_name=data.get("release_name", ""),
            snapshot_oid=data.get("snapshot_oid", ""),
            manifest_oid=data.get("manifest_oid", ""),
            pipeline_name=data.get("pipeline_name", ""),
            released_at=data.get("released_at", ""),
            source_ref=data.get("source_ref", ""),
            adapters=data.get("adapters", {}),
        )


def _release_dir(synix_dir: Path, release_name: str) -> Path:
    return synix_dir / "releases" / release_name


def _load_current_receipt(release_dir: Path) -> ReleaseReceipt | None:
    receipt_path = release_dir / "receipt.json"
    if not receipt_path.exists():
        return None
    data = json.loads(receipt_path.read_text(encoding="utf-8"))
    return ReleaseReceipt.from_dict(data)


def _write_pending(release_dir: Path, closure: ReleaseClosure, source_ref: str) -> Path:
    release_dir.mkdir(parents=True, exist_ok=True)
    pending_path = release_dir / ".pending.json"
    atomic_write(
        pending_path,
        json.dumps(
            {
                "snapshot_oid": closure.snapshot_oid,
                "manifest_oid": closure.manifest_oid,
                "source_ref": source_ref,
                "started_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        ),
    )
    return pending_path


def _write_receipt(release_dir: Path, receipt: ReleaseReceipt) -> Path:
    receipt_path = release_dir / "receipt.json"
    atomic_write(
        receipt_path,
        json.dumps(receipt.to_dict(), indent=2, sort_keys=True),
    )
    return receipt_path


def _append_history(release_dir: Path, receipt: ReleaseReceipt) -> None:
    history_dir = release_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    ts = receipt.released_at.replace(":", "").replace("+", "p")
    history_path = history_dir / f"{ts}.json"
    atomic_write(
        history_path,
        json.dumps(receipt.to_dict(), indent=2, sort_keys=True),
    )


def execute_release(
    synix_dir: str | Path,
    *,
    ref: str = "HEAD",
    release_name: str,
    target: str | Path | None = None,
) -> ReleaseReceipt:
    """Execute a full release: resolve ref, build closure, dispatch adapters, write receipt.

    Parameters
    ----------
    synix_dir:
        Path to the .synix directory.
    ref:
        Source ref to release (default HEAD).
    release_name:
        Name of the release target (e.g. "local", "prod").
    target:
        Optional override for the release target directory.
        Defaults to `.synix/releases/<release_name>/`.
    """
    synix_path = Path(synix_dir)
    ref_store = RefStore(synix_path)

    # 1. Resolve ref -> snapshot OID
    snapshot_oid = ref_store.read_ref(ref)
    if snapshot_oid is None:
        raise ValueError(f"ref {ref!r} does not resolve to a snapshot")

    # 2. Build closure
    closure = ReleaseClosure.from_snapshot(synix_path, snapshot_oid)

    # Warn if releasing a snapshot that had DLQ'd artifacts
    if closure.dlq_entries:
        count = len(closure.dlq_entries)
        layers = {e.get("layer_name", "unknown") for e in closure.dlq_entries}
        logger.warning(
            "Releasing snapshot with %d DLQ'd artifact(s) in layer(s): %s. "
            "Downstream artifacts were built from incomplete inputs. "
            "Inspect the manifest or build log for details.",
            count,
            ", ".join(sorted(layers)),
        )

    # 3. Resolve target directory
    release_dir = _release_dir(synix_path, release_name)
    target_path = Path(target) if target else release_dir

    # 4. Write pending transaction
    pending_path = _write_pending(release_dir, closure, ref)

    # 5. Load current receipt for delta planning
    current_receipt = _load_current_receipt(release_dir)
    current_adapter_receipts: dict[str, AdapterReceipt] = {}
    if current_receipt:
        for name, data in current_receipt.adapters.items():
            current_adapter_receipts[name] = AdapterReceipt(
                adapter=data.get("adapter", ""),
                projection_name=name,
                target=data.get("target", ""),
                artifacts_applied=data.get("artifacts_applied", 0),
                status=data.get("status", ""),
            )

    # 6. Dispatch adapters for each projection
    adapter_receipts: dict[str, dict[str, Any]] = {}
    try:
        for proj_name, declaration in closure.projections.items():
            adapter = get_adapter(declaration.adapter)
            prev = current_adapter_receipts.get(proj_name)

            plan = adapter.plan(closure, declaration, prev)
            receipt = adapter.apply(plan, target_path)

            if not adapter.verify(receipt, target_path):
                raise RuntimeError(
                    f"Adapter {declaration.adapter!r} verification failed for "
                    f"projection {proj_name!r}. Release aborted."
                )

            adapter_receipts[proj_name] = receipt.to_dict()

    except Exception:
        # Leave .pending.json for diagnosis
        logger.error("Release %r failed — .pending.json preserved at %s", release_name, pending_path)
        raise

    # 7. Build full receipt
    released_at = datetime.now(UTC).isoformat()
    full_receipt = ReleaseReceipt(
        release_name=release_name,
        snapshot_oid=closure.snapshot_oid,
        manifest_oid=closure.manifest_oid,
        pipeline_name=closure.pipeline_name,
        released_at=released_at,
        source_ref=ref,
        adapters=adapter_receipts,
    )

    # 8. Write receipt + history
    _write_receipt(release_dir, full_receipt)
    _append_history(release_dir, full_receipt)

    # 9. Advance release ref
    ref_store.write_ref(f"refs/releases/{release_name}", snapshot_oid)

    # 10. Clean up pending
    if pending_path.exists():
        pending_path.unlink()

    return full_receipt


def list_releases(synix_dir: str | Path) -> list[dict[str, Any]]:
    """List all releases with their receipt info."""
    synix_path = Path(synix_dir)
    releases_dir = synix_path / "releases"
    if not releases_dir.exists():
        return []

    results = []
    for release_path in sorted(releases_dir.iterdir()):
        if not release_path.is_dir():
            continue
        receipt_path = release_path / "receipt.json"
        if not receipt_path.exists():
            continue
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
        results.append(data)
    return results


def get_release(synix_dir: str | Path, release_name: str) -> ReleaseReceipt | None:
    """Load a release receipt by name."""
    release_dir = _release_dir(Path(synix_dir), release_name)
    return _load_current_receipt(release_dir)
