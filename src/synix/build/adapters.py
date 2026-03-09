"""ProjectionAdapter contract and built-in adapter registry.

Every projection is materialized by an adapter that follows the
plan / apply / verify lifecycle.  Two built-in adapters ship:
``synix_search`` and ``flat_file``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ReleasePlan:
    """Describes what an adapter will do during apply()."""

    adapter: str
    projection_name: str
    action: str  # "rebuild", "update", "noop"
    summary: str  # human-readable description
    artifacts_count: int = 0
    details: dict = field(default_factory=dict)


@dataclass
class AdapterReceipt:
    """Proof of what an adapter did during apply()."""

    adapter: str
    projection_name: str
    target: str
    artifacts_applied: int
    status: str  # "success", "failed"
    applied_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "projection_name": self.projection_name,
            "target": self.target,
            "artifacts_applied": self.artifacts_applied,
            "status": self.status,
            "applied_at": self.applied_at,
            "details": self.details,
        }


class ProjectionAdapter(ABC):
    """Target-specific materialization. Every adapter follows this contract."""

    @abstractmethod
    def plan(
        self,
        closure,  # ReleaseClosure (imported at call site to avoid circular)
        declaration,  # ProjectionDeclaration
        current_receipt: AdapterReceipt | None,
    ) -> ReleasePlan:
        """Diff desired state against current state."""
        ...

    @abstractmethod
    def apply(
        self,
        plan: ReleasePlan,
        target: str | Path,
    ) -> AdapterReceipt:
        """Execute the plan. Returns a receipt."""
        ...

    @abstractmethod
    def verify(self, receipt: AdapterReceipt, target: str | Path) -> bool:
        """Confirm the release is live and consistent."""
        ...


# ---------------------------------------------------------------------------
# Registry — lazy resolution to respect build/ -> search/ import boundary
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: dict[str, type[ProjectionAdapter]] = {}

# Built-in adapters resolved lazily via import path
_BUILTIN_ADAPTERS: dict[str, tuple[str, str]] = {
    "synix_search": ("synix.search.adapter", "SynixSearchAdapter"),
    "flat_file": ("synix.build.flat_file_adapter", "FlatFileAdapter"),
}


def register_adapter(name: str, adapter_cls: type[ProjectionAdapter]) -> None:
    _ADAPTER_REGISTRY[name] = adapter_cls


def get_adapter(name: str) -> ProjectionAdapter:
    cls = _ADAPTER_REGISTRY.get(name)
    if cls is not None:
        return cls()

    # Lazy resolution for built-in adapters
    builtin = _BUILTIN_ADAPTERS.get(name)
    if builtin is not None:
        import importlib

        module = importlib.import_module(builtin[0])
        cls = getattr(module, builtin[1])
        _ADAPTER_REGISTRY[name] = cls
        return cls()

    available = sorted(set(_ADAPTER_REGISTRY) | set(_BUILTIN_ADAPTERS))
    raise ValueError(f"Unknown adapter: {name!r}. Available: {available}")
