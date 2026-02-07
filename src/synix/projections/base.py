"""Base projection interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from synix import Artifact


class BaseProjection(ABC):
    """Abstract base for projections that materialize artifacts into output surfaces."""

    @abstractmethod
    def materialize(self, artifacts: list[Artifact], config: dict) -> None:
        """Materialize artifacts into the projection's output format."""
        ...
