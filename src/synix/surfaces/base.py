"""Base classes for artifact publishing surfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synix.db.artifacts import Record


@dataclass
class PublishResult:
    """Result of publishing artifacts to a surface."""

    success: bool
    location: str  # Where published (file path, URL, etc.)
    count: int  # Records published
    error: str | None = None


@dataclass
class Surface(ABC):
    """Base class for artifact publishing surfaces.

    Surfaces are destinations where pipeline outputs can be published.
    Examples: files, APIs, MCP servers, etc.
    """

    name: str = ""  # Subclasses can override with computed default

    @abstractmethod
    def publish(self, records: list["Record"], run_id: str) -> PublishResult:
        """Publish records to this surface.

        Args:
            records: Records to publish.
            run_id: Current pipeline run ID.

        Returns:
            PublishResult with status and location.
        """
        ...
