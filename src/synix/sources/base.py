"""Abstract base class for source importers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synix.db.artifacts import Record


@dataclass
class Source(ABC):
    """Abstract base class for source importers.

    Sources parse external data files and yield Records.
    They are the leaves of the provenance DAG - records
    created by sources have no upstream sources.
    """

    name: str
    file_path: Path
    format: str

    @abstractmethod
    def parse(self, run_id: str) -> Iterator["Record"]:
        """Parse the source file and yield Records.

        Args:
            run_id: Current run ID for tracking.

        Yields:
            Records parsed from the source file.
        """
        ...

    def validate(self) -> None:
        """Validate that the source file exists and is readable."""
        if not self.file_path.exists():
            msg = f"Source file not found: {self.file_path}"
            raise FileNotFoundError(msg)
        if not self.file_path.is_file():
            msg = f"Source path is not a file: {self.file_path}"
            raise ValueError(msg)


def expand_path(path: str) -> Path:
    """Expand user home directory and resolve path."""
    return Path(path).expanduser().resolve()
