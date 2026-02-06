"""File surface for publishing artifacts to the filesystem."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from synix.surfaces.base import PublishResult, Surface

if TYPE_CHECKING:
    from synix.db.artifacts import Record


@dataclass
class FileSurface(Surface):
    """Publish records to the filesystem.

    Supports template variables in path:
    - {step_name} — source step name (first record's step)
    - {run_id} — pipeline run ID
    - {timestamp} — ISO timestamp (YYYY-MM-DD_HH-MM-SS)
    - {date} — ISO date (YYYY-MM-DD)

    Formats:
    - markdown — records as markdown sections
    - json — records as JSON array
    - text — raw content concatenated
    """

    path_template: str = ""  # e.g., "output/{step_name}.md"
    format: str = "markdown"  # markdown, json, text

    def __post_init__(self) -> None:
        """Set name from path if not provided."""
        if not self.path_template:
            msg = "path_template is required"
            raise ValueError(msg)
        if not self.name:
            self.name = Path(self.path_template).stem

    def publish(self, records: list["Record"], run_id: str) -> PublishResult:
        """Publish records to a file.

        Args:
            records: Records to publish.
            run_id: Current pipeline run ID.

        Returns:
            PublishResult with file path and status.
        """
        if not records:
            return PublishResult(
                success=True,
                location="",
                count=0,
                error="No records to publish",
            )

        try:
            # Expand template
            path = self._expand_template(self.path_template, records, run_id)

            # Format content
            if self.format == "markdown":
                content = self._format_markdown(records)
            elif self.format == "json":
                content = self._format_json(records)
            else:  # text
                content = self._format_text(records)

            # Write file
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            path_obj.write_text(content)

            return PublishResult(
                success=True,
                location=str(path_obj.absolute()),
                count=len(records),
            )

        except Exception as e:
            return PublishResult(
                success=False,
                location=self.path_template,
                count=0,
                error=str(e),
            )

    def _expand_template(
        self,
        template: str,
        records: list["Record"],
        run_id: str,
    ) -> str:
        """Expand template variables in path.

        Args:
            template: Path template with {variables}.
            records: Records being published.
            run_id: Pipeline run ID.

        Returns:
            Expanded path string.
        """
        now = datetime.now()

        # Get step name from first record
        step_name = records[0].step_name if records else "unknown"

        # Build replacement dict
        replacements = {
            "step_name": step_name,
            "run_id": run_id[:8] if len(run_id) > 8 else run_id,
            "timestamp": now.strftime("%Y-%m-%d_%H-%M-%S"),
            "date": now.strftime("%Y-%m-%d"),
        }

        # Expand template
        result = template
        for key, value in replacements.items():
            result = result.replace(f"{{{key}}}", value)

        return result

    def _format_markdown(self, records: list["Record"]) -> str:
        """Format records as markdown.

        Args:
            records: Records to format.

        Returns:
            Markdown string.
        """
        parts = []
        for record in records:
            # Get metadata for title
            meta = record.metadata_
            title = meta.get("meta.source.title", record.step_name)

            parts.append(f"## {title}\n\n{record.content}")

        return "\n\n---\n\n".join(parts)

    def _format_json(self, records: list["Record"]) -> str:
        """Format records as JSON.

        Args:
            records: Records to format.

        Returns:
            JSON string.
        """
        data = []
        for record in records:
            data.append(
                {
                    "id": str(record.id),
                    "step_name": record.step_name,
                    "content": record.content,
                    "metadata": record.metadata_,
                    "created_at": str(record.created_at),
                }
            )

        return json.dumps(data, indent=2)

    def _format_text(self, records: list["Record"]) -> str:
        """Format records as plain text.

        Args:
            records: Records to format.

        Returns:
            Plain text string.
        """
        return "\n\n".join(r.content for r in records)


def parse_file_surface(name: str, surface_uri: str) -> FileSurface:
    """Parse a file:// URI into a FileSurface.

    Args:
        name: Artifact name.
        surface_uri: URI like "file://output/report.md"

    Returns:
        Configured FileSurface.
    """
    if not surface_uri.startswith("file://"):
        msg = f"Expected file:// URI, got: {surface_uri}"
        raise ValueError(msg)

    path = surface_uri[7:]  # Remove "file://"

    # Determine format from extension
    ext = Path(path).suffix.lower()
    if ext == ".json":
        fmt = "json"
    elif ext in (".md", ".markdown"):
        fmt = "markdown"
    else:
        fmt = "text"

    return FileSurface(
        name=name,
        path_template=path,
        format=fmt,
    )
