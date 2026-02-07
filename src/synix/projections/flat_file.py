"""Flat file projection — renders core memory as context document."""

from __future__ import annotations

from pathlib import Path

from synix import Artifact
from synix.projections.base import BaseProjection


class FlatFileProjection(BaseProjection):
    """Renders artifacts as a flat file context document."""

    def materialize(self, artifacts: list[Artifact], config: dict) -> None:
        """Render artifacts as a text file.

        Config should contain 'output_path' — where to write the file.
        The output format is determined by the artifact content, not this projection.
        """
        output_path = Path(config.get("output_path", "./build/context.md"))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        parts: list[str] = []
        for artifact in artifacts:
            parts.append(artifact.content)

        content = "\n\n".join(parts)
        output_path.write_text(content)
