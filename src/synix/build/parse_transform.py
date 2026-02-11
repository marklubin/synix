"""Parse transform — discovers and parses source files."""

from __future__ import annotations

from pathlib import Path

from synix.adapters.registry import get_supported_extensions, parse_file
from synix.build.transforms import BaseTransform, register_transform
from synix.core.models import Artifact


@register_transform("parse")
class ParseTransform(BaseTransform):
    """Discover and parse source files from source_dir into transcript Artifacts."""

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        """Parse all recognized export files in source_dir."""
        source_dir = Path(config["source_dir"])
        artifacts: list[Artifact] = []

        # Collect all files matching supported extensions
        extensions = get_supported_extensions()
        files: list[Path] = []
        for ext in sorted(extensions):
            files.extend(source_dir.rglob(f"*{ext}"))

        # Sort for deterministic ordering and parse each file
        for filepath in sorted(set(files)):
            artifacts.extend(parse_file(filepath))

        return artifacts
