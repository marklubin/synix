"""Parse transform — discovers and parses source files."""

from __future__ import annotations

import json
from pathlib import Path

from synix import Artifact
from synix.sources.chatgpt import parse_chatgpt
from synix.sources.claude import parse_claude
from synix.transforms.base import BaseTransform, register_transform


@register_transform("parse")
class ParseTransform(BaseTransform):
    """Discover and parse source files from source_dir into transcript Artifacts."""

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        """Parse all recognized export files in source_dir."""
        source_dir = Path(config["source_dir"])
        artifacts: list[Artifact] = []

        for filepath in sorted(source_dir.rglob("*.json")):
            artifacts.extend(self._parse_file(filepath))

        return artifacts

    def _parse_file(self, filepath: Path) -> list[Artifact]:
        """Detect format and parse a single JSON file."""
        try:
            data = json.loads(filepath.read_text())
        except (json.JSONDecodeError, OSError):
            return []

        # ChatGPT format: top-level list of objects with "mapping" keys
        if isinstance(data, list) and data and isinstance(data[0], dict) and "mapping" in data[0]:
            return parse_chatgpt(filepath)

        # Claude format: object with "conversations" key containing chat_messages
        if (
            isinstance(data, dict)
            and "conversations" in data
            and isinstance(data["conversations"], list)
            and data["conversations"]
            and "chat_messages" in data["conversations"][0]
        ):
            return parse_claude(filepath)

        return []
