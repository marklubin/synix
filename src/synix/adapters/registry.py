"""Adapter registry — pluggable source format detection and parsing.

Maps file extensions to parser functions. Provides auto-detection for JSON files
(ChatGPT vs Claude format) and direct parsing for text/markdown files.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from synix.core.models import Artifact

# Registry: extension (with dot) -> parser function
_ADAPTERS: dict[str, Callable[[Path], list[Artifact]]] = {}

# All extensions handled by the registry
_ALL_EXTENSIONS: set[str] = set()


def register_adapter(extensions: list[str]):
    """Decorator to register a parser function for one or more file extensions.

    Usage::

        @register_adapter([".txt", ".md"])
        def parse_text(filepath: Path) -> list[Artifact]:
            ...
    """

    def decorator(fn: Callable[[Path], list[Artifact]]):
        for ext in extensions:
            normalized = ext if ext.startswith(".") else f".{ext}"
            _ADAPTERS[normalized] = fn
            _ALL_EXTENSIONS.add(normalized)
        return fn

    return decorator


def get_adapter(filepath: Path) -> Callable[[Path], list[Artifact]] | None:
    """Return the parser function for a file based on its extension, or None."""
    ext = filepath.suffix.lower()
    return _ADAPTERS.get(ext)


def get_supported_extensions() -> set[str]:
    """Return the set of all registered file extensions."""
    return set(_ALL_EXTENSIONS)


def parse_file(filepath: Path) -> list[Artifact]:
    """Auto-detect format and parse a file into Artifacts.

    For JSON files, inspects the content structure to decide between
    ChatGPT and Claude parsers. For text/markdown, uses the text parser.

    Returns an empty list for unrecognized formats or parse errors.
    """
    filepath = Path(filepath)
    adapter = get_adapter(filepath)
    if adapter is None:
        return []

    try:
        return adapter(filepath)
    except Exception:
        return []


def _parse_json_autodetect(filepath: Path) -> list[Artifact]:
    """Detect and parse a JSON file as either ChatGPT or Claude export.

    ChatGPT: top-level list of objects with "mapping" keys.
    Claude: object with "conversations" key containing chat_messages.
    """
    from synix.adapters.chatgpt import parse_chatgpt
    from synix.adapters.claude import parse_claude

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


# --- Register built-in adapters ---

# JSON files use auto-detection (ChatGPT vs Claude)
register_adapter([".json"])(_parse_json_autodetect)

# Text/Markdown files use the text parser
# Import here to avoid circular imports; the function is lightweight
from synix.adapters.text import parse_text  # noqa: E402

register_adapter([".txt", ".md"])(parse_text)
