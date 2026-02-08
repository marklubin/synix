"""Text/Markdown source parser — .txt and .md files to transcript Artifacts."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from synix.core.models import Artifact

# Pattern to match YAML frontmatter between --- markers
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?\n)---\s*\n", re.DOTALL)

# Pattern to match a date at the start of a filename: YYYY-MM-DD
_DATE_FROM_FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")

# Pattern to detect conversation turns
_TURN_RE = re.compile(r"^(User|Human|Assistant|AI|System):\s*", re.MULTILINE)


def parse_text(filepath: str | Path) -> list[Artifact]:
    """Parse a text/markdown file into transcript Artifacts.

    Supports:
    - YAML frontmatter (title, date, tags, etc.)
    - Date inference from filename (e.g., 2025-01-15-meeting-notes.md)
    - Optional turn detection (lines starting with "User:" / "Assistant:" etc.)
    - Plain text files (.txt) and markdown files (.md)
    """
    filepath = Path(filepath)
    raw = filepath.read_text(encoding="utf-8")

    # Parse frontmatter
    frontmatter: dict = {}
    body = raw
    fm_match = _FRONTMATTER_RE.match(raw)
    if fm_match:
        frontmatter = _parse_frontmatter(fm_match.group(1))
        body = raw[fm_match.end():]

    # Determine metadata
    title = frontmatter.get("title", _title_from_filename(filepath.stem))
    date = frontmatter.get("date", _date_from_filename(filepath.stem))
    tags = frontmatter.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    # Normalize date to string
    if isinstance(date, datetime):
        date = date.strftime("%Y-%m-%d")
    elif hasattr(date, "isoformat"):
        # date object (not datetime)
        date = date.isoformat()
    else:
        date = str(date) if date else ""

    # Detect conversation turns
    has_turns = bool(_TURN_RE.search(body))

    # Build content — use body as-is (preserving the original text)
    content = body.strip() + "\n" if body.strip() else ""

    if not content:
        return []

    # Count words
    word_count = len(content.split())

    # Count messages if conversation format
    message_count = len(_TURN_RE.findall(body)) if has_turns else 0

    # Sanitize filename stem for artifact ID
    safe_stem = _sanitize_id(filepath.stem)
    artifact_id = f"t-text-{safe_stem}"

    metadata: dict = {
        "source": "text",
        "title": title,
        "date": date,
        "word_count": word_count,
    }
    if tags:
        metadata["tags"] = tags
    if has_turns:
        metadata["has_turns"] = True
        metadata["message_count"] = message_count

    return [
        Artifact(
            artifact_id=artifact_id,
            artifact_type="transcript",
            content=content,
            metadata=metadata,
        )
    ]


def _parse_frontmatter(text: str) -> dict:
    """Parse simple YAML frontmatter without requiring pyyaml.

    Handles:
    - key: value (strings, numbers, dates)
    - key: [item1, item2] (inline lists)
    - tags as comma-separated or bracket-delimited lists
    """
    result: dict = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        colon_idx = line.find(":")
        if colon_idx < 0:
            continue

        key = line[:colon_idx].strip()
        value = line[colon_idx + 1:].strip()

        if not key:
            continue

        # Try to parse the value
        result[key] = _parse_value(value)

    return result


def _parse_value(value: str):
    """Parse a YAML-like value into a Python type."""
    if not value:
        return ""

    # Remove surrounding quotes
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]

    # Inline list: [item1, item2, ...]
    if value.startswith("[") and value.endswith("]"):
        items = value[1:-1].split(",")
        return [_parse_value(item.strip()) for item in items if item.strip()]

    # Boolean
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # Date: YYYY-MM-DD
    date_match = re.match(r"^\d{4}-\d{2}-\d{2}$", value)
    if date_match:
        return value  # Keep as string for consistency

    return value


def _title_from_filename(stem: str) -> str:
    """Derive a title from the filename stem.

    Strips leading date (YYYY-MM-DD-) and converts hyphens/underscores to spaces.
    Capitalizes the first letter of each word.
    """
    # Remove leading date
    cleaned = re.sub(r"^\d{4}-\d{2}-\d{2}[-_]?", "", stem)
    if not cleaned:
        return stem
    # Replace separators with spaces
    cleaned = re.sub(r"[-_]+", " ", cleaned)
    return cleaned.strip().title()


def _date_from_filename(stem: str) -> str:
    """Extract a date from the filename stem if it starts with YYYY-MM-DD."""
    match = _DATE_FROM_FILENAME_RE.match(stem)
    if match:
        return match.group(1)
    return ""


def _sanitize_id(stem: str) -> str:
    """Sanitize a filename stem for use as an artifact ID component.

    Keeps alphanumeric, hyphens, and underscores. Replaces others with hyphens.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "-", stem)
    # Collapse multiple hyphens
    sanitized = re.sub(r"-+", "-", sanitized)
    return sanitized.strip("-")
