"""Claude Code JSONL session parser — .jsonl session files → transcript Artifacts."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from synix.core.models import Artifact

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 80_000
MIN_MEANINGFUL_TURNS = 2


def _extract_text(content) -> str:
    """Extract human-readable text from message content.

    Content can be a plain string or an array of content blocks.
    Only text blocks are kept — tool_use, tool_result, thinking, etc. are skipped.
    """
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text.strip():
                    parts.append(text.strip())
        return "\n\n".join(parts)

    return ""


def _middle_cut(text: str, max_chars: int) -> str:
    """Truncate text preserving start and end, cutting from the middle."""
    if len(text) <= max_chars:
        return text

    keep = max_chars - 50  # room for marker
    head = keep // 2
    tail = keep - head
    return text[:head] + "\n\n[... middle truncated ...]\n\n" + text[-tail:]


def parse_claude_code(filepath: str | Path, max_chars: int = DEFAULT_MAX_CHARS) -> list[Artifact]:
    """Parse a Claude Code .jsonl session file into a transcript Artifact.

    Claude Code sessions are line-delimited JSON with entries like::

        {"type": "user"|"assistant", "message": {"role": "...", "content": ...},
         "timestamp": "...", "sessionId": "...", "cwd": "...", ...}

    Returns a list with one Artifact for valid sessions, or empty list for
    sessions with < 2 meaningful turns or non-Claude-Code JSONL files.
    """
    filepath = Path(filepath)
    turns: list[str] = []
    session_id = filepath.stem
    slug = ""
    date = ""
    git_branch = ""
    cwd = ""
    is_claude_code = False

    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Malformed JSON at %s:%d, skipping line", filepath, line_num)
                continue

            if not isinstance(entry, dict):
                continue

            entry_type = entry.get("type")
            if entry_type not in ("user", "assistant"):
                continue

            # If we see a user/assistant entry with a message dict, it's Claude Code format
            message = entry.get("message")
            if not isinstance(message, dict):
                continue
            is_claude_code = True

            # Extract metadata from first relevant entry
            if not date:
                ts = entry.get("timestamp", "")
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        date = dt.strftime("%Y-%m-%d")
                    except (ValueError, TypeError):
                        pass

            if not slug:
                slug = entry.get("slug", "")
            if not git_branch:
                git_branch = entry.get("gitBranch", "")
            if not cwd:
                cwd = entry.get("cwd", "")

            content = message.get("content", "")
            text = _extract_text(content)
            if not text:
                continue

            role = message.get("role", entry_type)
            prefix = "User" if role == "user" else "Assistant"
            turns.append(f"{prefix}: {text}")

    if not is_claude_code:
        logger.debug("JSONL file %s does not contain Claude Code session data, skipping", filepath)
        return []

    if len(turns) < MIN_MEANINGFUL_TURNS:
        return []

    transcript = "\n\n".join(turns)
    transcript = _middle_cut(transcript, max_chars)

    title = slug if slug else session_id
    metadata = {
        "source": "claude-code",
        "session_id": session_id,
        "title": title,
        "date": date,
        "cwd": cwd,
        "git_branch": git_branch,
        "message_count": len(turns),
    }

    return [
        Artifact(
            label=f"t-claude-code-{session_id}",
            artifact_type="transcript",
            content=transcript,
            metadata=metadata,
        )
    ]
