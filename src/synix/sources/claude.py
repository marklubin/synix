"""Claude export parser — JSON → transcript Artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from synix import Artifact


def parse_claude(filepath: str | Path) -> list[Artifact]:
    """Parse Claude export JSON into transcript Artifacts.

    The Claude export is ``{"conversations": [...]}``. Each conversation has a
    flat ``chat_messages`` array with ``sender`` and ``text`` fields.
    """
    filepath = Path(filepath)
    data = json.loads(filepath.read_text())

    conversations = data.get("conversations", [])
    artifacts: list[Artifact] = []

    for conv in conversations:
        uuid = conv["uuid"]
        title = conv.get("name", conv.get("title", "Untitled"))
        created_at = conv.get("created_at", "")
        chat_messages = conv.get("chat_messages", [])

        if not chat_messages:
            continue

        # Format transcript
        parts: list[str] = []
        for msg in chat_messages:
            sender = msg.get("sender", "unknown")
            text = msg.get("text")
            if text is None:
                # Fallback: try "content" field (nested content blocks)
                raw_content = msg.get("content")
                if isinstance(raw_content, str):
                    text = raw_content
                elif isinstance(raw_content, list):
                    # List of content blocks — extract text from each
                    text = " ".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in raw_content
                    )
                else:
                    text = ""
            if text.strip():
                parts.append(f"{sender}: {text}")

        if not parts:
            continue

        content = "\n\n".join(parts) + "\n"
        date_str = created_at[:10] if created_at else ""

        artifacts.append(Artifact(
            artifact_id=f"t-claude-{uuid}",
            artifact_type="transcript",
            content=content,
            metadata={
                "source": "claude",
                "source_conversation_id": uuid,
                "title": title,
                "date": date_str,
                "message_count": len(chat_messages),
            },
        ))

    return artifacts
