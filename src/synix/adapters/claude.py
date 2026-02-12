"""Claude export parser — JSON → transcript Artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from synix.core.models import Artifact

_SENDER_MAP = {"human": "user", "Human": "user"}


def parse_claude(filepath: str | Path) -> list[Artifact]:
    """Parse Claude export JSON into transcript Artifacts.

    The Claude export is ``{"conversations": [...]}``. Each conversation has a
    flat ``chat_messages`` array with ``sender`` and ``text`` fields.

    Sender normalization: ``human`` / ``Human`` are mapped to ``user`` so that
    transcripts from all sources use a consistent ``user`` / ``assistant``
    vocabulary.
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
            sender = _SENDER_MAP.get(sender, sender)
            text = msg.get("text")
            if text is None:
                # Fallback: try "content" field (nested content blocks)
                raw_content = msg.get("content")
                if isinstance(raw_content, str):
                    text = raw_content
                elif isinstance(raw_content, list):
                    # List of content blocks — extract text from each
                    text = " ".join(
                        block.get("text", "") if isinstance(block, dict) else str(block) for block in raw_content
                    )
                else:
                    text = ""
            if text.strip():
                parts.append(f"{sender}: {text}")

        if not parts:
            continue

        content = "\n\n".join(parts) + "\n"

        # Parse ISO timestamp robustly
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                date_str = created_at[:10] if len(created_at) >= 10 else ""
        else:
            date_str = ""

        # Derive last_message_date from individual message timestamps
        last_message_date = ""
        last_dt = None
        for msg in chat_messages:
            msg_ts = msg.get("created_at", "")
            if msg_ts:
                try:
                    msg_dt = datetime.fromisoformat(msg_ts.replace("Z", "+00:00"))
                    if last_dt is None or msg_dt > last_dt:
                        last_dt = msg_dt
                except (ValueError, TypeError):
                    pass
        if last_dt is not None:
            last_message_date = last_dt.strftime("%Y-%m-%d")

        artifacts.append(
            Artifact(
                artifact_id=f"t-claude-{uuid}",
                artifact_type="transcript",
                content=content,
                metadata={
                    "source": "claude",
                    "source_conversation_id": uuid,
                    "title": title,
                    "date": date_str,
                    "last_message_date": last_message_date,
                    "message_count": len(chat_messages),
                },
            )
        )

    return artifacts
