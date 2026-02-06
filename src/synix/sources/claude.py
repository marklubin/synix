"""Claude export parser.

Claude exports have this structure:
{
    "conversations": [
        {
            "uuid": "...",
            "title": "...",
            "created_at": "2024-03-15T10:00:00Z",
            "updated_at": "2024-03-15T11:00:00Z",
            "chat_messages": [
                {
                    "uuid": "...",
                    "sender": "human" | "assistant",
                    "text": "...",
                    "created_at": "..."
                }
            ]
        }
    ]
}
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import uuid4

from synix.db.artifacts import Record
from synix.sources.base import Source


@dataclass
class ClaudeExportSource(Source):
    """Parser for Claude conversation exports."""

    format: str = "claude-export"

    def parse(self, run_id: str) -> Iterator[Record]:
        """Parse Claude export and yield conversation records.

        Each conversation becomes one record with all messages concatenated.
        """
        self.validate()

        with open(self.file_path, encoding="utf-8") as f:
            data = json.load(f)

        conversations = data.get("conversations", [])
        if not conversations:
            # Try alternate structure (direct list)
            if isinstance(data, list):
                conversations = data

        for conv in conversations:
            record = self._parse_conversation(conv, run_id)
            if record:
                yield record

    def _parse_conversation(self, conv: dict[str, Any], run_id: str) -> Record | None:
        """Parse a single conversation into a Record."""
        conv_id = conv.get("uuid", conv.get("id", str(uuid4())))
        title = conv.get("title", "Untitled")
        created_at = conv.get("created_at", conv.get("create_time"))

        # Get messages - Claude uses "chat_messages"
        messages = conv.get("chat_messages", conv.get("messages", []))
        if not messages:
            return None

        # Format messages into content
        content_parts = [f"# {title}\n"]
        for msg in messages:
            sender = msg.get("sender", msg.get("role", "unknown"))
            text = msg.get("text", msg.get("content", ""))

            # Handle content that might be a list
            if isinstance(text, list):
                text = "\n".join(
                    part.get("text", str(part)) if isinstance(part, dict) else str(part)
                    for part in text
                )

            # Normalize sender name
            if sender in ("human", "user"):
                sender = "User"
            elif sender == "assistant":
                sender = "Assistant"

            content_parts.append(f"## {sender}\n{text}\n")

        content = "\n".join(content_parts)

        # Create record
        record = Record(
            id=str(uuid4()),
            content=content,
            content_fingerprint=Record.compute_fingerprint(content),
            step_name=self.name,
            branch="main",
            materialization_key=f"source:{self.name}:{conv_id}",
            run_id=run_id,
        )

        # Parse created_at
        created_dt = None
        if created_at:
            if isinstance(created_at, (int, float)):
                created_dt = datetime.fromtimestamp(created_at)
            elif isinstance(created_at, str):
                try:
                    created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except ValueError:
                    pass

        # Set metadata
        record.metadata_ = {
            "meta.chat.conversation_id": conv_id,
            "meta.chat.title": title,
            "meta.chat.message_count": len(messages),
            "meta.time.created_at": created_dt.isoformat() if created_dt else None,
            "meta.source.type": "claude-export",
            "meta.source.file": str(self.file_path),
        }

        return record


def create_claude_source(name: str, file: str) -> ClaudeExportSource:
    """Factory function to create a Claude export source.

    Args:
        name: Step name for this source.
        file: Path to the Claude export JSON file.

    Returns:
        Configured ClaudeExportSource.
    """
    from synix.sources.base import expand_path

    return ClaudeExportSource(
        name=name,
        file_path=expand_path(file),
    )
