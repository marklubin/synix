"""ChatGPT export parser.

ChatGPT exports have this structure (conversations.json):
[
    {
        "id": "...",
        "title": "...",
        "create_time": 1710500000,
        "update_time": 1710501000,
        "mapping": {
            "msg-id-1": {
                "id": "msg-id-1",
                "message": {
                    "id": "...",
                    "author": {"role": "user"},
                    "content": {
                        "content_type": "text",
                        "parts": ["Hello!"]
                    },
                    "create_time": 1710500000
                },
                "parent": null,
                "children": ["msg-id-2"]
            }
        }
    }
]
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
class ChatGPTExportSource(Source):
    """Parser for ChatGPT conversation exports."""

    format: str = "chatgpt-export"

    def parse(self, run_id: str) -> Iterator[Record]:
        """Parse ChatGPT export and yield conversation records.

        Each conversation becomes one record with all messages concatenated.
        """
        self.validate()

        with open(self.file_path, encoding="utf-8") as f:
            data = json.load(f)

        # ChatGPT export is a list of conversations
        conversations = data if isinstance(data, list) else data.get("conversations", [])

        for conv in conversations:
            record = self._parse_conversation(conv, run_id)
            if record:
                yield record

    def _parse_conversation(self, conv: dict[str, Any], run_id: str) -> Record | None:
        """Parse a single conversation into a Record."""
        conv_id = conv.get("id", str(uuid4()))
        title = conv.get("title", "Untitled")
        create_time = conv.get("create_time")

        # Extract messages from mapping structure
        mapping = conv.get("mapping", {})
        messages = self._extract_messages_from_mapping(mapping)

        if not messages:
            return None

        # Format messages into content
        content_parts = [f"# {title}\n"]
        for msg in messages:
            role = msg.get("role", "unknown")
            text = msg.get("text", "")

            # Normalize role name
            if role == "user":
                role = "User"
            elif role == "assistant":
                role = "Assistant"
            elif role == "system":
                continue  # Skip system messages

            if text.strip():
                content_parts.append(f"## {role}\n{text}\n")

        content = "\n".join(content_parts)

        if len(content_parts) <= 1:
            # Only title, no actual messages
            return None

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

        # Parse create_time
        created_dt = None
        if create_time:
            if isinstance(create_time, (int, float)):
                created_dt = datetime.fromtimestamp(create_time)

        # Set metadata
        record.metadata_ = {
            "meta.chat.conversation_id": conv_id,
            "meta.chat.title": title,
            "meta.chat.message_count": len(messages),
            "meta.time.created_at": created_dt.isoformat() if created_dt else None,
            "meta.source.type": "chatgpt-export",
            "meta.source.file": str(self.file_path),
        }

        return record

    def _extract_messages_from_mapping(self, mapping: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract ordered messages from ChatGPT's nested mapping structure."""
        messages = []

        # Build parent-child graph
        children_map: dict[str, list[str]] = {}
        for node_id, node in mapping.items():
            parent = node.get("parent")
            if parent:
                if parent not in children_map:
                    children_map[parent] = []
                children_map[parent].append(node_id)

        # Find root nodes (no parent)
        roots = [nid for nid, node in mapping.items() if node.get("parent") is None]

        # Walk the tree in order
        def walk(node_id: str) -> None:
            node = mapping.get(node_id, {})
            message = node.get("message")

            if message and message.get("content"):
                author = message.get("author", {})
                role = author.get("role", "unknown")
                content = message.get("content", {})

                # Extract text from parts
                parts = content.get("parts", [])
                text = "\n".join(str(p) for p in parts if p)

                if text.strip():
                    messages.append(
                        {
                            "role": role,
                            "text": text,
                            "create_time": message.get("create_time"),
                        }
                    )

            # Walk children
            for child_id in children_map.get(node_id, []):
                walk(child_id)

        for root in roots:
            walk(root)

        return messages


def create_chatgpt_source(name: str, file: str) -> ChatGPTExportSource:
    """Factory function to create a ChatGPT export source.

    Args:
        name: Step name for this source.
        file: Path to the ChatGPT export JSON file.

    Returns:
        Configured ChatGPTExportSource.
    """
    from synix.sources.base import expand_path

    return ChatGPTExportSource(
        name=name,
        file_path=expand_path(file),
    )
