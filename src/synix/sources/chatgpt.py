"""ChatGPT export parser — conversations.json → transcript Artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from synix import Artifact


def parse_chatgpt(filepath: str | Path) -> list[Artifact]:
    """Parse ChatGPT conversations.json export into transcript Artifacts.

    The ChatGPT export is a JSON array of conversation objects. Each conversation
    has a tree-structured ``mapping`` of messages. We linearize by following the
    first child at each node (the main thread).
    """
    filepath = Path(filepath)
    data = json.loads(filepath.read_text())

    artifacts: list[Artifact] = []
    for conv in data:
        conv_id = conv.get("conversation_id", conv.get("id"))
        title = conv.get("title", "Untitled")
        create_time = conv.get("create_time", 0)
        mapping = conv.get("mapping", {})

        # Linearize the message tree
        messages = _linearize_mapping(mapping)

        if not messages:
            continue

        # Format transcript
        parts: list[str] = []
        for msg in messages:
            role = msg["author"]["role"]
            text_parts = msg["content"].get("parts", [])
            text = "".join(str(p) for p in text_parts if isinstance(p, str))
            if text.strip():
                parts.append(f"{role}: {text}")

        if not parts:
            continue

        content = "\n\n".join(parts) + "\n"
        date_str = datetime.fromtimestamp(create_time).strftime("%Y-%m-%d") if create_time else ""

        artifacts.append(Artifact(
            artifact_id=f"t-chatgpt-{conv_id}",
            artifact_type="transcript",
            content=content,
            metadata={
                "source": "chatgpt",
                "source_conversation_id": conv_id,
                "title": title,
                "date": date_str,
                "message_count": len(parts),
            },
        ))

    return artifacts


def _linearize_mapping(mapping: dict) -> list[dict]:
    """Walk the ChatGPT message tree from root, following first child at each step.

    Returns messages in chronological order (root → leaf), skipping nodes
    where ``message`` is None (synthetic root nodes).
    """
    # Find root node(s) — parent is None
    root_id = None
    for node_id, node in mapping.items():
        if node.get("parent") is None:
            root_id = node_id
            break

    if root_id is None:
        return []

    messages: list[dict] = []
    current_id = root_id

    while current_id is not None:
        node = mapping.get(current_id)
        if node is None:
            break

        msg = node.get("message")
        if msg is not None:
            messages.append(msg)

        children = node.get("children", [])
        current_id = children[0] if children else None

    return messages
