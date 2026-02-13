"""ChatGPT export parser — conversations.json → transcript Artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from synix.core.models import Artifact

_ALLOWED_ROLES = {"user", "assistant"}


def parse_chatgpt(filepath: str | Path) -> list[Artifact]:
    """Parse ChatGPT conversations.json export into transcript Artifacts.

    The ChatGPT export is a JSON array of conversation objects. Each conversation
    has a tree-structured ``mapping`` of messages.

    Linearization strategy:
    - If the conversation has a ``current_node`` field pointing to the active leaf,
      we walk parent pointers from that node back to root, then reverse to get
      chronological order. This correctly handles regeneration branches.
    - Otherwise, we fall back to following the first child at each node from root
      (the legacy main-thread heuristic).

    Only ``user`` and ``assistant`` messages are included; system, tool, and other
    roles are filtered out.
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
        current_node = conv.get("current_node")
        messages = _linearize_mapping(mapping, current_node=current_node)

        if not messages:
            continue

        # Format transcript — only user and assistant roles
        parts: list[str] = []
        for msg in messages:
            role = msg["author"]["role"]
            if role not in _ALLOWED_ROLES:
                continue
            text_parts = msg["content"].get("parts", [])
            text = "".join(str(p) for p in text_parts if isinstance(p, str))
            if text.strip():
                parts.append(f"{role}: {text}")

        if not parts:
            continue

        content = "\n\n".join(parts) + "\n"
        date_str = datetime.fromtimestamp(create_time, tz=UTC).strftime("%Y-%m-%d") if create_time else ""

        # Derive last_message_date from individual message timestamps
        last_message_date = ""
        max_ts = 0.0
        for msg in messages:
            ts = msg.get("create_time")
            if ts and ts > max_ts:
                max_ts = ts
        if max_ts:
            last_message_date = datetime.fromtimestamp(max_ts, tz=UTC).strftime("%Y-%m-%d")

        metadata = {
            "source": "chatgpt",
            "source_conversation_id": conv_id,
            "title": title,
            "date": date_str,
            "last_message_date": last_message_date,
            "message_count": len(parts),
        }
        # Forward optional top-level fields (e.g., customer_id)
        if conv.get("customer_id"):
            metadata["customer_id"] = conv["customer_id"]

        artifacts.append(
            Artifact(
                label=f"t-chatgpt-{conv_id}",
                artifact_type="transcript",
                content=content,
                metadata=metadata,
            )
        )

    return artifacts


def _linearize_mapping(mapping: dict, *, current_node: str | None = None) -> list[dict]:
    """Linearize the ChatGPT message tree into a chronological message list.

    When *current_node* is provided and found in *mapping*, we walk parent
    pointers from that leaf back to the root and reverse the result.  This
    correctly selects the active branch when the user regenerated a response
    (which creates a fork in the tree).

    When *current_node* is ``None`` or not present in *mapping*, we fall back
    to the legacy strategy of following ``children[0]`` at every node from the
    root.
    """
    # --- Strategy 1: walk from current_node to root via parent pointers ---
    if current_node and current_node in mapping:
        messages: list[dict] = []
        node_id: str | None = current_node
        while node_id is not None:
            node = mapping.get(node_id)
            if node is None:
                break
            msg = node.get("message")
            if msg is not None:
                messages.append(msg)
            node_id = node.get("parent")
        messages.reverse()
        return messages

    # --- Strategy 2 (fallback): follow first child from root ---
    root_id = None
    for nid, node in mapping.items():
        if node.get("parent") is None:
            root_id = nid
            break

    if root_id is None:
        return []

    messages = []
    current_id: str | None = root_id

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
