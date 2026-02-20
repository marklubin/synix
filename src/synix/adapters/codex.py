"""Codex trace parser — Codex JSONL traces/history -> transcript Artifacts."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from synix.core.models import Artifact

DEFAULT_MAX_CHARS = 80_000


def _safe_label(value: str) -> str:
    """Sanitize a string for use in an artifact label."""
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    return cleaned or "unknown"


def _middle_cut(text: str, max_chars: int) -> str:
    """Truncate text preserving start and end, cutting from the middle."""
    if len(text) <= max_chars:
        return text
    keep = max_chars - 50
    head = keep // 2
    tail = keep - head
    return text[:head] + "\n\n[... middle truncated ...]\n\n" + text[-tail:]


def _parse_iso_date(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.astimezone(UTC).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return ""


def _extract_codex_message_text(content: object) -> str:
    """Extract text blocks from Codex message content list."""
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts)


def _build_artifacts(
    *,
    sessions: dict[str, list[dict]],
    base_metadata: dict[str, dict],
    max_chars: int,
    source_path: Path,
) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for session_id in sorted(sessions):
        turns = sessions[session_id]
        if not turns:
            continue

        safe_session = _safe_label(session_id)
        transcript_lines = [f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['text']}" for t in turns]
        transcript = _middle_cut("\n\n".join(transcript_lines), max_chars=max_chars)

        date = turns[0].get("date", "")
        meta = dict(base_metadata.get(session_id, {}))
        meta.update(
            {
                "source": "codex",
                "session_id": session_id,
                "date": date,
                "message_count": len(turns),
                "source_path": source_path.name,
            }
        )
        artifacts.append(
            Artifact(
                label=f"t-codex-{safe_session}",
                artifact_type="transcript",
                content=transcript,
                metadata=meta,
            )
        )

        for idx, turn in enumerate(turns, start=1):
            turn_meta = {
                "source": "codex",
                "session_id": session_id,
                "turn_index": idx,
                "role": turn["role"],
                "timestamp": turn.get("timestamp", ""),
                "phase": turn.get("phase", ""),
                "date": turn.get("date", ""),
                "source_path": source_path.name,
            }
            role_prefix = "User" if turn["role"] == "user" else "Assistant"
            artifacts.append(
                Artifact(
                    label=f"t-codex-turn-{safe_session}-{idx:05d}",
                    artifact_type="transcript_turn",
                    content=f"{role_prefix}: {turn['text']}",
                    metadata=turn_meta,
                )
            )
    return artifacts


def _parse_history(events: list[dict], filepath: Path, max_chars: int) -> list[Artifact]:
    sessions: dict[str, list[dict]] = defaultdict(list)
    for event in events:
        session_id = event.get("session_id")
        text = event.get("text")
        ts = event.get("ts")
        if not isinstance(session_id, str) or not isinstance(text, str) or not text.strip():
            continue
        iso_ts = ""
        date = ""
        if isinstance(ts, (int, float)):
            dt = datetime.fromtimestamp(ts, tz=UTC)
            iso_ts = dt.isoformat()
            date = dt.strftime("%Y-%m-%d")
        sessions[session_id].append(
            {
                "role": "user",
                "text": text.strip(),
                "timestamp": iso_ts,
                "date": date,
                "phase": "",
            }
        )

    return _build_artifacts(sessions=sessions, base_metadata={}, max_chars=max_chars, source_path=filepath)


def _default_session_id(filepath: Path) -> str:
    stem = filepath.stem
    if stem.startswith("rollout-"):
        parts = stem.split("-")
        if len(parts) >= 2:
            return "-".join(parts[-5:])
    return stem


def _parse_rollout(events: list[dict], filepath: Path, max_chars: int) -> list[Artifact]:
    default_session_id = _default_session_id(filepath)
    sessions: dict[str, list[dict]] = defaultdict(list)
    base_metadata: dict[str, dict] = defaultdict(dict)
    known_session_id: str | None = None

    for event in events:
        event_type = event.get("type")
        payload = event.get("payload")
        timestamp = event.get("timestamp", "")
        date = _parse_iso_date(timestamp) if isinstance(timestamp, str) else ""

        if event_type == "session_meta" and isinstance(payload, dict):
            sid = payload.get("id") if isinstance(payload.get("id"), str) else default_session_id
            known_session_id = sid
            base_metadata[sid].update(
                {
                    "cwd": payload.get("cwd", ""),
                    "cli_version": payload.get("cli_version", ""),
                    "model_provider": payload.get("model_provider", ""),
                }
            )
            continue

        if event_type != "response_item" or not isinstance(payload, dict):
            continue
        if payload.get("type") != "message":
            continue
        role = payload.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _extract_codex_message_text(payload.get("content"))
        if not text:
            continue
        sid = known_session_id or default_session_id
        sessions[sid].append(
            {
                "role": role,
                "text": text,
                "timestamp": timestamp if isinstance(timestamp, str) else "",
                "date": date,
                "phase": payload.get("phase", ""),
            }
        )

    return _build_artifacts(
        sessions=sessions,
        base_metadata=base_metadata,
        max_chars=max_chars,
        source_path=filepath,
    )


def parse_codex(filepath: str | Path, max_chars: int = DEFAULT_MAX_CHARS) -> list[Artifact]:
    """Parse Codex history/session JSONL into transcript and transcript_turn artifacts."""
    path = Path(filepath)
    events: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                events.append(parsed)

    if not events:
        return []

    # Codex history.jsonl format: session_id + ts + text.
    if all(isinstance(e.get("session_id"), str) and "text" in e for e in events):
        return _parse_history(events, path, max_chars)

    # Codex rollout format: top-level envelope with type + payload.
    if any("payload" in e and isinstance(e.get("type"), str) for e in events):
        return _parse_rollout(events, path, max_chars)

    return []
