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


def _parse_iso_epoch(ts: str) -> float | None:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.timestamp()
    except (TypeError, ValueError):
        return None


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
        turns = sorted(
            sessions[session_id],
            key=lambda t: (
                t.get("sort_ts") if isinstance(t.get("sort_ts"), float | int) else float("inf"),
                t.get("seq", 0),
            ),
        )
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
                # Structured aliases for downstream generic processing.
                "meta.source.adapter": "codex",
                "meta.chat.session_id": session_id,
                "meta.time.date": date,
                "meta.source.path": source_path.name,
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
                "meta.source.adapter": "codex",
                "meta.chat.session_id": session_id,
                "meta.chat.role": turn["role"],
                "meta.time.timestamp": turn.get("timestamp", ""),
                "meta.time.date": turn.get("date", ""),
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


def _is_history_row(row: dict, filepath: Path) -> bool:
    # Tighten detection: Codex history parser only applies to history.jsonl shape.
    if filepath.name != "history.jsonl":
        return False
    return (
        isinstance(row.get("session_id"), str)
        and isinstance(row.get("text"), str)
        and isinstance(row.get("ts"), (int, float))
    )


def _parse_history_row(
    row: dict,
    *,
    sessions: dict[str, list[dict]],
    seq: int,
) -> None:
    # history.jsonl rows are user prompt history records in Codex.
    session_id = row.get("session_id")
    text = row.get("text")
    ts = row.get("ts")
    if not isinstance(session_id, str) or not isinstance(text, str) or not text.strip():
        return
    if not isinstance(ts, (int, float)):
        return
    dt = datetime.fromtimestamp(ts, tz=UTC)
    sessions[session_id].append(
        {
            "role": "user",
            "text": text.strip(),
            "timestamp": dt.isoformat(),
            "date": dt.strftime("%Y-%m-%d"),
            "phase": "",
            "sort_ts": float(ts),
            "seq": seq,
        }
    )


def _default_session_id(filepath: Path) -> str:
    stem = filepath.stem
    match = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$", stem)
    if match:
        return match.group(1)
    return stem


def _is_rollout_row(row: dict) -> bool:
    return isinstance(row.get("type"), str) and "payload" in row


def _parse_rollout_row(
    row: dict,
    *,
    default_session_id: str,
    sessions: dict[str, list[dict]],
    base_metadata: dict[str, dict],
    state: dict[str, str | None],
    seq: int,
) -> None:
    event_type = row.get("type")
    payload = row.get("payload")
    timestamp = row.get("timestamp", "")
    date = _parse_iso_date(timestamp) if isinstance(timestamp, str) else ""
    sort_ts = _parse_iso_epoch(timestamp) if isinstance(timestamp, str) else None

    if event_type == "session_meta" and isinstance(payload, dict):
        sid = payload.get("id") if isinstance(payload.get("id"), str) else default_session_id
        state["known_session_id"] = sid
        base_metadata[sid].update(
            {
                "cwd": payload.get("cwd", ""),
                "cli_version": payload.get("cli_version", ""),
                "model_provider": payload.get("model_provider", ""),
            }
        )
        return

    if event_type != "response_item" or not isinstance(payload, dict):
        return
    if payload.get("type") != "message":
        return
    role = payload.get("role")
    if role not in ("user", "assistant"):
        return
    text = _extract_codex_message_text(payload.get("content"))
    if not text:
        return

    sid = state.get("known_session_id") or default_session_id
    sessions[sid].append(
        {
            "role": role,
            "text": text,
            "timestamp": timestamp if isinstance(timestamp, str) else "",
            "date": date,
            "phase": payload.get("phase", ""),
            "sort_ts": sort_ts,
            "seq": seq,
        }
    )


def parse_codex(filepath: str | Path, max_chars: int = DEFAULT_MAX_CHARS) -> list[Artifact]:
    """Parse Codex history/session JSONL into transcript and transcript_turn artifacts."""
    path = Path(filepath)
    mode: str | None = None
    default_session_id = _default_session_id(path)
    sessions: dict[str, list[dict]] = defaultdict(list)
    base_metadata: dict[str, dict] = defaultdict(dict)
    state: dict[str, str | None] = {"known_session_id": None}
    parsed_rows = 0

    with open(path, encoding="utf-8") as f:
        for seq, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue

            if mode is None:
                if _is_rollout_row(parsed):
                    mode = "rollout"
                elif _is_history_row(parsed, path):
                    mode = "history"
                else:
                    # Unknown JSONL shape; avoid false-positive classification.
                    return []

            if mode == "rollout":
                _parse_rollout_row(
                    parsed,
                    default_session_id=default_session_id,
                    sessions=sessions,
                    base_metadata=base_metadata,
                    state=state,
                    seq=seq,
                )
                parsed_rows += 1
            elif mode == "history":
                _parse_history_row(parsed, sessions=sessions, seq=seq)
                parsed_rows += 1

    if parsed_rows == 0:
        return []
    return _build_artifacts(sessions=sessions, base_metadata=base_metadata, max_chars=max_chars, source_path=path)
