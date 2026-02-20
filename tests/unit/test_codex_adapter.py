"""Tests for Codex JSONL adapter and .jsonl auto-detection."""

from __future__ import annotations

import json
from pathlib import Path

from synix.adapters.codex import parse_codex
from synix.adapters.registry import parse_file


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_parse_codex_history_dual_output(tmp_path):
    path = tmp_path / "history.jsonl"
    _write_jsonl(
        path,
        [
            {"session_id": "sess-a", "ts": 1770863525, "text": "How do I run tests?"},
            {"session_id": "sess-a", "ts": 1770863547, "text": "Use pytest -q."},
            {"session_id": "sess-b", "ts": 1770881940, "text": "Review this design doc"},
        ],
    )

    artifacts = parse_codex(path)
    transcripts = [a for a in artifacts if a.artifact_type == "transcript"]
    turns = [a for a in artifacts if a.artifact_type == "transcript_turn"]

    assert len(transcripts) == 2
    assert len(turns) == 3
    assert any(a.label == "t-codex-sess-a" for a in transcripts)
    assert all(t.metadata["source"] == "codex" for t in turns)
    assert all(t.metadata["role"] == "user" for t in turns)


def test_parse_codex_rollout_dual_output_and_metadata(tmp_path):
    path = tmp_path / "rollout-2026-02-19T14-43-28-019c7812-90eb-7753-a196-dd5938d96f1e.jsonl"
    _write_jsonl(
        path,
        [
            {
                "timestamp": "2026-02-19T14:43:28.100Z",
                "type": "session_meta",
                "payload": {
                    "id": "019c7812-90eb-7753-a196-dd5938d96f1e",
                    "cwd": "/home/mark/synix/docs",
                    "cli_version": "0.101.0",
                    "model_provider": "openai",
                },
            },
            {
                "timestamp": "2026-02-19T14:44:00.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "review this doc"}],
                },
            },
            {
                "timestamp": "2026-02-19T14:44:01.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "I will review it now."}],
                    "phase": "commentary",
                },
            },
            {
                "timestamp": "2026-02-19T14:44:05.000Z",
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Review completed."}],
                    "phase": "final_answer",
                },
            },
        ],
    )

    artifacts = parse_codex(path)
    transcripts = [a for a in artifacts if a.artifact_type == "transcript"]
    turns = [a for a in artifacts if a.artifact_type == "transcript_turn"]

    assert len(transcripts) == 1
    assert len(turns) == 3
    transcript = transcripts[0]
    assert transcript.metadata["source"] == "codex"
    assert transcript.metadata["session_id"] == "019c7812-90eb-7753-a196-dd5938d96f1e"
    assert transcript.metadata["cwd"] == "/home/mark/synix/docs"
    assert transcript.metadata["cli_version"] == "0.101.0"
    assert transcript.metadata["model_provider"] == "openai"
    assert "I will review it now." in transcript.content
    assert "Review completed." in transcript.content

    assistant_turns = [t for t in turns if t.metadata["role"] == "assistant"]
    assert len(assistant_turns) == 2
    assert {t.metadata["phase"] for t in assistant_turns} == {"commentary", "final_answer"}


def test_parse_codex_skips_malformed_lines(tmp_path):
    path = tmp_path / "history.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"session_id":"sess-a","ts":1770863525,"text":"hello"}\n')
        f.write("{not-json\n")
        f.write('{"session_id":"sess-a","ts":1770863547,"text":"world"}\n')

    artifacts = parse_codex(path)
    transcripts = [a for a in artifacts if a.artifact_type == "transcript"]
    assert len(transcripts) == 1
    assert "hello" in transcripts[0].content
    assert "world" in transcripts[0].content


def test_parse_codex_non_codex_jsonl_returns_empty(tmp_path):
    path = tmp_path / "analytics.jsonl"
    _write_jsonl(
        path,
        [
            {"event": "page_view", "url": "/home", "ts": 1234567890},
            {"event": "click", "target": "button", "ts": 1234567891},
        ],
    )
    assert parse_codex(path) == []


def test_registry_jsonl_autodetect_codex(tmp_path):
    path = tmp_path / "history.jsonl"
    _write_jsonl(path, [{"session_id": "sess-a", "ts": 1770863525, "text": "hello"}])

    artifacts = parse_file(path)
    assert len(artifacts) == 2  # one transcript + one turn
    assert all(a.metadata["source"] == "codex" for a in artifacts)


def test_registry_jsonl_autodetect_claude_code(tmp_path):
    path = tmp_path / "session.jsonl"
    _write_jsonl(
        path,
        [
            {
                "type": "user",
                "message": {"role": "user", "content": "hello"},
                "timestamp": "2025-06-15T10:30:00Z",
                "sessionId": "abc123",
                "slug": "test-session",
                "cwd": "/home/user/project",
                "gitBranch": "main",
            },
            {
                "type": "assistant",
                "message": {"role": "assistant", "content": "hi"},
                "timestamp": "2025-06-15T10:31:00Z",
                "sessionId": "abc123",
                "slug": "test-session",
                "cwd": "/home/user/project",
                "gitBranch": "main",
            },
        ],
    )

    artifacts = parse_file(path)
    assert len(artifacts) == 1
    assert artifacts[0].metadata["source"] == "claude-code"
