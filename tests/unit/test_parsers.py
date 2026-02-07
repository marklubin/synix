"""Tests for source parsers — ChatGPT and Claude export formats."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synix.sources.chatgpt import parse_chatgpt
from synix.sources.claude import parse_claude


class TestChatGPTParsing:
    """Tests for ChatGPT export parser."""

    def test_chatgpt_parse_basic(self, chatgpt_fixture_path):
        """Parse fixture produces 3 transcripts."""
        artifacts = parse_chatgpt(chatgpt_fixture_path)
        assert len(artifacts) == 3
        assert all(a.artifact_type == "transcript" for a in artifacts)

    def test_chatgpt_metadata(self, chatgpt_fixture_path):
        """Conversation ID, title, date, message_count all populated."""
        artifacts = parse_chatgpt(chatgpt_fixture_path)
        first = artifacts[0]

        assert first.metadata["source"] == "chatgpt"
        assert first.metadata["source_conversation_id"] == "conv-gpt-001"
        assert first.metadata["title"] == "Machine learning fundamentals"
        assert first.metadata["date"]  # non-empty date string
        assert first.metadata["message_count"] == 4  # 2 user + 2 assistant

    def test_chatgpt_message_ordering(self, chatgpt_fixture_path):
        """Messages in chronological order — user question before assistant answer."""
        artifacts = parse_chatgpt(chatgpt_fixture_path)
        first = artifacts[0]

        lines = first.content.strip().split("\n\n")
        # First message should be from user
        assert lines[0].startswith("user:")
        # Second should be assistant
        assert lines[1].startswith("assistant:")

    def test_chatgpt_empty_conversation(self, tmp_path):
        """Handle conversation with no messages gracefully."""
        empty_conv = [{
            "id": "conv-empty",
            "title": "Empty",
            "create_time": 1710500000,
            "mapping": {
                "msg-root": {
                    "id": "msg-root",
                    "message": None,
                    "parent": None,
                    "children": [],
                }
            },
        }]
        filepath = tmp_path / "empty.json"
        filepath.write_text(json.dumps(empty_conv))

        artifacts = parse_chatgpt(filepath)
        assert len(artifacts) == 0

    def test_chatgpt_artifact_ids_prefixed(self, chatgpt_fixture_path):
        """All artifact IDs have t-chatgpt- prefix."""
        artifacts = parse_chatgpt(chatgpt_fixture_path)
        assert all(a.artifact_id.startswith("t-chatgpt-") for a in artifacts)

    def test_chatgpt_content_hash_computed(self, chatgpt_fixture_path):
        """Content hash is auto-computed by Artifact.__post_init__."""
        artifacts = parse_chatgpt(chatgpt_fixture_path)
        assert all(a.content_hash.startswith("sha256:") for a in artifacts)


class TestClaudeParsing:
    """Tests for Claude export parser."""

    def test_claude_parse_basic(self, claude_fixture_path):
        """Parse fixture produces 5 transcripts."""
        artifacts = parse_claude(claude_fixture_path)
        assert len(artifacts) == 5
        assert all(a.artifact_type == "transcript" for a in artifacts)

    def test_claude_metadata(self, claude_fixture_path):
        """UUID, title, date, message_count all populated."""
        artifacts = parse_claude(claude_fixture_path)
        first = artifacts[0]

        assert first.metadata["source"] == "claude"
        assert first.metadata["source_conversation_id"] == "conv-001"
        assert first.metadata["title"] == "Rust ownership discussion"
        assert first.metadata["date"] == "2024-03-15"
        assert first.metadata["message_count"] == 4

    def test_claude_artifact_ids_prefixed(self, claude_fixture_path):
        """All artifact IDs have t-claude- prefix."""
        artifacts = parse_claude(claude_fixture_path)
        assert all(a.artifact_id.startswith("t-claude-") for a in artifacts)


class TestMixedSources:
    """Tests for parsing both formats together."""

    def test_mixed_sources_no_collisions(self, chatgpt_fixture_path, claude_fixture_path):
        """Parse both formats — no collisions in artifact IDs."""
        chatgpt_artifacts = parse_chatgpt(chatgpt_fixture_path)
        claude_artifacts = parse_claude(claude_fixture_path)

        all_ids = [a.artifact_id for a in chatgpt_artifacts + claude_artifacts]
        assert len(all_ids) == len(set(all_ids)), "Artifact ID collision detected"

    def test_mixed_sources_correct_total(self, chatgpt_fixture_path, claude_fixture_path):
        """3 ChatGPT + 5 Claude = 8 total artifacts."""
        chatgpt_artifacts = parse_chatgpt(chatgpt_fixture_path)
        claude_artifacts = parse_claude(claude_fixture_path)
        assert len(chatgpt_artifacts) + len(claude_artifacts) == 8
