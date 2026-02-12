"""Tests for source parsers — ChatGPT and Claude export formats."""

from __future__ import annotations

import json

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

    def test_chatgpt_last_message_date(self, chatgpt_fixture_path):
        """last_message_date derived from max per-message create_time."""
        artifacts = parse_chatgpt(chatgpt_fixture_path)
        first = artifacts[0]
        # conv-gpt-001 last message create_time is 1710500004
        assert first.metadata["last_message_date"]
        # date and last_message_date may differ for long conversations
        assert "last_message_date" in first.metadata
        # All artifacts should have the field
        assert all("last_message_date" in a.metadata for a in artifacts)

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
        empty_conv = [
            {
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
            }
        ]
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

    def test_chatgpt_regeneration_follows_current_node(self, tmp_path):
        """When a response is regenerated, current_node selects the active branch."""
        conv = [
            {
                "id": "conv-regen",
                "title": "Regeneration test",
                "create_time": 1710500000,
                "current_node": "m4",
                "mapping": {
                    "root": {
                        "id": "root",
                        "message": None,
                        "parent": None,
                        "children": ["m1"],
                    },
                    "m1": {
                        "id": "m1",
                        "message": {
                            "id": "m1",
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["Hello"]},
                            "create_time": 1710500001,
                        },
                        "parent": "root",
                        "children": ["m2"],
                    },
                    "m2": {
                        "id": "m2",
                        "message": {
                            "id": "m2",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["First response"]},
                            "create_time": 1710500002,
                        },
                        "parent": "m1",
                        "children": ["m3", "m4"],
                    },
                    "m3": {
                        "id": "m3",
                        "message": {
                            "id": "m3",
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["Abandoned branch"]},
                            "create_time": 1710500003,
                        },
                        "parent": "m2",
                        "children": [],
                    },
                    "m4": {
                        "id": "m4",
                        "message": {
                            "id": "m4",
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["Current branch"]},
                            "create_time": 1710500004,
                        },
                        "parent": "m2",
                        "children": [],
                    },
                },
            }
        ]
        filepath = tmp_path / "regen.json"
        filepath.write_text(json.dumps(conv))

        artifacts = parse_chatgpt(filepath)
        assert len(artifacts) == 1
        content = artifacts[0].content
        assert "Current branch" in content
        assert "Abandoned branch" not in content

    def test_chatgpt_filters_system_and_tool_roles(self, tmp_path):
        """Only user and assistant messages appear in transcript."""
        conv = [
            {
                "id": "conv-roles",
                "title": "Role filtering test",
                "create_time": 1710500000,
                "current_node": "m4",
                "mapping": {
                    "root": {
                        "id": "root",
                        "message": None,
                        "parent": None,
                        "children": ["m1"],
                    },
                    "m1": {
                        "id": "m1",
                        "message": {
                            "id": "m1",
                            "author": {"role": "system"},
                            "content": {"content_type": "text", "parts": ["System prompt text"]},
                            "create_time": 1710500001,
                        },
                        "parent": "root",
                        "children": ["m2"],
                    },
                    "m2": {
                        "id": "m2",
                        "message": {
                            "id": "m2",
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["User question"]},
                            "create_time": 1710500002,
                        },
                        "parent": "m1",
                        "children": ["m3"],
                    },
                    "m3": {
                        "id": "m3",
                        "message": {
                            "id": "m3",
                            "author": {"role": "tool"},
                            "content": {"content_type": "text", "parts": ["Tool output"]},
                            "create_time": 1710500003,
                        },
                        "parent": "m2",
                        "children": ["m4"],
                    },
                    "m4": {
                        "id": "m4",
                        "message": {
                            "id": "m4",
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["Assistant answer"]},
                            "create_time": 1710500004,
                        },
                        "parent": "m3",
                        "children": [],
                    },
                },
            }
        ]
        filepath = tmp_path / "roles.json"
        filepath.write_text(json.dumps(conv))

        artifacts = parse_chatgpt(filepath)
        assert len(artifacts) == 1
        content = artifacts[0].content
        assert "User question" in content
        assert "Assistant answer" in content
        assert "System prompt text" not in content
        assert "Tool output" not in content
        # message_count should only reflect user + assistant
        assert artifacts[0].metadata["message_count"] == 2


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

    def test_claude_last_message_date(self, claude_fixture_path):
        """last_message_date derived from max per-message created_at."""
        artifacts = parse_claude(claude_fixture_path)
        first = artifacts[0]
        # conv-001 last message is 2024-03-15T10:03:00Z
        assert first.metadata["last_message_date"] == "2024-03-15"
        # All artifacts should have the field
        assert all("last_message_date" in a.metadata for a in artifacts)

    def test_claude_artifact_ids_prefixed(self, claude_fixture_path):
        """All artifact IDs have t-claude- prefix."""
        artifacts = parse_claude(claude_fixture_path)
        assert all(a.artifact_id.startswith("t-claude-") for a in artifacts)

    def test_claude_sender_normalized(self, claude_fixture_path):
        """Claude 'human' sender is normalized to 'user' in transcripts."""
        artifacts = parse_claude(claude_fixture_path)
        for artifact in artifacts:
            assert "human:" not in artifact.content, (
                f"Artifact {artifact.artifact_id} contains un-normalized 'human:' sender"
            )
            # All user messages should use "user:" prefix
            lines = artifact.content.strip().split("\n\n")
            for line in lines:
                role = line.split(":")[0]
                assert role in ("user", "assistant"), f"Unexpected role '{role}' in artifact {artifact.artifact_id}"


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
