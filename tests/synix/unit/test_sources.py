"""Unit tests for source parsers."""

import pytest


class TestClaudeExportSource:
    """Tests for Claude export parser."""

    def test_parse_conversations(self, claude_export_file):
        """Claude export parses conversations."""
        from synix.sources.claude import ClaudeExportSource

        source = ClaudeExportSource(
            name="claude",
            file_path=claude_export_file,
        )

        records = list(source.parse("run-123"))

        assert len(records) == 3
        assert all(r.step_name == "claude" for r in records)

    def test_extracts_metadata(self, claude_export_file):
        """Claude parser extracts conversation metadata."""
        from synix.sources.claude import ClaudeExportSource

        source = ClaudeExportSource(
            name="claude",
            file_path=claude_export_file,
        )

        records = list(source.parse("run-123"))
        record = records[0]

        assert "meta.chat.conversation_id" in record.metadata_
        assert "meta.chat.title" in record.metadata_
        assert "meta.time.created_at" in record.metadata_
        assert record.metadata_["meta.source.type"] == "claude-export"

    def test_formats_messages(self, claude_export_file):
        """Claude parser formats messages into readable content."""
        from synix.sources.claude import ClaudeExportSource

        source = ClaudeExportSource(
            name="claude",
            file_path=claude_export_file,
        )

        records = list(source.parse("run-123"))
        content = records[0].content

        assert "User" in content or "## User" in content
        assert "Assistant" in content or "## Assistant" in content

    def test_materialization_key_unique(self, claude_export_file):
        """Each record has unique materialization key."""
        from synix.sources.claude import ClaudeExportSource

        source = ClaudeExportSource(
            name="claude",
            file_path=claude_export_file,
        )

        records = list(source.parse("run-123"))
        keys = [r.materialization_key for r in records]

        assert len(keys) == len(set(keys))

    def test_missing_file_raises_error(self, tmp_path):
        """Missing file raises FileNotFoundError."""
        from synix.sources.claude import ClaudeExportSource

        source = ClaudeExportSource(
            name="claude",
            file_path=tmp_path / "nonexistent.json",
        )

        with pytest.raises(FileNotFoundError):
            source.validate()


class TestChatGPTExportSource:
    """Tests for ChatGPT export parser."""

    def test_parse_conversations(self, chatgpt_export_file):
        """ChatGPT export parses conversations."""
        from synix.sources.chatgpt import ChatGPTExportSource

        source = ChatGPTExportSource(
            name="chatgpt",
            file_path=chatgpt_export_file,
        )

        records = list(source.parse("run-123"))

        assert len(records) == 2  # Based on fixture
        assert all(r.step_name == "chatgpt" for r in records)

    def test_extracts_metadata(self, chatgpt_export_file):
        """ChatGPT parser extracts conversation metadata."""
        from synix.sources.chatgpt import ChatGPTExportSource

        source = ChatGPTExportSource(
            name="chatgpt",
            file_path=chatgpt_export_file,
        )

        records = list(source.parse("run-123"))
        record = records[0]

        assert "meta.chat.conversation_id" in record.metadata_
        assert "meta.chat.title" in record.metadata_
        assert "meta.time.created_at" in record.metadata_
        assert record.metadata_["meta.source.type"] == "chatgpt-export"

    def test_handles_nested_mapping(self, chatgpt_export_file):
        """ChatGPT parser handles nested mapping structure."""
        from synix.sources.chatgpt import ChatGPTExportSource

        source = ChatGPTExportSource(
            name="chatgpt",
            file_path=chatgpt_export_file,
        )

        records = list(source.parse("run-123"))
        content = records[0].content

        # Should have extracted messages in order
        assert "User" in content
        assert "Assistant" in content

    def test_formats_messages_in_order(self, chatgpt_export_file):
        """ChatGPT parser maintains message order from parent-child structure."""
        from synix.sources.chatgpt import ChatGPTExportSource

        source = ChatGPTExportSource(
            name="chatgpt",
            file_path=chatgpt_export_file,
        )

        records = list(source.parse("run-123"))
        content = records[0].content

        # User message should appear before assistant response
        user_pos = content.find("User")
        asst_pos = content.find("Assistant")

        assert user_pos < asst_pos
