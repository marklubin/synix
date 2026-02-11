"""Tests for text/markdown source adapter and adapter registry."""

from __future__ import annotations

import json
from pathlib import Path

from synix.adapters.registry import get_adapter, get_supported_extensions, parse_file
from synix.adapters.text import parse_text

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "text_samples"


# ---------------------------------------------------------------------------
# Text/Markdown parser tests
# ---------------------------------------------------------------------------


class TestParseTextPlain:
    """Tests for parsing plain text files without frontmatter."""

    def test_plain_text_parse(self):
        """Parse journal.txt — produces one transcript artifact."""
        artifacts = parse_text(FIXTURES_DIR / "journal.txt")
        assert len(artifacts) == 1
        art = artifacts[0]
        assert art.artifact_type == "transcript"
        assert art.metadata["source"] == "text"

    def test_plain_text_artifact_id(self):
        """Artifact ID uses t-text- prefix with sanitized filename stem."""
        artifacts = parse_text(FIXTURES_DIR / "journal.txt")
        assert artifacts[0].artifact_id == "t-text-journal"

    def test_plain_text_content(self):
        """Content preserves original text."""
        artifacts = parse_text(FIXTURES_DIR / "journal.txt")
        content = artifacts[0].content
        assert "expense tracker CLI" in content
        assert "Rust ownership" in content

    def test_plain_text_word_count(self):
        """Word count is populated in metadata."""
        artifacts = parse_text(FIXTURES_DIR / "journal.txt")
        assert artifacts[0].metadata["word_count"] > 50

    def test_plain_text_no_frontmatter_metadata(self):
        """Without frontmatter, title derived from filename, no date."""
        artifacts = parse_text(FIXTURES_DIR / "journal.txt")
        meta = artifacts[0].metadata
        assert meta["title"] == "Journal"
        assert meta["date"] == ""  # no date in filename

    def test_plain_text_no_turns(self):
        """Plain text without turn markers has no has_turns flag."""
        artifacts = parse_text(FIXTURES_DIR / "journal.txt")
        assert "has_turns" not in artifacts[0].metadata

    def test_plain_text_content_hash(self):
        """Content hash is auto-computed by Artifact.__post_init__."""
        artifacts = parse_text(FIXTURES_DIR / "journal.txt")
        assert artifacts[0].content_hash.startswith("sha256:")


class TestParseMarkdownFrontmatter:
    """Tests for parsing markdown files with YAML frontmatter."""

    def test_markdown_with_frontmatter(self):
        """Parse meeting-notes.md — produces one transcript artifact."""
        artifacts = parse_text(FIXTURES_DIR / "2025-01-15-meeting-notes.md")
        assert len(artifacts) == 1

    def test_frontmatter_title(self):
        """Title extracted from frontmatter."""
        artifacts = parse_text(FIXTURES_DIR / "2025-01-15-meeting-notes.md")
        assert artifacts[0].metadata["title"] == "Q1 Planning Meeting Notes"

    def test_frontmatter_date(self):
        """Date extracted from frontmatter."""
        artifacts = parse_text(FIXTURES_DIR / "2025-01-15-meeting-notes.md")
        assert artifacts[0].metadata["date"] == "2025-01-15"

    def test_frontmatter_tags(self):
        """Tags extracted from frontmatter as a list."""
        artifacts = parse_text(FIXTURES_DIR / "2025-01-15-meeting-notes.md")
        tags = artifacts[0].metadata["tags"]
        assert isinstance(tags, list)
        assert "planning" in tags
        assert "engineering" in tags
        assert "q1" in tags

    def test_frontmatter_not_in_content(self):
        """Frontmatter is stripped from the artifact content."""
        artifacts = parse_text(FIXTURES_DIR / "2025-01-15-meeting-notes.md")
        content = artifacts[0].content
        assert "---" not in content.split("\n")[0]
        assert "tags:" not in content

    def test_markdown_content_preserved(self):
        """Markdown body is preserved in content."""
        artifacts = parse_text(FIXTURES_DIR / "2025-01-15-meeting-notes.md")
        content = artifacts[0].content
        assert "# Q1 Planning Meeting" in content
        assert "CockroachDB" in content

    def test_artifact_id_from_filename(self):
        """Artifact ID derived from filename stem."""
        artifacts = parse_text(FIXTURES_DIR / "2025-01-15-meeting-notes.md")
        assert artifacts[0].artifact_id == "t-text-2025-01-15-meeting-notes"


class TestDateInference:
    """Tests for date inference from filename."""

    def test_date_from_filename(self):
        """Date YYYY-MM-DD extracted from filename start."""
        artifacts = parse_text(FIXTURES_DIR / "2025-01-15-meeting-notes.md")
        # Frontmatter date takes precedence — but both match here
        assert artifacts[0].metadata["date"] == "2025-01-15"

    def test_date_from_filename_no_frontmatter(self, tmp_path):
        """When no frontmatter date, fall back to filename date."""
        filepath = tmp_path / "2024-06-20-standup.txt"
        filepath.write_text("We discussed the release timeline.\n")
        artifacts = parse_text(filepath)
        assert artifacts[0].metadata["date"] == "2024-06-20"

    def test_no_date_in_filename(self):
        """Files without date in name have empty date string."""
        artifacts = parse_text(FIXTURES_DIR / "journal.txt")
        assert artifacts[0].metadata["date"] == ""

    def test_title_from_filename_strips_date(self, tmp_path):
        """Title derived from filename strips the leading date."""
        filepath = tmp_path / "2024-06-20-standup-notes.txt"
        filepath.write_text("Discussed sprint progress.\n")
        artifacts = parse_text(filepath)
        assert artifacts[0].metadata["title"] == "Standup Notes"


class TestTurnDetection:
    """Tests for conversation turn detection."""

    def test_conversation_has_turns(self):
        """Conversation with User:/Assistant: markers detected."""
        artifacts = parse_text(FIXTURES_DIR / "conversation.md")
        assert artifacts[0].metadata.get("has_turns") is True

    def test_conversation_message_count(self):
        """Message count reflects number of turn markers."""
        artifacts = parse_text(FIXTURES_DIR / "conversation.md")
        # 2 User + 2 Assistant = 4 turns
        assert artifacts[0].metadata["message_count"] == 4

    def test_human_ai_markers(self, tmp_path):
        """Human: and AI: markers are also detected as turns."""
        filepath = tmp_path / "chat.txt"
        filepath.write_text(
            "Human: What is Python used for\n\n"
            "AI: Python is a general-purpose programming language.\n"
        )
        artifacts = parse_text(filepath)
        assert artifacts[0].metadata.get("has_turns") is True
        assert artifacts[0].metadata["message_count"] == 2

    def test_system_marker_detected(self, tmp_path):
        """System: marker is also detected as a turn."""
        filepath = tmp_path / "system-chat.txt"
        filepath.write_text(
            "System: You are a helpful assistant.\n\n"
            "User: Hello\n\n"
            "Assistant: Hi there\n"
        )
        artifacts = parse_text(filepath)
        assert artifacts[0].metadata["message_count"] == 3

    def test_plain_text_no_turns(self):
        """Plain text without turn markers has no message_count."""
        artifacts = parse_text(FIXTURES_DIR / "journal.txt")
        assert "message_count" not in artifacts[0].metadata


class TestEdgeCases:
    """Edge case tests for the text parser."""

    def test_empty_file(self, tmp_path):
        """Empty file returns no artifacts."""
        filepath = tmp_path / "empty.txt"
        filepath.write_text("")
        artifacts = parse_text(filepath)
        assert len(artifacts) == 0

    def test_whitespace_only_file(self, tmp_path):
        """Whitespace-only file returns no artifacts."""
        filepath = tmp_path / "blank.txt"
        filepath.write_text("   \n\n   \n")
        artifacts = parse_text(filepath)
        assert len(artifacts) == 0

    def test_frontmatter_only(self, tmp_path):
        """File with only frontmatter and no body returns no artifacts."""
        filepath = tmp_path / "meta-only.md"
        filepath.write_text("---\ntitle: Empty\n---\n")
        artifacts = parse_text(filepath)
        assert len(artifacts) == 0

    def test_special_chars_in_filename(self, tmp_path):
        """Special characters in filename are sanitized in artifact ID."""
        filepath = tmp_path / "meeting (draft) v2.txt"
        filepath.write_text("Some meeting notes.\n")
        artifacts = parse_text(filepath)
        art_id = artifacts[0].artifact_id
        assert art_id.startswith("t-text-")
        assert "(" not in art_id
        assert ")" not in art_id
        assert " " not in art_id

    def test_frontmatter_with_quotes(self, tmp_path):
        """Frontmatter values with quotes are parsed correctly."""
        filepath = tmp_path / "quoted.md"
        filepath.write_text(
            '---\ntitle: "My Quoted Title"\ndate: 2025-03-01\n---\n\nSome content here.\n'
        )
        artifacts = parse_text(filepath)
        assert artifacts[0].metadata["title"] == "My Quoted Title"

    def test_tags_as_comma_string(self, tmp_path):
        """Tags specified as comma-separated string are parsed as list."""
        filepath = tmp_path / "comma-tags.md"
        filepath.write_text(
            "---\ntitle: Test\ntags: work, personal, ideas\n---\n\nContent.\n"
        )
        artifacts = parse_text(filepath)
        tags = artifacts[0].metadata["tags"]
        assert isinstance(tags, list)
        assert "work" in tags
        assert "personal" in tags
        assert "ideas" in tags


# ---------------------------------------------------------------------------
# Adapter registry tests
# ---------------------------------------------------------------------------


class TestAdapterRegistry:
    """Tests for the adapter registry and auto-detection."""

    def test_supported_extensions(self):
        """Registry supports .json, .txt, and .md extensions."""
        exts = get_supported_extensions()
        assert ".json" in exts
        assert ".txt" in exts
        assert ".md" in exts

    def test_get_adapter_json(self, tmp_path):
        """get_adapter returns a callable for .json files."""
        adapter = get_adapter(tmp_path / "export.json")
        assert adapter is not None
        assert callable(adapter)

    def test_get_adapter_txt(self, tmp_path):
        """get_adapter returns a callable for .txt files."""
        adapter = get_adapter(tmp_path / "notes.txt")
        assert adapter is not None

    def test_get_adapter_md(self, tmp_path):
        """get_adapter returns a callable for .md files."""
        adapter = get_adapter(tmp_path / "doc.md")
        assert adapter is not None

    def test_get_adapter_unknown(self, tmp_path):
        """get_adapter returns None for unrecognized extensions."""
        adapter = get_adapter(tmp_path / "data.csv")
        assert adapter is None

    def test_parse_file_text(self):
        """parse_file dispatches .txt to text parser."""
        artifacts = parse_file(FIXTURES_DIR / "journal.txt")
        assert len(artifacts) == 1
        assert artifacts[0].metadata["source"] == "text"

    def test_parse_file_md(self):
        """parse_file dispatches .md to text parser."""
        artifacts = parse_file(FIXTURES_DIR / "2025-01-15-meeting-notes.md")
        assert len(artifacts) == 1
        assert artifacts[0].metadata["source"] == "text"

    def test_parse_file_unknown_extension(self, tmp_path):
        """parse_file returns empty list for unknown extensions."""
        filepath = tmp_path / "data.csv"
        filepath.write_text("a,b,c\n1,2,3\n")
        artifacts = parse_file(filepath)
        assert artifacts == []

    def test_parse_file_invalid_json(self, tmp_path):
        """parse_file handles invalid JSON gracefully."""
        filepath = tmp_path / "bad.json"
        filepath.write_text("not valid json {{{")
        artifacts = parse_file(filepath)
        assert artifacts == []


class TestRegistryBackwardCompat:
    """Tests for backward compatibility — JSON files still parsed by chatgpt/claude parsers."""

    def test_chatgpt_json_via_registry(self):
        """ChatGPT JSON export parsed correctly through registry."""
        chatgpt_path = Path(__file__).parent.parent / "synix" / "fixtures" / "chatgpt_export.json"
        artifacts = parse_file(chatgpt_path)
        assert len(artifacts) > 0
        assert all(a.artifact_id.startswith("t-chatgpt-") for a in artifacts)
        assert all(a.metadata["source"] == "chatgpt" for a in artifacts)

    def test_claude_json_via_registry(self):
        """Claude JSON export parsed correctly through registry."""
        claude_path = Path(__file__).parent.parent / "synix" / "fixtures" / "claude_export.json"
        artifacts = parse_file(claude_path)
        assert len(artifacts) > 0
        assert all(a.artifact_id.startswith("t-claude-") for a in artifacts)
        assert all(a.metadata["source"] == "claude" for a in artifacts)

    def test_unrecognized_json_returns_empty(self, tmp_path):
        """JSON file that is neither ChatGPT nor Claude returns empty list."""
        filepath = tmp_path / "random.json"
        filepath.write_text(json.dumps({"foo": "bar"}))
        artifacts = parse_file(filepath)
        assert artifacts == []

    def test_chatgpt_artifact_count_unchanged(self):
        """Same number of artifacts from ChatGPT fixture as direct parser."""
        from synix.adapters.chatgpt import parse_chatgpt

        chatgpt_path = Path(__file__).parent.parent / "synix" / "fixtures" / "chatgpt_export.json"
        direct = parse_chatgpt(chatgpt_path)
        via_registry = parse_file(chatgpt_path)
        assert len(direct) == len(via_registry)

    def test_claude_artifact_count_unchanged(self):
        """Same number of artifacts from Claude fixture as direct parser."""
        from synix.adapters.claude import parse_claude

        claude_path = Path(__file__).parent.parent / "synix" / "fixtures" / "claude_export.json"
        direct = parse_claude(claude_path)
        via_registry = parse_file(claude_path)
        assert len(direct) == len(via_registry)
