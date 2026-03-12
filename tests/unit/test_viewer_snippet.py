"""Unit tests for the viewer snippet extraction."""

from synix.viewer._snippet import make_snippet


class TestMakeSnippet:
    def test_extracts_context(self):
        content = "The quick brown fox jumps over the lazy dog"
        result = make_snippet(content, "fox", max_chars=30)
        assert "fox" in result.lower()
        assert len(result) <= 80  # some slack for mark tags

    def test_highlights_terms(self):
        content = "The quick brown fox jumps"
        result = make_snippet(content, "fox")
        assert "<mark>fox</mark>" in result

    def test_no_match_returns_beginning(self):
        content = "The quick brown fox"
        result = make_snippet(content, "zebra", max_chars=10)
        assert result == content[:10]

    def test_multiple_terms(self):
        content = "The quick brown fox jumps over the lazy dog"
        result = make_snippet(content, "fox dog")
        assert "<mark>" in result

    def test_case_insensitive(self):
        content = "Memory systems are important"
        result = make_snippet(content, "MEMORY")
        assert "<mark>" in result

    def test_empty_content(self):
        assert make_snippet("", "query") == ""

    def test_empty_query(self):
        content = "Some content"
        result = make_snippet(content, "")
        assert result == content[:200]
