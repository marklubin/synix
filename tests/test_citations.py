"""Unit tests for the citation URI scheme module."""

from __future__ import annotations

import pytest

from synix.core.citations import (
    Citation,
    extract_citations,
    make_uri,
    parse_uri,
    render_markdown,
)

# ---------------------------------------------------------------------------
# parse_uri tests
# ---------------------------------------------------------------------------


class TestParseUri:
    def test_valid_uri(self):
        c = parse_uri("synix://intel-acme-analytics")
        assert isinstance(c, Citation)
        assert c.uri == "synix://intel-acme-analytics"
        assert c.scheme == "synix"
        assert c.ref == "intel-acme-analytics"

    def test_simple_ref(self):
        c = parse_uri("synix://foo")
        assert c.ref == "foo"
        assert c.scheme == "synix"

    def test_invalid_uri_raises(self):
        with pytest.raises(ValueError, match="Invalid synix URI"):
            parse_uri("invalid")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="Invalid synix URI"):
            parse_uri("")

    def test_wrong_scheme_raises(self):
        with pytest.raises(ValueError, match="Invalid synix URI"):
            parse_uri("http://example.com")

    def test_dots_in_label(self):
        c = parse_uri("synix://acme.analytics")
        assert c.ref == "acme.analytics"

    def test_dots_and_hyphens(self):
        c = parse_uri("synix://intel-acme.v2.1")
        assert c.ref == "intel-acme.v2.1"

    def test_underscores(self):
        c = parse_uri("synix://my_artifact_name")
        assert c.ref == "my_artifact_name"

    def test_missing_ref_raises(self):
        with pytest.raises(ValueError, match="Invalid synix URI"):
            parse_uri("synix://")


# ---------------------------------------------------------------------------
# make_uri tests
# ---------------------------------------------------------------------------


class TestMakeUri:
    def test_basic(self):
        assert make_uri("intel-acme-analytics") == "synix://intel-acme-analytics"

    def test_simple_label(self):
        assert make_uri("foo") == "synix://foo"

    def test_roundtrip_with_parse(self):
        label = "ep-conversation-42"
        uri = make_uri(label)
        c = parse_uri(uri)
        assert c.ref == label

    def test_dotted_label(self):
        assert make_uri("acme.v2") == "synix://acme.v2"


# ---------------------------------------------------------------------------
# render_markdown tests
# ---------------------------------------------------------------------------


class TestRenderMarkdown:
    def test_default_display_text(self):
        result = render_markdown("synix://foo")
        assert result == "[foo](synix://foo)"

    def test_custom_display_text(self):
        result = render_markdown("synix://foo", "custom text")
        assert result == "[custom text](synix://foo)"

    def test_hyphenated_ref(self):
        result = render_markdown("synix://intel-acme-analytics")
        assert result == "[intel-acme-analytics](synix://intel-acme-analytics)"

    def test_dotted_ref(self):
        result = render_markdown("synix://acme.v2")
        assert result == "[acme.v2](synix://acme.v2)"


# ---------------------------------------------------------------------------
# extract_citations tests
# ---------------------------------------------------------------------------


class TestExtractCitations:
    def test_markdown_links(self):
        text = "text [link](synix://a) more [x](synix://b-c) end"
        citations = extract_citations(text)
        assert len(citations) == 2
        refs = [c.ref for c in citations]
        assert "a" in refs
        assert "b-c" in refs

    def test_no_citations(self):
        citations = extract_citations("no citations here")
        assert citations == []

    def test_plain_uri_without_markdown(self):
        citations = extract_citations("plain synix://abc text")
        assert len(citations) == 1
        assert citations[0].ref == "abc"
        assert citations[0].uri == "synix://abc"
        assert citations[0].scheme == "synix"

    def test_multiple_plain_uris(self):
        text = "see synix://first and synix://second-ref for details"
        citations = extract_citations(text)
        assert len(citations) == 2
        refs = [c.ref for c in citations]
        assert "first" in refs
        assert "second-ref" in refs

    def test_empty_string(self):
        assert extract_citations("") == []

    def test_dots_in_label(self):
        """Labels can contain dots (e.g., domain names, version numbers)."""
        text = "see synix://acme.analytics for details"
        citations = extract_citations(text)
        assert len(citations) == 1
        assert citations[0].ref == "acme.analytics"

    def test_dots_and_hyphens_in_label(self):
        text = "[link](synix://intel-acme.v2.1)"
        citations = extract_citations(text)
        assert len(citations) == 1
        assert citations[0].ref == "intel-acme.v2.1"

    def test_mixed_markdown_and_plain(self):
        text = "ref [link](synix://a) and plain synix://b here"
        citations = extract_citations(text)
        assert len(citations) == 2
