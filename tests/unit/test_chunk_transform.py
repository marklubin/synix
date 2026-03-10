"""Unit tests for the Chunk transform."""

import pytest

from synix.core.models import Artifact, TransformContext
from synix.ext.chunk import Chunk


def _art(label: str, content: str, **meta) -> Artifact:
    """Helper to create a test artifact."""
    return Artifact(label=label, artifact_type="document", content=content, metadata=meta)


class TestFixedChunking:
    def test_basic_fixed_chunks(self):
        chunk = Chunk("c", depends_on=[], chunk_size=10, chunk_overlap=0)
        result = chunk.execute([_art("doc", "0123456789abcdefghij")], TransformContext())
        assert len(result) == 2
        assert result[0].content == "0123456789"
        assert result[1].content == "abcdefghij"

    def test_fixed_chunks_with_overlap(self):
        chunk = Chunk("c", depends_on=[], chunk_size=10, chunk_overlap=3)
        text = "0123456789abcdef"  # 16 chars
        result = chunk.execute([_art("doc", text)], TransformContext())
        # step = 10 - 3 = 7, starts at 0, 7, 14
        assert len(result) == 3
        assert result[0].content == "0123456789"  # 0:10
        assert result[1].content == "789abcdef"  # 7:17 -> 7:16
        assert result[2].content == "ef"  # 14:24 -> 14:16

    def test_short_text_single_chunk(self):
        chunk = Chunk("c", depends_on=[], chunk_size=100, chunk_overlap=20)
        result = chunk.execute([_art("doc", "short")], TransformContext())
        assert len(result) == 1
        assert result[0].content == "short"

    def test_empty_content_returns_one_chunk(self):
        chunk = Chunk("c", depends_on=[], chunk_size=100, chunk_overlap=0)
        result = chunk.execute([_art("doc", "")], TransformContext())
        assert len(result) == 1
        assert result[0].content == ""

    def test_exact_chunk_size(self):
        chunk = Chunk("c", depends_on=[], chunk_size=5, chunk_overlap=0)
        result = chunk.execute([_art("doc", "abcde")], TransformContext())
        assert len(result) == 1
        assert result[0].content == "abcde"


class TestSeparatorChunking:
    def test_paragraph_separator(self):
        chunk = Chunk("c", depends_on=[], separator="\n\n")
        text = "para one\n\npara two\n\npara three"
        result = chunk.execute([_art("doc", text)], TransformContext())
        assert len(result) == 3
        assert result[0].content == "para one"
        assert result[1].content == "para two"
        assert result[2].content == "para three"

    def test_separator_filters_empty_segments(self):
        chunk = Chunk("c", depends_on=[], separator="\n\n")
        text = "para one\n\n\n\npara two"
        result = chunk.execute([_art("doc", text)], TransformContext())
        assert len(result) == 2

    def test_separator_empty_content(self):
        chunk = Chunk("c", depends_on=[], separator="\n\n")
        result = chunk.execute([_art("doc", "")], TransformContext())
        assert len(result) == 1
        assert result[0].content == ""

    def test_separator_no_match(self):
        chunk = Chunk("c", depends_on=[], separator="---")
        result = chunk.execute([_art("doc", "no separators here")], TransformContext())
        assert len(result) == 1
        assert result[0].content == "no separators here"


class TestCustomChunker:
    def test_callable_chunker(self):
        def by_line(text: str) -> list[str]:
            return text.strip().split("\n")

        chunk = Chunk("c", depends_on=[], chunker=by_line)
        result = chunk.execute([_art("doc", "line1\nline2\nline3")], TransformContext())
        assert len(result) == 3
        assert result[0].content == "line1"
        assert result[2].content == "line3"

    def test_callable_returning_empty_falls_back(self):
        """If chunker returns empty list, fall back to [text]."""
        chunk = Chunk("c", depends_on=[], chunker=lambda t: [])
        result = chunk.execute([_art("doc", "something")], TransformContext())
        assert len(result) == 1
        assert result[0].content == "something"


class TestProvenance:
    def test_input_ids_track_source(self):
        source = _art("doc-1", "hello world")
        chunk = Chunk("c", depends_on=[], chunk_size=5, chunk_overlap=0)
        result = chunk.execute([source], TransformContext())
        for art in result:
            assert art.input_ids == [source.artifact_id]


class TestMetadata:
    def test_chunk_metadata_fields(self):
        source = _art("doc-1", "0123456789ab", category="finance")
        chunk = Chunk("c", depends_on=[], chunk_size=5, chunk_overlap=0)
        result = chunk.execute([source], TransformContext())
        assert len(result) == 3
        for i, art in enumerate(result):
            assert art.metadata["source_label"] == "doc-1"
            assert art.metadata["chunk_index"] == i
            assert art.metadata["chunk_total"] == 3
            # Input metadata propagated
            assert art.metadata["category"] == "finance"

    def test_custom_metadata_fn(self):
        def add_meta(inp, idx, total):
            return {"position": "first" if idx == 0 else "rest"}

        chunk = Chunk("c", depends_on=[], chunk_size=100, chunk_overlap=0, metadata_fn=add_meta)
        result = chunk.execute([_art("doc", "content")], TransformContext())
        assert result[0].metadata["position"] == "first"

    def test_metadata_fn_overrides_propagated(self):
        """metadata_fn values override propagated input metadata."""
        source = _art("doc", "content", tag="old")
        chunk = Chunk(
            "c",
            depends_on=[],
            chunk_size=100,
            chunk_overlap=0,
            metadata_fn=lambda inp, i, t: {"tag": "new"},
        )
        result = chunk.execute([source], TransformContext())
        assert result[0].metadata["tag"] == "new"

    def test_metadata_fn_cannot_override_reserved_keys(self):
        """metadata_fn must not overwrite source_label, chunk_index, or chunk_total."""
        chunk = Chunk(
            "c",
            depends_on=[],
            chunk_size=100,
            chunk_overlap=0,
            metadata_fn=lambda inp, i, t: {"source_label": "hijacked"},
        )
        with pytest.raises(ValueError, match="reserved chunk keys"):
            chunk.execute([_art("doc", "content")], TransformContext())


class TestLabels:
    def test_default_labels_are_content_based(self):
        chunk = Chunk("chunks", depends_on=[], chunk_size=5, chunk_overlap=0)
        result = chunk.execute([_art("readme", "0123456789")], TransformContext())
        # Labels use content hash, not position index
        for art in result:
            assert art.label.startswith("chunks-readme-")
            # 8 hex chars from SHA256
            suffix = art.label.split("chunks-readme-")[1]
            assert len(suffix) == 8
            assert all(c in "0123456789abcdef" for c in suffix)

    def test_labels_stable_across_chunk_count_changes(self):
        """Same text produces same label regardless of surrounding chunks."""
        chunk = Chunk("c", depends_on=[], separator="\n\n")
        # Two paragraphs
        r1 = chunk.execute([_art("doc", "para one\n\npara two")], TransformContext())
        # Three paragraphs — para one is still there
        r2 = chunk.execute([_art("doc", "para one\n\npara two\n\npara three")], TransformContext())
        assert r1[0].label == r2[0].label  # "para one" has same label in both
        assert r1[1].label == r2[1].label  # "para two" has same label in both

    def test_duplicate_content_gets_unique_labels(self):
        """Identical chunks get disambiguated labels."""
        chunk = Chunk("c", depends_on=[], separator="---")
        result = chunk.execute([_art("doc", "same---same---same")], TransformContext())
        assert len(result) == 3
        labels = [a.label for a in result]
        assert len(set(labels)) == 3  # all unique

    def test_custom_label_fn(self):
        def my_labels(inp, idx, total):
            return f"{inp.label}_part{idx + 1}of{total}"

        chunk = Chunk("c", depends_on=[], chunk_size=100, chunk_overlap=0, label_fn=my_labels)
        result = chunk.execute([_art("doc", "content")], TransformContext())
        assert result[0].label == "doc_part1of1"

    def test_artifact_type(self):
        chunk = Chunk("c", depends_on=[], chunk_size=100, chunk_overlap=0, artifact_type="passage")
        result = chunk.execute([_art("doc", "content")], TransformContext())
        assert result[0].artifact_type == "passage"


class TestSplit:
    def test_split_one_unit_per_input(self):
        chunk = Chunk("c", depends_on=[], chunk_size=10, chunk_overlap=0)
        inputs = [_art("a", "aaa"), _art("b", "bbb"), _art("c", "ccc")]
        units = chunk.split(inputs, TransformContext())
        assert len(units) == 3
        for (unit_inputs, extras), original in zip(units, inputs):
            assert len(unit_inputs) == 1
            assert unit_inputs[0] is original
            assert extras == {}

    def test_split_empty_inputs_produces_no_units(self):
        """Empty inputs → no units (nothing to chunk, avoids execute([]) crash)."""
        chunk = Chunk("c", depends_on=[], chunk_size=10, chunk_overlap=0)
        units = chunk.split([], TransformContext())
        assert len(units) == 0


class TestCacheKey:
    def test_different_chunk_size_different_key(self):
        c1 = Chunk("c", depends_on=[], chunk_size=100, chunk_overlap=0)
        c2 = Chunk("c", depends_on=[], chunk_size=200, chunk_overlap=0)
        assert c1.get_cache_key({}) != c2.get_cache_key({})

    def test_different_separator_different_key(self):
        c1 = Chunk("c", depends_on=[], separator="\n")
        c2 = Chunk("c", depends_on=[], separator="\n\n")
        assert c1.get_cache_key({}) != c2.get_cache_key({})

    def test_different_chunker_different_key(self):
        c1 = Chunk("c", depends_on=[], chunker=lambda t: t.split("."))
        c2 = Chunk("c", depends_on=[], chunker=lambda t: t.split(","))
        assert c1.get_cache_key({}) != c2.get_cache_key({})

    def test_same_config_same_key(self):
        c1 = Chunk("c", depends_on=[], chunk_size=100, chunk_overlap=20)
        c2 = Chunk("c", depends_on=[], chunk_size=100, chunk_overlap=20)
        assert c1.get_cache_key({}) == c2.get_cache_key({})


class TestValidation:
    def test_chunk_size_zero_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            Chunk("c", depends_on=[], chunk_size=0, chunk_overlap=0)

    def test_chunk_size_negative_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            Chunk("c", depends_on=[], chunk_size=-1, chunk_overlap=0)

    def test_chunk_overlap_negative_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            Chunk("c", depends_on=[], chunk_size=10, chunk_overlap=-1)

    def test_overlap_ge_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap.*must be less than chunk_size"):
            Chunk("c", depends_on=[], chunk_size=10, chunk_overlap=10)

    def test_overlap_gt_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap.*must be less than chunk_size"):
            Chunk("c", depends_on=[], chunk_size=10, chunk_overlap=15)

    def test_custom_chunker_skips_validation(self):
        """When chunker is provided, chunk_size/overlap validation is skipped."""
        Chunk("c", depends_on=[], chunker=lambda t: [t], chunk_size=0, chunk_overlap=0)

    def test_separator_skips_validation(self):
        """When separator is provided, chunk_size/overlap validation is skipped."""
        Chunk("c", depends_on=[], separator="\n", chunk_size=0, chunk_overlap=0)

    def test_empty_separator_raises(self):
        with pytest.raises(ValueError, match="separator must be a non-empty string"):
            Chunk("c", depends_on=[], separator="")


class TestExecuteValidation:
    def test_execute_zero_inputs_raises(self):
        chunk = Chunk("c", depends_on=[], chunk_size=10, chunk_overlap=0)
        with pytest.raises(ValueError, match="expects exactly 1 input"):
            chunk.execute([], TransformContext())

    def test_execute_multiple_inputs_raises(self):
        chunk = Chunk("c", depends_on=[], chunk_size=10, chunk_overlap=0)
        with pytest.raises(ValueError, match="expects exactly 1 input"):
            chunk.execute([_art("a", "aaa"), _art("b", "bbb")], TransformContext())


class TestFingerprintConsistency:
    def test_fingerprint_changes_with_chunker(self):
        """compute_fingerprint must change when chunker callable changes."""
        c1 = Chunk("c", depends_on=[], chunker=lambda t: t.split("."))
        c2 = Chunk("c", depends_on=[], chunker=lambda t: t.split(","))
        fp1 = c1.compute_fingerprint({})
        fp2 = c2.compute_fingerprint({})
        assert fp1.digest != fp2.digest

    def test_fingerprint_and_cache_key_agree_on_identity(self):
        """If cache key differs, fingerprint must also differ (and vice versa)."""
        fn_a = lambda t: t.split(".")  # noqa: E731
        fn_b = lambda t: t.split(",")  # noqa: E731
        c1 = Chunk("c", depends_on=[], chunker=fn_a)
        c2 = Chunk("c", depends_on=[], chunker=fn_b)
        keys_differ = c1.get_cache_key({}) != c2.get_cache_key({})
        fps_differ = c1.compute_fingerprint({}).digest != c2.compute_fingerprint({}).digest
        assert keys_differ == fps_differ


class TestEstimate:
    def test_estimate_output_count(self):
        chunk = Chunk("c", depends_on=[], chunk_size=100, chunk_overlap=0)
        assert chunk.estimate_output_count(10) == 30
