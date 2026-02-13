"""Tests for the fingerprint module — Fingerprint type, helpers, and transform integration."""

from __future__ import annotations

from synix.build.fingerprint import (
    Fingerprint,
    compute_build_fingerprint,
    compute_digest,
    compute_projection_fingerprint,
    fingerprint_value,
)


class TestFingerprint:
    """Tests for the Fingerprint dataclass."""

    def test_matches_same(self):
        """Identical fingerprints match."""
        fp = Fingerprint(scheme="synix:test:v1", digest="abc123", components={"a": "1"})
        assert fp.matches(fp) is True

    def test_matches_equal(self):
        """Equal but distinct fingerprints match."""
        fp1 = Fingerprint(scheme="synix:test:v1", digest="abc123", components={"a": "1"})
        fp2 = Fingerprint(scheme="synix:test:v1", digest="abc123", components={"a": "1"})
        assert fp1.matches(fp2) is True

    def test_matches_different_digest(self):
        """Different digest means no match."""
        fp1 = Fingerprint(scheme="synix:test:v1", digest="abc123", components={"a": "1"})
        fp2 = Fingerprint(scheme="synix:test:v1", digest="def456", components={"a": "2"})
        assert fp1.matches(fp2) is False

    def test_matches_different_scheme(self):
        """Different scheme means no match, even with same digest."""
        fp1 = Fingerprint(scheme="synix:test:v1", digest="abc123", components={"a": "1"})
        fp2 = Fingerprint(scheme="synix:test:v2", digest="abc123", components={"a": "1"})
        assert fp1.matches(fp2) is False

    def test_matches_none(self):
        """Matching against None returns False."""
        fp = Fingerprint(scheme="synix:test:v1", digest="abc123", components={"a": "1"})
        assert fp.matches(None) is False

    def test_explain_diff_none(self):
        """Diff against None returns 'no stored fingerprint'."""
        fp = Fingerprint(scheme="synix:test:v1", digest="abc123", components={"a": "1"})
        assert fp.explain_diff(None) == ["no stored fingerprint"]

    def test_explain_diff_scheme_changed(self):
        """Diff with scheme mismatch reports scheme change."""
        fp1 = Fingerprint(scheme="synix:test:v2", digest="abc123", components={"a": "1"})
        fp2 = Fingerprint(scheme="synix:test:v1", digest="abc123", components={"a": "1"})
        result = fp1.explain_diff(fp2)
        assert len(result) == 1
        assert "scheme changed" in result[0]

    def test_explain_diff_component_changed(self):
        """Diff identifies which components changed."""
        fp1 = Fingerprint(scheme="synix:test:v1", digest="aaa", components={"source": "new", "prompt": "same"})
        fp2 = Fingerprint(scheme="synix:test:v1", digest="bbb", components={"source": "old", "prompt": "same"})
        result = fp1.explain_diff(fp2)
        assert result == ["source changed"]

    def test_explain_diff_multiple_components(self):
        """Diff reports all changed components."""
        fp1 = Fingerprint(
            scheme="synix:test:v1", digest="aaa", components={"source": "new", "prompt": "new2", "model": "same"}
        )
        fp2 = Fingerprint(
            scheme="synix:test:v1", digest="bbb", components={"source": "old", "prompt": "old2", "model": "same"}
        )
        result = fp1.explain_diff(fp2)
        assert "source changed" in result
        assert "prompt changed" in result
        assert len(result) == 2

    def test_explain_diff_added_component(self):
        """Diff detects added components."""
        fp1 = Fingerprint(scheme="synix:test:v1", digest="aaa", components={"source": "x", "config": "y"})
        fp2 = Fingerprint(scheme="synix:test:v1", digest="bbb", components={"source": "x"})
        result = fp1.explain_diff(fp2)
        assert "config changed" in result

    def test_to_dict_from_dict_roundtrip(self):
        """Serialization roundtrip preserves all fields."""
        fp = Fingerprint(
            scheme="synix:transform:v1",
            digest="abc123def456",
            components={"source": "aaa", "prompt": "bbb"},
        )
        d = fp.to_dict()
        restored = Fingerprint.from_dict(d)
        assert restored is not None
        assert restored.scheme == fp.scheme
        assert restored.digest == fp.digest
        assert restored.components == fp.components
        assert fp.matches(restored)

    def test_from_dict_none_on_empty(self):
        """from_dict returns None for empty/missing data."""
        assert Fingerprint.from_dict({}) is None
        assert Fingerprint.from_dict(None) is None  # type: ignore[arg-type]

    def test_from_dict_none_on_no_scheme(self):
        """from_dict returns None when scheme is missing."""
        assert Fingerprint.from_dict({"digest": "abc"}) is None


class TestComputeDigest:
    """Tests for compute_digest helper."""

    def test_deterministic(self):
        """Same components produce same digest."""
        d1 = compute_digest({"a": "1", "b": "2"})
        d2 = compute_digest({"a": "1", "b": "2"})
        assert d1 == d2

    def test_sorted_order(self):
        """Component order doesn't matter (sorted internally)."""
        d1 = compute_digest({"b": "2", "a": "1"})
        d2 = compute_digest({"a": "1", "b": "2"})
        assert d1 == d2

    def test_different_values(self):
        """Different values produce different digests."""
        d1 = compute_digest({"a": "1"})
        d2 = compute_digest({"a": "2"})
        assert d1 != d2

    def test_different_keys(self):
        """Different keys produce different digests."""
        d1 = compute_digest({"a": "1"})
        d2 = compute_digest({"b": "1"})
        assert d1 != d2


class TestFingerprintValue:
    """Tests for fingerprint_value helper."""

    def test_string(self):
        """Strings produce consistent hashes."""
        h1 = fingerprint_value("hello")
        h2 = fingerprint_value("hello")
        assert h1 == h2
        assert len(h1) == 16  # SHA256 prefix

    def test_string_crlf_normalization(self):
        """CRLF is normalized to LF."""
        h1 = fingerprint_value("hello\r\nworld")
        h2 = fingerprint_value("hello\nworld")
        assert h1 == h2

    def test_string_trailing_whitespace(self):
        """Trailing whitespace is stripped."""
        h1 = fingerprint_value("hello  ")
        h2 = fingerprint_value("hello")
        assert h1 == h2

    def test_dict(self):
        """Dicts are deterministically hashed (sorted keys)."""
        h1 = fingerprint_value({"b": 2, "a": 1})
        h2 = fingerprint_value({"a": 1, "b": 2})
        assert h1 == h2

    def test_list(self):
        """Lists are sorted before hashing."""
        h1 = fingerprint_value(["b", "a"])
        h2 = fingerprint_value(["a", "b"])
        assert h1 == h2

    def test_int(self):
        h1 = fingerprint_value(42)
        h2 = fingerprint_value(42)
        assert h1 == h2

    def test_none(self):
        h = fingerprint_value(None)
        assert len(h) == 16

    def test_different_values_different_hashes(self):
        """Different values produce different hashes."""
        assert fingerprint_value("hello") != fingerprint_value("world")
        assert fingerprint_value(None) != fingerprint_value("something")
        assert fingerprint_value({"a": 1}) != fingerprint_value({"b": 2})

    def test_bool(self):
        h1 = fingerprint_value(True)
        h2 = fingerprint_value(True)
        assert h1 == h2


class TestComputeBuildFingerprint:
    """Tests for compute_build_fingerprint."""

    def test_basic(self):
        """Build fingerprint includes transform digest and input hashes."""
        transform_fp = Fingerprint(
            scheme="synix:transform:v1",
            digest="transform_digest_123",
            components={"source": "aaa"},
        )
        build_fp = compute_build_fingerprint(transform_fp, ["sha256:input1", "sha256:input2"])
        assert build_fp.scheme == "synix:build:v1"
        assert "transform" in build_fp.components
        assert "inputs" in build_fp.components
        assert build_fp.components["transform"] == "transform_digest_123"

    def test_different_inputs_different_fingerprint(self):
        """Different input hashes produce different build fingerprints."""
        transform_fp = Fingerprint(scheme="synix:transform:v1", digest="same", components={})
        fp1 = compute_build_fingerprint(transform_fp, ["sha256:a"])
        fp2 = compute_build_fingerprint(transform_fp, ["sha256:b"])
        assert not fp1.matches(fp2)

    def test_different_transform_different_fingerprint(self):
        """Different transform fingerprints produce different build fingerprints."""
        tfp1 = Fingerprint(scheme="synix:transform:v1", digest="aaa", components={})
        tfp2 = Fingerprint(scheme="synix:transform:v1", digest="bbb", components={})
        fp1 = compute_build_fingerprint(tfp1, ["sha256:same"])
        fp2 = compute_build_fingerprint(tfp2, ["sha256:same"])
        assert not fp1.matches(fp2)


class TestComputeProjectionFingerprint:
    """Tests for compute_projection_fingerprint."""

    def test_basic(self):
        """Projection fingerprint has correct scheme."""
        fp = compute_projection_fingerprint(["sha256:a", "sha256:b"])
        assert fp.scheme == "synix:projection:v1"
        assert "sources" in fp.components

    def test_with_config(self):
        """Config is included when provided."""
        fp = compute_projection_fingerprint(["sha256:a"], config={"key": "val"})
        assert "config" in fp.components

    def test_without_config(self):
        """No config component when config is None."""
        fp = compute_projection_fingerprint(["sha256:a"])
        assert "config" not in fp.components


class TestTransformFingerprint:
    """Tests for BaseTransform.compute_fingerprint()."""

    def test_compute_fingerprint_includes_source(self):
        """Transform fingerprint includes source code hash."""
        # Import to trigger registration
        import synix.build.llm_transforms  # noqa: F401
        import synix.build.parse_transform  # noqa: F401
        from synix.build.transforms import get_transform

        transform = get_transform("episode_summary")
        fp = transform.compute_fingerprint({"llm_config": {"model": "test"}}, "episode_summary")
        assert fp.scheme == "synix:transform:v1"
        assert "source" in fp.components
        assert "prompt" in fp.components
        assert "model" in fp.components

    def test_different_prompt_different_fingerprint(self):
        """Changing prompt template changes the fingerprint."""
        import synix.build.llm_transforms  # noqa: F401
        import synix.build.parse_transform  # noqa: F401
        from synix.build.transforms import get_transform

        transform = get_transform("episode_summary")
        fp1 = transform.compute_fingerprint({"llm_config": {}}, "episode_summary")
        fp2 = transform.compute_fingerprint({"llm_config": {}}, "monthly_rollup")
        assert not fp1.matches(fp2)
        assert fp1.components["prompt"] != fp2.components["prompt"]

    def test_different_model_different_fingerprint(self):
        """Changing model config changes the fingerprint."""
        import synix.build.llm_transforms  # noqa: F401
        import synix.build.parse_transform  # noqa: F401
        from synix.build.transforms import get_transform

        transform = get_transform("episode_summary")
        fp1 = transform.compute_fingerprint({"llm_config": {"model": "a"}}, "episode_summary")
        fp2 = transform.compute_fingerprint({"llm_config": {"model": "b"}}, "episode_summary")
        assert not fp1.matches(fp2)

    def test_same_config_same_fingerprint(self):
        """Identical config produces identical fingerprint (deterministic)."""
        import synix.build.llm_transforms  # noqa: F401
        import synix.build.parse_transform  # noqa: F401
        from synix.build.transforms import get_transform

        transform = get_transform("episode_summary")
        config = {"llm_config": {"model": "test", "temperature": 0.3}}
        fp1 = transform.compute_fingerprint(config, "episode_summary")
        fp2 = transform.compute_fingerprint(config, "episode_summary")
        assert fp1.matches(fp2)

    def test_no_prompt_name_omits_prompt_component(self):
        """Without prompt_name, no prompt component in fingerprint."""
        import synix.build.parse_transform  # noqa: F401
        from synix.build.transforms import get_transform

        transform = get_transform("parse")
        fp = transform.compute_fingerprint({})
        assert "prompt" not in fp.components
