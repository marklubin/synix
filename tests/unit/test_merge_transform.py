"""Tests for the merge transform — similarity grouping with constraints."""

from __future__ import annotations

# Ensure merge transform is registered
import synix.build.merge_transform  # noqa: F401
from synix import Artifact
from synix.build.merge_transform import (
    MergeTransform,
    _build_merge_groups,
    _parse_constraints,
    jaccard_similarity,
)
from synix.build.transforms import get_transform


class TestSimilarityComputation:
    """Tests for the Jaccard similarity function."""

    def test_identical_texts(self):
        """Identical texts have similarity 1.0."""
        text = "The BillingEngine service threw ERR-4401 during checkout"
        assert jaccard_similarity(text, text) == 1.0

    def test_completely_different_texts(self):
        """Completely unrelated texts have very low similarity."""
        text_a = "The BillingEngine service threw ERR-4401 during checkout"
        text_b = "Weather forecast shows sunny skies tomorrow afternoon"
        sim = jaccard_similarity(text_a, text_b)
        assert sim < 0.2

    def test_similar_texts_high_score(self):
        """Texts sharing many words and bigrams get high similarity."""
        text_a = "BillingEngine ERR-4401 error during checkout process week 47"
        text_b = "BillingEngine ERR-4401 failure during checkout process week 47"
        sim = jaccard_similarity(text_a, text_b)
        assert sim >= 0.7

    def test_partially_similar_texts(self):
        """Texts with some overlap get moderate similarity."""
        text_a = "Customer reported billing error with ERR-4401 code"
        text_b = "Customer filed complaint about shipping delay"
        sim = jaccard_similarity(text_a, text_b)
        assert 0.0 < sim < 0.5

    def test_empty_texts_similarity(self):
        """Two empty texts have similarity 1.0."""
        assert jaccard_similarity("", "") == 1.0

    def test_one_empty_text(self):
        """One empty text and one non-empty have similarity 0.0."""
        assert jaccard_similarity("hello world", "") == 0.0
        assert jaccard_similarity("", "hello world") == 0.0

    def test_symmetry(self):
        """Similarity is symmetric: sim(a, b) == sim(b, a)."""
        a = "BillingEngine threw ERR-4401 during checkout"
        b = "Checkout failed with ERR-4401 from BillingEngine"
        assert jaccard_similarity(a, b) == jaccard_similarity(b, a)


class TestConstraintParsing:
    """Tests for constraint string parsing."""

    def test_parse_different_customer_id(self):
        """Standard constraint string extracts field name."""
        constraints = ["NEVER merge records with different customer_id"]
        fields = _parse_constraints(constraints)
        assert fields == ["customer_id"]

    def test_parse_distinct_field(self):
        """'distinct' keyword also works."""
        constraints = ["Do not merge records with distinct account_type"]
        fields = _parse_constraints(constraints)
        assert fields == ["account_type"]

    def test_parse_multiple_constraints(self):
        """Multiple constraints extract multiple fields."""
        constraints = [
            "NEVER merge records with different customer_id",
            "NEVER merge records with different region",
        ]
        fields = _parse_constraints(constraints)
        assert fields == ["customer_id", "region"]

    def test_parse_no_match(self):
        """Constraint without 'different'/'distinct' pattern returns empty."""
        constraints = ["Always merge records together"]
        fields = _parse_constraints(constraints)
        assert fields == []

    def test_parse_empty_constraints(self):
        """Empty list returns empty."""
        assert _parse_constraints([]) == []


class TestMergeTransformRegistration:
    """Tests for transform registry integration."""

    def test_merge_transform_registered(self):
        """MergeTransform is available via the registry."""
        transform = get_transform("merge")
        assert isinstance(transform, MergeTransform)


class TestMergeSimilarArtifacts:
    """Tests for merging similar artifacts."""

    def test_merge_similar_artifacts(self):
        """Two artifacts with very similar content get merged."""
        art_a = Artifact(
            label="extract-conv_8834",
            artifact_type="extract",
            content="BillingEngine threw ERR-4401 during checkout process week 47. "
            "Customer reported unable to complete payment transaction.",
            metadata={"customer_id": "alice", "week": "47"},
        )
        art_b = Artifact(
            label="extract-conv_8891",
            artifact_type="extract",
            content="BillingEngine threw ERR-4401 during checkout process week 47. "
            "Customer experienced payment failure during transaction.",
            metadata={"customer_id": "bob", "week": "47"},
        )

        transform = get_transform("merge")
        results = transform.execute(
            [art_a, art_b],
            {
                "similarity_threshold": 0.5,
                "constraints": [],
            },
        )

        # Should merge into one artifact
        assert len(results) == 1
        merged = results[0]
        assert merged.artifact_type == "merge"
        assert merged.label.startswith("merge-")
        assert "extract-conv_8834" in merged.content
        assert "extract-conv_8891" in merged.content

    def test_no_merge_below_threshold(self):
        """Dissimilar artifacts stay separate."""
        art_a = Artifact(
            label="extract-conv_001",
            artifact_type="extract",
            content="BillingEngine threw ERR-4401 during checkout process",
            metadata={"customer_id": "alice"},
        )
        art_b = Artifact(
            label="extract-conv_002",
            artifact_type="extract",
            content="Weather forecast shows sunny skies tomorrow afternoon",
            metadata={"customer_id": "bob"},
        )

        transform = get_transform("merge")
        results = transform.execute(
            [art_a, art_b],
            {
                "similarity_threshold": 0.85,
                "constraints": [],
            },
        )

        # Should remain separate
        assert len(results) == 2

    def test_constraint_prevents_merge(self):
        """customer_id constraint keeps different customers apart even if similar."""
        art_a = Artifact(
            label="extract-conv_8834",
            artifact_type="extract",
            content="BillingEngine threw ERR-4401 during checkout process week 47. Customer reported payment failure.",
            metadata={"customer_id": "alice", "week": "47"},
        )
        art_b = Artifact(
            label="extract-conv_8891",
            artifact_type="extract",
            content="BillingEngine threw ERR-4401 during checkout process week 47. Customer reported payment failure.",
            metadata={"customer_id": "bob", "week": "47"},
        )

        transform = get_transform("merge")
        results = transform.execute(
            [art_a, art_b],
            {
                "similarity_threshold": 0.5,
                "constraints": ["NEVER merge records with different customer_id"],
            },
        )

        # Should stay separate despite high similarity
        assert len(results) == 2
        # Each should be passed through as-is (singletons)
        labels = {r.label for r in results}
        assert "extract-conv_8834" in labels
        assert "extract-conv_8891" in labels

    def test_threshold_affects_grouping(self):
        """Lower threshold = more merging, higher threshold = less merging."""
        # Create artifacts with moderate similarity
        art_a = Artifact(
            label="a",
            artifact_type="extract",
            content="BillingEngine error ERR-4401 during checkout process failed payment",
            metadata={},
        )
        art_b = Artifact(
            label="b",
            artifact_type="extract",
            content="BillingEngine error ERR-4401 during payment checkout issue reported",
            metadata={},
        )

        transform = get_transform("merge")

        # With low threshold, they merge
        results_low = transform.execute(
            [art_a, art_b],
            {
                "similarity_threshold": 0.3,
                "constraints": [],
            },
        )

        # With very high threshold, they stay separate
        results_high = transform.execute(
            [art_a, art_b],
            {
                "similarity_threshold": 0.99,
                "constraints": [],
            },
        )

        assert len(results_low) <= len(results_high)
        # Low threshold should merge them
        assert len(results_low) == 1
        # Very high threshold keeps them apart
        assert len(results_high) == 2


class TestMergeMetadata:
    """Tests for merge artifact metadata tracking."""

    def test_merge_metadata_tracks_sources(self):
        """Merged artifact metadata includes source artifact IDs and customer IDs."""
        art_a = Artifact(
            label="extract-conv_100",
            artifact_type="extract",
            content="Same product same error same week BillingEngine ERR-4401",
            metadata={"customer_id": "alice"},
        )
        art_b = Artifact(
            label="extract-conv_200",
            artifact_type="extract",
            content="Same product same error same week BillingEngine ERR-4401",
            metadata={"customer_id": "bob"},
        )

        transform = get_transform("merge")
        results = transform.execute(
            [art_a, art_b],
            {
                "similarity_threshold": 0.5,
                "constraints": [],
            },
        )

        assert len(results) == 1
        merged = results[0]
        assert "extract-conv_100" in merged.metadata["source_labels"]
        assert "extract-conv_200" in merged.metadata["source_labels"]
        assert set(merged.metadata["source_customer_ids"]) == {"alice", "bob"}
        assert merged.metadata["merge_count"] == 2
        assert 0.0 <= merged.metadata["similarity_score"] <= 1.0

    def test_singleton_groups_pass_through(self):
        """Artifacts with no similar peers are returned as-is (not wrapped)."""
        art_a = Artifact(
            label="extract-conv_001",
            artifact_type="extract",
            content="BillingEngine threw ERR-4401",
            metadata={"customer_id": "alice"},
        )
        art_b = Artifact(
            label="extract-conv_002",
            artifact_type="extract",
            content="Completely unrelated shipping delay weather topic",
            metadata={"customer_id": "bob"},
        )

        transform = get_transform("merge")
        results = transform.execute(
            [art_a, art_b],
            {
                "similarity_threshold": 0.85,
                "constraints": [],
            },
        )

        assert len(results) == 2
        # Singletons should retain their original label and type
        result_labels = {r.label for r in results}
        assert "extract-conv_001" in result_labels
        assert "extract-conv_002" in result_labels
        for r in results:
            assert r.artifact_type == "extract"  # original type preserved

    def test_merge_input_ids(self):
        """Merged artifact tracks artifact IDs of all source artifacts."""
        art_a = Artifact(
            label="a",
            artifact_type="extract",
            content="Same exact content for merge test purposes here",
            metadata={},
        )
        art_b = Artifact(
            label="b",
            artifact_type="extract",
            content="Same exact content for merge test purposes here",
            metadata={},
        )

        transform = get_transform("merge")
        results = transform.execute(
            [art_a, art_b],
            {
                "similarity_threshold": 0.5,
                "constraints": [],
            },
        )

        assert len(results) == 1
        merged = results[0]
        assert len(merged.input_ids) == 2
        assert art_a.artifact_id in merged.input_ids
        assert art_b.artifact_id in merged.input_ids


class TestMergeCacheKey:
    """Tests for cache key computation."""

    def test_cache_key_includes_threshold(self):
        """Changing similarity_threshold changes the cache key."""
        transform = get_transform("merge")
        key_85 = transform.get_cache_key({"similarity_threshold": 0.85})
        key_92 = transform.get_cache_key({"similarity_threshold": 0.92})
        assert key_85 != key_92

    def test_cache_key_includes_constraints(self):
        """Changing constraints changes the cache key."""
        transform = get_transform("merge")
        key_no_constraint = transform.get_cache_key(
            {
                "similarity_threshold": 0.85,
                "constraints": [],
            }
        )
        key_with_constraint = transform.get_cache_key(
            {
                "similarity_threshold": 0.85,
                "constraints": ["NEVER merge records with different customer_id"],
            }
        )
        assert key_no_constraint != key_with_constraint

    def test_cache_key_deterministic(self):
        """Same config produces same cache key."""
        transform = get_transform("merge")
        config = {
            "similarity_threshold": 0.85,
            "constraints": ["NEVER merge records with different customer_id"],
        }
        key1 = transform.get_cache_key(config)
        key2 = transform.get_cache_key(config)
        assert key1 == key2

    def test_cache_key_constraint_order_independent(self):
        """Constraints in different order produce same cache key (they are sorted)."""
        transform = get_transform("merge")
        key_a = transform.get_cache_key(
            {
                "constraints": ["no different customer_id", "no different region"],
            }
        )
        key_b = transform.get_cache_key(
            {
                "constraints": ["no different region", "no different customer_id"],
            }
        )
        assert key_a == key_b


class TestMergeEdgeCases:
    """Tests for edge cases."""

    def test_empty_inputs(self):
        """Empty input list returns empty output."""
        transform = get_transform("merge")
        results = transform.execute([], {"similarity_threshold": 0.85})
        assert results == []

    def test_single_input(self):
        """Single artifact passes through unchanged."""
        art = Artifact(
            label="extract-only",
            artifact_type="extract",
            content="Some content here",
            metadata={"customer_id": "alice"},
        )
        transform = get_transform("merge")
        results = transform.execute([art], {"similarity_threshold": 0.85})
        assert len(results) == 1
        assert results[0].label == "extract-only"

    def test_three_way_merge(self):
        """Three highly similar artifacts can merge into one group."""
        arts = [
            Artifact(
                label=f"extract-{i}",
                artifact_type="extract",
                content="BillingEngine ERR-4401 checkout failure payment process week 47",
                metadata={"customer_id": f"customer_{i}"},
            )
            for i in range(3)
        ]

        transform = get_transform("merge")
        results = transform.execute(
            arts,
            {
                "similarity_threshold": 0.5,
                "constraints": [],
            },
        )

        # All three should merge into one
        assert len(results) == 1
        assert results[0].metadata["merge_count"] == 3

    def test_constraint_with_missing_metadata_field(self):
        """Constraint on a field that doesn't exist in metadata doesn't block merge."""
        art_a = Artifact(
            label="a",
            artifact_type="extract",
            content="Same content for test BillingEngine ERR-4401",
            metadata={},  # No customer_id
        )
        art_b = Artifact(
            label="b",
            artifact_type="extract",
            content="Same content for test BillingEngine ERR-4401",
            metadata={},  # No customer_id
        )

        transform = get_transform("merge")
        results = transform.execute(
            [art_a, art_b],
            {
                "similarity_threshold": 0.5,
                "constraints": ["NEVER merge records with different customer_id"],
            },
        )

        # Missing field doesn't violate constraint, so they should merge
        assert len(results) == 1

    def test_default_threshold(self):
        """Default threshold is 0.85 when not specified."""
        transform = get_transform("merge")
        # Create artifacts with identical content (similarity = 1.0 > 0.85 default)
        arts = [
            Artifact(label="a", artifact_type="extract", content="Exact same content here for testing", metadata={}),
            Artifact(label="b", artifact_type="extract", content="Exact same content here for testing", metadata={}),
        ]
        results = transform.execute(arts, {})
        assert len(results) == 1  # merged with default threshold


class TestBuildMergeGroups:
    """Tests for the internal _build_merge_groups function."""

    def test_all_similar_one_group(self):
        """All similar artifacts form one group."""
        arts = [
            Artifact(label=f"a{i}", artifact_type="x", content="Same content repeated here", metadata={})
            for i in range(4)
        ]
        groups = _build_merge_groups(arts, threshold=0.5, constraint_fields=[])
        assert len(groups) == 1
        assert len(groups[0]) == 4

    def test_two_distinct_groups(self):
        """Two clusters of similar artifacts form two groups."""
        cluster_a = [
            Artifact(
                label=f"billing-{i}",
                artifact_type="x",
                content="BillingEngine ERR-4401 checkout failure payment",
                metadata={},
            )
            for i in range(2)
        ]
        cluster_b = [
            Artifact(
                label=f"shipping-{i}",
                artifact_type="x",
                content="Shipping delay warehouse logistics tracking number",
                metadata={},
            )
            for i in range(2)
        ]

        groups = _build_merge_groups(cluster_a + cluster_b, threshold=0.5, constraint_fields=[])
        # Should form 2 groups
        assert len(groups) == 2

    def test_empty_input(self):
        """Empty input returns empty groups."""
        assert _build_merge_groups([], threshold=0.5, constraint_fields=[]) == []
