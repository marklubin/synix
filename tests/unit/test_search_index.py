"""Tests for SQLite FTS5 search index and shadow swap."""

from __future__ import annotations

import pytest

from synix import Artifact
from synix.artifacts.provenance import ProvenanceTracker
from synix.search.index import SearchIndex
from synix.search.indexer import ShadowIndexManager, SearchIndexProjection


class TestSearchIndex:
    def test_materialize_and_query(self, tmp_build_dir):
        """Index 10 artifacts, query returns relevant results."""
        index = SearchIndex(tmp_build_dir / "search.db")
        index.create()

        topics = [
            "machine learning and neural networks",
            "Docker containerization and deployment",
            "Rust programming and ownership model",
            "Python web development with Flask",
            "distributed systems and microservices",
            "database optimization with PostgreSQL",
            "cloud computing on AWS",
            "frontend development with React",
            "cybersecurity best practices",
            "Anthropic and Claude AI assistant",
        ]
        for i, topic in enumerate(topics):
            artifact = Artifact(
                artifact_id=f"ep-{i:03d}",
                artifact_type="episode",
                content=f"In this conversation, the user discussed {topic}.",
                metadata={"layer_name": "episodes"},
            )
            index.insert(artifact, "episodes", 1)

        results = index.query("machine learning")
        assert len(results) > 0
        assert "machine learning" in results[0].content.lower()

        results2 = index.query("Anthropic Claude")
        assert len(results2) > 0
        assert "anthropic" in results2[0].content.lower()

        index.close()

    def test_layer_filtering(self, tmp_build_dir):
        """Query with layer filter, only matching layers returned."""
        index = SearchIndex(tmp_build_dir / "search.db")
        index.create()

        index.insert(
            Artifact(artifact_id="ep-001", artifact_type="episode",
                     content="Discussion about Python programming"),
            "episodes", 1,
        )
        index.insert(
            Artifact(artifact_id="monthly-001", artifact_type="rollup",
                     content="Monthly summary covering Python and other topics"),
            "monthly", 2,
        )

        # Filter to episodes only
        results = index.query("Python", layers=["episodes"])
        assert len(results) == 1
        assert results[0].layer_name == "episodes"

        # Filter to monthly only
        results2 = index.query("Python", layers=["monthly"])
        assert len(results2) == 1
        assert results2[0].layer_name == "monthly"

        # No filter — both
        results_all = index.query("Python")
        assert len(results_all) == 2

        index.close()

    def test_provenance_always_included(self, tmp_build_dir):
        """Every search result has provenance_chain field when tracker provided."""
        index = SearchIndex(tmp_build_dir / "search.db")
        index.create()

        index.insert(
            Artifact(artifact_id="ep-001", artifact_type="episode",
                     content="Discussion about AI systems"),
            "episodes", 1,
        )

        tracker = ProvenanceTracker(tmp_build_dir)
        tracker.record("ep-001", parent_ids=["t-001"], prompt_id="ep_v1")
        tracker.record("t-001", parent_ids=[])

        results = index.query("AI", provenance_tracker=tracker)
        assert len(results) > 0
        assert len(results[0].provenance_chain) > 0
        assert "ep-001" in results[0].provenance_chain
        assert "t-001" in results[0].provenance_chain

        index.close()

    def test_ranking(self, tmp_build_dir):
        """More relevant results rank higher."""
        index = SearchIndex(tmp_build_dir / "search.db")
        index.create()

        # One artifact mentions "Rust" many times
        index.insert(
            Artifact(artifact_id="ep-rust", artifact_type="episode",
                     content="Rust Rust Rust. The user loves Rust programming. Rust ownership is great."),
            "episodes", 1,
        )
        # Another mentions it once among other things
        index.insert(
            Artifact(artifact_id="ep-mixed", artifact_type="episode",
                     content="The user discussed Python, JavaScript, and briefly mentioned Rust."),
            "episodes", 1,
        )

        results = index.query("Rust")
        assert len(results) == 2
        # The Rust-heavy article should rank first (lower rank = more relevant)
        assert results[0].artifact_id == "ep-rust"

        index.close()

    def test_empty_query_handling(self, tmp_build_dir):
        """Empty or no-match query returns empty results."""
        index = SearchIndex(tmp_build_dir / "search.db")
        index.create()

        index.insert(
            Artifact(artifact_id="ep-001", artifact_type="episode",
                     content="Discussion about Python programming"),
            "episodes", 1,
        )

        results = index.query("xyznonexistent")
        assert len(results) == 0

        index.close()

    def test_rebuild_replaces_index(self, tmp_build_dir):
        """Materialize twice, second run replaces first cleanly."""
        index = SearchIndex(tmp_build_dir / "search.db")

        # First build
        index.create()
        index.insert(
            Artifact(artifact_id="ep-001", artifact_type="episode",
                     content="First version about Python"),
            "episodes", 1,
        )
        results1 = index.query("Python")
        assert len(results1) == 1

        # Second build (create drops and recreates)
        index.create()
        index.insert(
            Artifact(artifact_id="ep-002", artifact_type="episode",
                     content="Second version about Python and Rust"),
            "episodes", 1,
        )
        results2 = index.query("Python")
        assert len(results2) == 1
        assert results2[0].artifact_id == "ep-002"

        # Old data is gone
        results_old = index.query("First version")
        assert len(results_old) == 0

        index.close()


class TestShadowIndexManager:
    """Tests for shadow index build-and-swap pattern."""

    def test_begin_build_creates_shadow(self, tmp_build_dir):
        """begin_build() creates a shadow index file."""
        manager = ShadowIndexManager(tmp_build_dir)
        shadow = manager.begin_build()

        assert manager.shadow_path.exists()
        shadow.insert(
            Artifact(artifact_id="ep-001", artifact_type="episode",
                     content="Test content"),
            "episodes", 1,
        )
        # Shadow file exists, main does not yet
        assert not manager.main_path.exists()
        manager.rollback()

    def test_commit_swaps_to_main(self, tmp_build_dir):
        """commit() replaces main with shadow atomically."""
        manager = ShadowIndexManager(tmp_build_dir)
        shadow = manager.begin_build()
        shadow.insert(
            Artifact(artifact_id="ep-001", artifact_type="episode",
                     content="Shadow content about Python"),
            "episodes", 1,
        )
        manager.commit()

        # Shadow file gone, main file exists
        assert not manager.shadow_path.exists()
        assert manager.main_path.exists()

        # Main file has the data
        index = SearchIndex(manager.main_path)
        results = index.query("Python")
        assert len(results) == 1
        assert results[0].artifact_id == "ep-001"
        index.close()

    def test_commit_replaces_existing_main(self, tmp_build_dir):
        """commit() replaces an existing main index with the shadow."""
        # Create an initial main index
        old_index = SearchIndex(tmp_build_dir / "search.db")
        old_index.create()
        old_index.insert(
            Artifact(artifact_id="old-001", artifact_type="episode",
                     content="Old data about Rust"),
            "episodes", 1,
        )
        old_index.close()

        # Build a shadow with different content
        manager = ShadowIndexManager(tmp_build_dir)
        shadow = manager.begin_build()
        shadow.insert(
            Artifact(artifact_id="new-001", artifact_type="episode",
                     content="New data about Python"),
            "episodes", 1,
        )
        manager.commit()

        # Main now has the new data
        index = SearchIndex(tmp_build_dir / "search.db")
        results = index.query("Python")
        assert len(results) == 1
        assert results[0].artifact_id == "new-001"

        # Old data is gone
        results_old = index.query("Rust")
        assert len(results_old) == 0
        index.close()

    def test_rollback_preserves_old_index(self, tmp_build_dir):
        """rollback() keeps old main index intact and removes shadow."""
        # Create initial main index
        old_index = SearchIndex(tmp_build_dir / "search.db")
        old_index.create()
        old_index.insert(
            Artifact(artifact_id="old-001", artifact_type="episode",
                     content="Preserved data about Docker"),
            "episodes", 1,
        )
        old_index.close()

        # Start and then rollback a shadow build
        manager = ShadowIndexManager(tmp_build_dir)
        shadow = manager.begin_build()
        shadow.insert(
            Artifact(artifact_id="new-001", artifact_type="episode",
                     content="New data that should be discarded"),
            "episodes", 1,
        )
        manager.rollback()

        # Shadow is gone
        assert not manager.shadow_path.exists()

        # Old main is preserved
        index = SearchIndex(tmp_build_dir / "search.db")
        results = index.query("Docker")
        assert len(results) == 1
        assert results[0].artifact_id == "old-001"
        index.close()

    def test_rollback_without_begin(self, tmp_build_dir):
        """rollback() is safe even without begin_build()."""
        manager = ShadowIndexManager(tmp_build_dir)
        manager.rollback()  # should not raise

    def test_commit_without_begin_raises(self, tmp_build_dir):
        """commit() without begin_build() raises RuntimeError."""
        manager = ShadowIndexManager(tmp_build_dir)
        with pytest.raises(RuntimeError, match="No shadow build"):
            manager.commit()

    def test_stale_shadow_cleaned_on_begin(self, tmp_build_dir):
        """begin_build() removes a stale shadow file from a previous failed build."""
        shadow_path = tmp_build_dir / "search_shadow.db"
        shadow_path.write_text("stale data")

        manager = ShadowIndexManager(tmp_build_dir)
        shadow = manager.begin_build()

        # Stale file was replaced with a valid SQLite DB
        results = shadow.query("anything")
        assert results == []
        manager.rollback()


class TestSearchIndexProjectionShadow:
    """Tests that SearchIndexProjection uses the shadow pattern."""

    def test_materialize_uses_shadow(self, tmp_build_dir):
        """materialize() builds via shadow and swaps atomically."""
        proj = SearchIndexProjection(tmp_build_dir)

        artifacts = [
            Artifact(artifact_id="ep-001", artifact_type="episode",
                     content="Content about machine learning",
                     metadata={"layer_name": "episodes", "layer_level": 1}),
        ]

        proj.materialize(artifacts, {"sources": [{"layer": "episodes", "level": 1}]})

        # Shadow should be gone after successful materialize
        assert not (tmp_build_dir / "search_shadow.db").exists()
        # Main should exist
        assert (tmp_build_dir / "search.db").exists()

        # Query works
        results = proj.query("machine learning")
        assert len(results) == 1
        proj.close()

    def test_materialize_preserves_old_on_failure(self, tmp_build_dir):
        """If materialize fails, old index is preserved."""
        # Create an initial index
        proj = SearchIndexProjection(tmp_build_dir)
        proj.materialize(
            [Artifact(artifact_id="old-001", artifact_type="episode",
                      content="Original data about Rust",
                      metadata={"layer_name": "episodes", "layer_level": 1})],
            {"sources": [{"layer": "episodes", "level": 1}]},
        )
        proj.close()

        # Now simulate a failure by passing an artifact that will cause an error
        # We'll monkeypatch the SearchIndex.insert to fail mid-build
        original_insert = SearchIndex.insert

        call_count = 0
        def failing_insert(self, artifact, layer_name, layer_level):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise RuntimeError("Simulated failure")
            original_insert(self, artifact, layer_name, layer_level)

        SearchIndex.insert = failing_insert
        try:
            proj2 = SearchIndexProjection(tmp_build_dir)
            with pytest.raises(RuntimeError, match="Simulated failure"):
                proj2.materialize(
                    [
                        Artifact(artifact_id="new-001", artifact_type="episode",
                                 content="First new artifact",
                                 metadata={"layer_name": "episodes", "layer_level": 1}),
                        Artifact(artifact_id="new-002", artifact_type="episode",
                                 content="Second new artifact that will fail",
                                 metadata={"layer_name": "episodes", "layer_level": 1}),
                    ],
                    {"sources": [{"layer": "episodes", "level": 1}]},
                )
            proj2.close()
        finally:
            SearchIndex.insert = original_insert

        # Shadow should be cleaned up
        assert not (tmp_build_dir / "search_shadow.db").exists()

        # Old index should still have the original data
        index = SearchIndex(tmp_build_dir / "search.db")
        results = index.query("Rust")
        assert len(results) == 1
        assert results[0].artifact_id == "old-001"
        index.close()

    def test_query_after_materialize(self, tmp_build_dir):
        """Query works correctly after materialize replaces the index."""
        proj = SearchIndexProjection(tmp_build_dir)

        # First build
        proj.materialize(
            [Artifact(artifact_id="ep-001", artifact_type="episode",
                      content="First build data",
                      metadata={"layer_name": "episodes", "layer_level": 1})],
            {"sources": [{"layer": "episodes", "level": 1}]},
        )
        results1 = proj.query("First build")
        assert len(results1) == 1

        # Second build replaces content
        proj.materialize(
            [Artifact(artifact_id="ep-002", artifact_type="episode",
                      content="Second build data",
                      metadata={"layer_name": "episodes", "layer_level": 1})],
            {"sources": [{"layer": "episodes", "level": 1}]},
        )
        results2 = proj.query("Second build")
        assert len(results2) == 1
        assert results2[0].artifact_id == "ep-002"

        # Old data gone
        results_old = proj.query("First build")
        assert len(results_old) == 0

        proj.close()
