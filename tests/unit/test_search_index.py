"""Tests for SQLite FTS5 search index."""

from __future__ import annotations

from synix import Artifact
from synix.artifacts.provenance import ProvenanceTracker
from synix.search.index import SearchIndex


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
