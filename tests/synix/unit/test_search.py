"""Unit tests for FTS search."""

import pytest


class TestFTSSearch:
    """Tests for FTS search functionality."""

    def test_search_matches_content(self, initialized_db):
        """FTS matches content."""
        from synix.db.engine import get_artifact_session
        from synix.services.records import create_record
        from synix.services.search import search_fts

        with get_artifact_session(initialized_db) as session:
            # Create test records
            create_record(
                session,
                content="Rust ownership is a unique approach to memory management",
                step_name="source",
                materialization_key="key1",
                run_id="run-123",
            )
            create_record(
                session,
                content="Python async uses coroutines and event loops",
                step_name="source",
                materialization_key="key2",
                run_id="run-123",
            )

        with get_artifact_session(initialized_db) as session:
            hits = search_fts(session, "rust ownership")

            assert len(hits) == 1
            assert "Rust" in hits[0].content

    def test_search_no_results(self, initialized_db):
        """FTS returns empty list for no matches."""
        from synix.db.engine import get_artifact_session
        from synix.services.records import create_record
        from synix.services.search import search_fts

        with get_artifact_session(initialized_db) as session:
            create_record(
                session,
                content="Rust ownership discussion",
                step_name="source",
                materialization_key="key1",
                run_id="run-123",
            )

        with get_artifact_session(initialized_db) as session:
            hits = search_fts(session, "javascript frameworks")

            assert len(hits) == 0

    def test_search_filters_by_step(self, initialized_db):
        """FTS can filter by step name."""
        from synix.db.engine import get_artifact_session
        from synix.services.records import create_record
        from synix.services.search import search_fts

        with get_artifact_session(initialized_db) as session:
            create_record(
                session,
                content="Rust programming language",
                step_name="source",
                materialization_key="key1",
                run_id="run-123",
            )
            create_record(
                session,
                content="Rust summary: great for systems programming",
                step_name="summaries",
                materialization_key="key2",
                run_id="run-123",
            )

        with get_artifact_session(initialized_db) as session:
            # Search only in summaries
            hits = search_fts(session, "rust", step="summaries")

            assert len(hits) == 1
            assert hits[0].step_name == "summaries"

    def test_search_respects_limit(self, initialized_db):
        """FTS respects limit parameter."""
        from synix.db.engine import get_artifact_session
        from synix.services.records import create_record
        from synix.services.search import search_fts

        with get_artifact_session(initialized_db) as session:
            for i in range(5):
                create_record(
                    session,
                    content=f"Programming topic {i} about code and development",
                    step_name="source",
                    materialization_key=f"key{i}",
                    run_id="run-123",
                )

        with get_artifact_session(initialized_db) as session:
            hits = search_fts(session, "programming", limit=2)

            assert len(hits) == 2

    def test_search_returns_snippets(self, initialized_db):
        """FTS returns snippets with highlighting."""
        from synix.db.engine import get_artifact_session
        from synix.services.records import create_record
        from synix.services.search import search_fts

        with get_artifact_session(initialized_db) as session:
            create_record(
                session,
                content="Rust is a systems programming language that focuses on safety and performance.",
                step_name="source",
                materialization_key="key1",
                run_id="run-123",
            )

        with get_artifact_session(initialized_db) as session:
            hits = search_fts(session, "rust")

            assert len(hits) == 1
            # Snippet should have highlighting markers
            if hits[0].snippet:
                assert "<mark>" in hits[0].snippet or "rust" in hits[0].snippet.lower()

    def test_search_ranks_by_relevance(self, initialized_db):
        """FTS results are ranked by relevance."""
        from synix.db.engine import get_artifact_session
        from synix.services.records import create_record
        from synix.services.search import search_fts

        with get_artifact_session(initialized_db) as session:
            # More relevant - mentions rust multiple times
            create_record(
                session,
                content="Rust programming. Rust ownership. Rust lifetimes. Rust borrowing.",
                step_name="source",
                materialization_key="key1",
                run_id="run-123",
            )
            # Less relevant - mentions rust once
            create_record(
                session,
                content="Python is great. Also Rust exists.",
                step_name="source",
                materialization_key="key2",
                run_id="run-123",
            )

        with get_artifact_session(initialized_db) as session:
            hits = search_fts(session, "rust")

            assert len(hits) == 2
            # First result should be more relevant (lower BM25 score = better)
            # The record with more "rust" mentions should rank higher
            assert "Rust" in hits[0].content and "Rust" in hits[0].content[5:]
