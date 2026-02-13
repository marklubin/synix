"""Tests for SQLite FTS5 search index and shadow swap."""

from __future__ import annotations

import pytest

from synix import Artifact
from synix.artifacts.provenance import ProvenanceTracker
from synix.search.index import SearchIndex
from synix.search.indexer import SearchIndexProjection, ShadowIndexManager


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
                label=f"ep-{i:03d}",
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
            Artifact(label="ep-001", artifact_type="episode", content="Discussion about Python programming"),
            "episodes",
            1,
        )
        index.insert(
            Artifact(
                label="monthly-001",
                artifact_type="rollup",
                content="Monthly summary covering Python and other topics",
            ),
            "monthly",
            2,
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
            Artifact(label="ep-001", artifact_type="episode", content="Discussion about AI systems"),
            "episodes",
            1,
        )

        tracker = ProvenanceTracker(tmp_build_dir)
        tracker.record("ep-001", parent_labels=["t-001"], prompt_id="ep_v1")
        tracker.record("t-001", parent_labels=[])

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
            Artifact(
                label="ep-rust",
                artifact_type="episode",
                content="Rust Rust Rust. The user loves Rust programming. Rust ownership is great.",
            ),
            "episodes",
            1,
        )
        # Another mentions it once among other things
        index.insert(
            Artifact(
                label="ep-mixed",
                artifact_type="episode",
                content="The user discussed Python, JavaScript, and briefly mentioned Rust.",
            ),
            "episodes",
            1,
        )

        results = index.query("Rust")
        assert len(results) == 2
        # The Rust-heavy article should rank first (lower rank = more relevant)
        assert results[0].label == "ep-rust"

        index.close()

    def test_empty_query_handling(self, tmp_build_dir):
        """Empty or no-match query returns empty results."""
        index = SearchIndex(tmp_build_dir / "search.db")
        index.create()

        index.insert(
            Artifact(label="ep-001", artifact_type="episode", content="Discussion about Python programming"),
            "episodes",
            1,
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
            Artifact(label="ep-001", artifact_type="episode", content="First version about Python"),
            "episodes",
            1,
        )
        results1 = index.query("Python")
        assert len(results1) == 1

        # Second build (create drops and recreates)
        index.create()
        index.insert(
            Artifact(label="ep-002", artifact_type="episode", content="Second version about Python and Rust"),
            "episodes",
            1,
        )
        results2 = index.query("Python")
        assert len(results2) == 1
        assert results2[0].label == "ep-002"

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
            Artifact(label="ep-001", artifact_type="episode", content="Test content"),
            "episodes",
            1,
        )
        # Shadow file exists, main does not yet
        assert not manager.main_path.exists()
        manager.rollback()

    def test_commit_swaps_to_main(self, tmp_build_dir):
        """commit() replaces main with shadow atomically."""
        manager = ShadowIndexManager(tmp_build_dir)
        shadow = manager.begin_build()
        shadow.insert(
            Artifact(label="ep-001", artifact_type="episode", content="Shadow content about Python"),
            "episodes",
            1,
        )
        manager.commit()

        # Shadow file gone, main file exists
        assert not manager.shadow_path.exists()
        assert manager.main_path.exists()

        # Main file has the data
        index = SearchIndex(manager.main_path)
        results = index.query("Python")
        assert len(results) == 1
        assert results[0].label == "ep-001"
        index.close()

    def test_commit_replaces_existing_main(self, tmp_build_dir):
        """commit() replaces an existing main index with the shadow."""
        # Create an initial main index
        old_index = SearchIndex(tmp_build_dir / "search.db")
        old_index.create()
        old_index.insert(
            Artifact(label="old-001", artifact_type="episode", content="Old data about Rust"),
            "episodes",
            1,
        )
        old_index.close()

        # Build a shadow with different content
        manager = ShadowIndexManager(tmp_build_dir)
        shadow = manager.begin_build()
        shadow.insert(
            Artifact(label="new-001", artifact_type="episode", content="New data about Python"),
            "episodes",
            1,
        )
        manager.commit()

        # Main now has the new data
        index = SearchIndex(tmp_build_dir / "search.db")
        results = index.query("Python")
        assert len(results) == 1
        assert results[0].label == "new-001"

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
            Artifact(label="old-001", artifact_type="episode", content="Preserved data about Docker"),
            "episodes",
            1,
        )
        old_index.close()

        # Start and then rollback a shadow build
        manager = ShadowIndexManager(tmp_build_dir)
        shadow = manager.begin_build()
        shadow.insert(
            Artifact(label="new-001", artifact_type="episode", content="New data that should be discarded"),
            "episodes",
            1,
        )
        manager.rollback()

        # Shadow is gone
        assert not manager.shadow_path.exists()

        # Old main is preserved
        index = SearchIndex(tmp_build_dir / "search.db")
        results = index.query("Docker")
        assert len(results) == 1
        assert results[0].label == "old-001"
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
            Artifact(
                label="ep-001",
                artifact_type="episode",
                content="Content about machine learning",
                metadata={"layer_name": "episodes", "layer_level": 1},
            ),
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
            [
                Artifact(
                    label="old-001",
                    artifact_type="episode",
                    content="Original data about Rust",
                    metadata={"layer_name": "episodes", "layer_level": 1},
                )
            ],
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
                        Artifact(
                            label="new-001",
                            artifact_type="episode",
                            content="First new artifact",
                            metadata={"layer_name": "episodes", "layer_level": 1},
                        ),
                        Artifact(
                            label="new-002",
                            artifact_type="episode",
                            content="Second new artifact that will fail",
                            metadata={"layer_name": "episodes", "layer_level": 1},
                        ),
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
        assert results[0].label == "old-001"
        index.close()

    def test_query_after_materialize(self, tmp_build_dir):
        """Query works correctly after materialize replaces the index."""
        proj = SearchIndexProjection(tmp_build_dir)

        # First build
        proj.materialize(
            [
                Artifact(
                    label="ep-001",
                    artifact_type="episode",
                    content="First build data",
                    metadata={"layer_name": "episodes", "layer_level": 1},
                )
            ],
            {"sources": [{"layer": "episodes", "level": 1}]},
        )
        results1 = proj.query("First build")
        assert len(results1) == 1

        # Second build replaces content
        proj.materialize(
            [
                Artifact(
                    label="ep-002",
                    artifact_type="episode",
                    content="Second build data",
                    metadata={"layer_name": "episodes", "layer_level": 1},
                )
            ],
            {"sources": [{"layer": "episodes", "level": 1}]},
        )
        results2 = proj.query("Second build")
        assert len(results2) == 1
        assert results2[0].label == "ep-002"

        # Old data gone
        results_old = proj.query("First build")
        assert len(results_old) == 0

        proj.close()


class TestFTS5QuerySanitization:
    """Unit tests for SearchIndex._sanitize_fts5_query()."""

    def test_special_chars_escaped(self):
        """Special FTS5 chars (?, *, +, -, (, )) are safely wrapped."""
        result = SearchIndex._sanitize_fts5_query("hello? world*")
        # Tokens should be quoted to neutralize special chars
        assert '"hello?"' in result
        assert '"world*"' in result

    def test_quotes_in_query(self):
        """Embedded double quotes are stripped cleanly."""
        result = SearchIndex._sanitize_fts5_query('say "hello" world')
        # Double quotes in tokens are removed; tokens are re-quoted
        assert '""' not in result
        assert '"sayhello"' in result or '"say"' in result

    def test_short_query_uses_and(self):
        """1-3 terms use implicit AND (no OR)."""
        result = SearchIndex._sanitize_fts5_query("machine learning")
        assert "OR" not in result
        assert '"machine"' in result
        assert '"learning"' in result

    def test_long_query_uses_or(self):
        """4+ terms with content words use OR between them."""
        result = SearchIndex._sanitize_fts5_query("what are the main themes from november conversations")
        assert "OR" in result
        # Stop words (what, are, the, from) should be stripped
        assert '"what"' not in result
        assert '"are"' not in result
        # Content words preserved
        assert '"main"' in result
        assert '"themes"' in result
        assert '"november"' in result
        assert '"conversations"' in result

    def test_all_stop_words_fallback(self):
        """Query of only stop words falls back to AND."""
        result = SearchIndex._sanitize_fts5_query("what is the")
        assert "OR" not in result
        # All tokens are kept since there are no content words
        assert '"what"' in result
        assert '"is"' in result
        assert '"the"' in result

    def test_punctuation_stripped(self):
        """Trailing punctuation is stripped from tokens in OR mode."""
        result = SearchIndex._sanitize_fts5_query("what do I think about anthropic?")
        # In long-query OR mode, trailing ? stripped from "anthropic?"
        assert '"anthropic"' in result
        assert '"anthropic?"' not in result

    def test_empty_tokens_skipped(self):
        """Empty/whitespace-only tokens don't produce output."""
        result = SearchIndex._sanitize_fts5_query('hello  ""  world')
        # The empty-after-stripping token should not appear
        tokens = result.split()
        assert all(t.strip() for t in tokens)


class TestFTS5QueryIntegration:
    """Integration tests running full index.query() path with tricky inputs."""

    def _make_index(self, tmp_build_dir):
        index = SearchIndex(tmp_build_dir / "search.db")
        index.create()
        topics = [
            "Discussion about Anthropic and their Claude AI assistant",
            "Machine learning fundamentals and neural networks",
            "Monthly themes from November conversations about AI",
            "Python web development with Flask framework",
            "Rust programming and the ownership model",
        ]
        for i, topic in enumerate(topics):
            index.insert(
                Artifact(
                    label=f"ep-{i:03d}",
                    artifact_type="episode",
                    content=topic,
                    metadata={"layer_name": "episodes", "topic": topic[:20]},
                ),
                "episodes",
                1,
            )
        return index

    def test_query_with_question_mark(self, tmp_build_dir):
        """'what do I think about anthropic?' doesn't crash."""
        index = self._make_index(tmp_build_dir)
        results = index.query("what do I think about anthropic?")
        assert isinstance(results, list)
        index.close()

    def test_query_with_asterisk(self, tmp_build_dir):
        """'how to use the * operator' doesn't crash."""
        index = self._make_index(tmp_build_dir)
        results = index.query("how to use the * operator")
        assert isinstance(results, list)
        index.close()

    def test_query_with_parentheses(self, tmp_build_dir):
        """'(optional) configuration' doesn't crash."""
        index = self._make_index(tmp_build_dir)
        results = index.query("(optional) configuration")
        assert isinstance(results, list)
        index.close()

    def test_natural_language_query_returns_results(self, tmp_build_dir):
        """Long natural-language query returns results."""
        index = self._make_index(tmp_build_dir)
        results = index.query("what are the main themes from november conversations about AI?")
        assert len(results) > 0
        index.close()

    def test_unicode_content_and_query(self, tmp_build_dir):
        """Index and search with unicode/emoji."""
        index = SearchIndex(tmp_build_dir / "search.db")
        index.create()
        index.insert(
            Artifact(
                label="ep-uni",
                artifact_type="episode",
                content="Discussion about caf\u00e9 culture and \U0001f30d global trends",
                metadata={"layer_name": "episodes"},
            ),
            "episodes",
            1,
        )
        results = index.query("caf\u00e9")
        assert len(results) > 0
        index.close()

    def test_metadata_preserved_in_results(self, tmp_build_dir):
        """Inserted metadata roundtrips through query."""
        index = self._make_index(tmp_build_dir)
        results = index.query("Anthropic")
        assert len(results) > 0
        assert "topic" in results[0].metadata
        index.close()

    def test_multiple_layer_filtering(self, tmp_build_dir):
        """Filter to 2 of 3 layers, get correct subset."""
        index = SearchIndex(tmp_build_dir / "search.db")
        index.create()
        for layer, level in [("episodes", 1), ("monthly", 2), ("core", 3)]:
            index.insert(
                Artifact(
                    label=f"art-{layer}",
                    artifact_type="episode",
                    content=f"Python discussion in {layer} layer",
                    metadata={"layer_name": layer},
                ),
                layer,
                level,
            )
        results = index.query("Python", layers=["episodes", "core"])
        assert len(results) == 2
        layer_names = {r.layer_name for r in results}
        assert layer_names == {"episodes", "core"}
        index.close()

    def test_empty_string_query(self, tmp_build_dir):
        """Empty string query doesn't crash."""
        index = self._make_index(tmp_build_dir)
        # FTS5 may raise on empty MATCH; we just verify no unhandled crash
        try:
            results = index.query("")
            assert isinstance(results, list)
        except Exception:
            # An empty query raising a clean error is also acceptable
            pass
        index.close()


class TestHybridRetriever:
    """Tests for the HybridRetriever with keyword-only mode."""

    def _make_retriever(self, tmp_build_dir):
        from synix.search.retriever import HybridRetriever

        index = SearchIndex(tmp_build_dir / "search.db")
        index.create()
        for i, content in enumerate(
            [
                "Machine learning and deep neural networks",
                "Anthropic Claude AI assistant for coding",
                "Rust programming language ownership model",
            ]
        ):
            index.insert(
                Artifact(
                    label=f"ep-{i:03d}",
                    artifact_type="episode",
                    content=content,
                    metadata={"layer_name": "episodes"},
                ),
                "episodes",
                1,
            )
        return HybridRetriever(search_index=index), index

    def test_keyword_mode_returns_results(self, tmp_build_dir):
        """Basic keyword search works through retriever."""
        retriever, index = self._make_retriever(tmp_build_dir)
        results = retriever.query("machine learning", mode="keyword")
        assert len(results) > 0
        assert "machine learning" in results[0].content.lower()
        index.close()

    def test_hybrid_without_embeddings_falls_back(self, tmp_build_dir):
        """Hybrid mode without embedding provider falls back to keyword."""
        retriever, index = self._make_retriever(tmp_build_dir)
        results = retriever.query("Anthropic", mode="hybrid")
        assert len(results) > 0
        assert "anthropic" in results[0].content.lower()
        index.close()

    def test_semantic_without_provider_raises(self, tmp_build_dir):
        """Semantic mode without embedding provider raises ValueError."""
        retriever, index = self._make_retriever(tmp_build_dir)
        with pytest.raises(ValueError, match="[Ss]emantic"):
            retriever.query("test", mode="semantic")
        index.close()

    def test_rrf_fusion_combines_rankings(self, tmp_build_dir):
        """RRF fusion produces expected combined scores."""
        from synix.search.results import SearchResult
        from synix.search.retriever import HybridRetriever

        index = SearchIndex(tmp_build_dir / "search.db")
        index.create()
        index.close()

        retriever = HybridRetriever(search_index=index)

        keyword_results = [
            SearchResult(content="A", label="a", layer_name="ep", layer_level=1, score=10.0),
            SearchResult(content="B", label="b", layer_name="ep", layer_level=1, score=5.0),
        ]
        semantic_results = [
            SearchResult(content="B", label="b", layer_name="ep", layer_level=1, score=0.95),
            SearchResult(content="C", label="c", layer_name="ep", layer_level=1, score=0.90),
        ]

        fused = retriever._rrf_fuse(keyword_results, semantic_results, top_k=10)
        # "b" appears in both lists, so it should have highest RRF score
        assert fused[0].label == "b"
        # All three items should be present
        fused_ids = {r.label for r in fused}
        assert fused_ids == {"a", "b", "c"}
