"""Tests for shadow index swap pattern (S07).

Verifies that the search index is built to a shadow file and atomically
swapped into place on success, preserving the old index on failure.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from synix import Artifact
from synix.search.indexer import SearchIndex, SearchIndexProjection, ShadowIndexManager


def _make_artifact(label: str, content: str, layer: str = "episodes") -> Artifact:
    """Helper to create a test artifact."""
    return Artifact(
        label=label,
        artifact_type="episode",
        content=content,
        metadata={"layer_name": layer, "layer_level": 1},
    )


class TestShadowIndexSwap:
    """Tests for the shadow index swap pattern end-to-end."""

    def test_successful_build_swaps_index(self, tmp_build_dir):
        """After materialize, search.db exists and search_shadow.db does not."""
        proj = SearchIndexProjection(tmp_build_dir)
        artifacts = [
            _make_artifact("ep-001", "Content about machine learning"),
            _make_artifact("ep-002", "Content about distributed systems"),
        ]
        config = {"sources": [{"layer": "episodes", "level": 1}]}

        proj.materialize(artifacts, config)

        # Main index exists with correct data
        assert (tmp_build_dir / "search.db").exists()
        # Shadow file is gone
        assert not (tmp_build_dir / "search_shadow.db").exists()

        # Verify data is queryable in the final index
        results = proj.query("machine learning")
        assert len(results) == 1
        assert results[0].label == "ep-001"

        results2 = proj.query("distributed systems")
        assert len(results2) == 1
        assert results2[0].label == "ep-002"

        proj.close()

    def test_failed_build_preserves_old_index(self, tmp_build_dir):
        """If materialize fails mid-write, old search.db is unchanged."""
        # Build an initial index with known content
        proj = SearchIndexProjection(tmp_build_dir)
        old_artifacts = [_make_artifact("old-001", "Original data about Rust ownership")]
        config = {"sources": [{"layer": "episodes", "level": 1}]}
        proj.materialize(old_artifacts, config)
        proj.close()

        # Record the content hash of the old database to verify it's untouched
        old_db_path = tmp_build_dir / "search.db"
        old_db_stat = old_db_path.stat()
        old_db_mtime = old_db_stat.st_mtime

        # Read old content to verify later
        old_index = SearchIndex(old_db_path)
        old_results = old_index.query("Rust")
        assert len(old_results) == 1
        old_index.close()

        # Monkeypatch insert to fail on the second artifact
        original_insert = SearchIndex.insert
        call_count = 0

        def failing_insert(self, artifact, layer_name, layer_level):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise RuntimeError("Simulated mid-build failure")
            original_insert(self, artifact, layer_name, layer_level)

        SearchIndex.insert = failing_insert
        try:
            proj2 = SearchIndexProjection(tmp_build_dir)
            with pytest.raises(RuntimeError, match="Simulated mid-build failure"):
                proj2.materialize(
                    [
                        _make_artifact("new-001", "New data one"),
                        _make_artifact("new-002", "New data two that triggers failure"),
                    ],
                    config,
                )
            proj2.close()
        finally:
            SearchIndex.insert = original_insert

        # Shadow file should be cleaned up
        assert not (tmp_build_dir / "search_shadow.db").exists()

        # Old index should be completely unchanged
        assert old_db_path.exists()
        verify_index = SearchIndex(old_db_path)
        results = verify_index.query("Rust")
        assert len(results) == 1
        assert results[0].label == "old-001"

        # New data should NOT be in the index
        new_results = verify_index.query("New data")
        assert len(new_results) == 0
        verify_index.close()

    def test_first_build_no_old_index(self, tmp_build_dir):
        """Works correctly when no previous index exists (first ever build)."""
        # Verify no index exists yet
        assert not (tmp_build_dir / "search.db").exists()
        assert not (tmp_build_dir / "search_shadow.db").exists()

        proj = SearchIndexProjection(tmp_build_dir)
        artifacts = [
            _make_artifact("ep-001", "First ever build content about Python"),
        ]
        config = {"sources": [{"layer": "episodes", "level": 1}]}

        proj.materialize(artifacts, config)

        # Index was created successfully
        assert (tmp_build_dir / "search.db").exists()
        assert not (tmp_build_dir / "search_shadow.db").exists()

        results = proj.query("Python")
        assert len(results) == 1
        assert results[0].label == "ep-001"

        proj.close()

    def test_concurrent_query_during_build(self, tmp_build_dir):
        """Old index remains queryable while a shadow build is in progress."""
        # Build initial index
        proj = SearchIndexProjection(tmp_build_dir)
        proj.materialize(
            [_make_artifact("old-001", "Original searchable content about Docker")],
            {"sources": [{"layer": "episodes", "level": 1}]},
        )
        proj.close()

        # Start a shadow build manually (simulating a long build)
        manager = ShadowIndexManager(tmp_build_dir)
        shadow = manager.begin_build()
        shadow.insert(
            _make_artifact("new-001", "New shadow content about Kubernetes"),
            "episodes",
            1,
        )

        # While shadow is being built, the old main index should still be queryable
        reader = SearchIndex(tmp_build_dir / "search.db")
        results = reader.query("Docker")
        assert len(results) == 1
        assert results[0].label == "old-001"

        # New content should NOT be visible in the main index yet
        new_results = reader.query("Kubernetes")
        assert len(new_results) == 0
        reader.close()

        # Complete the shadow build
        manager.commit()

        # Now the main index should have the new content
        reader2 = SearchIndex(tmp_build_dir / "search.db")
        new_results2 = reader2.query("Kubernetes")
        assert len(new_results2) == 1
        assert new_results2[0].label == "new-001"
        reader2.close()

    def test_rollback_cleans_up_shadow(self, tmp_build_dir):
        """Shadow file is deleted on rollback, no residual state."""
        manager = ShadowIndexManager(tmp_build_dir)
        shadow = manager.begin_build()

        # Write some data to the shadow
        shadow.insert(
            _make_artifact("ep-001", "Data that will be rolled back"),
            "episodes",
            1,
        )

        # Shadow file should exist before rollback
        assert (tmp_build_dir / "search_shadow.db").exists()

        manager.rollback()

        # Shadow file should be gone after rollback
        assert not (tmp_build_dir / "search_shadow.db").exists()

        # No main index should have been created either (since there was no prior one)
        assert not (tmp_build_dir / "search.db").exists()

    def test_commit_replaces_atomically(self, tmp_build_dir):
        """os.replace is used for atomic swap (verified via mock)."""
        import os as real_os

        manager = ShadowIndexManager(tmp_build_dir)
        shadow = manager.begin_build()
        shadow.insert(
            _make_artifact("ep-001", "Atomic swap test content"),
            "episodes",
            1,
        )

        shadow_path_str = str(manager.shadow_path)
        main_path_str = str(manager.main_path)

        # Capture a direct reference to the real os.replace before patching
        _real_replace = real_os.replace

        with patch("synix.search.indexer.os.replace", wraps=_real_replace) as mock_replace:
            manager.commit()

            # Verify os.replace was called with the correct paths
            mock_replace.assert_called_once_with(shadow_path_str, main_path_str)

        # And verify it actually worked
        assert (tmp_build_dir / "search.db").exists()
        assert not (tmp_build_dir / "search_shadow.db").exists()

        index = SearchIndex(tmp_build_dir / "search.db")
        results = index.query("Atomic swap")
        assert len(results) == 1
        index.close()
