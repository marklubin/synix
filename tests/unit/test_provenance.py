"""Tests for provenance tracking."""

from __future__ import annotations

from synix.artifacts.provenance import ProvenanceTracker


class TestProvenanceTracker:
    def test_record_and_retrieve(self, tmp_build_dir):
        """Record provenance, get_parents returns correct IDs."""
        tracker = ProvenanceTracker(tmp_build_dir)
        tracker.record(
            artifact_id="ep-conv001",
            parent_ids=["t-chatgpt-conv001"],
            prompt_id="episode_summary_v1",
            model_config={"model": "claude-sonnet-4-20250514"},
        )

        parents = tracker.get_parents("ep-conv001")
        assert parents == ["t-chatgpt-conv001"]

        # Non-existent artifact returns empty list
        assert tracker.get_parents("nonexistent") == []

    def test_chain_walking(self, tmp_build_dir):
        """3-level chain: core → monthly → episode → transcript. get_chain returns full path."""
        tracker = ProvenanceTracker(tmp_build_dir)

        # transcript has no parents (root)
        tracker.record("t-001", parent_ids=[])
        # episode depends on transcript
        tracker.record("ep-001", parent_ids=["t-001"], prompt_id="episode_v1")
        # monthly depends on episode
        tracker.record("monthly-2025-01", parent_ids=["ep-001"], prompt_id="monthly_v1")
        # core depends on monthly
        tracker.record("core-memory", parent_ids=["monthly-2025-01"], prompt_id="core_v1")

        chain = tracker.get_chain("core-memory")
        chain_ids = [r.artifact_id for r in chain]

        assert "core-memory" in chain_ids
        assert "monthly-2025-01" in chain_ids
        assert "ep-001" in chain_ids
        assert "t-001" in chain_ids
        assert len(chain) == 4

    def test_chain_multiple_parents(self, tmp_build_dir):
        """Monthly rollup with 5 episode inputs, all in chain."""
        tracker = ProvenanceTracker(tmp_build_dir)

        # 5 episodes, each from a transcript
        for i in range(5):
            tracker.record(f"t-{i}", parent_ids=[])
            tracker.record(f"ep-{i}", parent_ids=[f"t-{i}"], prompt_id="episode_v1")

        # Monthly rollup depends on all 5 episodes
        episode_ids = [f"ep-{i}" for i in range(5)]
        tracker.record("monthly-2025-01", parent_ids=episode_ids, prompt_id="monthly_v1")

        chain = tracker.get_chain("monthly-2025-01")
        chain_ids = {r.artifact_id for r in chain}

        # Should include: monthly + 5 episodes + 5 transcripts = 11
        assert "monthly-2025-01" in chain_ids
        for i in range(5):
            assert f"ep-{i}" in chain_ids
            assert f"t-{i}" in chain_ids
        assert len(chain) == 11

    def test_persistence(self, tmp_build_dir):
        """Record provenance, reload from disk, data intact."""
        tracker1 = ProvenanceTracker(tmp_build_dir)
        tracker1.record(
            "ep-001",
            parent_ids=["t-001", "t-002"],
            prompt_id="episode_v1",
            model_config={"model": "claude-sonnet-4-20250514"},
        )

        # Create new tracker from same directory
        tracker2 = ProvenanceTracker(tmp_build_dir)
        parents = tracker2.get_parents("ep-001")
        assert parents == ["t-001", "t-002"]

        record = tracker2.get_record("ep-001")
        assert record is not None
        assert record.prompt_id == "episode_v1"
        assert record.model_config == {"model": "claude-sonnet-4-20250514"}
