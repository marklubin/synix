"""Tests for DAG resolution."""

from __future__ import annotations

import pytest

from synix import Artifact, Pipeline, SearchIndex, SearchSurface, Source, SynixSearch
from synix.artifacts.store import ArtifactStore
from synix.build.fingerprint import Fingerprint, compute_build_fingerprint, compute_digest
from synix.ext import CoreSynthesis, EpisodeSummary, MonthlyRollup, TopicalRollup
from synix.pipeline.dag import needs_rebuild, resolve_build_order


class TestResolveBuildOrder:
    def test_topological_sort_simple(self):
        """4 layers in correct order."""
        pipeline = Pipeline("test")
        transcripts = Source("transcripts")
        episodes = EpisodeSummary("episodes", depends_on=[transcripts])
        monthly = MonthlyRollup("monthly", depends_on=[episodes])
        core = CoreSynthesis("core", depends_on=[monthly])
        pipeline.add(transcripts, episodes, monthly, core)

        order = resolve_build_order(pipeline)
        names = [l.name for l in order]

        assert names.index("transcripts") < names.index("episodes")
        assert names.index("episodes") < names.index("monthly")
        assert names.index("monthly") < names.index("core")

    def test_topological_sort_diamond(self):
        """Diamond dependency, no duplicate."""
        pipeline = Pipeline("test")
        transcripts = Source("transcripts")
        episodes = EpisodeSummary("episodes", depends_on=[transcripts])
        monthly = MonthlyRollup("monthly", depends_on=[episodes])
        topical = TopicalRollup("topical", depends_on=[episodes])
        core = CoreSynthesis("core", depends_on=[monthly, topical])
        pipeline.add(transcripts, episodes, monthly, topical, core)

        order = resolve_build_order(pipeline)
        names = [l.name for l in order]

        # No duplicates
        assert len(names) == len(set(names))
        # transcripts before episodes
        assert names.index("transcripts") < names.index("episodes")
        # episodes before both level-2 layers
        assert names.index("episodes") < names.index("monthly")
        assert names.index("episodes") < names.index("topical")
        # both level-2 before core
        assert names.index("monthly") < names.index("core")
        assert names.index("topical") < names.index("core")

    def test_cycle_detection(self):
        """Circular dependency raises ValueError."""
        pipeline = Pipeline("test")
        a = Source("a")
        b = EpisodeSummary("b", depends_on=[a])
        c = MonthlyRollup("c", depends_on=[b])
        # Force a cycle: a -> c
        a.depends_on = [c]
        pipeline.add(a, b, c)

        with pytest.raises(ValueError, match="[Cc]ircular"):
            resolve_build_order(pipeline)

    def test_pipeline_registers_search_surfaces_outside_build_dag(self):
        """Search surfaces are tracked separately from build DAG layers."""
        pipeline = Pipeline("test")
        transcripts = Source("transcripts")
        episodes = EpisodeSummary("episodes", depends_on=[transcripts])
        episode_search = SearchSurface("episode-search", sources=[episodes], modes=["fulltext"])
        topics = TopicalRollup("topics", depends_on=[episodes], uses=[episode_search])
        pipeline.add(transcripts, episodes, episode_search, topics)

        assert [layer.name for layer in pipeline.layers] == ["transcripts", "episodes", "topics"]
        assert [surface.name for surface in pipeline.surfaces] == ["episode-search"]
        assert pipeline.projections == []

        order = resolve_build_order(pipeline)
        assert [layer.name for layer in order] == ["transcripts", "episodes", "topics"]

    def test_pipeline_keeps_search_index_in_projections(self):
        """SearchIndex remains a projection and is never treated as a search surface."""
        pipeline = Pipeline("test")
        transcripts = Source("transcripts")
        episodes = EpisodeSummary("episodes", depends_on=[transcripts])
        episode_search = SearchSurface("episode-search", sources=[episodes], modes=["fulltext"])
        memory_index = SearchIndex("memory-index", sources=[episodes], search=["fulltext"])

        pipeline.add(transcripts, episodes, episode_search, memory_index)

        assert [surface.name for surface in pipeline.surfaces] == ["episode-search"]
        assert [projection.name for projection in pipeline.projections] == ["memory-index"]

    def test_pipeline_keeps_synix_search_in_projections(self):
        """SynixSearch remains a projection backed by a declared search surface."""
        pipeline = Pipeline("test")
        transcripts = Source("transcripts")
        episodes = EpisodeSummary("episodes", depends_on=[transcripts])
        episode_search = SearchSurface("episode-search", sources=[episodes], modes=["fulltext"])
        search_output = SynixSearch("search", surface=episode_search)

        pipeline.add(transcripts, episodes, episode_search, search_output)

        assert [surface.name for surface in pipeline.surfaces] == ["episode-search"]
        assert [projection.name for projection in pipeline.projections] == ["search"]


class TestNeedsRebuild:
    """Tests for needs_rebuild — returns tuple[bool, list[str]]."""

    def _make_transform_fp(self, source="aaa", prompt="bbb"):
        components = {"source": source, "prompt": prompt}
        return Fingerprint(
            scheme="synix:transform:v1",
            digest=compute_digest(components),
            components=components,
        )

    def test_new_artifact(self, tmp_build_dir):
        """Empty store, everything needs rebuild."""
        store = ArtifactStore(tmp_build_dir)
        result, reasons = needs_rebuild("nonexistent", ["sha256:abc"], store)
        assert result is True
        assert "new artifact" in reasons

    def test_fingerprint_match_cached(self, tmp_build_dir):
        """Matching build fingerprint = cached."""
        store = ArtifactStore(tmp_build_dir)
        transform_fp = self._make_transform_fp()
        build_fp = compute_build_fingerprint(transform_fp, ["sha256:input1"])

        artifact = Artifact(
            label="ep-001",
            artifact_type="episode",
            content="cached content",
            input_ids=["sha256:input1"],
            prompt_id="ep_v1",
            metadata={"build_fingerprint": build_fp.to_dict()},
        )
        store.save_artifact(artifact, layer_name="episodes", layer_level=1)

        result, reasons = needs_rebuild("ep-001", ["sha256:input1"], store, current_build_fingerprint=build_fp)
        assert result is False
        assert reasons == []

    def test_fingerprint_mismatch_rebuilds(self, tmp_build_dir):
        """Different build fingerprint = rebuild with reasons."""
        store = ArtifactStore(tmp_build_dir)
        old_transform_fp = self._make_transform_fp(source="old")
        old_build_fp = compute_build_fingerprint(old_transform_fp, ["sha256:input1"])

        artifact = Artifact(
            label="ep-001",
            artifact_type="episode",
            content="cached content",
            input_ids=["sha256:input1"],
            prompt_id="ep_v1",
            metadata={"build_fingerprint": old_build_fp.to_dict()},
        )
        store.save_artifact(artifact, layer_name="episodes", layer_level=1)

        new_transform_fp = self._make_transform_fp(source="new")
        new_build_fp = compute_build_fingerprint(new_transform_fp, ["sha256:input1"])

        result, reasons = needs_rebuild("ep-001", ["sha256:input1"], store, current_build_fingerprint=new_build_fp)
        assert result is True
        assert "transform changed" in reasons

    def test_no_stored_fingerprint_forces_rebuild(self, tmp_build_dir):
        """Artifact without fingerprint in metadata forces rebuild."""
        store = ArtifactStore(tmp_build_dir)

        artifact = Artifact(
            label="ep-001",
            artifact_type="episode",
            content="old content",
            input_ids=["sha256:input1"],
            prompt_id="ep_v1",
        )
        store.save_artifact(artifact, layer_name="episodes", layer_level=1)

        transform_fp = self._make_transform_fp()
        build_fp = compute_build_fingerprint(transform_fp, ["sha256:input1"])

        result, reasons = needs_rebuild("ep-001", ["sha256:input1"], store, current_build_fingerprint=build_fp)
        assert result is True
        assert "no stored fingerprint" in reasons

    def test_scheme_mismatch_rebuilds(self, tmp_build_dir):
        """Different fingerprint scheme = rebuild."""
        store = ArtifactStore(tmp_build_dir)
        old_fp = Fingerprint(
            scheme="synix:build:v0",
            digest="somedigest",
            components={"transform": "aaa", "inputs": "bbb"},
        )
        artifact = Artifact(
            label="ep-001",
            artifact_type="episode",
            content="old content",
            input_ids=["sha256:input1"],
            prompt_id="ep_v1",
            metadata={"build_fingerprint": old_fp.to_dict()},
        )
        store.save_artifact(artifact, layer_name="episodes", layer_level=1)

        transform_fp = self._make_transform_fp()
        new_build_fp = compute_build_fingerprint(transform_fp, ["sha256:input1"])

        result, reasons = needs_rebuild("ep-001", ["sha256:input1"], store, current_build_fingerprint=new_build_fp)
        assert result is True
        assert any("scheme changed" in r for r in reasons)

    def test_input_hash_change_without_fingerprint(self, tmp_build_dir):
        """Without fingerprint, input hash change is detected."""
        store = ArtifactStore(tmp_build_dir)
        artifact = Artifact(
            label="ep-001",
            artifact_type="episode",
            content="cached content",
            input_ids=["sha256:input1"],
            prompt_id="ep_v1",
        )
        store.save_artifact(artifact, layer_name="episodes", layer_level=1)

        # Same inputs, no fingerprint — cached
        result, reasons = needs_rebuild("ep-001", ["sha256:input1"], store)
        assert result is False

        # Changed inputs, no fingerprint — rebuild
        result, reasons = needs_rebuild("ep-001", ["sha256:changed"], store)
        assert result is True
        assert "inputs changed" in reasons

    def test_cascade_via_input_ids(self, tmp_build_dir):
        """Changing level 1 forces rebuild of 2 and 3 via changed input ids."""
        store = ArtifactStore(tmp_build_dir)
        transform_fp = self._make_transform_fp()

        # Level 1
        build_fp1 = compute_build_fingerprint(transform_fp, ["sha256:transcript1"])
        store.save_artifact(
            Artifact(
                label="ep-001",
                artifact_type="episode",
                content="episode v1",
                input_ids=["sha256:transcript1"],
                prompt_id="ep_v1",
                metadata={"build_fingerprint": build_fp1.to_dict()},
            ),
            layer_name="episodes",
            layer_level=1,
        )
        ep_hash = store.get_artifact_id("ep-001")

        # Level 2
        build_fp2 = compute_build_fingerprint(transform_fp, [ep_hash])
        store.save_artifact(
            Artifact(
                label="monthly-2025-01",
                artifact_type="rollup",
                content="monthly v1",
                input_ids=[ep_hash],
                prompt_id="monthly_v1",
                metadata={"build_fingerprint": build_fp2.to_dict()},
            ),
            layer_name="monthly",
            layer_level=2,
        )
        monthly_hash = store.get_artifact_id("monthly-2025-01")

        # Level 3
        build_fp3 = compute_build_fingerprint(transform_fp, [monthly_hash])
        store.save_artifact(
            Artifact(
                label="core-memory",
                artifact_type="core_memory",
                content="core v1",
                input_ids=[monthly_hash],
                prompt_id="core_v1",
                metadata={"build_fingerprint": build_fp3.to_dict()},
            ),
            layer_name="core",
            layer_level=3,
        )

        # If episode rebuilds, monthly sees changed input id
        new_ep_hash = "sha256:new_episode_hash"
        new_build_fp2 = compute_build_fingerprint(transform_fp, [new_ep_hash])
        result, _ = needs_rebuild("monthly-2025-01", [new_ep_hash], store, current_build_fingerprint=new_build_fp2)
        assert result is True

        # With original hashes, everything cached
        result, _ = needs_rebuild("ep-001", ["sha256:transcript1"], store, current_build_fingerprint=build_fp1)
        assert result is False
        result, _ = needs_rebuild("monthly-2025-01", [ep_hash], store, current_build_fingerprint=build_fp2)
        assert result is False
        result, _ = needs_rebuild("core-memory", [monthly_hash], store, current_build_fingerprint=build_fp3)
        assert result is False
