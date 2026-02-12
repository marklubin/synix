"""Tests for DAG resolution."""

from __future__ import annotations

import pytest

from synix import Artifact, Layer, Pipeline
from synix.artifacts.store import ArtifactStore
from synix.pipeline.dag import needs_rebuild, resolve_build_order


class TestResolveBuildOrder:
    def test_topological_sort_simple(self):
        """4 layers in correct order."""
        pipeline = Pipeline("test")
        pipeline.add_layer(Layer(name="transcripts", level=0, transform="parse"))
        pipeline.add_layer(Layer(name="episodes", level=1, depends_on=["transcripts"], transform="summarize"))
        pipeline.add_layer(Layer(name="monthly", level=2, depends_on=["episodes"], transform="rollup"))
        pipeline.add_layer(Layer(name="core", level=3, depends_on=["monthly"], transform="synthesize"))

        order = resolve_build_order(pipeline)
        names = [l.name for l in order]

        assert names.index("transcripts") < names.index("episodes")
        assert names.index("episodes") < names.index("monthly")
        assert names.index("monthly") < names.index("core")

    def test_topological_sort_diamond(self):
        """Diamond dependency, no duplicate."""
        pipeline = Pipeline("test")
        pipeline.add_layer(Layer(name="transcripts", level=0, transform="parse"))
        pipeline.add_layer(Layer(name="episodes", level=1, depends_on=["transcripts"], transform="summarize"))
        pipeline.add_layer(Layer(name="monthly", level=2, depends_on=["episodes"], transform="rollup"))
        pipeline.add_layer(Layer(name="topical", level=2, depends_on=["episodes"], transform="topic_rollup"))
        pipeline.add_layer(Layer(name="core", level=3, depends_on=["monthly", "topical"], transform="synthesize"))

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
        pipeline.add_layer(Layer(name="a", level=0, depends_on=["c"], transform="x"))
        pipeline.add_layer(Layer(name="b", level=1, depends_on=["a"], transform="x"))
        pipeline.add_layer(Layer(name="c", level=2, depends_on=["b"], transform="x"))

        with pytest.raises(ValueError, match="[Cc]ircular"):
            resolve_build_order(pipeline)


class TestNeedsRebuild:
    def test_rebuild_detection_all_new(self, tmp_build_dir):
        """Empty store, everything needs rebuild."""
        store = ArtifactStore(tmp_build_dir)
        assert needs_rebuild("nonexistent", ["sha256:abc"], "prompt_v1", store) is True

    def test_rebuild_detection_all_cached(self, tmp_build_dir):
        """Matching hashes and prompt, nothing needs rebuild."""
        store = ArtifactStore(tmp_build_dir)
        artifact = Artifact(
            artifact_id="ep-001",
            artifact_type="episode",
            content="cached content",
            input_hashes=["sha256:input1", "sha256:input2"],
            prompt_id="episode_summary_v1",
        )
        store.save_artifact(artifact, layer_name="episodes", layer_level=1)

        result = needs_rebuild(
            "ep-001",
            ["sha256:input1", "sha256:input2"],
            "episode_summary_v1",
            store,
        )
        assert result is False

    def test_rebuild_detection_partial(self, tmp_build_dir):
        """Change prompt, artifact needs rebuild."""
        store = ArtifactStore(tmp_build_dir)
        artifact = Artifact(
            artifact_id="ep-001",
            artifact_type="episode",
            content="cached content",
            input_hashes=["sha256:input1"],
            prompt_id="episode_summary_v1",
        )
        store.save_artifact(artifact, layer_name="episodes", layer_level=1)

        # Same inputs, different prompt → needs rebuild
        assert needs_rebuild("ep-001", ["sha256:input1"], "episode_summary_v2", store) is True
        # Same prompt, different inputs → needs rebuild
        assert needs_rebuild("ep-001", ["sha256:changed"], "episode_summary_v1", store) is True

    def test_rebuild_cascades(self, tmp_build_dir):
        """Changing level 1 forces rebuild of 2 and 3 via changed input hashes."""
        store = ArtifactStore(tmp_build_dir)

        # Level 1 artifact (episode)
        store.save_artifact(
            Artifact(
                artifact_id="ep-001",
                artifact_type="episode",
                content="episode v1",
                input_hashes=["sha256:transcript1"],
                prompt_id="ep_v1",
            ),
            layer_name="episodes",
            layer_level=1,
        )
        ep_hash = store.get_content_hash("ep-001")

        # Level 2 artifact (monthly) depends on episode hash
        store.save_artifact(
            Artifact(
                artifact_id="monthly-2025-01",
                artifact_type="rollup",
                content="monthly v1",
                input_hashes=[ep_hash],
                prompt_id="monthly_v1",
            ),
            layer_name="monthly",
            layer_level=2,
        )
        monthly_hash = store.get_content_hash("monthly-2025-01")

        # Level 3 artifact (core) depends on monthly hash
        store.save_artifact(
            Artifact(
                artifact_id="core-memory",
                artifact_type="core_memory",
                content="core v1",
                input_hashes=[monthly_hash],
                prompt_id="core_v1",
            ),
            layer_name="core",
            layer_level=3,
        )

        # If episode rebuilds with new content, its hash changes
        new_ep_hash = "sha256:new_episode_hash"

        # Monthly sees changed input hash → needs rebuild
        assert needs_rebuild("monthly-2025-01", [new_ep_hash], "monthly_v1", store) is True

        # Core also needs rebuild because monthly will produce new hash
        new_monthly_hash = "sha256:new_monthly_hash"
        assert needs_rebuild("core-memory", [new_monthly_hash], "core_v1", store) is True

        # But with original hashes, everything is cached
        assert needs_rebuild("ep-001", ["sha256:transcript1"], "ep_v1", store) is False
        assert needs_rebuild("monthly-2025-01", [ep_hash], "monthly_v1", store) is False
        assert needs_rebuild("core-memory", [monthly_hash], "core_v1", store) is False
