"""Tests for artifact storage."""

from __future__ import annotations

import hashlib

from synix import Artifact
from synix.artifacts.store import ArtifactStore


class TestArtifactStore:
    def test_save_and_load_roundtrip(self, tmp_build_dir):
        """Save artifact, load by ID, content matches."""
        store = ArtifactStore(tmp_build_dir)
        artifact = Artifact(
            artifact_id="t-chatgpt-conv001",
            artifact_type="transcript",
            content="Hello, world!",
            metadata={"source": "chatgpt"},
        )
        store.save_artifact(artifact, layer_name="transcripts", layer_level=0)

        loaded = store.load_artifact("t-chatgpt-conv001")
        assert loaded is not None
        assert loaded.artifact_id == "t-chatgpt-conv001"
        assert loaded.artifact_type == "transcript"
        assert loaded.content == "Hello, world!"
        assert loaded.metadata == {"source": "chatgpt"}

    def test_load_nonexistent_returns_none(self, tmp_build_dir):
        """Missing ID returns None."""
        store = ArtifactStore(tmp_build_dir)
        assert store.load_artifact("nonexistent-id") is None

    def test_list_by_layer(self, tmp_build_dir):
        """Save 5 artifacts across 3 layers, list each layer correctly."""
        store = ArtifactStore(tmp_build_dir)

        # 2 transcripts
        for i in range(2):
            store.save_artifact(
                Artifact(artifact_id=f"t-{i}", artifact_type="transcript", content=f"transcript {i}"),
                layer_name="transcripts",
                layer_level=0,
            )

        # 2 episodes
        for i in range(2):
            store.save_artifact(
                Artifact(artifact_id=f"ep-{i}", artifact_type="episode", content=f"episode {i}"),
                layer_name="episodes",
                layer_level=1,
            )

        # 1 core
        store.save_artifact(
            Artifact(artifact_id="core-memory", artifact_type="core_memory", content="core"),
            layer_name="core",
            layer_level=3,
        )

        assert len(store.list_artifacts("transcripts")) == 2
        assert len(store.list_artifacts("episodes")) == 2
        assert len(store.list_artifacts("core")) == 1
        assert len(store.list_artifacts("nonexistent")) == 0

    def test_content_hash_computed(self, tmp_build_dir):
        """Save artifact, hash is SHA256 of content."""
        store = ArtifactStore(tmp_build_dir)
        content = "Test content for hashing"
        artifact = Artifact(artifact_id="hash-test", artifact_type="transcript", content=content)

        store.save_artifact(artifact, layer_name="transcripts", layer_level=0)

        expected_hash = f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"
        loaded = store.load_artifact("hash-test")
        assert loaded is not None
        assert loaded.content_hash == expected_hash
        assert store.get_content_hash("hash-test") == expected_hash

    def test_manifest_persistence(self, tmp_build_dir):
        """Save artifacts, create new store instance from same dir, manifest intact."""
        store1 = ArtifactStore(tmp_build_dir)
        store1.save_artifact(
            Artifact(artifact_id="persist-1", artifact_type="transcript", content="persisted"),
            layer_name="transcripts",
            layer_level=0,
        )
        store1.save_artifact(
            Artifact(artifact_id="persist-2", artifact_type="episode", content="also persisted"),
            layer_name="episodes",
            layer_level=1,
        )

        # Create new store instance from same directory
        store2 = ArtifactStore(tmp_build_dir)
        loaded1 = store2.load_artifact("persist-1")
        loaded2 = store2.load_artifact("persist-2")
        assert loaded1 is not None
        assert loaded1.content == "persisted"
        assert loaded2 is not None
        assert loaded2.content == "also persisted"

    def test_overwrite_artifact(self, tmp_build_dir):
        """Save same ID twice, latest wins, manifest updated."""
        store = ArtifactStore(tmp_build_dir)

        store.save_artifact(
            Artifact(artifact_id="overwrite-me", artifact_type="transcript", content="version 1"),
            layer_name="transcripts",
            layer_level=0,
        )
        original_hash = store.get_content_hash("overwrite-me")

        store.save_artifact(
            Artifact(artifact_id="overwrite-me", artifact_type="transcript", content="version 2"),
            layer_name="transcripts",
            layer_level=0,
        )

        loaded = store.load_artifact("overwrite-me")
        assert loaded is not None
        assert loaded.content == "version 2"
        assert store.get_content_hash("overwrite-me") != original_hash


class TestResolvePrefix:
    """Tests for git-like prefix resolution on artifact IDs and content hashes."""

    def _store_with_artifacts(self, tmp_build_dir):
        store = ArtifactStore(tmp_build_dir)
        store.save_artifact(
            Artifact(artifact_id="t-text-alice", artifact_type="transcript", content="Alice bio"),
            layer_name="bios",
            layer_level=0,
        )
        store.save_artifact(
            Artifact(artifact_id="t-text-bob", artifact_type="transcript", content="Bob bio"),
            layer_name="bios",
            layer_level=0,
        )
        store.save_artifact(
            Artifact(artifact_id="ep-alice-001", artifact_type="episode", content="Episode about Alice"),
            layer_name="episodes",
            layer_level=1,
        )
        return store

    def test_exact_match(self, tmp_build_dir):
        store = self._store_with_artifacts(tmp_build_dir)
        assert store.resolve_prefix("t-text-alice") == "t-text-alice"

    def test_id_prefix_unique(self, tmp_build_dir):
        store = self._store_with_artifacts(tmp_build_dir)
        assert store.resolve_prefix("ep-") == "ep-alice-001"

    def test_id_prefix_ambiguous(self, tmp_build_dir):
        store = self._store_with_artifacts(tmp_build_dir)
        import pytest

        with pytest.raises(ValueError, match="ambiguous"):
            store.resolve_prefix("t-text-")

    def test_hash_prefix_unique(self, tmp_build_dir):
        store = self._store_with_artifacts(tmp_build_dir)
        # Get the actual hash of alice and use its first 8 chars
        full_hash = store.get_content_hash("t-text-alice").removeprefix("sha256:")
        prefix = full_hash[:8]
        resolved = store.resolve_prefix(prefix)
        assert resolved == "t-text-alice"

    def test_hash_prefix_with_sha256_prefix(self, tmp_build_dir):
        store = self._store_with_artifacts(tmp_build_dir)
        full_hash = store.get_content_hash("t-text-bob")
        # Pass the full "sha256:abcd..." format with a prefix
        short = full_hash[:15]  # "sha256:abcdef12"
        resolved = store.resolve_prefix(short)
        assert resolved == "t-text-bob"

    def test_no_match(self, tmp_build_dir):
        store = self._store_with_artifacts(tmp_build_dir)
        assert store.resolve_prefix("zzz-nonexistent") is None

    def test_empty_store(self, tmp_build_dir):
        store = ArtifactStore(tmp_build_dir)
        assert store.resolve_prefix("anything") is None
