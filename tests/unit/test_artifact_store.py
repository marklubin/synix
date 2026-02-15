"""Tests for artifact storage."""

from __future__ import annotations

import hashlib
import json

from synix import Artifact
from synix.build.artifacts import MANIFEST_FILENAME, ArtifactStore


class TestArtifactStore:
    def test_save_and_load_roundtrip(self, tmp_build_dir):
        """Save artifact, load by ID, content matches."""
        store = ArtifactStore(tmp_build_dir)
        artifact = Artifact(
            label="t-chatgpt-conv001",
            artifact_type="transcript",
            content="Hello, world!",
            metadata={"source": "chatgpt"},
        )
        store.save_artifact(artifact, layer_name="transcripts", layer_level=0)

        loaded = store.load_artifact("t-chatgpt-conv001")
        assert loaded is not None
        assert loaded.label == "t-chatgpt-conv001"
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
                Artifact(label=f"t-{i}", artifact_type="transcript", content=f"transcript {i}"),
                layer_name="transcripts",
                layer_level=0,
            )

        # 2 episodes
        for i in range(2):
            store.save_artifact(
                Artifact(label=f"ep-{i}", artifact_type="episode", content=f"episode {i}"),
                layer_name="episodes",
                layer_level=1,
            )

        # 1 core
        store.save_artifact(
            Artifact(label="core-memory", artifact_type="core_memory", content="core"),
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
        artifact = Artifact(label="hash-test", artifact_type="transcript", content=content)

        store.save_artifact(artifact, layer_name="transcripts", layer_level=0)

        expected_hash = f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"
        loaded = store.load_artifact("hash-test")
        assert loaded is not None
        assert loaded.artifact_id == expected_hash
        assert store.get_artifact_id("hash-test") == expected_hash

    def test_manifest_persistence(self, tmp_build_dir):
        """Save artifacts, create new store instance from same dir, manifest intact."""
        store1 = ArtifactStore(tmp_build_dir)
        store1.save_artifact(
            Artifact(label="persist-1", artifact_type="transcript", content="persisted"),
            layer_name="transcripts",
            layer_level=0,
        )
        store1.save_artifact(
            Artifact(label="persist-2", artifact_type="episode", content="also persisted"),
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
            Artifact(label="overwrite-me", artifact_type="transcript", content="version 1"),
            layer_name="transcripts",
            layer_level=0,
        )
        original_hash = store.get_artifact_id("overwrite-me")

        store.save_artifact(
            Artifact(label="overwrite-me", artifact_type="transcript", content="version 2"),
            layer_name="transcripts",
            layer_level=0,
        )

        loaded = store.load_artifact("overwrite-me")
        assert loaded is not None
        assert loaded.content == "version 2"
        assert store.get_artifact_id("overwrite-me") != original_hash


class TestResolvePrefix:
    """Tests for git-like prefix resolution on artifact IDs and content hashes."""

    def _store_with_artifacts(self, tmp_build_dir):
        store = ArtifactStore(tmp_build_dir)
        store.save_artifact(
            Artifact(label="t-text-alice", artifact_type="transcript", content="Alice bio"),
            layer_name="bios",
            layer_level=0,
        )
        store.save_artifact(
            Artifact(label="t-text-bob", artifact_type="transcript", content="Bob bio"),
            layer_name="bios",
            layer_level=0,
        )
        store.save_artifact(
            Artifact(label="ep-alice-001", artifact_type="episode", content="Episode about Alice"),
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
        full_hash = store.get_artifact_id("t-text-alice").removeprefix("sha256:")
        prefix = full_hash[:8]
        resolved = store.resolve_prefix(prefix)
        assert resolved == "t-text-alice"

    def test_hash_prefix_with_sha256_prefix(self, tmp_build_dir):
        store = self._store_with_artifacts(tmp_build_dir)
        full_hash = store.get_artifact_id("t-text-bob")
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


class TestManifestValidation:
    """Tests for manifest format collision handling (Issue 1)."""

    def test_foreign_manifest_flat_strings(self, tmp_build_dir):
        """Foreign manifest with string values (not dicts) → empty store, no crash."""
        manifest_path = tmp_build_dir / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(
                {
                    "package.json": "1.0.0",
                    "node_modules": "installed",
                    "build": "success",
                }
            )
        )
        store = ArtifactStore(tmp_build_dir)
        assert len(store._manifest) == 0

    def test_mixed_valid_invalid_entries(self, tmp_build_dir):
        """Mixed valid/invalid entries → valid entries survive."""
        manifest_path = tmp_build_dir / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(
                {
                    "good-artifact": {
                        "path": "layer0-transcripts/good-artifact.json",
                        "artifact_id": "sha256:abc123",
                        "layer": "transcripts",
                        "level": 0,
                    },
                    "bad-string": "not a dict",
                    "bad-missing-keys": {"some_key": "some_value"},
                }
            )
        )
        store = ArtifactStore(tmp_build_dir)
        assert len(store._manifest) == 1
        assert "good-artifact" in store._manifest

    def test_non_json_file(self, tmp_build_dir):
        """Non-JSON manifest → warn + empty store."""
        manifest_path = tmp_build_dir / MANIFEST_FILENAME
        manifest_path.write_text("this is not json {{{")
        store = ArtifactStore(tmp_build_dir)
        assert len(store._manifest) == 0

    def test_missing_file(self, tmp_build_dir):
        """Missing manifest → empty store (regression guard)."""
        # Don't create any manifest file
        (tmp_build_dir / MANIFEST_FILENAME).unlink(missing_ok=True)
        store = ArtifactStore(tmp_build_dir)
        assert len(store._manifest) == 0

    def test_non_dict_top_level(self, tmp_build_dir):
        """Manifest that is a JSON array → empty store."""
        manifest_path = tmp_build_dir / MANIFEST_FILENAME
        manifest_path.write_text(json.dumps(["item1", "item2"]))
        store = ArtifactStore(tmp_build_dir)
        assert len(store._manifest) == 0

    def test_underscore_prefixed_keys_skipped(self, tmp_build_dir):
        """Keys starting with _ are reserved metadata and skipped."""
        manifest_path = tmp_build_dir / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(
                {
                    "_version": "2.0",
                    "_generator": "other-tool",
                    "real-artifact": {
                        "path": "layer0-transcripts/real.json",
                        "layer": "transcripts",
                        "level": 0,
                    },
                }
            )
        )
        store = ArtifactStore(tmp_build_dir)
        assert len(store._manifest) == 1
        assert "real-artifact" in store._manifest
        assert "_version" not in store._manifest
