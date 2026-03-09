"""Unit tests for SnapshotView ref-resolved read API."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest

from synix.build.object_store import SCHEMA_VERSION, ObjectStore
from synix.build.refs import RefStore
from synix.build.snapshot_view import SnapshotView


def _canonical_json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _store_artifact(
    object_store: ObjectStore, label: str, content: str, *, parent_labels: list[str] | None = None
) -> str:
    """Store a content blob + artifact object and return the artifact oid."""
    content_oid = object_store.put_bytes(content.encode("utf-8"))
    content_hash = f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"
    artifact_payload = {
        "type": "artifact",
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "artifact_type": "episode",
        "artifact_id": content_hash,
        "content_oid": content_oid,
        "input_ids": [],
        "metadata": {"layer_name": "transcripts", "layer_level": 0},
        "parent_labels": parent_labels or [],
    }
    return object_store.put_json(artifact_payload)


def _create_snapshot(tmp_path, artifacts: dict[str, str] | None = None, content_map: dict[str, str] | None = None):
    """Create a full snapshot chain in tmp_path/.synix and return (synix_dir, snapshot_oid, manifest_oid, artifact_oids)."""
    synix_dir = tmp_path / ".synix"
    object_store = ObjectStore(synix_dir)
    ref_store = RefStore(synix_dir)

    if content_map is None:
        content_map = {"ep-1": "Episode one content", "ep-2": "Episode two content"}
    if artifacts is None:
        artifacts = {}
        for label, content in content_map.items():
            artifacts[label] = _store_artifact(object_store, label, content)

    manifest_payload = {
        "type": "manifest",
        "schema_version": SCHEMA_VERSION,
        "pipeline_name": "test-pipeline",
        "pipeline_fingerprint": "sha256:test",
        "artifacts": [{"label": lbl, "oid": oid} for lbl, oid in sorted(artifacts.items())],
        "projections": {},
    }
    manifest_oid = object_store.put_json(manifest_payload)

    snapshot_payload = {
        "type": "snapshot",
        "schema_version": SCHEMA_VERSION,
        "manifest_oid": manifest_oid,
        "parent_snapshot_oids": [],
        "created_at": datetime.now(UTC).isoformat(),
        "pipeline_name": "test-pipeline",
        "run_id": "20260307T120000000000Z",
    }
    snapshot_oid = object_store.put_json(snapshot_payload)

    ref_store.ensure_head()
    ref_store.write_ref("refs/heads/main", snapshot_oid)

    return synix_dir, snapshot_oid, manifest_oid, artifacts


class TestSnapshotViewOpen:
    def test_open_resolves_head_to_snapshot(self, tmp_path):
        """SnapshotView.open resolves HEAD and returns a valid view."""
        synix_dir, snapshot_oid, manifest_oid, _ = _create_snapshot(tmp_path)

        view = SnapshotView.open(synix_dir)

        assert view.snapshot_oid == snapshot_oid
        assert view.manifest_oid == manifest_oid

    def test_open_explicit_ref(self, tmp_path):
        """SnapshotView.open can resolve a named ref."""
        synix_dir, snapshot_oid, _, _ = _create_snapshot(tmp_path)
        ref_store = RefStore(synix_dir)
        ref_store.write_ref("refs/runs/test-run", snapshot_oid)

        view = SnapshotView.open(synix_dir, ref="refs/runs/test-run")

        assert view.snapshot_oid == snapshot_oid

    def test_open_unresolved_ref_raises(self, tmp_path):
        """SnapshotView.open raises ValueError for a ref that doesn't exist."""
        synix_dir = tmp_path / ".synix"
        ObjectStore(synix_dir)  # ensure dir exists
        RefStore(synix_dir)

        with pytest.raises(ValueError, match="does not resolve to a snapshot"):
            SnapshotView.open(synix_dir, ref="refs/heads/nonexistent")

    def test_open_ref_to_non_snapshot_raises(self, tmp_path):
        """SnapshotView.open raises ValueError if the ref points to a non-snapshot object."""
        synix_dir = tmp_path / ".synix"
        object_store = ObjectStore(synix_dir)
        ref_store = RefStore(synix_dir)

        manifest_payload = {
            "type": "manifest",
            "schema_version": SCHEMA_VERSION,
            "pipeline_name": "bad-ref",
            "pipeline_fingerprint": "sha256:test",
            "artifacts": [],
            "projections": {},
        }
        manifest_oid = object_store.put_json(manifest_payload)
        ref_store.ensure_head()
        ref_store.write_ref("refs/heads/main", manifest_oid)

        with pytest.raises(ValueError, match="expected 'snapshot'"):
            SnapshotView.open(synix_dir)


class TestSnapshotViewListArtifacts:
    def test_list_artifacts_returns_loaded_objects(self, tmp_path):
        """list_artifacts returns full artifact objects for each manifest entry."""
        synix_dir, _, _, artifact_oids = _create_snapshot(tmp_path)

        view = SnapshotView.open(synix_dir)
        artifacts = view.list_artifacts()

        assert len(artifacts) == 2
        labels = {a["label"] for a in artifacts}
        assert labels == {"ep-1", "ep-2"}
        for a in artifacts:
            assert a["type"] == "artifact"
            assert "content_oid" in a


class TestSnapshotViewGetArtifact:
    def test_get_artifact_returns_object_with_content(self, tmp_path):
        """get_artifact loads the artifact object and inlines the content."""
        synix_dir, _, _, _ = _create_snapshot(tmp_path)

        view = SnapshotView.open(synix_dir)
        artifact = view.get_artifact("ep-1")

        assert artifact["label"] == "ep-1"
        assert artifact["content"] == "Episode one content"
        assert artifact["type"] == "artifact"

    def test_get_artifact_missing_label_raises(self, tmp_path):
        """get_artifact raises KeyError for a nonexistent label."""
        synix_dir, _, _, _ = _create_snapshot(tmp_path)

        view = SnapshotView.open(synix_dir)

        with pytest.raises(KeyError, match="not found"):
            view.get_artifact("nonexistent")


class TestSnapshotViewGetContent:
    def test_get_content_returns_raw_text(self, tmp_path):
        """get_content returns just the text content for a label."""
        synix_dir, _, _, _ = _create_snapshot(tmp_path)

        view = SnapshotView.open(synix_dir)

        assert view.get_content("ep-1") == "Episode one content"
        assert view.get_content("ep-2") == "Episode two content"

    def test_get_content_missing_label_raises(self, tmp_path):
        """get_content raises KeyError for a nonexistent label."""
        synix_dir, _, _, _ = _create_snapshot(tmp_path)

        view = SnapshotView.open(synix_dir)

        with pytest.raises(KeyError, match="not found"):
            view.get_content("nonexistent")


class TestSnapshotViewGetManifest:
    def test_get_manifest_returns_full_dict(self, tmp_path):
        """get_manifest returns the complete manifest dictionary."""
        synix_dir, _, _, _ = _create_snapshot(tmp_path)

        view = SnapshotView.open(synix_dir)
        manifest = view.get_manifest()

        assert manifest["type"] == "manifest"
        assert manifest["pipeline_name"] == "test-pipeline"
        assert len(manifest["artifacts"]) == 2

    def test_get_manifest_returns_copy(self, tmp_path):
        """get_manifest returns a copy, not the internal dict."""
        synix_dir, _, _, _ = _create_snapshot(tmp_path)

        view = SnapshotView.open(synix_dir)
        m1 = view.get_manifest()
        m2 = view.get_manifest()

        assert m1 is not m2
        assert m1 == m2


class TestSnapshotViewGetProvenance:
    def test_provenance_walks_parent_labels(self, tmp_path):
        """get_provenance returns BFS-ordered label chain."""
        synix_dir = tmp_path / ".synix"
        object_store = ObjectStore(synix_dir)
        ref_store = RefStore(synix_dir)

        # Create a chain: ep-1 -> roll-1 (roll-1 has parent_labels=["ep-1"])
        ep_oid = _store_artifact(object_store, "ep-1", "Episode content")
        roll_oid = _store_artifact(object_store, "roll-1", "Rollup content", parent_labels=["ep-1"])

        artifacts = {"ep-1": ep_oid, "roll-1": roll_oid}
        manifest_payload = {
            "type": "manifest",
            "schema_version": SCHEMA_VERSION,
            "pipeline_name": "prov-test",
            "pipeline_fingerprint": "sha256:test",
            "artifacts": [{"label": lbl, "oid": oid} for lbl, oid in sorted(artifacts.items())],
            "projections": {},
        }
        manifest_oid = object_store.put_json(manifest_payload)
        snapshot_payload = {
            "type": "snapshot",
            "schema_version": SCHEMA_VERSION,
            "manifest_oid": manifest_oid,
            "parent_snapshot_oids": [],
            "created_at": datetime.now(UTC).isoformat(),
            "pipeline_name": "prov-test",
            "run_id": "20260307T120000000001Z",
        }
        snapshot_oid = object_store.put_json(snapshot_payload)
        ref_store.ensure_head()
        ref_store.write_ref("refs/heads/main", snapshot_oid)

        view = SnapshotView.open(synix_dir)
        chain = view.get_provenance("roll-1")

        assert chain == ["roll-1", "ep-1"]

    def test_provenance_no_parents(self, tmp_path):
        """get_provenance for a root artifact returns just itself."""
        synix_dir, _, _, _ = _create_snapshot(tmp_path)

        view = SnapshotView.open(synix_dir)
        chain = view.get_provenance("ep-1")

        assert chain == ["ep-1"]

    def test_provenance_handles_missing_parent_gracefully(self, tmp_path):
        """get_provenance handles parent labels not in the manifest."""
        synix_dir = tmp_path / ".synix"
        object_store = ObjectStore(synix_dir)
        ref_store = RefStore(synix_dir)

        # Create artifact with parent_labels pointing to a non-manifest label
        orphan_oid = _store_artifact(object_store, "orphan", "Orphan content", parent_labels=["missing-parent"])
        artifacts = {"orphan": orphan_oid}
        manifest_payload = {
            "type": "manifest",
            "schema_version": SCHEMA_VERSION,
            "pipeline_name": "orphan-test",
            "pipeline_fingerprint": "sha256:test",
            "artifacts": [{"label": "orphan", "oid": orphan_oid}],
            "projections": {},
        }
        manifest_oid = object_store.put_json(manifest_payload)
        snapshot_payload = {
            "type": "snapshot",
            "schema_version": SCHEMA_VERSION,
            "manifest_oid": manifest_oid,
            "parent_snapshot_oids": [],
            "created_at": datetime.now(UTC).isoformat(),
            "pipeline_name": "orphan-test",
            "run_id": "20260307T120000000002Z",
        }
        snapshot_oid = object_store.put_json(snapshot_payload)
        ref_store.ensure_head()
        ref_store.write_ref("refs/heads/main", snapshot_oid)

        view = SnapshotView.open(synix_dir)
        chain = view.get_provenance("orphan")

        # missing-parent is visited but can't be resolved further
        assert chain == ["orphan", "missing-parent"]


class TestSnapshotViewResolvePrefix:
    def test_exact_match(self, tmp_path):
        """resolve_prefix returns the label on exact match."""
        synix_dir, _, _, _ = _create_snapshot(tmp_path)

        view = SnapshotView.open(synix_dir)

        assert view.resolve_prefix("ep-1") == "ep-1"

    def test_label_prefix_unique(self, tmp_path):
        """resolve_prefix resolves a unique label prefix."""
        synix_dir = tmp_path / ".synix"
        object_store = ObjectStore(synix_dir)
        ref_store = RefStore(synix_dir)

        a_oid = _store_artifact(object_store, "alpha-one", "Alpha content")
        b_oid = _store_artifact(object_store, "beta-one", "Beta content")
        artifacts = {"alpha-one": a_oid, "beta-one": b_oid}

        manifest_payload = {
            "type": "manifest",
            "schema_version": SCHEMA_VERSION,
            "pipeline_name": "prefix-test",
            "pipeline_fingerprint": "sha256:test",
            "artifacts": [{"label": lbl, "oid": oid} for lbl, oid in sorted(artifacts.items())],
            "projections": {},
        }
        manifest_oid = object_store.put_json(manifest_payload)
        snapshot_payload = {
            "type": "snapshot",
            "schema_version": SCHEMA_VERSION,
            "manifest_oid": manifest_oid,
            "parent_snapshot_oids": [],
            "created_at": datetime.now(UTC).isoformat(),
            "pipeline_name": "prefix-test",
            "run_id": "20260307T120000000003Z",
        }
        snapshot_oid = object_store.put_json(snapshot_payload)
        ref_store.ensure_head()
        ref_store.write_ref("refs/heads/main", snapshot_oid)

        view = SnapshotView.open(synix_dir)

        assert view.resolve_prefix("alpha") == "alpha-one"
        assert view.resolve_prefix("beta") == "beta-one"

    def test_ambiguous_label_prefix_raises(self, tmp_path):
        """resolve_prefix raises ValueError for ambiguous label prefix."""
        synix_dir, _, _, _ = _create_snapshot(tmp_path)

        view = SnapshotView.open(synix_dir)

        with pytest.raises(ValueError, match="ambiguous prefix"):
            view.resolve_prefix("ep-")

    def test_hash_prefix_unique(self, tmp_path):
        """resolve_prefix resolves a unique hash prefix."""
        synix_dir, _, _, _ = _create_snapshot(tmp_path)
        object_store = ObjectStore(synix_dir)

        view = SnapshotView.open(synix_dir)

        # Get the artifact_id hash for ep-1
        ep1_oid = view._artifact_oids["ep-1"]
        ep1_obj = object_store.get_json(ep1_oid)
        aid_hash = ep1_obj["artifact_id"].removeprefix("sha256:")

        # Use a short prefix of the hash
        short_prefix = aid_hash[:8]
        result = view.resolve_prefix(short_prefix)
        # It should resolve to ep-1 (assuming no hash collision with ep-2 on 8 chars)
        assert result is not None

    def test_no_match_returns_none(self, tmp_path):
        """resolve_prefix returns None when nothing matches."""
        synix_dir, _, _, _ = _create_snapshot(tmp_path)

        view = SnapshotView.open(synix_dir)

        assert view.resolve_prefix("zzz-nonexistent") is None

    def test_sha256_prefix_stripped(self, tmp_path):
        """resolve_prefix strips 'sha256:' before hash matching."""
        synix_dir, _, _, _ = _create_snapshot(tmp_path)
        object_store = ObjectStore(synix_dir)

        view = SnapshotView.open(synix_dir)

        ep1_oid = view._artifact_oids["ep-1"]
        ep1_obj = object_store.get_json(ep1_oid)
        aid_hash = ep1_obj["artifact_id"].removeprefix("sha256:")

        result = view.resolve_prefix(f"sha256:{aid_hash[:8]}")
        assert result is not None
