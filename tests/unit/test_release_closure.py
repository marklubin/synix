"""Unit tests for ReleaseClosure — fully-resolved artifact bundle from snapshots."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest

from synix.build.object_store import SCHEMA_VERSION, ObjectStore
from synix.build.refs import RefStore
from synix.build.release import ProjectionDeclaration, ReleaseClosure, ResolvedArtifact


def _canonical_json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _store_artifact(
    object_store: ObjectStore,
    label: str,
    content: str,
    *,
    artifact_type: str = "episode",
    parent_labels: list[str] | None = None,
    layer_name: str = "transcripts",
    layer_level: int = 0,
) -> str:
    """Store a content blob + artifact object and return the artifact oid."""
    content_oid = object_store.put_bytes(content.encode("utf-8"))
    content_hash = f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"
    artifact_payload = {
        "type": "artifact",
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "artifact_type": artifact_type,
        "artifact_id": content_hash,
        "content_oid": content_oid,
        "input_ids": [],
        "metadata": {"layer_name": layer_name, "layer_level": layer_level},
        "parent_labels": parent_labels or [],
    }
    return object_store.put_json(artifact_payload)


def _create_snapshot(
    tmp_path,
    *,
    artifacts: dict[str, str] | None = None,
    content_map: dict[str, str] | None = None,
    artifact_oids: dict[str, str] | None = None,
    projections: dict | None = None,
    pipeline_name: str = "test-pipeline",
):
    """Create a full snapshot chain and return (synix_dir, snapshot_oid, manifest_oid, artifact_oids)."""
    synix_dir = tmp_path / ".synix"
    object_store = ObjectStore(synix_dir)
    ref_store = RefStore(synix_dir)

    if artifact_oids is None:
        if content_map is None:
            content_map = {"ep-1": "Episode one content", "ep-2": "Episode two content"}
        artifact_oids = {}
        for label, content in content_map.items():
            artifact_oids[label] = _store_artifact(object_store, label, content)

    manifest_payload = {
        "type": "manifest",
        "schema_version": SCHEMA_VERSION,
        "pipeline_name": pipeline_name,
        "pipeline_fingerprint": "sha256:test",
        "artifacts": [{"label": lbl, "oid": oid} for lbl, oid in sorted(artifact_oids.items())],
        "projections": projections if projections is not None else {},
    }
    manifest_oid = object_store.put_json(manifest_payload)

    snapshot_payload = {
        "type": "snapshot",
        "schema_version": SCHEMA_VERSION,
        "manifest_oid": manifest_oid,
        "parent_snapshot_oids": [],
        "created_at": datetime.now(UTC).isoformat(),
        "pipeline_name": pipeline_name,
        "run_id": "20260307T120000000000Z",
    }
    snapshot_oid = object_store.put_json(snapshot_payload)

    ref_store.ensure_head()
    ref_store.write_ref("refs/heads/main", snapshot_oid)

    return synix_dir, snapshot_oid, manifest_oid, artifact_oids


class TestFromSnapshotLoadsArtifacts:
    def test_from_snapshot_loads_artifacts(self, tmp_path):
        """Create a snapshot with 2 artifacts, build closure, verify both resolved."""
        synix_dir, snapshot_oid, _, _ = _create_snapshot(tmp_path)

        closure = ReleaseClosure.from_snapshot(synix_dir, snapshot_oid)

        assert len(closure.artifacts) == 2
        assert "ep-1" in closure.artifacts
        assert "ep-2" in closure.artifacts

        ep1 = closure.artifacts["ep-1"]
        assert isinstance(ep1, ResolvedArtifact)
        assert ep1.label == "ep-1"
        assert ep1.content == "Episode one content"
        assert ep1.artifact_type == "episode"
        assert ep1.layer_name == "transcripts"
        assert ep1.layer_level == 0
        assert ep1.artifact_id.startswith("sha256:")

        ep2 = closure.artifacts["ep-2"]
        assert ep2.content == "Episode two content"

        # Verify closure-level fields
        assert closure.snapshot_oid == snapshot_oid
        assert closure.pipeline_name == "test-pipeline"


class TestFromSnapshotResolvesProvenance:
    def test_from_snapshot_resolves_provenance(self, tmp_path):
        """Create artifacts with parent_labels chain, verify provenance_chain."""
        synix_dir = tmp_path / ".synix"
        object_store = ObjectStore(synix_dir)

        # ep-1 is a root, roll-1 depends on ep-1
        ep_oid = _store_artifact(object_store, "ep-1", "Episode content", layer_name="episodes", layer_level=1)
        roll_oid = _store_artifact(
            object_store,
            "roll-1",
            "Rollup content",
            artifact_type="rollup",
            parent_labels=["ep-1"],
            layer_name="rollups",
            layer_level=2,
        )

        artifact_oids = {"ep-1": ep_oid, "roll-1": roll_oid}
        synix_dir, snapshot_oid, _, _ = _create_snapshot(tmp_path, artifact_oids=artifact_oids)

        closure = ReleaseClosure.from_snapshot(synix_dir, snapshot_oid)

        # roll-1 provenance should walk to ep-1
        assert closure.artifacts["roll-1"].provenance_chain == ["roll-1", "ep-1"]
        # ep-1 is a root — provenance is just itself
        assert closure.artifacts["ep-1"].provenance_chain == ["ep-1"]


class TestFromSnapshotParsesProjections:
    def test_from_snapshot_parses_projections(self, tmp_path):
        """Create manifest with structured projections, verify parsing."""
        projections = {
            "search": {
                "adapter": "synix_search",
                "input_artifacts": ["ep-1", "ep-2"],
                "config": {"modes": ["fulltext"]},
                "config_fingerprint": "sha256:abc123",
                "precomputed_oid": None,
            },
            "context-doc": {
                "adapter": "flat_file",
                "input_artifacts": ["ep-1"],
                "config": {"output_template": "context.md"},
                "config_fingerprint": "sha256:def456",
                "precomputed_oid": "aabbccdd" * 8,
            },
        }
        synix_dir, snapshot_oid, _, _ = _create_snapshot(tmp_path, projections=projections)

        closure = ReleaseClosure.from_snapshot(synix_dir, snapshot_oid)

        assert len(closure.projections) == 2

        search_proj = closure.projections["search"]
        assert isinstance(search_proj, ProjectionDeclaration)
        assert search_proj.name == "search"
        assert search_proj.adapter == "synix_search"
        assert search_proj.input_artifacts == ["ep-1", "ep-2"]
        assert search_proj.config == {"modes": ["fulltext"]}
        assert search_proj.config_fingerprint == "sha256:abc123"
        assert search_proj.precomputed_oid is None

        ctx_proj = closure.projections["context-doc"]
        assert ctx_proj.adapter == "flat_file"
        assert ctx_proj.precomputed_oid == "aabbccdd" * 8


class TestFromRefResolvesHead:
    def test_from_ref_resolves_head(self, tmp_path):
        """Create a snapshot pointed to by HEAD, verify from_ref works."""
        synix_dir, snapshot_oid, _, _ = _create_snapshot(tmp_path)

        closure = ReleaseClosure.from_ref(synix_dir, ref="HEAD")

        assert closure.snapshot_oid == snapshot_oid
        assert len(closure.artifacts) == 2


class TestFromRefUnresolvedRaises:
    def test_from_ref_unresolved_raises(self, tmp_path):
        """Verify ValueError for non-existent ref."""
        synix_dir = tmp_path / ".synix"
        ObjectStore(synix_dir)  # ensure dir exists
        RefStore(synix_dir)

        with pytest.raises(ValueError, match="does not resolve to a snapshot"):
            ReleaseClosure.from_ref(synix_dir, ref="refs/heads/nonexistent")


class TestFromSnapshotNonSnapshotRaises:
    def test_from_snapshot_non_snapshot_raises(self, tmp_path):
        """Verify ValueError when OID points to non-snapshot object."""
        synix_dir = tmp_path / ".synix"
        object_store = ObjectStore(synix_dir)

        manifest_payload = {
            "type": "manifest",
            "schema_version": SCHEMA_VERSION,
            "pipeline_name": "bad-ref",
            "pipeline_fingerprint": "sha256:test",
            "artifacts": [],
            "projections": {},
        }
        manifest_oid = object_store.put_json(manifest_payload)

        with pytest.raises(ValueError, match="expected 'snapshot'"):
            ReleaseClosure.from_snapshot(synix_dir, manifest_oid)


class TestArtifactsForProjection:
    def test_artifacts_for_projection(self, tmp_path):
        """Create closure with 3 artifacts, projection referencing 2, verify filtering."""
        synix_dir = tmp_path / ".synix"
        object_store = ObjectStore(synix_dir)

        ep1_oid = _store_artifact(object_store, "ep-1", "Episode one")
        ep2_oid = _store_artifact(object_store, "ep-2", "Episode two")
        ep3_oid = _store_artifact(object_store, "ep-3", "Episode three")

        artifact_oids = {"ep-1": ep1_oid, "ep-2": ep2_oid, "ep-3": ep3_oid}
        projections = {
            "search": {
                "adapter": "synix_search",
                "input_artifacts": ["ep-1", "ep-3"],
                "config": {},
                "config_fingerprint": "sha256:test",
                "precomputed_oid": None,
            },
        }
        synix_dir, snapshot_oid, _, _ = _create_snapshot(tmp_path, artifact_oids=artifact_oids, projections=projections)

        closure = ReleaseClosure.from_snapshot(synix_dir, snapshot_oid)
        filtered = closure.artifacts_for_projection("search")

        assert len(filtered) == 2
        assert "ep-1" in filtered
        assert "ep-3" in filtered
        assert "ep-2" not in filtered


class TestArtifactsForProjectionMissingRaises:
    def test_artifacts_for_projection_missing_raises(self, tmp_path):
        """Verify KeyError for unknown projection name."""
        synix_dir, snapshot_oid, _, _ = _create_snapshot(tmp_path)

        closure = ReleaseClosure.from_snapshot(synix_dir, snapshot_oid)

        with pytest.raises(KeyError, match="not found in closure"):
            closure.artifacts_for_projection("nonexistent")


class TestEmptyProjections:
    def test_empty_projections(self, tmp_path):
        """Closure with no projections works fine."""
        synix_dir, snapshot_oid, _, _ = _create_snapshot(tmp_path, projections={})

        closure = ReleaseClosure.from_snapshot(synix_dir, snapshot_oid)

        assert closure.projections == {}
        assert len(closure.artifacts) == 2


class TestProvenanceHandlesMissingParent:
    def test_provenance_handles_missing_parent(self, tmp_path):
        """Parent label not in manifest doesn't crash, chain includes missing label."""
        synix_dir = tmp_path / ".synix"
        object_store = ObjectStore(synix_dir)

        # Create artifact with parent_labels pointing to a non-manifest label
        orphan_oid = _store_artifact(object_store, "orphan", "Orphan content", parent_labels=["missing-parent"])

        artifact_oids = {"orphan": orphan_oid}
        synix_dir, snapshot_oid, _, _ = _create_snapshot(tmp_path, artifact_oids=artifact_oids)

        closure = ReleaseClosure.from_snapshot(synix_dir, snapshot_oid)

        chain = closure.artifacts["orphan"].provenance_chain
        # missing-parent is visited but can't be resolved further
        assert chain == ["orphan", "missing-parent"]
