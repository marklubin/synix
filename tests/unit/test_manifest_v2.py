"""Dedicated tests for manifest schema v2 structured projection entries."""

from __future__ import annotations

import pytest

from synix.build.object_store import SCHEMA_VERSION, ObjectStore


def _valid_manifest(projections: dict | None = None) -> dict:
    """Return a valid v2 manifest payload with optional projections override."""
    return {
        "type": "manifest",
        "schema_version": SCHEMA_VERSION,
        "pipeline_name": "manifest-v2-test",
        "pipeline_fingerprint": "sha256:test",
        "artifacts": [{"label": "a", "oid": "0" * 64}],
        "projections": projections if projections is not None else {},
    }


def _valid_projection_entry(**overrides: object) -> dict:
    """Return a valid structured projection entry with optional overrides."""
    entry: dict = {
        "adapter": "search_index",
        "input_artifacts": ["a"],
        "config": {"search": ["fulltext"]},
        "config_fingerprint": "sha256:abc123",
        "precomputed_oid": None,
    }
    entry.update(overrides)
    return entry


class TestManifestV2ProjectionEntries:
    def test_empty_projections_dict_is_valid(self, tmp_path):
        """An empty projections dict is valid for v2 manifests."""
        store = ObjectStore(tmp_path / ".synix")
        oid = store.put_json(_valid_manifest(projections={}))
        loaded = store.get_json(oid)
        assert loaded["projections"] == {}

    def test_valid_projection_entry_round_trips(self, tmp_path):
        """A well-formed structured projection entry passes validation."""
        store = ObjectStore(tmp_path / ".synix")
        proj = _valid_projection_entry()
        manifest = _valid_manifest(projections={"memory-index": proj})
        oid = store.put_json(manifest)
        loaded = store.get_json(oid)
        assert loaded["projections"]["memory-index"]["adapter"] == "search_index"
        assert loaded["projections"]["memory-index"]["input_artifacts"] == ["a"]
        assert loaded["projections"]["memory-index"]["config"] == {"search": ["fulltext"]}
        assert loaded["projections"]["memory-index"]["config_fingerprint"] == "sha256:abc123"
        assert loaded["projections"]["memory-index"]["precomputed_oid"] is None

    def test_precomputed_oid_string_is_valid(self, tmp_path):
        """A non-None string precomputed_oid is valid."""
        store = ObjectStore(tmp_path / ".synix")
        proj = _valid_projection_entry(precomputed_oid="f" * 64)
        manifest = _valid_manifest(projections={"idx": proj})
        oid = store.put_json(manifest)
        assert store.get_json(oid)["projections"]["idx"]["precomputed_oid"] == "f" * 64

    def test_precomputed_oid_absent_is_valid(self, tmp_path):
        """A projection entry without precomputed_oid at all is valid."""
        store = ObjectStore(tmp_path / ".synix")
        proj = _valid_projection_entry()
        del proj["precomputed_oid"]
        manifest = _valid_manifest(projections={"idx": proj})
        oid = store.put_json(manifest)
        assert "precomputed_oid" not in store.get_json(oid)["projections"]["idx"]

    def test_missing_adapter_field_raises(self, tmp_path):
        """A projection entry missing 'adapter' is rejected."""
        store = ObjectStore(tmp_path / ".synix")
        proj = _valid_projection_entry()
        del proj["adapter"]
        with pytest.raises(ValueError, match="missing required field 'adapter'"):
            store.put_json(_valid_manifest(projections={"bad": proj}))

    def test_missing_input_artifacts_field_raises(self, tmp_path):
        """A projection entry missing 'input_artifacts' is rejected."""
        store = ObjectStore(tmp_path / ".synix")
        proj = _valid_projection_entry()
        del proj["input_artifacts"]
        with pytest.raises(ValueError, match="missing required field 'input_artifacts'"):
            store.put_json(_valid_manifest(projections={"bad": proj}))

    def test_missing_config_field_raises(self, tmp_path):
        """A projection entry missing 'config' is rejected."""
        store = ObjectStore(tmp_path / ".synix")
        proj = _valid_projection_entry()
        del proj["config"]
        with pytest.raises(ValueError, match="missing required field 'config'"):
            store.put_json(_valid_manifest(projections={"bad": proj}))

    def test_missing_config_fingerprint_field_raises(self, tmp_path):
        """A projection entry missing 'config_fingerprint' is rejected."""
        store = ObjectStore(tmp_path / ".synix")
        proj = _valid_projection_entry()
        del proj["config_fingerprint"]
        with pytest.raises(ValueError, match="missing required field 'config_fingerprint'"):
            store.put_json(_valid_manifest(projections={"bad": proj}))

    def test_wrong_type_adapter_raises(self, tmp_path):
        """A projection entry with non-string adapter is rejected."""
        store = ObjectStore(tmp_path / ".synix")
        proj = _valid_projection_entry(adapter=42)
        with pytest.raises(ValueError, match="field 'adapter'.*must be of type str"):
            store.put_json(_valid_manifest(projections={"bad": proj}))

    def test_wrong_type_input_artifacts_raises(self, tmp_path):
        """A projection entry with non-list input_artifacts is rejected."""
        store = ObjectStore(tmp_path / ".synix")
        proj = _valid_projection_entry(input_artifacts="not-a-list")
        with pytest.raises(ValueError, match="field 'input_artifacts'.*must be of type list"):
            store.put_json(_valid_manifest(projections={"bad": proj}))

    def test_wrong_type_config_raises(self, tmp_path):
        """A projection entry with non-dict config is rejected."""
        store = ObjectStore(tmp_path / ".synix")
        proj = _valid_projection_entry(config="not-a-dict")
        with pytest.raises(ValueError, match="field 'config'.*must be of type dict"):
            store.put_json(_valid_manifest(projections={"bad": proj}))

    def test_wrong_type_config_fingerprint_raises(self, tmp_path):
        """A projection entry with non-string config_fingerprint is rejected."""
        store = ObjectStore(tmp_path / ".synix")
        proj = _valid_projection_entry(config_fingerprint=123)
        with pytest.raises(ValueError, match="field 'config_fingerprint'.*must be of type str"):
            store.put_json(_valid_manifest(projections={"bad": proj}))

    def test_wrong_type_precomputed_oid_raises(self, tmp_path):
        """A projection entry with non-string/non-None precomputed_oid is rejected."""
        store = ObjectStore(tmp_path / ".synix")
        proj = _valid_projection_entry(precomputed_oid=42)
        with pytest.raises(ValueError, match="'precomputed_oid'.*must be a string or None"):
            store.put_json(_valid_manifest(projections={"bad": proj}))

    def test_projection_entry_not_a_dict_raises(self, tmp_path):
        """A projection entry that is not a dict is rejected."""
        store = ObjectStore(tmp_path / ".synix")
        with pytest.raises(ValueError, match="must be an object"):
            store.put_json(_valid_manifest(projections={"bad": "not-a-dict"}))

    def test_null_adapter_raises(self, tmp_path):
        """A projection entry with adapter=None is rejected."""
        store = ObjectStore(tmp_path / ".synix")
        proj = _valid_projection_entry(adapter=None)
        with pytest.raises(ValueError, match="field 'adapter'.*got None"):
            store.put_json(_valid_manifest(projections={"bad": proj}))

    def test_multiple_projections_all_validated(self, tmp_path):
        """All projection entries are validated, not just the first."""
        store = ObjectStore(tmp_path / ".synix")
        good = _valid_projection_entry()
        bad = _valid_projection_entry()
        del bad["adapter"]
        with pytest.raises(ValueError, match="missing required field 'adapter'"):
            store.put_json(_valid_manifest(projections={"good": good, "bad": bad}))


class TestManifestV1BackwardCompat:
    def test_v1_manifest_with_plain_projections_dict_readable(self, tmp_path):
        """A v1 manifest stored with allow_older_schema can be read back without v2 validation."""
        store = ObjectStore(tmp_path / ".synix")
        # Write a v1 manifest directly as raw bytes to bypass put_json v2 validation
        import hashlib
        import json

        v1_manifest = {
            "type": "manifest",
            "schema_version": 1,
            "pipeline_name": "legacy",
            "pipeline_fingerprint": "sha256:legacy",
            "artifacts": [{"label": "x", "oid": "a" * 64}],
            "projections": {},
        }
        encoded = json.dumps(v1_manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        oid = hashlib.sha256(encoded).hexdigest()
        path = store._path_for_oid(oid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)

        # get_json uses allow_older_schema=True, so this should work
        loaded = store.get_json(oid)
        assert loaded["schema_version"] == 1
        assert loaded["projections"] == {}

    def test_v1_manifest_with_unstructured_projections_readable(self, tmp_path):
        """A v1 manifest with plain string values in projections can be read."""
        store = ObjectStore(tmp_path / ".synix")
        import hashlib
        import json

        v1_manifest = {
            "type": "manifest",
            "schema_version": 1,
            "pipeline_name": "legacy",
            "pipeline_fingerprint": "sha256:legacy",
            "artifacts": [{"label": "x", "oid": "a" * 64}],
            "projections": {"memory-index": "some-oid-string"},
        }
        encoded = json.dumps(v1_manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        oid = hashlib.sha256(encoded).hexdigest()
        path = store._path_for_oid(oid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)

        loaded = store.get_json(oid)
        assert loaded["schema_version"] == 1
        assert loaded["projections"]["memory-index"] == "some-oid-string"

    def test_put_json_rejects_v1_schema_version(self, tmp_path):
        """put_json (which uses allow_older_schema=False) rejects v1 payloads now that SCHEMA_VERSION is 2."""
        store = ObjectStore(tmp_path / ".synix")
        with pytest.raises(ValueError, match="schema_version=2"):
            store.put_json(
                {
                    "type": "manifest",
                    "schema_version": 1,
                    "pipeline_name": "old",
                    "pipeline_fingerprint": "sha256:old",
                    "artifacts": [],
                    "projections": {},
                }
            )
