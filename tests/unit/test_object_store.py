"""Unit tests for the immutable Synix object store."""

from __future__ import annotations

import hashlib

import pytest

from synix.build.object_store import SCHEMA_VERSION, ObjectStore


class TestObjectStore:
    def test_put_bytes_round_trips(self, tmp_path):
        """Raw bytes can be stored and loaded by oid."""
        store = ObjectStore(tmp_path / ".synix")

        oid = store.put_bytes(b"hello, snapshots")

        assert store.get_bytes(oid) == b"hello, snapshots"

    def test_put_json_is_canonical(self, tmp_path):
        """Equivalent JSON payloads produce the same oid."""
        store = ObjectStore(tmp_path / ".synix")

        payload1 = {
            "type": "manifest",
            "schema_version": SCHEMA_VERSION,
            "pipeline_name": "test",
            "pipeline_fingerprint": "sha256:test",
            "artifacts": {"a": "0" * 64},
            "projections": {},
        }
        payload2 = {
            "projections": {},
            "pipeline_fingerprint": "sha256:test",
            "pipeline_name": "test",
            "artifacts": {"a": "0" * 64},
            "schema_version": SCHEMA_VERSION,
            "type": "manifest",
        }

        oid1 = store.put_json(payload1)
        oid2 = store.put_json(payload2)

        assert oid1 == oid2
        assert store.get_json(oid1) == payload1

    def test_put_json_rejects_missing_schema_version(self, tmp_path):
        """Snapshot objects must declare a schema version."""
        store = ObjectStore(tmp_path / ".synix")

        with pytest.raises(ValueError, match="schema_version"):
            store.put_json({"type": "manifest", "pipeline_name": "test"})

    def test_put_bytes_oid_matches_sha256(self, tmp_path):
        """Raw byte object ids are pinned to sha256(content)."""
        store = ObjectStore(tmp_path / ".synix")
        payload = b"synix-blob-contract"

        oid = store.put_bytes(payload)

        assert oid == hashlib.sha256(payload).hexdigest()

    def test_put_text_rejects_non_string(self, tmp_path):
        """Text objects must be explicitly textual."""
        store = ObjectStore(tmp_path / ".synix")

        with pytest.raises(TypeError, match="text must be a string"):
            store.put_text(b"not-text")  # type: ignore[arg-type]
