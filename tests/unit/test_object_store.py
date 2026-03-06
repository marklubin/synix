"""Unit tests for the immutable Synix object store."""

from __future__ import annotations

from synix.build.object_store import ObjectStore


class TestObjectStore:
    def test_put_bytes_round_trips(self, tmp_path):
        """Raw bytes can be stored and loaded by oid."""
        store = ObjectStore(tmp_path / ".synix")

        oid = store.put_bytes(b"hello, snapshots")

        assert store.get_bytes(oid) == b"hello, snapshots"

    def test_put_json_is_canonical(self, tmp_path):
        """Equivalent JSON payloads produce the same oid."""
        store = ObjectStore(tmp_path / ".synix")

        oid1 = store.put_json({"b": 2, "a": 1})
        oid2 = store.put_json({"a": 1, "b": 2})

        assert oid1 == oid2
        assert store.get_json(oid1) == {"a": 1, "b": 2}
