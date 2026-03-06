"""Unit tests for Synix refs and HEAD resolution."""

from __future__ import annotations

from synix.build.refs import DEFAULT_HEAD_REF, RefStore


class TestRefStore:
    def test_ensure_head_creates_default_symbolic_ref(self, tmp_path):
        """HEAD is created as a symbolic ref to the default build branch."""
        store = RefStore(tmp_path / ".synix")

        head_target = store.ensure_head()

        assert head_target == DEFAULT_HEAD_REF
        assert store.read_head_target() == DEFAULT_HEAD_REF
        assert (tmp_path / ".synix" / "HEAD").read_text() == f"ref: {DEFAULT_HEAD_REF}\n"

    def test_read_ref_resolves_head_symbolically(self, tmp_path):
        """Reading HEAD resolves through the symbolic ref chain."""
        store = RefStore(tmp_path / ".synix")
        store.ensure_head()
        store.write_ref(DEFAULT_HEAD_REF, "snapshot-oid-1")

        assert store.read_ref("HEAD") == "snapshot-oid-1"
        assert store.read_ref(DEFAULT_HEAD_REF) == "snapshot-oid-1"

    def test_iter_refs_lists_run_refs(self, tmp_path):
        """Run refs are discoverable under refs/runs."""
        store = RefStore(tmp_path / ".synix")
        store.write_ref("refs/runs/run-1", "snapshot-oid-1")
        store.write_ref("refs/runs/run-2", "snapshot-oid-2")

        assert store.iter_refs("refs/runs") == [
            ("refs/runs/run-1", "snapshot-oid-1"),
            ("refs/runs/run-2", "snapshot-oid-2"),
        ]
