"""Unit tests for Synix refs and HEAD resolution."""

from __future__ import annotations

import pytest

from synix.build.refs import DEFAULT_HEAD_REF, RefStore, synix_dir_for_build_dir

OID1 = "1" * 64
OID2 = "2" * 64


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
        store.write_ref(DEFAULT_HEAD_REF, OID1)

        assert store.read_ref("HEAD") == OID1
        assert store.read_ref(DEFAULT_HEAD_REF) == OID1

    def test_iter_refs_lists_run_refs(self, tmp_path):
        """Run refs are discoverable under refs/runs."""
        store = RefStore(tmp_path / ".synix")
        store.write_ref("refs/runs/run-1", OID1)
        store.write_ref("refs/runs/run-2", OID2)

        assert store.iter_refs("refs/runs") == [
            ("refs/runs/run-1", OID1),
            ("refs/runs/run-2", OID2),
        ]

    def test_read_ref_rejects_symbolic_cycles(self, tmp_path):
        """Corrupt symbolic refs fail loudly instead of looping forever."""
        store = RefStore(tmp_path / ".synix")
        store.ensure_head()
        store.write_head("refs/heads/loop")
        (tmp_path / ".synix" / "refs" / "heads").mkdir(parents=True, exist_ok=True)
        (tmp_path / ".synix" / "refs" / "heads" / "loop").write_text("ref: refs/heads/loop\n", encoding="utf-8")

        with pytest.raises(ValueError, match="cycle"):
            store.read_ref("HEAD")

    def test_synix_dir_prefers_persistent_sibling_store(self, tmp_path):
        """Default store placement stays outside build/ so clean does not wipe history."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        assert synix_dir_for_build_dir(build_dir) == tmp_path / ".synix"

    def test_synix_dir_rejects_ambiguous_legacy_and_nested_store(self, tmp_path):
        """Ambiguous store discovery should fail loudly instead of silently forking history."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        (tmp_path / ".synix").mkdir()
        (build_dir / ".synix").mkdir()

        with pytest.raises(ValueError, match="ambiguous snapshot store resolution"):
            synix_dir_for_build_dir(build_dir)
