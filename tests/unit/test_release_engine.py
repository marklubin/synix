"""Unit tests for the release engine orchestration."""

from __future__ import annotations

import json

import pytest

from synix.build.release_engine import (
    ReleaseReceipt,
    execute_release,
    get_release,
    list_releases,
)
from synix.core.models import Artifact
from tests.helpers.snapshot_factory import create_test_snapshot


def _make_artifact(label, content, atype="episode", layer_name="episodes"):
    return Artifact(
        label=label,
        artifact_type=atype,
        content=content,
        metadata={"layer_name": layer_name, "layer_level": 0},
    )


def _make_snapshot_with_projections(tmp_path, projections=None):
    """Create a snapshot with artifacts and structured projection declarations."""
    ep1 = _make_artifact("ep-1", "Episode one content")
    ep2 = _make_artifact("ep-2", "Episode two content")
    core = _make_artifact("core-1", "Core memory content", atype="core", layer_name="core")

    if projections is None:
        projections = {
            "search": {
                "adapter": "synix_search",
                "input_artifacts": ["ep-1", "ep-2"],
                "config": {"modes": ["fulltext"]},
                "config_fingerprint": "sha256:test",
                "precomputed_oid": None,
            },
            "context-doc": {
                "adapter": "flat_file",
                "input_artifacts": ["core-1"],
                "config": {"output_path": "context.md"},
                "config_fingerprint": "sha256:test2",
                "precomputed_oid": None,
            },
        }

    synix_dir = create_test_snapshot(
        tmp_path,
        {"episodes": [ep1, ep2], "core": [core]},
        projections=projections,
    )
    return synix_dir


class TestReleaseReceipt:
    def test_round_trip(self):
        receipt = ReleaseReceipt(
            release_name="local",
            snapshot_oid="a" * 64,
            manifest_oid="b" * 64,
            pipeline_name="test",
            released_at="2026-03-07T12:00:00Z",
            source_ref="HEAD",
            adapters={"search": {"adapter": "synix_search", "status": "success"}},
        )
        d = receipt.to_dict()
        restored = ReleaseReceipt.from_dict(d)
        assert restored.release_name == "local"
        assert restored.snapshot_oid == "a" * 64
        assert restored.adapters["search"]["status"] == "success"


class TestExecuteRelease:
    def test_release_creates_receipt(self, tmp_path):
        """execute_release writes a receipt.json."""
        synix_dir = _make_snapshot_with_projections(tmp_path)
        receipt = execute_release(synix_dir, release_name="local")

        assert receipt.release_name == "local"
        assert receipt.pipeline_name == "test-pipeline"
        assert receipt.snapshot_oid
        assert receipt.manifest_oid
        assert len(receipt.adapters) == 2

        # Receipt file on disk
        receipt_path = synix_dir / "releases" / "local" / "receipt.json"
        assert receipt_path.exists()
        data = json.loads(receipt_path.read_text())
        assert data["release_name"] == "local"

    def test_release_creates_search_db(self, tmp_path):
        """execute_release materializes search.db via synix_search adapter."""
        synix_dir = _make_snapshot_with_projections(tmp_path)
        execute_release(synix_dir, release_name="local")

        db_path = synix_dir / "releases" / "local" / "search.db"
        assert db_path.exists()

    def test_release_creates_context_md(self, tmp_path):
        """execute_release materializes context.md via flat_file adapter."""
        synix_dir = _make_snapshot_with_projections(tmp_path)
        execute_release(synix_dir, release_name="local")

        context_path = synix_dir / "releases" / "local" / "context.md"
        assert context_path.exists()
        content = context_path.read_text()
        assert "Core memory content" in content

    def test_release_advances_ref(self, tmp_path):
        """execute_release advances refs/releases/<name>."""
        from synix.build.refs import RefStore

        synix_dir = _make_snapshot_with_projections(tmp_path)
        receipt = execute_release(synix_dir, release_name="local")

        ref_store = RefStore(synix_dir)
        oid = ref_store.read_ref("refs/releases/local")
        assert oid == receipt.snapshot_oid

    def test_release_writes_history(self, tmp_path):
        """execute_release appends to history directory."""
        synix_dir = _make_snapshot_with_projections(tmp_path)
        execute_release(synix_dir, release_name="local")

        history_dir = synix_dir / "releases" / "local" / "history"
        assert history_dir.exists()
        history_files = list(history_dir.glob("*.json"))
        assert len(history_files) == 1

    def test_release_cleans_pending(self, tmp_path):
        """execute_release removes .pending.json on success."""
        synix_dir = _make_snapshot_with_projections(tmp_path)
        execute_release(synix_dir, release_name="local")

        pending = synix_dir / "releases" / "local" / ".pending.json"
        assert not pending.exists()

    def test_release_unresolved_ref_raises(self, tmp_path):
        """execute_release raises ValueError for non-existent ref."""
        synix_dir = _make_snapshot_with_projections(tmp_path)

        with pytest.raises(ValueError, match="does not resolve"):
            execute_release(synix_dir, ref="refs/heads/nonexistent", release_name="local")

    def test_release_idempotent(self, tmp_path):
        """Releasing the same snapshot twice produces valid state."""
        synix_dir = _make_snapshot_with_projections(tmp_path)
        r1 = execute_release(synix_dir, release_name="local")
        r2 = execute_release(synix_dir, release_name="local")

        assert r1.snapshot_oid == r2.snapshot_oid
        # History should have 2 entries
        history_dir = synix_dir / "releases" / "local" / "history"
        assert len(list(history_dir.glob("*.json"))) == 2

    def test_release_no_projections(self, tmp_path):
        """Release with no projections produces empty adapter receipts."""
        synix_dir = _make_snapshot_with_projections(tmp_path, projections={})
        receipt = execute_release(synix_dir, release_name="local")

        assert len(receipt.adapters) == 0
        # Receipt still written
        assert (synix_dir / "releases" / "local" / "receipt.json").exists()


class TestListReleases:
    def test_list_empty(self, tmp_path):
        synix_dir = tmp_path / ".synix"
        synix_dir.mkdir()
        assert list_releases(synix_dir) == []

    def test_list_after_release(self, tmp_path):
        synix_dir = _make_snapshot_with_projections(tmp_path)
        execute_release(synix_dir, release_name="local")

        releases = list_releases(synix_dir)
        assert len(releases) == 1
        assert releases[0]["release_name"] == "local"


class TestGetRelease:
    def test_get_existing(self, tmp_path):
        synix_dir = _make_snapshot_with_projections(tmp_path)
        execute_release(synix_dir, release_name="local")

        receipt = get_release(synix_dir, "local")
        assert receipt is not None
        assert receipt.release_name == "local"

    def test_get_nonexistent(self, tmp_path):
        synix_dir = tmp_path / ".synix"
        synix_dir.mkdir()
        assert get_release(synix_dir, "nonexistent") is None
