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


class TestReleaseFailureAtomicity:
    """Tests that release failures leave no partial state."""

    def _register_fake_adapters(self, adapter_map):
        """Register fake adapters in the adapter registry.

        adapter_map: dict of adapter_name -> ProjectionAdapter instance
        """
        from synix.build.adapters import _ADAPTER_REGISTRY

        for name, adapter in adapter_map.items():
            _ADAPTER_REGISTRY[name] = type(adapter)

    def _cleanup_registry(self, *names):
        from synix.build.adapters import _ADAPTER_REGISTRY

        for name in names:
            _ADAPTER_REGISTRY.pop(name, None)

    def test_adapter_apply_failure_no_receipt_no_ref(self, tmp_path):
        """When adapter.apply() raises, no receipt.json is written,
        no release ref is advanced, and .pending.json is preserved."""

        from synix.build.adapters import ProjectionAdapter, ReleasePlan
        from synix.build.refs import RefStore

        class FailingApplyAdapter(ProjectionAdapter):
            def plan(self, closure, declaration, current_receipt):
                return ReleasePlan(
                    adapter="failing_apply",
                    projection_name=declaration.name,
                    action="rebuild",
                    summary="will fail",
                    artifacts_count=1,
                )

            def apply(self, plan, target):
                raise RuntimeError("disk full")

            def verify(self, receipt, target):
                return True

        # Build snapshot with a single projection using the failing adapter
        projections = {
            "search": {
                "adapter": "failing_apply",
                "input_artifacts": ["ep-1"],
                "config": {"modes": ["fulltext"]},
                "config_fingerprint": "sha256:test",
                "precomputed_oid": None,
            },
        }
        synix_dir = _make_snapshot_with_projections(tmp_path, projections=projections)

        self._register_fake_adapters({"failing_apply": FailingApplyAdapter()})
        try:
            with pytest.raises(RuntimeError, match="disk full"):
                execute_release(synix_dir, release_name="local")

            # No receipt written
            receipt_path = synix_dir / "releases" / "local" / "receipt.json"
            assert not receipt_path.exists()

            # No release ref advanced
            ref_store = RefStore(synix_dir)
            assert ref_store.read_ref("refs/releases/local") is None

            # .pending.json preserved for diagnosis
            pending = synix_dir / "releases" / "local" / ".pending.json"
            assert pending.exists()
        finally:
            self._cleanup_registry("failing_apply")

    def test_adapter_verify_failure_no_receipt_no_ref(self, tmp_path):
        """When adapter.verify() returns False, no receipt.json is written,
        no release ref is advanced, and .pending.json is preserved."""
        from synix.build.adapters import AdapterReceipt, ProjectionAdapter, ReleasePlan
        from synix.build.refs import RefStore

        class FailingVerifyAdapter(ProjectionAdapter):
            def plan(self, closure, declaration, current_receipt):
                return ReleasePlan(
                    adapter="failing_verify",
                    projection_name=declaration.name,
                    action="rebuild",
                    summary="will fail verify",
                    artifacts_count=1,
                )

            def apply(self, plan, target):
                return AdapterReceipt(
                    adapter="failing_verify",
                    projection_name=plan.projection_name,
                    target=str(target),
                    artifacts_applied=1,
                    status="success",
                )

            def verify(self, receipt, target):
                return False

        projections = {
            "search": {
                "adapter": "failing_verify",
                "input_artifacts": ["ep-1"],
                "config": {"modes": ["fulltext"]},
                "config_fingerprint": "sha256:test",
                "precomputed_oid": None,
            },
        }
        synix_dir = _make_snapshot_with_projections(tmp_path, projections=projections)

        self._register_fake_adapters({"failing_verify": FailingVerifyAdapter()})
        try:
            with pytest.raises(RuntimeError, match="verification failed"):
                execute_release(synix_dir, release_name="local")

            # No receipt written
            receipt_path = synix_dir / "releases" / "local" / "receipt.json"
            assert not receipt_path.exists()

            # No release ref advanced
            ref_store = RefStore(synix_dir)
            assert ref_store.read_ref("refs/releases/local") is None

            # .pending.json preserved
            pending = synix_dir / "releases" / "local" / ".pending.json"
            assert pending.exists()
        finally:
            self._cleanup_registry("failing_verify")

    def test_idempotent_re_release_same_snapshot(self, tmp_path):
        """Releasing the same snapshot twice to the same target produces
        a valid receipt both times, and the ref points to the same snapshot."""
        from synix.build.refs import RefStore

        synix_dir = _make_snapshot_with_projections(tmp_path)
        r1 = execute_release(synix_dir, release_name="local")
        r2 = execute_release(synix_dir, release_name="local")

        # Both receipts are valid
        assert r1.release_name == "local"
        assert r2.release_name == "local"
        assert r1.snapshot_oid == r2.snapshot_oid
        assert r1.manifest_oid == r2.manifest_oid
        assert len(r1.adapters) == len(r2.adapters)

        # Receipt on disk matches second release
        receipt = get_release(synix_dir, "local")
        assert receipt is not None
        assert receipt.snapshot_oid == r2.snapshot_oid
        assert receipt.released_at == r2.released_at

        # Ref still points to the same snapshot
        ref_store = RefStore(synix_dir)
        oid = ref_store.read_ref("refs/releases/local")
        assert oid == r1.snapshot_oid

        # History has 2 entries
        history_dir = synix_dir / "releases" / "local" / "history"
        assert len(list(history_dir.glob("*.json"))) == 2

    def test_no_partial_receipts_on_second_adapter_failure(self, tmp_path):
        """With 2 projections, if the second adapter fails, no receipt.json
        is written — the first adapter's success is NOT persisted."""
        from synix.build.adapters import AdapterReceipt, ProjectionAdapter, ReleasePlan
        from synix.build.refs import RefStore

        class SuccessAdapter(ProjectionAdapter):
            def plan(self, closure, declaration, current_receipt):
                return ReleasePlan(
                    adapter="success_adapter",
                    projection_name=declaration.name,
                    action="rebuild",
                    summary="will succeed",
                    artifacts_count=1,
                )

            def apply(self, plan, target):
                return AdapterReceipt(
                    adapter="success_adapter",
                    projection_name=plan.projection_name,
                    target=str(target),
                    artifacts_applied=1,
                    status="success",
                )

            def verify(self, receipt, target):
                return True

        class BoomAdapter(ProjectionAdapter):
            def plan(self, closure, declaration, current_receipt):
                return ReleasePlan(
                    adapter="boom_adapter",
                    projection_name=declaration.name,
                    action="rebuild",
                    summary="will explode",
                    artifacts_count=1,
                )

            def apply(self, plan, target):
                raise RuntimeError("adapter explosion")

            def verify(self, receipt, target):
                return True

        # Two projections: first uses SuccessAdapter, second uses BoomAdapter.
        # Dict ordering is insertion order in Python 3.7+, so "alpha" runs first.
        projections = {
            "alpha": {
                "adapter": "success_adapter",
                "input_artifacts": ["ep-1"],
                "config": {},
                "config_fingerprint": "sha256:a",
                "precomputed_oid": None,
            },
            "beta": {
                "adapter": "boom_adapter",
                "input_artifacts": ["ep-2"],
                "config": {},
                "config_fingerprint": "sha256:b",
                "precomputed_oid": None,
            },
        }
        synix_dir = _make_snapshot_with_projections(tmp_path, projections=projections)

        self._register_fake_adapters(
            {
                "success_adapter": SuccessAdapter(),
                "boom_adapter": BoomAdapter(),
            }
        )
        try:
            with pytest.raises(RuntimeError, match="adapter explosion"):
                execute_release(synix_dir, release_name="local")

            # No receipt.json at all — first adapter's success is not persisted
            receipt_path = synix_dir / "releases" / "local" / "receipt.json"
            assert not receipt_path.exists()

            # No release ref advanced
            ref_store = RefStore(synix_dir)
            assert ref_store.read_ref("refs/releases/local") is None

            # .pending.json preserved
            pending = synix_dir / "releases" / "local" / ".pending.json"
            assert pending.exists()

            # No history entries
            history_dir = synix_dir / "releases" / "local" / "history"
            assert not history_dir.exists() or len(list(history_dir.glob("*.json"))) == 0
        finally:
            self._cleanup_registry("success_adapter", "boom_adapter")
