"""Unit tests for batch build state management."""

from __future__ import annotations

import json

from synix.build.batch_state import BatchState, BuildInstance


class TestBuildInstance:
    def test_to_dict_round_trips(self):
        instance = BuildInstance(
            build_id="batch-abc123",
            pipeline_hash="fp:deadbeef",
            status="submitted",
            created_at=1000.0,
            layers_completed=["transcripts"],
            current_layer="episodes",
            failed_requests=2,
            error=None,
        )
        d = instance.to_dict()
        restored = BuildInstance.from_dict(d)
        assert restored.build_id == "batch-abc123"
        assert restored.pipeline_hash == "fp:deadbeef"
        assert restored.status == "submitted"
        assert restored.layers_completed == ["transcripts"]
        assert restored.current_layer == "episodes"
        assert restored.failed_requests == 2
        assert restored.error is None

    def test_failed_requests_defaults_to_zero(self):
        d = {"build_id": "b1", "pipeline_hash": "fp:1234"}
        instance = BuildInstance.from_dict(d)
        assert instance.failed_requests == 0

    def test_completed_with_errors_status(self):
        instance = BuildInstance(
            build_id="b1",
            pipeline_hash="fp:1234",
            status="completed_with_errors",
            failed_requests=3,
        )
        d = instance.to_dict()
        restored = BuildInstance.from_dict(d)
        assert restored.status == "completed_with_errors"
        assert restored.failed_requests == 3


class TestBatchStateRequests:
    def test_queue_and_retrieve_by_layer(self, tmp_path):
        state = BatchState(tmp_path, "b1")
        state.queue_request("k1", "episodes", {"model": "gpt-4o"}, "ep-1")
        state.queue_request("k2", "rollups", {"model": "gpt-4o"}, "rollup-1")
        state.queue_request("k3", "episodes", {"model": "gpt-4o"}, "ep-2")

        ep_pending = state.get_pending("episodes")
        assert set(ep_pending.keys()) == {"k1", "k3"}

        all_pending = state.get_pending()
        assert len(all_pending) == 3

    def test_has_pending(self, tmp_path):
        state = BatchState(tmp_path, "b1")
        assert not state.has_pending()
        state.queue_request("k1", "layer", {}, "desc")
        assert state.has_pending()


class TestBatchStateTracking:
    def test_record_batch_maps_keys_and_clears_pending(self, tmp_path):
        state = BatchState(tmp_path, "b1")
        state.queue_request("k1", "episodes", {}, "ep-1")
        state.queue_request("k2", "episodes", {}, "ep-2")

        state.record_batch("batch-001", "episodes", ["k1", "k2"])

        assert state.get_pending("episodes") == {}
        assert state.get_batch_for_request("k1") == "batch-001"
        assert state.get_batch_for_request("k2") == "batch-001"

        batch = state.get_batch("batch-001")
        assert batch is not None
        assert batch["layer"] == "episodes"
        assert batch["keys"] == ["k1", "k2"]
        assert batch["status"] == "submitted"

    def test_update_batch_status(self, tmp_path):
        state = BatchState(tmp_path, "b1")
        state.record_batch("batch-001", "episodes", ["k1"])
        state.update_batch_status("batch-001", "completed")
        assert state.get_batch("batch-001")["status"] == "completed"

    def test_get_batches_by_layer(self, tmp_path):
        state = BatchState(tmp_path, "b1")
        state.record_batch("b1", "episodes", ["k1"])
        state.record_batch("b2", "rollups", ["k2"])
        state.record_batch("b3", "episodes", ["k3"])

        ep_batches = state.get_batches("episodes")
        assert set(ep_batches.keys()) == {"b1", "b3"}


class TestBatchStateResults:
    def test_store_and_retrieve_results(self, tmp_path):
        state = BatchState(tmp_path, "b1")
        state.store_result("k1", "summary text", "gpt-4o-mini", {"input": 100, "output": 50})

        result = state.get_result("k1")
        assert result is not None
        assert result["content"] == "summary text"
        assert result["model"] == "gpt-4o-mini"
        assert result["tokens"] == {"input": 100, "output": 50}

    def test_has_result(self, tmp_path):
        state = BatchState(tmp_path, "b1")
        assert not state.has_result("k1")
        state.store_result("k1", "text", "model", {})
        assert state.has_result("k1")


class TestBatchStateErrors:
    def test_store_and_retrieve_errors(self, tmp_path):
        state = BatchState(tmp_path, "b1")
        state.store_error("k1", "content_filter", "Content was filtered")

        error = state.get_error("k1")
        assert error is not None
        assert error["code"] == "content_filter"
        assert error["message"] == "Content was filtered"

    def test_get_all_errors(self, tmp_path):
        state = BatchState(tmp_path, "b1")
        state.store_error("k1", "err1", "msg1")
        state.store_error("k2", "err2", "msg2")

        errors = state.get_errors()
        assert len(errors) == 2


class TestBatchStatePersistence:
    def test_save_and_reload(self, tmp_path):
        state = BatchState(tmp_path, "b1")
        state.queue_request("k1", "episodes", {"m": "gpt-4o"}, "ep-1")
        state.record_batch("batch-001", "episodes", ["k1"])
        state.store_result("k1", "output", "gpt-4o", {"input": 10, "output": 5})
        state.store_error("k2", "err", "msg")
        state.save()

        state2 = BatchState(tmp_path, "b1")
        assert state2.get_result("k1") is not None
        assert state2.get_result("k1")["content"] == "output"
        assert state2.get_batch_for_request("k1") == "batch-001"
        assert state2.get_error("k2") is not None

    def test_manifest_save_and_reload(self, tmp_path):
        state = BatchState(tmp_path, "b1")
        instance = BuildInstance(
            build_id="b1",
            pipeline_hash="fp:1234",
            status="completed_with_errors",
            failed_requests=5,
        )
        state.save_manifest(instance)

        state2 = BatchState(tmp_path, "b1")
        loaded = state2.load_manifest()
        assert loaded is not None
        assert loaded.build_id == "b1"
        assert loaded.status == "completed_with_errors"
        assert loaded.failed_requests == 5

    def test_load_manifest_returns_none_when_missing(self, tmp_path):
        state = BatchState(tmp_path, "b1")
        assert state.load_manifest() is None


class TestBatchStateCorruptionQuarantine:
    def test_corrupted_state_json_is_quarantined(self, tmp_path):
        instance_dir = tmp_path / "builds" / "b1"
        instance_dir.mkdir(parents=True)
        state_path = instance_dir / "batch_state.json"
        state_path.write_text("{invalid json!!!}")

        import pytest

        with pytest.raises(RuntimeError, match="Corrupted batch state"):
            BatchState(tmp_path, "b1")

        # Original file should be renamed
        assert not state_path.exists()
        corrupt_files = list(instance_dir.glob("batch_state.corrupt.*"))
        assert len(corrupt_files) == 1

    def test_corrupted_manifest_is_quarantined(self, tmp_path):
        instance_dir = tmp_path / "builds" / "b1"
        instance_dir.mkdir(parents=True)
        manifest_path = instance_dir / "manifest.json"
        manifest_path.write_text("not json at all")

        state = BatchState(tmp_path, "b1")

        import pytest

        with pytest.raises(RuntimeError, match="Corrupted manifest"):
            state.load_manifest()

        assert not manifest_path.exists()
        corrupt_files = list(instance_dir.glob("manifest.corrupt.*"))
        assert len(corrupt_files) == 1

    def test_corrupted_manifest_with_invalid_keys(self, tmp_path):
        instance_dir = tmp_path / "builds" / "b1"
        instance_dir.mkdir(parents=True)
        manifest_path = instance_dir / "manifest.json"
        # Valid JSON but missing required keys
        manifest_path.write_text('{"foo": "bar"}')

        state = BatchState(tmp_path, "b1")

        import pytest

        with pytest.raises(RuntimeError, match="Corrupted manifest"):
            state.load_manifest()


class TestBatchStateAtomicWrite:
    def test_state_file_content_after_save(self, tmp_path):
        state = BatchState(tmp_path, "b1")
        state.queue_request("k1", "layer", {"x": 1}, "desc")
        state.save()

        state_path = tmp_path / "builds" / "b1" / "batch_state.json"
        assert state_path.exists()
        data = json.loads(state_path.read_text())
        assert "k1" in data["pending"]


class TestBatchStateListBuilds:
    def test_list_builds_sorted(self, tmp_path):
        for bid, ts in [("b2", 2000.0), ("b1", 1000.0), ("b3", 3000.0)]:
            state = BatchState(tmp_path, bid)
            instance = BuildInstance(build_id=bid, pipeline_hash="fp:1234", created_at=ts)
            state.save_manifest(instance)

        builds = BatchState.list_builds(tmp_path)
        assert len(builds) == 3
        # Sorted by directory name (b1, b2, b3)
        assert [b.build_id for b in builds] == ["b1", "b2", "b3"]

    def test_list_builds_empty(self, tmp_path):
        assert BatchState.list_builds(tmp_path) == []

    def test_list_builds_skips_corrupted(self, tmp_path):
        # Create a valid build
        state = BatchState(tmp_path, "b1")
        state.save_manifest(BuildInstance(build_id="b1", pipeline_hash="fp:1234"))

        # Create a corrupted build
        bad_dir = tmp_path / "builds" / "b2"
        bad_dir.mkdir(parents=True)
        (bad_dir / "manifest.json").write_text("not json")

        builds = BatchState.list_builds(tmp_path)
        assert len(builds) == 1
        assert builds[0].build_id == "b1"


class TestPipelineHash:
    def test_compute_pipeline_hash_changes_with_layers(self):
        from synix.core.models import Pipeline, Source, Transform

        class DummyTransform(Transform):
            def execute(self, inputs, config):
                return []

        src = Source("src")
        p1 = Pipeline("test", source_dir="./src", build_dir="./build")
        p1.add(src)

        p2 = Pipeline("test", source_dir="./src", build_dir="./build")
        t = DummyTransform("t1", depends_on=[src])
        p2.add(src, t)

        h1 = BatchState.compute_pipeline_hash(p1)
        h2 = BatchState.compute_pipeline_hash(p2)
        assert h1 != h2
