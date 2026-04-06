"""Unit tests for the synix viewer server."""

import json

import pytest

from synix.viewer.server import ViewerState, create_app

# ---------------------------------------------------------------------------
# Fixtures for build-log tests
# ---------------------------------------------------------------------------


@pytest.fixture
def build_log_client(viewer_release, tmp_path):
    """Flask test client with a project that has JSONL build logs."""

    # Create a mock project that has _synix_dir pointing to tmp_path
    class MockProject:
        _synix_dir = tmp_path / ".synix"
        _pipeline = None

    mock_project = MockProject()
    (mock_project._synix_dir / "logs").mkdir(parents=True, exist_ok=True)

    # Write a sample JSONL log
    log_events = [
        {"event": "run_start", "pipeline": "demo", "layer_count": 3, "timestamp": "2026-04-06T04:11:36Z"},
        {"event": "layer_start", "layer": "conversations", "level": 0, "timestamp": "2026-04-06T04:11:36.001Z"},
        {"event": "artifact_built", "layer": "conversations", "label": "t-text-foo", "timestamp": "2026-04-06T04:11:36.010Z"},
        {"event": "artifact_built", "layer": "conversations", "label": "t-text-bar", "timestamp": "2026-04-06T04:11:36.020Z"},
        {"event": "layer_finish", "layer": "conversations", "built": 2, "cached": 0, "time_seconds": 0.034, "timestamp": "2026-04-06T04:11:36.034Z"},
        {"event": "layer_start", "layer": "episodes", "level": 1, "timestamp": "2026-04-06T04:11:36.035Z"},
        {"event": "llm_call_start", "layer": "episodes", "artifact_desc": "episode ep-foo", "model": "Qwen/Qwen3.5-2B", "timestamp": "2026-04-06T04:11:36.036Z"},
        {"event": "llm_call_finish", "layer": "episodes", "artifact_desc": "episode ep-foo", "duration_seconds": 4.5, "input_tokens": 168, "output_tokens": 295, "timestamp": "2026-04-06T04:11:40.536Z"},
        {"event": "artifact_built", "layer": "episodes", "label": "ep-foo", "timestamp": "2026-04-06T04:11:40.540Z"},
        {"event": "layer_finish", "layer": "episodes", "built": 1, "cached": 0, "time_seconds": 4.505, "timestamp": "2026-04-06T04:11:40.540Z"},
        {"event": "run_finish", "total_time": 4.54, "total_llm_calls": 1, "total_tokens": 463, "total_cost_estimate": 0.003, "timestamp": "2026-04-06T04:11:40.540Z"},
    ]

    run_id = "20260406T041136000000Z-abcd1234"
    log_path = mock_project._synix_dir / "logs" / f"{run_id}.jsonl"
    with open(log_path, "w") as f:
        for event in log_events:
            f.write(json.dumps(event) + "\n")

    state = ViewerState(viewer_release, "Test Viewer", project=mock_project)
    app = create_app(state)
    app.config["TESTING"] = True
    return app.test_client()


class TestStatus:
    def test_returns_loaded(self, viewer_client):
        resp = viewer_client.get("/api/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["loaded"] is True
        assert data["title"] == "Test Viewer"
        assert data["artifact_count"] == 4


class TestLayers:
    def test_sorted_by_level(self, viewer_client):
        resp = viewer_client.get("/api/layers")
        assert resp.status_code == 200
        layers = resp.get_json()
        assert len(layers) == 3
        levels = [l["level"] for l in layers]
        assert levels == sorted(levels)

    def test_layer_counts(self, viewer_client):
        resp = viewer_client.get("/api/layers")
        layers = resp.get_json()
        counts = {l["name"]: l["count"] for l in layers}
        assert counts["transcripts"] == 2
        assert counts["episodes"] == 1
        assert counts["core"] == 1


class TestArtifacts:
    def test_by_layer(self, viewer_client):
        resp = viewer_client.get("/api/artifacts?layer=transcripts")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total"] == 2
        assert len(data["items"]) == 2
        assert all(item["layer"] == "transcripts" for item in data["items"])

    def test_pagination(self, viewer_client):
        resp = viewer_client.get("/api/artifacts?layer=transcripts&page=1&per_page=1")
        data = resp.get_json()
        assert len(data["items"]) == 1
        assert data["total"] == 2
        assert data["page"] == 1

    def test_sort_by_date_desc(self, viewer_client):
        resp = viewer_client.get("/api/artifacts?layer=transcripts&sort=date&order=desc")
        data = resp.get_json()
        dates = [item["date"] for item in data["items"]]
        assert dates == sorted(dates, reverse=True)

    def test_sort_by_title_asc(self, viewer_client):
        resp = viewer_client.get("/api/artifacts?layer=transcripts&sort=title&order=asc")
        data = resp.get_json()
        titles = [item["title"] for item in data["items"]]
        assert titles == sorted(titles)

    def test_requires_layer_param(self, viewer_client):
        resp = viewer_client.get("/api/artifacts")
        assert resp.status_code == 400


class TestArtifactDetail:
    def test_returns_full_artifact(self, viewer_client):
        resp = viewer_client.get("/api/artifact/transcript-001")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["label"] == "transcript-001"
        assert data["artifact_type"] == "transcript"
        assert "memory" in data["content"].lower()
        assert data["layer"] == "transcripts"
        assert isinstance(data["metadata"], dict)

    def test_not_found(self, viewer_client):
        resp = viewer_client.get("/api/artifact/nonexistent")
        assert resp.status_code == 404


class TestLineage:
    def test_parents_and_children(self, viewer_client):
        resp = viewer_client.get("/api/lineage/episode-001")
        assert resp.status_code == 200
        data = resp.get_json()
        parent_labels = [p["label"] for p in data["parents"]]
        assert "transcript-001" in parent_labels or "transcript-002" in parent_labels
        child_labels = [c["label"] for c in data["children"]]
        assert "core-001" in child_labels

    def test_no_relations(self, viewer_client):
        # core-001 has no children (it's a leaf artifact)
        resp = viewer_client.get("/api/lineage/core-001")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["children"] == []

    def test_not_found(self, viewer_client):
        resp = viewer_client.get("/api/lineage/nonexistent")
        assert resp.status_code == 404


class TestSearch:
    def test_keyword_search(self, viewer_client):
        resp = viewer_client.get("/api/search?q=memory")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["items"]) > 0
        assert "has_more" in data
        assert all("snippet" in item for item in data["items"])

    def test_search_with_layer_filter(self, viewer_client):
        resp = viewer_client.get("/api/search?q=memory&layer=transcripts")
        data = resp.get_json()
        assert all(item["layer"] == "transcripts" for item in data["items"])

    def test_search_pagination(self, viewer_client):
        resp = viewer_client.get("/api/search?q=memory&page=1&per_page=1")
        data = resp.get_json()
        assert len(data["items"]) <= 1
        assert data["page"] == 1

    def test_empty_query(self, viewer_client):
        resp = viewer_client.get("/api/search?q=")
        assert resp.status_code == 400


class TestSecurity:
    def test_artifact_content_returned_raw(self, viewer_client):
        """Verify the API returns raw content — client is responsible for sanitization."""
        resp = viewer_client.get("/api/artifact/transcript-001")
        data = resp.get_json()
        # Content is returned as-is; sanitization is client-side via marked config
        assert isinstance(data["content"], str)


class TestInputValidation:
    def test_negative_page(self, viewer_client):
        resp = viewer_client.get("/api/artifacts?layer=transcripts&page=-1")
        data = resp.get_json()
        assert data["page"] == 1  # clamped to 1

    def test_huge_per_page(self, viewer_client):
        resp = viewer_client.get("/api/artifacts?layer=transcripts&per_page=99999")
        data = resp.get_json()
        assert data["per_page"] <= 200  # clamped

    def test_invalid_sort(self, viewer_client):
        resp = viewer_client.get("/api/artifacts?layer=transcripts&sort=malicious")
        assert resp.status_code == 200  # should still work with default sort

    def test_invalid_order(self, viewer_client):
        resp = viewer_client.get("/api/artifacts?layer=transcripts&order=malicious")
        assert resp.status_code == 200  # should still work with default order

    def test_search_negative_page(self, viewer_client):
        resp = viewer_client.get("/api/search?q=memory&page=-5")
        data = resp.get_json()
        assert data["page"] == 1

    def test_search_huge_per_page(self, viewer_client):
        resp = viewer_client.get("/api/search?q=memory&per_page=50000")
        data = resp.get_json()
        assert data["per_page"] <= 200


class TestStatic:
    def test_index_html(self, viewer_client):
        resp = viewer_client.get("/")
        assert resp.status_code == 200
        assert b"<!DOCTYPE html>" in resp.data

    def test_static_js(self, viewer_client):
        resp = viewer_client.get("/static/app.js")
        assert resp.status_code == 200


class TestBuildLogs:
    """Tests for the /api/build-logs and /api/build-log endpoints."""

    def test_build_logs_list(self, build_log_client):
        resp = build_log_client.get("/api/build-logs")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "logs" in data
        assert len(data["logs"]) == 1
        log_entry = data["logs"][0]
        assert log_entry["run_id"] == "20260406T041136000000Z-abcd1234"
        assert log_entry["size_bytes"] > 0
        assert "timestamp" in log_entry

    def test_build_logs_empty_without_project(self, viewer_client):
        resp = viewer_client.get("/api/build-logs")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["logs"] == []

    def test_build_log_detail_by_run_id(self, build_log_client):
        resp = build_log_client.get("/api/build-log?run_id=20260406T041136000000Z-abcd1234")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["pipeline"] == "demo"
        assert data["total_time"] == 4.54
        assert len(data["layers"]) == 2

        # Check layer details
        conv = data["layers"][0]
        assert conv["name"] == "conversations"
        assert conv["level"] == 0
        assert conv["built"] == 2
        assert conv["cached"] == 0
        assert len(conv["artifacts"]) == 2
        assert "t-text-foo" in conv["artifacts"]

        ep = data["layers"][1]
        assert ep["name"] == "episodes"
        assert ep["level"] == 1
        assert ep["built"] == 1
        assert len(ep["llm_calls"]) == 1
        llm_call = ep["llm_calls"][0]
        assert llm_call["artifact"] == "episode ep-foo"
        assert llm_call["duration"] == 4.5
        assert llm_call["input_tokens"] == 168
        assert llm_call["output_tokens"] == 295
        assert llm_call["model"] == "Qwen/Qwen3.5-2B"

        # Summary
        assert data["summary"]["total_llm_calls"] == 1
        assert data["summary"]["total_tokens"] == 463
        assert data["summary"]["layers_count"] == 2

    def test_build_log_detail_most_recent(self, build_log_client):
        """When no run_id is provided, returns the most recent log."""
        resp = build_log_client.get("/api/build-log")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["pipeline"] == "demo"

    def test_build_log_not_found(self, build_log_client):
        resp = build_log_client.get("/api/build-log?run_id=nonexistent")
        assert resp.status_code == 404

    def test_build_log_no_project(self, viewer_client):
        resp = viewer_client.get("/api/build-log")
        assert resp.status_code == 404

    def test_build_log_timestamps(self, build_log_client):
        resp = build_log_client.get("/api/build-log?run_id=20260406T041136000000Z-abcd1234")
        data = resp.get_json()
        assert data["started_at"] == "2026-04-06T04:11:36Z"
        assert data["completed_at"] == "2026-04-06T04:11:40.540Z"
