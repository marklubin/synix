"""Integration tests for the synix viewer — full-flow verification."""

from __future__ import annotations

import pytest

from synix.build.release_engine import execute_release
from synix.core.models import Artifact
from synix.sdk import open_project
from synix.viewer import create_app
from tests.helpers.snapshot_factory import create_test_snapshot


@pytest.fixture
def integration_client(tmp_path):
    """Build a realistic multi-layer release for integration testing."""
    transcripts = [
        Artifact(
            label=f"t-{i:03d}",
            artifact_type="transcript",
            content=f"user: Question {i} about {'memory' if i % 2 == 0 else 'agents'}?\nassistant: Answer {i} about the topic.",
            metadata={"layer_level": 0, "title": f"Conversation {i}", "date": f"2026-01-{10+i:02d}"},
        )
        for i in range(1, 7)
    ]

    episodes = [
        Artifact(
            label="ep-memory",
            artifact_type="episode",
            content="# Memory Systems\n\nSynthesis of memory-related conversations. Memory is fundamental to agent architectures.",
            metadata={"layer_level": 1, "title": "Memory Systems", "date": "2026-02-01"},
        ),
        Artifact(
            label="ep-agents",
            artifact_type="episode",
            content="# Agent Architecture\n\nSynthesis of agent-related conversations. Agents use memory for reasoning.",
            metadata={"layer_level": 1, "title": "Agent Architecture", "date": "2026-02-15"},
        ),
    ]

    core = [
        Artifact(
            label="core-knowledge",
            artifact_type="core_memory",
            content="Core knowledge: Memory systems enable agent architectures to reason effectively.",
            metadata={"layer_level": 2, "title": "Core Knowledge", "date": "2026-03-01"},
        ),
    ]

    parent_map = {
        "ep-memory": ["t-002", "t-004", "t-006"],
        "ep-agents": ["t-001", "t-003", "t-005"],
        "core-knowledge": ["ep-memory", "ep-agents"],
    }

    synix_dir = create_test_snapshot(
        tmp_path,
        {"transcripts": transcripts, "episodes": episodes, "core": core},
        parent_labels_map=parent_map,
        projections={
            "search": {
                "adapter": "synix_search",
                "input_artifacts": [a.label for a in transcripts + episodes + core],
                "config": {"modes": ["fulltext"]},
                "config_fingerprint": "sha256:integration-test",
            },
        },
    )

    execute_release(synix_dir, ref="HEAD", release_name="integration")
    project = open_project(tmp_path)
    release = project.release("integration")
    app = create_app(release, title="Integration Test")
    app.config["TESTING"] = True
    client = app.test_client()
    # Wait for background cache build to complete
    import time
    for _ in range(100):
        resp = client.get("/api/status")
        if resp.get_json().get("loaded"):
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("Viewer caches did not become ready within 5s")
    return client


class TestFullFlow:
    """End-to-end flow: status -> layers -> browse -> detail -> lineage -> search."""

    def test_status_reflects_all_artifacts(self, integration_client):
        data = integration_client.get("/api/status").get_json()
        assert data["loaded"] is True
        assert data["artifact_count"] == 9  # 6 transcripts + 2 episodes + 1 core

    def test_layers_match_release(self, integration_client):
        layers = integration_client.get("/api/layers").get_json()
        assert len(layers) == 3
        layer_map = {l["name"]: l for l in layers}
        assert layer_map["transcripts"]["count"] == 6
        assert layer_map["episodes"]["count"] == 2
        assert layer_map["core"]["count"] == 1

    def test_browse_all_layers(self, integration_client):
        """Verify browsing each layer returns correct artifacts."""
        layers = integration_client.get("/api/layers").get_json()
        total = 0
        for layer in layers:
            resp = integration_client.get(f"/api/artifacts?layer={layer['name']}")
            data = resp.get_json()
            assert data["total"] == layer["count"]
            assert len(data["items"]) == layer["count"]
            total += data["total"]
        assert total == 9

    def test_artifact_detail_consistency(self, integration_client):
        """Artifact detail should match list metadata."""
        list_data = integration_client.get("/api/artifacts?layer=episodes").get_json()
        for item in list_data["items"]:
            detail = integration_client.get(f"/api/artifact/{item['label']}").get_json()
            assert detail["label"] == item["label"]
            assert detail["artifact_type"] == item["artifact_type"]
            assert detail["content"]  # content should be non-empty

    def test_lineage_consistency(self, integration_client):
        """Verify lineage relationships are consistent."""
        # core-knowledge should have parents (episodes) and no children
        lineage = integration_client.get("/api/lineage/core-knowledge").get_json()
        parent_labels = {p["label"] for p in lineage["parents"]}
        assert "ep-memory" in parent_labels or "ep-agents" in parent_labels
        assert lineage["children"] == []

        # ep-memory should have children (core-knowledge) and parents (transcripts)
        lineage = integration_client.get("/api/lineage/ep-memory").get_json()
        child_labels = {c["label"] for c in lineage["children"]}
        assert "core-knowledge" in child_labels

    def test_search_returns_relevant_results(self, integration_client):
        """Search should return artifacts containing the query term."""
        data = integration_client.get("/api/search?q=memory").get_json()
        assert data["total"] > 0
        # All results should have snippets
        assert all("snippet" in item for item in data["items"])
        # At least one result should contain "memory" in its snippet
        snippets = " ".join(item["snippet"] for item in data["items"])
        assert "memory" in snippets.lower()

    def test_search_layer_filter_narrows_results(self, integration_client):
        """Search with layer filter should only return results from that layer."""
        all_data = integration_client.get("/api/search?q=memory").get_json()
        filtered_data = integration_client.get("/api/search?q=memory&layer=episodes").get_json()
        assert filtered_data["total"] <= all_data["total"]
        assert all(item["layer"] == "episodes" for item in filtered_data["items"])

    def test_pagination_across_layers(self, integration_client):
        """Pagination should correctly page through artifacts."""
        page1 = integration_client.get("/api/artifacts?layer=transcripts&page=1&per_page=3").get_json()
        page2 = integration_client.get("/api/artifacts?layer=transcripts&page=2&per_page=3").get_json()
        assert len(page1["items"]) == 3
        assert len(page2["items"]) == 3
        # No overlap
        labels1 = {item["label"] for item in page1["items"]}
        labels2 = {item["label"] for item in page2["items"]}
        assert labels1.isdisjoint(labels2)
