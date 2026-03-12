"""Test fixtures for the synix viewer."""

from __future__ import annotations

from pathlib import Path

import pytest

from synix.build.release_engine import execute_release
from synix.core.models import Artifact
from synix.sdk import open_project
from synix.viewer import create_app
from tests.helpers.snapshot_factory import create_test_snapshot


@pytest.fixture
def viewer_release(tmp_path: Path):
    """Create a multi-layer test release with search index."""
    # Layer 0: transcripts
    t1 = Artifact(
        label="transcript-001",
        artifact_type="transcript",
        content="user: What is memory?\nassistant: Memory is the ability to store and retrieve information.",
        metadata={"layer_level": 0, "title": "Memory Discussion", "date": "2026-01-15", "message_count": 2},
    )
    t2 = Artifact(
        label="transcript-002",
        artifact_type="transcript",
        content="user: Tell me about agents.\nassistant: Agents are autonomous systems that can take actions.",
        metadata={"layer_level": 0, "title": "Agent Overview", "date": "2026-01-20", "message_count": 2},
    )

    # Layer 1: episodes
    ep1 = Artifact(
        label="episode-001",
        artifact_type="episode",
        content="# Memory and Agents\n\nA synthesis of conversations about memory systems and agent architectures.",
        metadata={"layer_level": 1, "title": "Memory and Agents", "date": "2026-02-01"},
    )

    # Layer 2: core
    core1 = Artifact(
        label="core-001",
        artifact_type="core_memory",
        content="Core knowledge about memory systems, agents, and their architectures.",
        metadata={"layer_level": 2, "title": "Core Memory", "date": "2026-03-01"},
    )

    synix_dir = create_test_snapshot(
        tmp_path,
        {
            "transcripts": [t1, t2],
            "episodes": [ep1],
            "core": [core1],
        },
        parent_labels_map={
            "episode-001": ["transcript-001", "transcript-002"],
            "core-001": ["episode-001"],
        },
        projections={
            "search": {
                "adapter": "synix_search",
                "input_artifacts": ["transcript-001", "transcript-002", "episode-001", "core-001"],
                "config": {"modes": ["fulltext"]},
                "config_fingerprint": "sha256:test-viewer",
            },
        },
    )

    execute_release(synix_dir, ref="HEAD", release_name="test")
    project = open_project(tmp_path)
    return project.release("test")


@pytest.fixture
def viewer_client(viewer_release):
    """Flask test client for the viewer.

    Waits for background cache build to complete before returning so
    tests get deterministic, fully-ready state.
    """
    app = create_app(viewer_release, title="Test Viewer")
    app.config["TESTING"] = True
    client = app.test_client()
    # Poll /api/status until caches are ready (background thread)
    import time
    for _ in range(100):
        resp = client.get("/api/status")
        if resp.get_json().get("loaded"):
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("Viewer caches did not become ready within 5s")
    return client
