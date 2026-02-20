"""Tests for synix.mesh.source — server source reading from session store."""

from __future__ import annotations

import pytest

from synix.mesh.source import GenericServerSource
from synix.mesh.store import SessionStore

pytestmark = pytest.mark.mesh


@pytest.fixture
def store(tmp_path):
    """Create a SessionStore with some sessions."""
    s = SessionStore(
        db_path=tmp_path / "sessions.db",
        sessions_dir=tmp_path / "sessions",
    )
    return s


@pytest.fixture
def sample_content():
    return b'{"role": "user", "content": "hello"}\n{"role": "assistant", "content": "hi"}\n'


class TestGenericServerSourceLoad:
    def test_load_from_store(self, store, sample_content):
        store.submit("sess-001", "project-a", sample_content, "node-1")

        source = GenericServerSource(store=store)
        artifacts = source.load({})

        assert len(artifacts) == 1
        art = artifacts[0]
        assert art.label == "t-mesh-sess-001"
        assert art.artifact_type == "transcript"
        assert art.content == sample_content.decode("utf-8")

    def test_empty_store_returns_empty(self, store):
        source = GenericServerSource(store=store)
        artifacts = source.load({})
        assert artifacts == []

    def test_no_store_returns_empty(self):
        source = GenericServerSource()
        artifacts = source.load({})
        assert artifacts == []

    def test_artifact_metadata(self, store, sample_content):
        store.submit("sess-042", "my-project", sample_content, "laptop-1")

        source = GenericServerSource(store=store)
        artifacts = source.load({})

        assert len(artifacts) == 1
        meta = artifacts[0].metadata
        assert meta["source"] == "mesh"
        assert meta["session_id"] == "sess-042"
        assert meta["project_dir"] == "my-project"
        assert meta["submitted_by"] == "laptop-1"

    def test_multiple_sessions(self, store):
        store.submit("s1", "proj", b"content one", "node-1")
        store.submit("s2", "proj", b"content two", "node-2")

        source = GenericServerSource(store=store)
        artifacts = source.load({})

        assert len(artifacts) == 2
        labels = {a.label for a in artifacts}
        assert labels == {"t-mesh-s1", "t-mesh-s2"}

    def test_only_loads_unprocessed(self, store, sample_content):
        store.submit("s1", "proj", sample_content, "node-1")
        store.submit("s2", "proj", b"other content", "node-2")
        store.mark_processed(["s1"])

        source = GenericServerSource(store=store)
        artifacts = source.load({})

        assert len(artifacts) == 1
        assert artifacts[0].label == "t-mesh-s2"

    def test_artifact_has_content_hash(self, store, sample_content):
        store.submit("s1", "proj", sample_content, "node-1")

        source = GenericServerSource(store=store)
        artifacts = source.load({})

        assert len(artifacts) == 1
        assert artifacts[0].artifact_id.startswith("sha256:")
