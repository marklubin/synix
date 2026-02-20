"""Tests for session sync — manifest endpoint, file download, sync loop, bootstrap."""

from __future__ import annotations

import gzip

import pytest

from synix.mesh.store import SessionStore

pytestmark = pytest.mark.mesh


@pytest.fixture
def sync_store(tmp_path):
    """Create a SessionStore with some sessions."""
    store = SessionStore(
        db_path=tmp_path / "sessions.db",
        sessions_dir=tmp_path / "sessions",
    )
    content_a = b'{"role":"user","content":"hello"}\n'
    content_b = b'{"role":"user","content":"world"}\n'
    content_c = b'{"role":"user","content":"project-b session"}\n'
    store.submit("sess-001", "default", content_a, "node-1")
    store.submit("sess-002", "default", content_b, "node-1")
    store.submit("sess-010", "project-b", content_c, "node-2")
    return store


@pytest.fixture
def app_with_sessions(tmp_path, sync_store):
    """Create a Starlette app backed by the sync_store."""
    from synix.mesh.config import MeshConfig

    config = MeshConfig(name="test-sync", token="msh_test123")
    # Monkey-patch mesh_dir to use tmp_path
    original_mesh_dir = config.__class__.mesh_dir
    config.__class__.mesh_dir = property(lambda self: tmp_path)

    # Manually set up the server directory structure
    server_dir = tmp_path / "server"
    server_dir.mkdir(exist_ok=True)

    # Move store's DB and sessions into place
    import shutil

    if (tmp_path / "sessions.db").exists():
        shutil.copy(tmp_path / "sessions.db", server_dir / "sessions.db")
    if (tmp_path / "sessions").exists():
        if (server_dir / "sessions").exists():
            shutil.rmtree(server_dir / "sessions")
        shutil.copytree(tmp_path / "sessions", server_dir / "sessions")

    from synix.mesh.server import create_app

    app = create_app(config)

    # Restore
    config.__class__.mesh_dir = original_mesh_dir
    return app


class TestSessionsManifest:
    def test_manifest_returns_all_sessions(self, sync_store):
        sessions = sync_store.list_all_sessions()
        assert len(sessions) == 3
        sids = {s["session_id"] for s in sessions}
        assert sids == {"sess-001", "sess-002", "sess-010"}

    def test_manifest_includes_sha256(self, sync_store):
        sessions = sync_store.list_all_sessions()
        for s in sessions:
            assert len(s["jsonl_sha256"]) == 64  # SHA-256 hex

    def test_manifest_includes_project_dir(self, sync_store):
        sessions = sync_store.list_all_sessions()
        project_dirs = {s["project_dir"] for s in sessions}
        assert project_dirs == {"default", "project-b"}


class TestGetSessionFilePath:
    def test_existing_file(self, sync_store, tmp_path):
        path = sync_store.get_session_file_path("sess-001", "default")
        assert path is not None
        assert path.exists()
        assert path.name == "sess-001.jsonl.gz"

    def test_missing_file(self, sync_store):
        path = sync_store.get_session_file_path("nonexistent", "default")
        assert path is None

    def test_wrong_project_dir(self, sync_store):
        path = sync_store.get_session_file_path("sess-001", "wrong-project")
        assert path is None


class TestBootstrapFromArchive:
    def test_basic_import(self, tmp_path):
        # Create archive with 2 sessions
        archive_dir = tmp_path / "archive"
        (archive_dir / "default").mkdir(parents=True)
        content_a = b"session A content"
        content_b = b"session B content"
        (archive_dir / "default" / "sess-001.jsonl.gz").write_bytes(gzip.compress(content_a))
        (archive_dir / "default" / "sess-002.jsonl.gz").write_bytes(gzip.compress(content_b))

        imported = SessionStore.bootstrap_from_archive(
            db_path=tmp_path / "new.db",
            sessions_dir=tmp_path / "new-sessions",
            archive_dir=archive_dir,
        )
        assert imported == 2

        # Verify the sessions are in the new store
        store = SessionStore(db_path=tmp_path / "new.db", sessions_dir=tmp_path / "new-sessions")
        assert store.count()["total"] == 2
        # All should be unprocessed
        assert store.count()["pending"] == 2

    def test_dedup_on_import(self, tmp_path):
        # Same content in two files → only one should import
        archive_dir = tmp_path / "archive"
        (archive_dir / "default").mkdir(parents=True)
        content = b"same content"
        (archive_dir / "default" / "sess-001.jsonl.gz").write_bytes(gzip.compress(content))
        (archive_dir / "default" / "sess-002.jsonl.gz").write_bytes(gzip.compress(content))

        imported = SessionStore.bootstrap_from_archive(
            db_path=tmp_path / "dedup.db",
            sessions_dir=tmp_path / "dedup-sessions",
            archive_dir=archive_dir,
        )
        assert imported == 1

    def test_corrupt_file_skipped(self, tmp_path):
        archive_dir = tmp_path / "archive"
        (archive_dir / "default").mkdir(parents=True)
        good = b"good content"
        (archive_dir / "default" / "good.jsonl.gz").write_bytes(gzip.compress(good))
        (archive_dir / "default" / "bad.jsonl.gz").write_bytes(b"not gzip data")

        imported = SessionStore.bootstrap_from_archive(
            db_path=tmp_path / "corrupt.db",
            sessions_dir=tmp_path / "corrupt-sessions",
            archive_dir=archive_dir,
        )
        assert imported == 1

    def test_empty_archive(self, tmp_path):
        archive_dir = tmp_path / "empty-archive"
        archive_dir.mkdir()
        imported = SessionStore.bootstrap_from_archive(
            db_path=tmp_path / "empty.db",
            sessions_dir=tmp_path / "empty-sessions",
            archive_dir=archive_dir,
        )
        assert imported == 0

    def test_nonexistent_archive(self, tmp_path):
        imported = SessionStore.bootstrap_from_archive(
            db_path=tmp_path / "ne.db",
            sessions_dir=tmp_path / "ne-sessions",
            archive_dir=tmp_path / "does-not-exist",
        )
        assert imported == 0

    def test_project_dir_preserved(self, tmp_path):
        archive_dir = tmp_path / "archive"
        (archive_dir / "project-x").mkdir(parents=True)
        content = b"project X content"
        (archive_dir / "project-x" / "sess-100.jsonl.gz").write_bytes(gzip.compress(content))

        imported = SessionStore.bootstrap_from_archive(
            db_path=tmp_path / "proj.db",
            sessions_dir=tmp_path / "proj-sessions",
            archive_dir=archive_dir,
        )
        assert imported == 1

        store = SessionStore(db_path=tmp_path / "proj.db", sessions_dir=tmp_path / "proj-sessions")
        sessions = store.list_all_sessions()
        assert len(sessions) == 1
        assert sessions[0]["project_dir"] == "project-x"


class TestCompositeKey:
    """Test that composite primary key (session_id, project_dir) works correctly."""

    def test_same_session_id_different_projects(self, tmp_path):
        store = SessionStore(
            db_path=tmp_path / "composite.db",
            sessions_dir=tmp_path / "sessions",
        )
        content_a = b"content for project A"
        content_b = b"content for project B"
        assert store.submit("sess-001", "project-a", content_a, "node-1") is True
        assert store.submit("sess-001", "project-b", content_b, "node-1") is True
        assert store.count()["total"] == 2

    def test_get_session_content_with_project_dir(self, tmp_path):
        store = SessionStore(
            db_path=tmp_path / "composite.db",
            sessions_dir=tmp_path / "sessions",
        )
        content_a = b"content for project A"
        content_b = b"content for project B"
        store.submit("sess-001", "project-a", content_a, "node-1")
        store.submit("sess-001", "project-b", content_b, "node-1")

        assert store.get_session_content("sess-001", "project-a") == content_a
        assert store.get_session_content("sess-001", "project-b") == content_b

    def test_mark_processed_with_composite_keys(self, tmp_path):
        store = SessionStore(
            db_path=tmp_path / "composite.db",
            sessions_dir=tmp_path / "sessions",
        )
        content_a = b"content A"
        content_b = b"content B"
        store.submit("sess-001", "project-a", content_a, "node-1")
        store.submit("sess-002", "project-a", content_b, "node-1")

        store.mark_processed([("sess-001", "project-a")])
        unprocessed = store.get_unprocessed()
        assert len(unprocessed) == 1
        assert unprocessed[0]["session_id"] == "sess-002"
