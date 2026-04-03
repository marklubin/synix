"""End-to-end tests for the Synix knowledge server.

Exercises the full flow: REST API health → ingest → build → search → context.
Uses Starlette TestClient (in-process, no real HTTP server).
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# REST API — Health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# REST API — Ingest
# ---------------------------------------------------------------------------


class TestBucketsEndpoint:
    def test_list_buckets(self, client):
        resp = client.get("/api/v1/buckets")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["buckets"]) == 3
        names = [b["name"] for b in data["buckets"]]
        assert "documents" in names
        assert "sessions" in names
        assert "reports" in names


class TestIngestEndpoint:
    def test_ingest_to_documents_bucket(self, client, server_project):
        project_dir, config = server_project
        resp = client.post(
            "/api/v1/ingest/documents",
            json={"content": "Test document content", "filename": "test-note.md"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["bucket"] == "documents"
        assert data["filename"] == "test-note.md"

        # Verify file was written
        dest = Path(project_dir) / "sources" / "documents" / "test-note.md"
        assert dest.exists()
        assert dest.read_text() == "Test document content"

    def test_ingest_to_sessions_bucket(self, client, server_project):
        project_dir, _ = server_project
        resp = client.post(
            "/api/v1/ingest/sessions",
            json={"content": '{"role":"user","content":"hello"}', "filename": "sess-001.jsonl"},
        )
        assert resp.status_code == 200
        dest = Path(project_dir) / "sources" / "sessions" / "sess-001.jsonl"
        assert dest.exists()

    def test_ingest_to_reports_bucket(self, client, server_project):
        project_dir, _ = server_project
        resp = client.post(
            "/api/v1/ingest/reports",
            json={"content": "# Daily Report\nAll clear.", "filename": "daily-2026-04-03.md"},
        )
        assert resp.status_code == 200
        dest = Path(project_dir) / "sources" / "reports" / "daily-2026-04-03.md"
        assert dest.exists()

    def test_ingest_to_unknown_bucket_404(self, client):
        resp = client.post(
            "/api/v1/ingest/nonexistent",
            json={"content": "x", "filename": "x.md"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["error"].lower()

    def test_ingest_missing_content_400(self, client):
        resp = client.post(
            "/api/v1/ingest/documents",
            json={"filename": "x.md"},
        )
        assert resp.status_code == 400

    def test_ingest_missing_filename_400(self, client):
        resp = client.post(
            "/api/v1/ingest/documents",
            json={"content": "hello"},
        )
        assert resp.status_code == 400

    def test_ingest_path_traversal_blocked(self, client, server_project):
        """Path traversal in filename must be sanitized."""
        project_dir, _ = server_project
        resp = client.post(
            "/api/v1/ingest/documents",
            json={"content": "malicious", "filename": "../../etc/passwd"},
        )
        assert resp.status_code == 200
        # File should land IN the bucket dir, not at ../../etc/passwd
        assert not (Path(project_dir) / "etc" / "passwd").exists()
        assert (Path(project_dir) / "sources" / "documents" / "passwd").exists()

    def test_ingest_path_traversal_subdirectory(self, client, server_project):
        """Subdirectory paths are stripped to just the filename."""
        project_dir, _ = server_project
        resp = client.post(
            "/api/v1/ingest/documents",
            json={"content": "test", "filename": "subdir/nested/file.md"},
        )
        assert resp.status_code == 200
        # Only the filename part should be kept
        assert (Path(project_dir) / "sources" / "documents" / "file.md").exists()

    def test_ingest_invalid_json_400(self, client):
        resp = client.post(
            "/api/v1/ingest/documents",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 400

    def test_ingest_creates_bucket_dir_if_missing(self, client, server_project):
        """Bucket dir doesn't need to pre-exist — ingest creates it."""
        project_dir, _ = server_project
        # Delete the documents dir
        docs_dir = Path(project_dir) / "sources" / "documents"
        if docs_dir.exists():
            import shutil

            shutil.rmtree(docs_dir)

        resp = client.post(
            "/api/v1/ingest/documents",
            json={"content": "recreated", "filename": "revived.md"},
        )
        assert resp.status_code == 200
        assert (Path(project_dir) / "sources" / "documents" / "revived.md").exists()


# ---------------------------------------------------------------------------
# REST API — Search & Context (requires built project)
# ---------------------------------------------------------------------------


class TestSearchEndpoint:
    def test_search_returns_results(self, built_server):
        client, _, _ = built_server
        resp = client.get("/api/v1/search?q=summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] > 0
        assert len(data["results"]) > 0
        assert "label" in data["results"][0]
        assert "score" in data["results"][0]

    def test_search_empty_query_400(self, client):
        resp = client.get("/api/v1/search")
        assert resp.status_code == 400

    def test_search_no_build_returns_error(self, client):
        resp = client.get("/api/v1/search?q=test")
        assert resp.status_code in (500, 503)
        assert "error" in resp.json()

    def test_flat_file_returns_context(self, built_server):
        client, project_dir, _ = built_server
        resp = client.get("/api/v1/flat-file/context-doc")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/markdown")
        content = resp.text
        assert len(content) > 0

    def test_flat_file_unknown_name_404(self, built_server):
        client, _, _ = built_server
        resp = client.get("/api/v1/flat-file/nonexistent-doc")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# MCP Tools — via direct function calls (same as TestClient would invoke)
# ---------------------------------------------------------------------------


class TestMCPToolsIntegration:
    """Test MCP tools through the actual function calls with real project state."""

    def test_list_buckets(self, client, server_project):
        from synix.server.mcp_tools import list_buckets

        result = list_buckets()
        assert "documents" in result
        assert "sessions" in result
        assert "reports" in result

    def test_ingest_via_mcp(self, client, server_project):
        from synix.server.mcp_tools import ingest

        project_dir, _ = server_project
        result = ingest("documents", "MCP ingest test", "mcp-test.md")
        assert "mcp-test.md" in result
        assert (Path(project_dir) / "sources" / "documents" / "mcp-test.md").exists()

    def test_ingest_invalid_bucket_via_mcp(self, client, server_project):
        from synix.server.mcp_tools import ingest

        try:
            ingest("fake-bucket", "content", "file.md")
            assert False, "Should have raised"
        except ValueError as e:
            assert "fake-bucket" in str(e)

    def test_ingest_path_traversal_via_mcp(self, client, server_project):
        from synix.server.mcp_tools import ingest

        project_dir, _ = server_project
        result = ingest("documents", "safe content", "../../../etc/shadow")
        # Should write "shadow" in the bucket dir, not traverse
        assert "shadow" in result
        assert (Path(project_dir) / "sources" / "documents" / "shadow").exists()
        assert not (Path(project_dir) / "etc").exists()

    def test_search_after_build(self, built_server):
        from synix.server.mcp_tools import _current_release, search

        # Verify release has artifacts
        rel = _current_release()
        layers = rel.layers()
        assert len(layers) > 0, f"No layers in release: {layers}"

        artifacts = list(rel.artifacts())
        assert len(artifacts) > 0, "No artifacts in release"

        # Search for a term from mock LLM responses
        result = search("summary")
        assert len(result) > 0

    def test_get_context_after_build(self, built_server):
        from synix.server.mcp_tools import get_context

        result = get_context("context-doc")
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Full Workflow — Ingest → Build → Search → Context
# ---------------------------------------------------------------------------


class TestFullWorkflow:
    """End-to-end: ingest via REST, build manually, search, get context."""

    def test_ingest_build_search_context(self, client, server_project, mock_llm):
        """Complete flow: ingest → build → search → context."""
        import synix
        from synix.server.mcp_tools import _state

        project_dir, config = server_project

        # 1. Ingest documents via REST
        for i, topic in enumerate(["neural networks", "distributed systems", "memory architecture"]):
            resp = client.post(
                "/api/v1/ingest/documents",
                json={
                    "content": f"Deep dive into {topic}. This covers the fundamentals.",
                    "filename": f"topic-{i}.md",
                },
            )
            assert resp.status_code == 200

        # Verify files landed in bucket
        docs_dir = Path(project_dir) / "sources" / "documents"
        assert len(list(docs_dir.glob("topic-*.md"))) == 3

        # 2. Copy ingested docs to pipeline source dir
        # (toy pipeline reads from ./sources; in production the Source would point at the bucket)
        import shutil

        for f in docs_dir.glob("topic-*.md"):
            shutil.copy(f, Path(project_dir) / "sources" / f.name)

        # 3. Build + release via SDK
        project = synix.open_project(str(project_dir))
        project.load_pipeline(str(Path(project_dir) / "pipeline.py"))
        result = project.build()
        assert result.built >= 3, f"Expected at least 3 artifacts, got {result.built}"
        project.release_to("local")

        # Re-open project to pick up release
        project = synix.open_project(str(project_dir))
        _state["project"] = project

        # 4. Verify context-doc flat file
        resp = client.get("/api/v1/flat-file/context-doc")
        assert resp.status_code == 200
        assert len(resp.text) > 0

        # 5. Verify search works
        from synix.server.mcp_tools import search

        result = search("summary")
        assert len(result) > 0

    def test_multiple_buckets_ingest(self, client, server_project):
        """Verify each bucket receives its documents independently."""
        project_dir, _ = server_project

        # Ingest to each bucket
        client.post("/api/v1/ingest/documents", json={"content": "doc", "filename": "d.md"})
        client.post("/api/v1/ingest/sessions", json={"content": "sess", "filename": "s.jsonl"})
        client.post("/api/v1/ingest/reports", json={"content": "report", "filename": "r.md"})

        # Verify isolation
        assert (Path(project_dir) / "sources" / "documents" / "d.md").read_text() == "doc"
        assert (Path(project_dir) / "sources" / "sessions" / "s.jsonl").read_text() == "sess"
        assert (Path(project_dir) / "sources" / "reports" / "r.md").read_text() == "report"


# ---------------------------------------------------------------------------
# Plugin Hook Simulation
# ---------------------------------------------------------------------------


class TestPluginHookSimulation:
    """Simulate what the Claude Code plugin hooks do, exercising the server endpoints."""

    def test_session_start_hook_fetches_context(self, built_server):
        """Simulate SessionStart: GET /api/v1/flat-file/context-doc."""
        client, _, _ = built_server
        resp = client.get("/api/v1/flat-file/context-doc")
        assert resp.status_code == 200
        context = resp.text
        assert len(context) > 0

        # Simulate what the hook script does: wrap in JSON
        import json

        hook_output = {"hookSpecificOutput": {"additionalContext": context}}
        serialized = json.dumps(hook_output)
        assert "additionalContext" in serialized

    def test_session_end_hook_pushes_transcript(self, client, server_project):
        """Simulate Stop hook: POST session transcript to /api/v1/ingest/sessions."""
        project_dir, _ = server_project

        # Simulate a session transcript (simplified)
        session_content = json.dumps([
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is..."},
        ])

        resp = client.post(
            "/api/v1/ingest/sessions",
            json={
                "content": session_content,
                "filename": "test-session-abc123.jsonl",
            },
        )
        assert resp.status_code == 200

        # Verify file landed
        dest = Path(project_dir) / "sources" / "sessions" / "test-session-abc123.jsonl"
        assert dest.exists()
        assert json.loads(dest.read_text())[0]["role"] == "user"

    def test_session_end_hook_idempotent(self, client, server_project):
        """Pushing the same session twice overwrites (idempotent)."""
        resp1 = client.post(
            "/api/v1/ingest/sessions",
            json={"content": "version 1", "filename": "sess-dedup.jsonl"},
        )
        assert resp1.status_code == 200

        resp2 = client.post(
            "/api/v1/ingest/sessions",
            json={"content": "version 2", "filename": "sess-dedup.jsonl"},
        )
        assert resp2.status_code == 200

        project_dir, _ = server_project
        content = (Path(project_dir) / "sources" / "sessions" / "sess-dedup.jsonl").read_text()
        assert content == "version 2"
