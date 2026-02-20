"""Tests for mesh auth middleware and helpers."""

from __future__ import annotations

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from synix.mesh.auth import AuthMiddleware, auth_headers, generate_token

TEST_TOKEN = "msh_testtoken1234567890abcdef"


def _make_app(token: str = TEST_TOKEN) -> Starlette:
    """Create a minimal Starlette app with auth middleware."""

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    async def protected(request: Request) -> JSONResponse:
        return JSONResponse({"data": "secret"})

    app = Starlette(
        routes=[
            Route("/api/v1/health", health),
            Route("/api/v1/sessions", protected, methods=["POST"]),
            Route("/api/v1/status", protected),
        ],
    )
    app.add_middleware(AuthMiddleware, token=token)
    return app


class TestAuthMiddleware:
    def test_health_bypasses_auth(self):
        client = TestClient(_make_app())
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_missing_header_returns_401(self):
        client = TestClient(_make_app())
        resp = client.get("/api/v1/status")
        assert resp.status_code == 401
        assert resp.json()["error"] == "unauthorized"

    def test_wrong_token_returns_401(self):
        client = TestClient(_make_app())
        resp = client.get("/api/v1/status", headers={"Authorization": "Bearer wrong_token"})
        assert resp.status_code == 401

    def test_valid_token_passes(self):
        client = TestClient(_make_app())
        resp = client.get("/api/v1/status", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
        assert resp.status_code == 200
        assert resp.json()["data"] == "secret"

    def test_malformed_bearer_returns_401(self):
        client = TestClient(_make_app())
        # No "Bearer " prefix
        resp = client.get("/api/v1/status", headers={"Authorization": TEST_TOKEN})
        assert resp.status_code == 401

    def test_empty_bearer_returns_401(self):
        client = TestClient(_make_app())
        resp = client.get("/api/v1/status", headers={"Authorization": "Bearer "})
        assert resp.status_code == 401

    def test_post_requires_auth(self):
        client = TestClient(_make_app())
        resp = client.post("/api/v1/sessions")
        assert resp.status_code == 401

    def test_post_with_auth_passes(self):
        client = TestClient(_make_app())
        resp = client.post("/api/v1/sessions", headers={"Authorization": f"Bearer {TEST_TOKEN}"})
        assert resp.status_code == 200


class TestGenerateToken:
    def test_token_has_prefix(self):
        token = generate_token()
        assert token.startswith("msh_")

    def test_token_length(self):
        token = generate_token()
        # msh_ (4) + 64 hex chars (32 bytes) = 68
        assert len(token) == 68

    def test_tokens_are_unique(self):
        tokens = {generate_token() for _ in range(10)}
        assert len(tokens) == 10


class TestAuthHeaders:
    def test_basic_headers(self):
        headers = auth_headers("msh_token123")
        assert headers["Authorization"] == "Bearer msh_token123"
        assert "X-Mesh-Node" not in headers

    def test_headers_with_node(self):
        headers = auth_headers("msh_token123", node_name="obispo")
        assert headers["Authorization"] == "Bearer msh_token123"
        assert headers["X-Mesh-Node"] == "obispo"
