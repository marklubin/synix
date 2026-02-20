"""E2E: Token validation — all other tests depend on auth working."""

from __future__ import annotations


class TestAuth:
    def test_health_no_auth(self, test_client):
        resp = test_client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_status_requires_auth(self, test_client):
        resp = test_client.get("/api/v1/status")
        assert resp.status_code == 401

    def test_status_with_valid_token(self, test_client, authed_headers):
        resp = test_client.get("/api/v1/status", headers=authed_headers)
        assert resp.status_code == 200

    def test_wrong_token_rejected(self, test_client):
        resp = test_client.get("/api/v1/status", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

    def test_sessions_requires_auth(self, test_client):
        resp = test_client.post("/api/v1/sessions", json={})
        assert resp.status_code == 401

    def test_heartbeat_requires_auth(self, test_client):
        resp = test_client.post("/api/v1/heartbeat", json={})
        assert resp.status_code == 401

    def test_cluster_requires_auth(self, test_client):
        resp = test_client.get("/api/v1/cluster")
        assert resp.status_code == 401

    def test_builds_status_requires_auth(self, test_client):
        resp = test_client.get("/api/v1/builds/status")
        assert resp.status_code == 401
