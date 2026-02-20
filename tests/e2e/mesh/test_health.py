"""E2E: Health endpoint + status reporting."""

from __future__ import annotations


class TestHealth:
    def test_health_returns_ok(self, test_client):
        resp = test_client.get("/api/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_status_includes_stats(self, test_client, authed_headers):
        resp = test_client.get("/api/v1/status", headers=authed_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "build_count" in data
        assert "sessions" in data
        assert "uptime_seconds" in data
        assert "scheduler" in data

    def test_status_session_counts(self, test_client, authed_headers):
        resp = test_client.get("/api/v1/status", headers=authed_headers)
        data = resp.json()
        sessions = data["sessions"]
        assert "total" in sessions
        assert "processed" in sessions
        assert "pending" in sessions
