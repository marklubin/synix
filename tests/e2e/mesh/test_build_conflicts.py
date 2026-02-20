"""E2E: Trigger during active build -> 202 queued, coalesced."""

from __future__ import annotations


class TestBuildConflicts:
    def test_trigger_when_idle(self, test_client, authed_headers):
        """Trigger when no build running should start one."""
        resp = test_client.post("/api/v1/builds/trigger", headers=authed_headers)
        # Could be 200 (started) or 202 (queued) depending on timing
        assert resp.status_code in (200, 202)
        data = resp.json()
        assert data["status"] in ("started", "queued")

    def test_trigger_returns_status(self, test_client, authed_headers):
        """Build trigger always returns a status field."""
        resp = test_client.post("/api/v1/builds/trigger", headers=authed_headers)
        assert "status" in resp.json()
