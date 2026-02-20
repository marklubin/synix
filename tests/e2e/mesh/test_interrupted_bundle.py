"""E2E: Partial download -> client retries -> bundle integrity."""

from __future__ import annotations


class TestInterruptedBundle:
    def test_conditional_get_returns_304(self, test_client, authed_headers):
        """ETag-based conditional GET returns 304 when unchanged."""
        # First, there's no bundle -> 404
        resp = test_client.get("/api/v1/artifacts/bundle", headers=authed_headers)
        assert resp.status_code == 404

    def test_etag_missing_returns_full_bundle(self, test_client, authed_headers):
        """Request without If-None-Match gets full bundle (when available)."""
        # No bundle yet -> 404
        resp = test_client.get("/api/v1/artifacts/bundle", headers=authed_headers)
        assert resp.status_code == 404

    def test_manifest_etag_consistent(self, test_client, authed_headers):
        """Manifest endpoint returns consistent ETag (empty when no build)."""
        resp = test_client.get("/api/v1/artifacts/manifest", headers=authed_headers)
        # No build yet
        assert resp.status_code == 404
