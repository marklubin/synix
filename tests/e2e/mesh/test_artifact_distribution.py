"""E2E: ETag -> bundle -> pull -> deploy."""

from __future__ import annotations


class TestArtifactDistribution:
    def test_no_bundle_returns_404(self, test_client, authed_headers):
        """Before any build, bundle should be 404."""
        resp = test_client.get("/api/v1/artifacts/bundle", headers=authed_headers)
        assert resp.status_code == 404

    def test_no_manifest_returns_404(self, test_client, authed_headers):
        """Before any build, manifest should be 404."""
        resp = test_client.get("/api/v1/artifacts/manifest", headers=authed_headers)
        assert resp.status_code == 404

    def test_search_no_index_returns_404(self, test_client, authed_headers):
        """Before any build, search should return 404."""
        resp = test_client.post(
            "/api/v1/search",
            json={"query": "test"},
            headers=authed_headers,
        )
        assert resp.status_code == 404
