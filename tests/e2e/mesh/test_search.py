"""E2E: Query -> results -> search index availability."""

from __future__ import annotations


class TestSearch:
    def test_search_requires_query(self, test_client, authed_headers):
        """Search without query returns 400."""
        resp = test_client.post("/api/v1/search", json={}, headers=authed_headers)
        assert resp.status_code == 400
        assert "query" in resp.json()["error"]

    def test_search_no_index_returns_404(self, test_client, authed_headers):
        """Search before any build returns 404."""
        resp = test_client.post(
            "/api/v1/search",
            json={"query": "test query"},
            headers=authed_headers,
        )
        assert resp.status_code == 404

    def test_search_accepts_mode_param(self, test_client, authed_headers):
        """Search accepts mode parameter (even if no index yet)."""
        resp = test_client.post(
            "/api/v1/search",
            json={"query": "test", "mode": "keyword"},
            headers=authed_headers,
        )
        # 404 (no index) is expected, not 400
        assert resp.status_code == 404

    def test_search_accepts_layers_param(self, test_client, authed_headers):
        """Search accepts layers filter parameter."""
        resp = test_client.post(
            "/api/v1/search",
            json={"query": "test", "layers": ["episodes"]},
            headers=authed_headers,
        )
        assert resp.status_code == 404

    def test_search_invalid_json(self, test_client, authed_headers):
        """Search with invalid JSON returns 400."""
        resp = test_client.post(
            "/api/v1/search",
            content=b"not json",
            headers={**authed_headers, "content-type": "application/json"},
        )
        assert resp.status_code == 400
