"""Unit tests for the synix viewer server."""


class TestStatus:
    def test_returns_loaded(self, viewer_client):
        resp = viewer_client.get("/api/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["loaded"] is True
        assert data["title"] == "Test Viewer"
        assert data["artifact_count"] == 4


class TestLayers:
    def test_sorted_by_level(self, viewer_client):
        resp = viewer_client.get("/api/layers")
        assert resp.status_code == 200
        layers = resp.get_json()
        assert len(layers) == 3
        levels = [l["level"] for l in layers]
        assert levels == sorted(levels)

    def test_layer_counts(self, viewer_client):
        resp = viewer_client.get("/api/layers")
        layers = resp.get_json()
        counts = {l["name"]: l["count"] for l in layers}
        assert counts["transcripts"] == 2
        assert counts["episodes"] == 1
        assert counts["core"] == 1


class TestArtifacts:
    def test_by_layer(self, viewer_client):
        resp = viewer_client.get("/api/artifacts?layer=transcripts")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["total"] == 2
        assert len(data["items"]) == 2
        assert all(item["layer"] == "transcripts" for item in data["items"])

    def test_pagination(self, viewer_client):
        resp = viewer_client.get("/api/artifacts?layer=transcripts&page=1&per_page=1")
        data = resp.get_json()
        assert len(data["items"]) == 1
        assert data["total"] == 2
        assert data["page"] == 1

    def test_sort_by_date_desc(self, viewer_client):
        resp = viewer_client.get("/api/artifacts?layer=transcripts&sort=date&order=desc")
        data = resp.get_json()
        dates = [item["date"] for item in data["items"]]
        assert dates == sorted(dates, reverse=True)

    def test_sort_by_title_asc(self, viewer_client):
        resp = viewer_client.get("/api/artifacts?layer=transcripts&sort=title&order=asc")
        data = resp.get_json()
        titles = [item["title"] for item in data["items"]]
        assert titles == sorted(titles)

    def test_requires_layer_param(self, viewer_client):
        resp = viewer_client.get("/api/artifacts")
        assert resp.status_code == 400


class TestArtifactDetail:
    def test_returns_full_artifact(self, viewer_client):
        resp = viewer_client.get("/api/artifact/transcript-001")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["label"] == "transcript-001"
        assert data["artifact_type"] == "transcript"
        assert "memory" in data["content"].lower()
        assert data["layer"] == "transcripts"
        assert isinstance(data["metadata"], dict)

    def test_not_found(self, viewer_client):
        resp = viewer_client.get("/api/artifact/nonexistent")
        assert resp.status_code == 404


class TestLineage:
    def test_parents_and_children(self, viewer_client):
        resp = viewer_client.get("/api/lineage/episode-001")
        assert resp.status_code == 200
        data = resp.get_json()
        parent_labels = [p["label"] for p in data["parents"]]
        assert "transcript-001" in parent_labels or "transcript-002" in parent_labels
        child_labels = [c["label"] for c in data["children"]]
        assert "core-001" in child_labels

    def test_no_relations(self, viewer_client):
        # core-001 has no children (it's a leaf artifact)
        resp = viewer_client.get("/api/lineage/core-001")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["children"] == []

    def test_not_found(self, viewer_client):
        resp = viewer_client.get("/api/lineage/nonexistent")
        assert resp.status_code == 404


class TestSearch:
    def test_keyword_search(self, viewer_client):
        resp = viewer_client.get("/api/search?q=memory")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["items"]) > 0
        assert "has_more" in data
        assert all("snippet" in item for item in data["items"])

    def test_search_with_layer_filter(self, viewer_client):
        resp = viewer_client.get("/api/search?q=memory&layer=transcripts")
        data = resp.get_json()
        assert all(item["layer"] == "transcripts" for item in data["items"])

    def test_search_pagination(self, viewer_client):
        resp = viewer_client.get("/api/search?q=memory&page=1&per_page=1")
        data = resp.get_json()
        assert len(data["items"]) <= 1
        assert data["page"] == 1

    def test_empty_query(self, viewer_client):
        resp = viewer_client.get("/api/search?q=")
        assert resp.status_code == 400


class TestSecurity:
    def test_artifact_content_returned_raw(self, viewer_client):
        """Verify the API returns raw content — client is responsible for sanitization."""
        resp = viewer_client.get("/api/artifact/transcript-001")
        data = resp.get_json()
        # Content is returned as-is; sanitization is client-side via marked config
        assert isinstance(data["content"], str)


class TestInputValidation:
    def test_negative_page(self, viewer_client):
        resp = viewer_client.get("/api/artifacts?layer=transcripts&page=-1")
        data = resp.get_json()
        assert data["page"] == 1  # clamped to 1

    def test_huge_per_page(self, viewer_client):
        resp = viewer_client.get("/api/artifacts?layer=transcripts&per_page=99999")
        data = resp.get_json()
        assert data["per_page"] <= 200  # clamped

    def test_invalid_sort(self, viewer_client):
        resp = viewer_client.get("/api/artifacts?layer=transcripts&sort=malicious")
        assert resp.status_code == 200  # should still work with default sort

    def test_invalid_order(self, viewer_client):
        resp = viewer_client.get("/api/artifacts?layer=transcripts&order=malicious")
        assert resp.status_code == 200  # should still work with default order

    def test_search_negative_page(self, viewer_client):
        resp = viewer_client.get("/api/search?q=memory&page=-5")
        data = resp.get_json()
        assert data["page"] == 1

    def test_search_huge_per_page(self, viewer_client):
        resp = viewer_client.get("/api/search?q=memory&per_page=50000")
        data = resp.get_json()
        assert data["per_page"] <= 200


class TestStatic:
    def test_index_html(self, viewer_client):
        resp = viewer_client.get("/")
        assert resp.status_code == 200
        assert b"<!DOCTYPE html>" in resp.data

    def test_static_js(self, viewer_client):
        resp = viewer_client.get("/static/app.js")
        assert resp.status_code == 200
