"""E2E: Submit -> store -> dedup -> build triggers."""

from __future__ import annotations

import base64
import hashlib


class TestSessionFlow:
    def _make_session_payload(self, session_id: str, content: str, project_dir: str = "default"):
        raw = content.encode()
        encoded = base64.b64encode(raw).decode()
        sha256 = hashlib.sha256(raw).hexdigest()
        return {
            "session_id": session_id,
            "project_dir": project_dir,
            "content": encoded,
            "sha256": sha256,
        }

    def test_submit_new_session(self, test_client, authed_headers):
        payload = self._make_session_payload("sess-001", "hello world")
        resp = test_client.post("/api/v1/sessions", json=payload, headers=authed_headers)
        assert resp.status_code == 201
        data = resp.json()
        assert data["new"] is True
        assert data["session_id"] == "sess-001"

    def test_submit_duplicate_detected(self, test_client, authed_headers):
        payload = self._make_session_payload("sess-dup", "duplicate content")
        resp1 = test_client.post("/api/v1/sessions", json=payload, headers=authed_headers)
        assert resp1.status_code == 201

        # Same content, same session_id -> dedup
        resp2 = test_client.post("/api/v1/sessions", json=payload, headers=authed_headers)
        assert resp2.status_code == 200
        assert resp2.json()["new"] is False

    def test_submit_missing_fields_rejected(self, test_client, authed_headers):
        resp = test_client.post("/api/v1/sessions", json={}, headers=authed_headers)
        assert resp.status_code == 400

    def test_submit_invalid_base64_rejected(self, test_client, authed_headers):
        payload = {
            "session_id": "bad-b64",
            "content": "not-valid-base64!!!",
        }
        resp = test_client.post("/api/v1/sessions", json=payload, headers=authed_headers)
        assert resp.status_code == 400

    def test_submit_sha256_mismatch_rejected(self, test_client, authed_headers):
        content = b"test content for sha check"
        payload = {
            "session_id": "bad-sha",
            "content": base64.b64encode(content).decode(),
            "sha256": "wrong_hash",
        }
        resp = test_client.post("/api/v1/sessions", json=payload, headers=authed_headers)
        assert resp.status_code == 400
        assert "mismatch" in resp.json()["error"]

    def test_build_status_after_submit(self, test_client, authed_headers):
        payload = self._make_session_payload("sess-status", "test status")
        test_client.post("/api/v1/sessions", json=payload, headers=authed_headers)
        resp = test_client.get("/api/v1/builds/status", headers=authed_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "state" in data
