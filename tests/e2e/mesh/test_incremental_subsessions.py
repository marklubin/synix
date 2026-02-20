"""E2E: Incremental subsession submission and processing."""

from __future__ import annotations

import base64
import hashlib


class TestIncrementalSubsessions:
    """Test subsession submission through the HTTP API."""

    def _make_payload(
        self,
        session_id: str,
        content: str,
        project_dir: str = "default",
        subsession_seq: int = 0,
    ):
        raw = content.encode()
        encoded = base64.b64encode(raw).decode()
        sha256 = hashlib.sha256(raw).hexdigest()
        return {
            "session_id": session_id,
            "project_dir": project_dir,
            "subsession_seq": subsession_seq,
            "content": encoded,
            "sha256": sha256,
        }

    def test_submit_subsessions(self, test_client, authed_headers):
        """Submit multiple subsessions for the same session_id."""
        p0 = self._make_payload("sess-inc", "part one", subsession_seq=0)
        p1 = self._make_payload("sess-inc", "part two", subsession_seq=1)
        p2 = self._make_payload("sess-inc", "part three", subsession_seq=2)

        r0 = test_client.post("/api/v1/sessions", json=p0, headers=authed_headers)
        r1 = test_client.post("/api/v1/sessions", json=p1, headers=authed_headers)
        r2 = test_client.post("/api/v1/sessions", json=p2, headers=authed_headers)

        assert r0.status_code == 201
        assert r1.status_code == 201
        assert r2.status_code == 201

        # All 3 should be pending
        status = test_client.get("/api/v1/status", headers=authed_headers)
        sessions = status.json()["sessions"]
        assert sessions["pending"] >= 3

    def test_subsession_manifest_includes_seq(self, test_client, authed_headers):
        """Session manifest includes subsession_seq for each entry."""
        p0 = self._make_payload("sess-man", "sub zero", subsession_seq=0)
        p1 = self._make_payload("sess-man", "sub one", subsession_seq=1)

        test_client.post("/api/v1/sessions", json=p0, headers=authed_headers)
        test_client.post("/api/v1/sessions", json=p1, headers=authed_headers)

        resp = test_client.get("/api/v1/sessions/manifest", headers=authed_headers)
        assert resp.status_code == 200

        manifest = resp.json()
        sess_entries = [s for s in manifest["sessions"] if s["session_id"] == "sess-man"]
        assert len(sess_entries) == 2
        seqs = {s["subsession_seq"] for s in sess_entries}
        assert seqs == {0, 1}

    def test_subsession_dedup(self, test_client, authed_headers):
        """Same content submitted twice is deduped."""
        p = self._make_payload("sess-dup", "identical", subsession_seq=0)
        r1 = test_client.post("/api/v1/sessions", json=p, headers=authed_headers)
        assert r1.status_code == 201
        assert r1.json()["new"] is True

        r2 = test_client.post("/api/v1/sessions", json=p, headers=authed_headers)
        assert r2.status_code == 200
        assert r2.json()["new"] is False

    def test_backward_compat_no_subsession_seq(self, test_client, authed_headers):
        """Submissions without subsession_seq default to 0."""
        raw = b"legacy content"
        payload = {
            "session_id": "sess-legacy",
            "project_dir": "default",
            "content": base64.b64encode(raw).decode(),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

        resp = test_client.post("/api/v1/sessions", json=payload, headers=authed_headers)
        assert resp.status_code == 201

        # Should appear in manifest with seq=0
        manifest = test_client.get("/api/v1/sessions/manifest", headers=authed_headers).json()
        entry = [s for s in manifest["sessions"] if s["session_id"] == "sess-legacy"]
        assert len(entry) == 1
        assert entry[0]["subsession_seq"] == 0

    def test_session_file_download_with_seq(self, test_client, authed_headers):
        """Can download individual subsession files."""
        p0 = self._make_payload("sess-dl", "download zero", subsession_seq=0)
        p1 = self._make_payload("sess-dl", "download one", subsession_seq=1)

        test_client.post("/api/v1/sessions", json=p0, headers=authed_headers)
        test_client.post("/api/v1/sessions", json=p1, headers=authed_headers)

        # Download seq=0
        r0 = test_client.get(
            "/api/v1/sessions/sess-dl/file",
            params={"project_dir": "default", "subsession_seq": "0"},
            headers=authed_headers,
        )
        assert r0.status_code == 200

        # Download seq=1
        r1 = test_client.get(
            "/api/v1/sessions/sess-dl/file",
            params={"project_dir": "default", "subsession_seq": "1"},
            headers=authed_headers,
        )
        assert r1.status_code == 200

        # Download nonexistent seq=99
        r99 = test_client.get(
            "/api/v1/sessions/sess-dl/file",
            params={"project_dir": "default", "subsession_seq": "99"},
            headers=authed_headers,
        )
        assert r99.status_code == 404
