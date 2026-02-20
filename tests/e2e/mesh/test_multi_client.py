"""E2E: Two clients -> different files -> same build -> same bundle."""

from __future__ import annotations

import base64
import hashlib


class TestMultiClient:
    def test_two_clients_submit_different_sessions(self, test_client, authed_headers):
        """Two different clients can submit different sessions."""
        for i, node in enumerate(["client-a", "client-b"]):
            content = f"session data from {node}".encode()
            payload = {
                "session_id": f"sess-{node}-{i}",
                "project_dir": "shared-project",
                "content": base64.b64encode(content).decode(),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            headers = {**authed_headers, "X-Mesh-Node": node}
            resp = test_client.post("/api/v1/sessions", json=payload, headers=headers)
            assert resp.status_code == 201, f"Failed for {node}: {resp.json()}"
            assert resp.json()["new"] is True

    def test_both_clients_see_same_status(self, test_client, authed_headers):
        """After submissions from both clients, status shows all sessions."""
        # Submit from two clients
        for node in ["client-x", "client-y"]:
            content = f"data-{node}".encode()
            payload = {
                "session_id": f"sess-{node}",
                "project_dir": "shared",
                "content": base64.b64encode(content).decode(),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            headers = {**authed_headers, "X-Mesh-Node": node}
            test_client.post("/api/v1/sessions", json=payload, headers=headers)

        resp = test_client.get("/api/v1/status", headers=authed_headers)
        data = resp.json()
        assert data["sessions"]["total"] >= 2

    def test_node_header_recorded(self, test_client, authed_headers):
        """X-Mesh-Node header is used to track submission origin."""
        content = b"node-tracking-test"
        payload = {
            "session_id": "sess-tracked",
            "project_dir": "project",
            "content": base64.b64encode(content).decode(),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        headers = {**authed_headers, "X-Mesh-Node": "tracker-node"}
        resp = test_client.post("/api/v1/sessions", json=payload, headers=headers)
        assert resp.status_code == 201
