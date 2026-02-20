"""E2E: Mismatched candidate ordering -> 409."""

from __future__ import annotations

from synix.mesh.cluster import cluster_config_hash


class TestConfigHashMismatch:
    def test_mismatched_config_hash_rejected(self, test_client, authed_headers):
        # Server has empty candidates -> its hash is sha256 of ""
        wrong_hash = cluster_config_hash(["different", "ordering"])
        payload = {
            "hostname": "confused-node",
            "term": {"counter": 1, "leader_id": "test-server"},
            "config_hash": wrong_hash,
        }
        resp = test_client.post("/api/v1/heartbeat", json=payload, headers=authed_headers)
        assert resp.status_code == 409
        assert "config_hash" in resp.json().get("error", "")

    def test_empty_config_hash_accepted(self, test_client, authed_headers):
        """Empty hash from client is accepted (backwards compat)."""
        payload = {
            "hostname": "new-node",
            "term": {"counter": 1, "leader_id": "test-server"},
            "config_hash": "",
        }
        resp = test_client.post("/api/v1/heartbeat", json=payload, headers=authed_headers)
        assert resp.status_code == 200
