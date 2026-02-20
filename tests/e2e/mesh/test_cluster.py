"""E2E: Heartbeats -> cluster state -> members visible."""

from __future__ import annotations


class TestCluster:
    def test_heartbeat_registers_member(self, test_client, authed_headers):
        payload = {
            "hostname": "node-alpha",
            "term": {"counter": 1, "leader_id": "test-server"},
            "config_hash": "",
        }
        resp = test_client.post("/api/v1/heartbeat", json=payload, headers=authed_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_cluster_state_shows_members(self, test_client, authed_headers):
        # Register a member via heartbeat
        payload = {
            "hostname": "node-beta",
            "term": {"counter": 1, "leader_id": "test-server"},
            "config_hash": "",
        }
        test_client.post("/api/v1/heartbeat", json=payload, headers=authed_headers)

        resp = test_client.get("/api/v1/cluster", headers=authed_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "node-beta" in data["members"]

    def test_heartbeat_missing_hostname_rejected(self, test_client, authed_headers):
        payload = {"term": 1, "config_hash": ""}
        resp = test_client.post("/api/v1/heartbeat", json=payload, headers=authed_headers)
        assert resp.status_code == 400

    def test_multiple_members_tracked(self, test_client, authed_headers):
        for name in ["node-a", "node-b", "node-c"]:
            payload = {
                "hostname": name,
                "term": {"counter": 1, "leader_id": "test-server"},
                "config_hash": "",
            }
            test_client.post("/api/v1/heartbeat", json=payload, headers=authed_headers)

        resp = test_client.get("/api/v1/cluster", headers=authed_headers)
        data = resp.json()
        assert data["member_count"] >= 3
