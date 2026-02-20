"""Tests for health monitoring module."""

from __future__ import annotations

import time

import pytest

from synix.mesh.health import HealthMonitor, MemberState

pytestmark = pytest.mark.mesh


class TestRecordHeartbeat:
    def test_sets_healthy(self):
        monitor = HealthMonitor(heartbeat_interval=10)
        monitor.record_heartbeat("node-a", term_counter=1, config_hash="abc")
        member = monitor.members["node-a"]
        assert member.status == "healthy"
        assert member.missed_beats == 0
        assert member.term_counter == 1
        assert member.config_hash == "abc"

    def test_creates_new_member(self):
        monitor = HealthMonitor()
        assert "node-x" not in monitor.members
        monitor.record_heartbeat("node-x")
        assert "node-x" in monitor.members

    def test_updates_existing_member(self):
        monitor = HealthMonitor()
        monitor.record_heartbeat("node-a", term_counter=1, config_hash="v1")
        monitor.record_heartbeat("node-a", term_counter=2, config_hash="v2")
        member = monitor.members["node-a"]
        assert member.term_counter == 2
        assert member.config_hash == "v2"


class TestCheckHealth:
    def test_recent_heartbeat_healthy(self):
        monitor = HealthMonitor(heartbeat_interval=10)
        monitor.record_heartbeat("node-a")
        dead = monitor.check_health()
        assert dead == []
        assert monitor.members["node-a"].status == "healthy"

    def test_missed_beats_suspect(self):
        monitor = HealthMonitor(heartbeat_interval=10, miss_threshold=3)
        monitor.record_heartbeat("node-a")
        # Simulate time passing: 1-2 missed beats
        monitor.members["node-a"].last_heartbeat = time.time() - 15
        dead = monitor.check_health()
        assert dead == []
        assert monitor.members["node-a"].status == "suspect"
        assert monitor.members["node-a"].missed_beats >= 1

    def test_many_missed_beats_dead(self):
        monitor = HealthMonitor(heartbeat_interval=10, miss_threshold=3)
        monitor.record_heartbeat("node-a")
        # Simulate time passing: 3+ missed beats
        monitor.members["node-a"].last_heartbeat = time.time() - 35
        dead = monitor.check_health()
        assert len(dead) == 1
        assert dead[0].hostname == "node-a"
        assert dead[0].status == "dead"

    def test_returns_dead_members_list(self):
        monitor = HealthMonitor(heartbeat_interval=10, miss_threshold=3)
        monitor.record_heartbeat("alive-node")
        monitor.record_heartbeat("dead-node")
        # Make dead-node old
        monitor.members["dead-node"].last_heartbeat = time.time() - 50
        dead = monitor.check_health()
        hostnames = [m.hostname for m in dead]
        assert "dead-node" in hostnames
        assert "alive-node" not in hostnames

    def test_zero_last_heartbeat_skipped(self):
        monitor = HealthMonitor(heartbeat_interval=10)
        monitor.members["ghost"] = MemberState(hostname="ghost")
        dead = monitor.check_health()
        assert dead == []
        assert monitor.members["ghost"].status == "unknown"


class TestGetMemberStates:
    def test_returns_all_members(self):
        monitor = HealthMonitor()
        monitor.record_heartbeat("node-a")
        monitor.record_heartbeat("node-b")
        states = monitor.get_member_states()
        assert len(states) == 2
        hostnames = {s["hostname"] for s in states}
        assert hostnames == {"node-a", "node-b"}

    def test_state_dict_fields(self):
        monitor = HealthMonitor()
        monitor.record_heartbeat("node-a", term_counter=3, config_hash="xyz")
        states = monitor.get_member_states()
        state = states[0]
        assert state["hostname"] == "node-a"
        assert state["status"] == "healthy"
        assert state["term_counter"] == 3
        assert state["config_hash"] == "xyz"
        assert "last_heartbeat" in state
        assert "missed_beats" in state


class TestRemoveMember:
    def test_remove_existing(self):
        monitor = HealthMonitor()
        monitor.record_heartbeat("node-a")
        assert "node-a" in monitor.members
        monitor.remove_member("node-a")
        assert "node-a" not in monitor.members

    def test_remove_nonexistent_no_error(self):
        monitor = HealthMonitor()
        monitor.remove_member("ghost")  # should not raise
