"""Tests for cluster module — election logic, config hash, state persistence."""

from __future__ import annotations

import pytest

from synix.mesh.cluster import (
    ClusterState,
    Term,
    cluster_config_hash,
    elect_leader,
)

pytestmark = pytest.mark.mesh


class TestClusterConfigHash:
    def test_deterministic(self):
        candidates = ["alpha", "bravo", "charlie"]
        assert cluster_config_hash(candidates) == cluster_config_hash(candidates)

    def test_ordering_matters(self):
        a = cluster_config_hash(["alpha", "bravo"])
        b = cluster_config_hash(["bravo", "alpha"])
        assert a != b


class TestElectLeader:
    def test_first_alive_wins(self):
        alive = {"alpha", "bravo"}
        ping = lambda host: host in alive
        result = elect_leader(["alpha", "bravo", "charlie"], ping, "charlie")
        assert result == "alpha"

    def test_skip_dead_higher_priority(self):
        alive = {"bravo"}
        ping = lambda host: host in alive
        result = elect_leader(["alpha", "bravo", "charlie"], ping, "charlie")
        assert result == "bravo"

    def test_all_dead_returns_none(self):
        ping = lambda host: False
        result = elect_leader(["alpha", "bravo", "charlie"], ping, "delta")
        assert result is None

    def test_self_wins_when_highest_priority(self):
        ping = lambda host: False  # no one else is alive
        result = elect_leader(["alpha", "bravo"], ping, "alpha")
        assert result == "alpha"

    def test_self_wins_when_higher_dead(self):
        alive = set()  # no one reachable
        ping = lambda host: host in alive
        result = elect_leader(["alpha", "bravo", "charlie"], ping, "bravo")
        # alpha is dead (ping returns False), bravo is self -> bravo wins
        assert result == "bravo"

    def test_higher_priority_alive_beats_self(self):
        alive = {"alpha"}
        ping = lambda host: host in alive
        result = elect_leader(["alpha", "bravo"], ping, "bravo")
        assert result == "alpha"


class TestClusterState:
    def test_save_and_load(self, tmp_path):
        state = ClusterState(
            term=Term(counter=5, leader_id="node-a"),
            candidates=["node-a", "node-b"],
            config_hash="abc123",
            role="server",
            server_url="http://node-a:7433",
            my_hostname="node-a",
        )
        path = tmp_path / "state.json"
        state.save(path)
        loaded = ClusterState.load(path)

        assert loaded.term.counter == 5
        assert loaded.term.leader_id == "node-a"
        assert loaded.candidates == ["node-a", "node-b"]
        assert loaded.config_hash == "abc123"
        assert loaded.role == "server"
        assert loaded.server_url == "http://node-a:7433"
        assert loaded.my_hostname == "node-a"

    def test_load_missing_file_returns_default(self, tmp_path):
        loaded = ClusterState.load(tmp_path / "nonexistent.json")
        assert loaded.term.counter == 0
        assert loaded.role == "client"

    def test_save_creates_parent_dirs(self, tmp_path):
        state = ClusterState(term=Term(counter=1, leader_id="x"))
        path = tmp_path / "nested" / "dir" / "state.json"
        state.save(path)
        assert path.exists()


class TestTerm:
    def test_to_dict_roundtrip(self):
        term = Term(counter=42, leader_id="leader-x")
        d = term.to_dict()
        restored = Term.from_dict(d)
        assert restored.counter == 42
        assert restored.leader_id == "leader-x"

    def test_from_dict_defaults(self):
        term = Term.from_dict({})
        assert term.counter == 0
        assert term.leader_id == ""
