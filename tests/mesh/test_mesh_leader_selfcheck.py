"""Tests for leader self-check logic."""

from __future__ import annotations

import pytest

from synix.mesh.cluster import Term, leader_self_check

pytestmark = pytest.mark.mesh


class TestLeaderSelfCheck:
    def test_no_higher_priority_remain_leader(self):
        # "charlie" is index 2, no one at 0 or 1 is alive
        ping = lambda host: False
        result = leader_self_check(
            my_hostname="charlie",
            my_term=Term(counter=1, leader_id="charlie"),
            candidates=["alpha", "bravo", "charlie"],
            ping_fn=ping,
        )
        assert result is True

    def test_higher_priority_alive_step_down(self):
        alive = {"alpha"}
        ping = lambda host: host in alive
        result = leader_self_check(
            my_hostname="bravo",
            my_term=Term(counter=1, leader_id="bravo"),
            candidates=["alpha", "bravo"],
            ping_fn=ping,
        )
        assert result is False

    def test_higher_priority_dead_remain_leader(self):
        ping = lambda host: False  # all dead
        result = leader_self_check(
            my_hostname="bravo",
            my_term=Term(counter=1, leader_id="bravo"),
            candidates=["alpha", "bravo"],
            ping_fn=ping,
        )
        assert result is True

    def test_all_higher_priority_dead_remain_leader(self):
        ping = lambda host: False
        result = leader_self_check(
            my_hostname="charlie",
            my_term=Term(counter=1, leader_id="charlie"),
            candidates=["alpha", "bravo", "charlie"],
            ping_fn=ping,
        )
        assert result is True

    def test_highest_priority_node_always_remains(self):
        # "alpha" is index 0, no one higher to check
        ping = lambda host: True  # doesn't matter
        result = leader_self_check(
            my_hostname="alpha",
            my_term=Term(counter=1, leader_id="alpha"),
            candidates=["alpha", "bravo", "charlie"],
            ping_fn=ping,
        )
        assert result is True

    def test_only_first_higher_alive_triggers_stepdown(self):
        # "charlie" is index 2, "bravo" (index 1) is alive
        alive = {"bravo"}
        ping = lambda host: host in alive
        result = leader_self_check(
            my_hostname="charlie",
            my_term=Term(counter=1, leader_id="charlie"),
            candidates=["alpha", "bravo", "charlie"],
            ping_fn=ping,
        )
        assert result is False
