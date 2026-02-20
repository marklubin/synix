"""E2E: Lower-priority leader detects higher-priority -> steps down."""

from __future__ import annotations

from synix.mesh.cluster import Term, leader_self_check, should_step_down


class TestLeaderSelfCheck:
    def test_leader_should_step_down_on_higher_term(self):
        """Leader should step down if request has higher term."""
        candidates = ["primary", "secondary"]
        my_term = Term(counter=1, leader_id="secondary")
        request_term = Term(counter=2, leader_id="primary")
        result = should_step_down("secondary", my_term, request_term, candidates)
        assert result is True

    def test_leader_should_not_step_down_when_highest_priority(self):
        """Highest-priority leader should not step down on same term."""
        candidates = ["primary", "secondary"]
        my_term = Term(counter=1, leader_id="primary")
        request_term = Term(counter=1, leader_id="primary")
        result = should_step_down("primary", my_term, request_term, candidates)
        assert result is False

    def test_leader_stays_when_lower_term_request(self):
        """Leader stays if request has a lower term."""
        candidates = ["primary", "secondary"]
        my_term = Term(counter=3, leader_id="secondary")
        request_term = Term(counter=2, leader_id="primary")
        result = should_step_down("secondary", my_term, request_term, candidates)
        assert result is False

    def test_step_down_same_term_higher_priority_leader(self):
        """Same term but higher-priority leader in request -> step down."""
        candidates = ["primary", "secondary", "tertiary"]
        my_term = Term(counter=5, leader_id="secondary")
        request_term = Term(counter=5, leader_id="primary")
        result = should_step_down("secondary", my_term, request_term, candidates)
        assert result is True

    def test_self_check_stays_when_higher_dead(self):
        """leader_self_check stays leader when higher-priority is dead."""
        candidates = ["node-a", "node-b", "node-c"]
        my_term = Term(counter=1, leader_id="node-b")

        def ping_fn(hostname: str) -> bool:
            return {"node-a": False, "node-b": True, "node-c": True}.get(hostname, False)

        # node-a is dead, so node-b should remain leader
        result = leader_self_check("node-b", my_term, candidates, ping_fn)
        assert result is True

    def test_self_check_steps_down_when_higher_alive(self):
        """leader_self_check returns False when higher-priority is alive."""
        candidates = ["node-a", "node-b", "node-c"]
        my_term = Term(counter=1, leader_id="node-b")

        def ping_fn(hostname: str) -> bool:
            return True

        result = leader_self_check("node-b", my_term, candidates, ping_fn)
        assert result is False  # node-a is alive and higher priority
