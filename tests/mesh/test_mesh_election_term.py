"""Tests for term validation and step-down logic."""

from __future__ import annotations

import pytest

from synix.mesh.cluster import Term, should_step_down, validate_term

pytestmark = pytest.mark.mesh


class TestValidateTerm:
    def test_stale_term_rejected(self):
        request = Term(counter=1, leader_id="a")
        server = Term(counter=5, leader_id="a")
        valid, reason = validate_term(request, server)
        assert not valid
        assert "stale term" in reason

    def test_matching_term_and_leader_accepted(self):
        request = Term(counter=3, leader_id="leader-x")
        server = Term(counter=3, leader_id="leader-x")
        valid, reason = validate_term(request, server)
        assert valid
        assert reason == ""

    def test_same_counter_different_leader_rejected(self):
        request = Term(counter=3, leader_id="leader-a")
        server = Term(counter=3, leader_id="leader-b")
        valid, reason = validate_term(request, server)
        assert not valid
        assert "leader_id mismatch" in reason

    def test_higher_term_always_accepted(self):
        request = Term(counter=10, leader_id="new-leader")
        server = Term(counter=3, leader_id="old-leader")
        valid, reason = validate_term(request, server)
        assert valid
        assert reason == ""


class TestShouldStepDown:
    def test_higher_term_step_down(self):
        candidates = ["alpha", "bravo", "charlie"]
        result = should_step_down(
            my_hostname="bravo",
            my_term=Term(counter=1, leader_id="bravo"),
            request_term=Term(counter=5, leader_id="alpha"),
            candidates=candidates,
        )
        assert result is True

    def test_same_term_higher_priority_leader_step_down(self):
        candidates = ["alpha", "bravo", "charlie"]
        result = should_step_down(
            my_hostname="bravo",
            my_term=Term(counter=3, leader_id="bravo"),
            request_term=Term(counter=3, leader_id="alpha"),
            candidates=candidates,
        )
        assert result is True

    def test_same_term_lower_priority_leader_no_step_down(self):
        candidates = ["alpha", "bravo", "charlie"]
        result = should_step_down(
            my_hostname="alpha",
            my_term=Term(counter=3, leader_id="alpha"),
            request_term=Term(counter=3, leader_id="charlie"),
            candidates=candidates,
        )
        assert result is False

    def test_same_term_same_leader_no_step_down(self):
        candidates = ["alpha", "bravo"]
        result = should_step_down(
            my_hostname="alpha",
            my_term=Term(counter=3, leader_id="alpha"),
            request_term=Term(counter=3, leader_id="alpha"),
            candidates=candidates,
        )
        assert result is False

    def test_lower_term_no_step_down(self):
        candidates = ["alpha", "bravo"]
        result = should_step_down(
            my_hostname="alpha",
            my_term=Term(counter=5, leader_id="alpha"),
            request_term=Term(counter=2, leader_id="bravo"),
            candidates=candidates,
        )
        assert result is False
