"""E2E: Two leaders claim same term -> client picks higher-priority."""

from __future__ import annotations

import pytest

from synix.mesh.cluster import resolve_equal_term_tiebreak


class TestEqualTermTiebreak:
    def test_lower_index_wins_same_term(self):
        """When two leaders have the same term, lower candidate index wins."""
        candidates = ["node-a", "node-b", "node-c"]
        winner = resolve_equal_term_tiebreak(5, "node-a", "node-b", candidates)
        assert winner == "node-a"

    def test_higher_index_loses(self):
        """Higher-index candidate loses tiebreak."""
        candidates = ["primary", "secondary", "tertiary"]
        winner = resolve_equal_term_tiebreak(3, "secondary", "tertiary", candidates)
        assert winner == "secondary"

    def test_same_leader_returns_same(self):
        """Same leader on both terms returns that leader."""
        candidates = ["node-a", "node-b"]
        winner = resolve_equal_term_tiebreak(2, "node-a", "node-a", candidates)
        assert winner == "node-a"

    def test_unknown_candidate_loses(self):
        """Leader not in candidates list loses to one that is."""
        candidates = ["node-a", "node-b"]
        winner = resolve_equal_term_tiebreak(1, "node-a", "node-z", candidates)
        assert winner == "node-a"

    def test_both_unknown_raises(self):
        """Both unknown candidates raises ValueError."""
        candidates = ["node-a", "node-b"]
        with pytest.raises(ValueError, match="Neither"):
            resolve_equal_term_tiebreak(1, "node-x", "node-z", candidates)
