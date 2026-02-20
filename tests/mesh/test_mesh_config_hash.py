"""Tests for config hash computation consistency."""

from __future__ import annotations

import pytest

from synix.mesh.cluster import cluster_config_hash
from synix.mesh.config import ClusterConfig

pytestmark = pytest.mark.mesh


class TestConfigHashConsistency:
    def test_matches_cluster_config_property(self):
        candidates = ["obispo", "salinas", "oxnard"]
        config = ClusterConfig(leader_candidates=candidates)
        assert cluster_config_hash(candidates) == config.config_hash

    def test_different_orderings_different_hashes(self):
        a = cluster_config_hash(["alpha", "bravo", "charlie"])
        b = cluster_config_hash(["charlie", "bravo", "alpha"])
        assert a != b

    def test_empty_candidate_list_produces_valid_hash(self):
        result = cluster_config_hash([])
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex length

    def test_single_candidate(self):
        result = cluster_config_hash(["only-one"])
        config = ClusterConfig(leader_candidates=["only-one"])
        assert result == config.config_hash
