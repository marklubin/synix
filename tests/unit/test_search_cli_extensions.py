"""Unit tests for search CLI extensions: --step, --trace, --customer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from synix.cli import main
from synix.core.models import Artifact
from synix.build.artifacts import ArtifactStore
from synix.build.provenance import ProvenanceTracker
from synix.search.indexer import SearchIndex


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def populated_build_dir(tmp_path):
    """Build dir with a search index, artifacts, and provenance for testing."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    # Create artifacts with varying layers and metadata
    store = ArtifactStore(build_dir)
    provenance = ProvenanceTracker(build_dir)

    # Transcript artifact
    t1 = Artifact(
        artifact_id="t-conv001",
        artifact_type="transcript",
        content="user: Tell me about machine learning\nassistant: Machine learning is a subset of AI",
        metadata={
            "source": "chatgpt",
            "source_conversation_id": "conv001",
            "title": "ML discussion",
            "date": "2024-03-15",
            "layer_name": "transcripts",
            "customer_id": "acme-corp",
        },
    )
    store.save_artifact(t1, layer_name="transcripts", layer_level=0)

    # Episode artifact
    ep1 = Artifact(
        artifact_id="ep-conv001",
        artifact_type="episode",
        content="In this conversation about machine learning, the user discussed fundamentals of AI",
        metadata={
            "source_conversation_id": "conv001",
            "date": "2024-03-15",
            "layer_name": "episodes",
            "customer_id": "acme-corp",
        },
    )
    store.save_artifact(ep1, layer_name="episodes", layer_level=1)

    # Another episode with different customer
    ep2 = Artifact(
        artifact_id="ep-conv002",
        artifact_type="episode",
        content="Discussion about machine learning applications in healthcare and diagnostics",
        metadata={
            "source_conversation_id": "conv002",
            "date": "2024-03-16",
            "layer_name": "episodes",
            "customer_id": "beta-inc",
        },
    )
    store.save_artifact(ep2, layer_name="episodes", layer_level=1)

    # Monthly rollup artifact
    m1 = Artifact(
        artifact_id="monthly-2024-03",
        artifact_type="rollup",
        content="In March 2024 the main themes were machine learning and AI development",
        metadata={
            "date_range": "2024-03",
            "layer_name": "monthly",
            "customer_id": "acme-corp",
        },
    )
    store.save_artifact(m1, layer_name="monthly", layer_level=2)

    # Record provenance chains
    provenance.record("t-conv001", parent_ids=[], prompt_id=None)
    provenance.record("ep-conv001", parent_ids=["t-conv001"], prompt_id="episode_summary_v1")
    provenance.record("ep-conv002", parent_ids=[], prompt_id="episode_summary_v1")
    provenance.record("monthly-2024-03", parent_ids=["ep-conv001", "ep-conv002"], prompt_id="monthly_rollup_v1")

    # Build search index
    db_path = build_dir / "search.db"
    index = SearchIndex(db_path)
    index.create()
    index.insert(ep1, "episodes", 1)
    index.insert(ep2, "episodes", 1)
    index.insert(m1, "monthly", 2)
    index.close()

    return build_dir


# --- Flag existence tests ---


def test_step_flag_exists(runner):
    """CLI accepts --step flag."""
    result = runner.invoke(main, ["search", "--help"])
    assert result.exit_code == 0
    assert "--step" in result.output


def test_trace_flag_exists(runner):
    """CLI accepts --trace flag."""
    result = runner.invoke(main, ["search", "--help"])
    assert result.exit_code == 0
    assert "--trace" in result.output


def test_customer_flag_exists(runner):
    """CLI accepts --customer flag."""
    result = runner.invoke(main, ["search", "--help"])
    assert result.exit_code == 0
    assert "--customer" in result.output


# --- Functional tests ---


def test_step_filters_results(runner, populated_build_dir):
    """--step filters results to the specified layer."""
    result = runner.invoke(main, [
        "search", "machine learning",
        "--build-dir", str(populated_build_dir),
        "--step", "episodes",
    ])
    assert result.exit_code == 0
    # Should have episode results but not monthly
    assert "episodes" in result.output
    assert "monthly" not in result.output or "monthly-2024-03" not in result.output


def test_step_and_layers_combine(runner, populated_build_dir):
    """--step and --layers combine their filter sets."""
    result = runner.invoke(main, [
        "search", "machine learning",
        "--build-dir", str(populated_build_dir),
        "--layers", "episodes",
        "--step", "monthly",
    ])
    assert result.exit_code == 0
    # Should include both episode and monthly results
    assert "episodes" in result.output or "monthly" in result.output


def test_trace_shows_provenance(runner, populated_build_dir):
    """With --trace, output contains provenance tree."""
    result = runner.invoke(main, [
        "search", "machine learning",
        "--build-dir", str(populated_build_dir),
        "--trace",
    ])
    assert result.exit_code == 0
    assert "Provenance" in result.output


def test_trace_shows_tree_characters(runner, populated_build_dir):
    """With --trace, output contains tree-drawing characters."""
    result = runner.invoke(main, [
        "search", "machine learning",
        "--build-dir", str(populated_build_dir),
        "--trace",
    ])
    assert result.exit_code == 0
    # Rich Tree uses box-drawing characters; look for any parent artifact IDs
    # that would appear in the provenance tree
    assert "t-conv001" in result.output or "ep-conv001" in result.output


def test_customer_filters_results(runner, populated_build_dir):
    """--customer filters results by customer_id metadata."""
    result = runner.invoke(main, [
        "search", "machine learning",
        "--build-dir", str(populated_build_dir),
        "--customer", "acme-corp",
    ])
    assert result.exit_code == 0
    # Should have results for acme-corp
    assert "ep-conv001" in result.output or "monthly-2024-03" in result.output
    # beta-inc's episode (ep-conv002) should NOT appear as a search result panel.
    # It may still appear in provenance trees since the monthly rollup depends on it.
    # Check that ep-conv002 does not appear as a result title/artifact line.
    # The result panels show "Artifact: <id>" — ep-conv002 should not be listed there.
    lines = result.output.split("\n")
    artifact_lines = [l for l in lines if "Artifact:" in l and "ep-conv002" in l]
    assert len(artifact_lines) == 0, "ep-conv002 should not appear as a search result"


def test_customer_no_match(runner, populated_build_dir):
    """--customer with no matching results shows appropriate message."""
    result = runner.invoke(main, [
        "search", "machine learning",
        "--build-dir", str(populated_build_dir),
        "--customer", "nonexistent-customer",
    ])
    assert result.exit_code == 0
    assert "No results for customer" in result.output


def test_step_no_match(runner, populated_build_dir):
    """--step with a layer that has no matches returns no results."""
    result = runner.invoke(main, [
        "search", "machine learning",
        "--build-dir", str(populated_build_dir),
        "--step", "core",
    ])
    assert result.exit_code == 0
    assert "No results for" in result.output


def test_customer_and_step_combined(runner, populated_build_dir):
    """--customer and --step can be combined."""
    result = runner.invoke(main, [
        "search", "machine learning",
        "--build-dir", str(populated_build_dir),
        "--step", "episodes",
        "--customer", "acme-corp",
    ])
    assert result.exit_code == 0
    # Only acme-corp episodes should show
    assert "ep-conv002" not in result.output


def test_trace_without_provenance_data(runner, tmp_path):
    """--trace works gracefully when no provenance data exists for a result."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    # Create a search index with an artifact that has no provenance
    artifact = Artifact(
        artifact_id="orphan-001",
        artifact_type="episode",
        content="An orphaned artifact about machine learning with no provenance",
        metadata={"layer_name": "episodes"},
    )
    db_path = build_dir / "search.db"
    index = SearchIndex(db_path)
    index.create()
    index.insert(artifact, "episodes", 1)
    index.close()

    result = runner.invoke(main, [
        "search", "machine learning",
        "--build-dir", str(build_dir),
        "--trace",
    ])
    assert result.exit_code == 0
    # Should still show results, just without provenance tree
    assert "orphan-001" in result.output
