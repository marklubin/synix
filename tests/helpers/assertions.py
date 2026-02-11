"""Shared assertion helpers for Synix demo E2E tests.

These helpers operate on structured run logs, artifact stores, and search
interfaces to verify the exact behaviors described in the demo scenario
test specifications.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Artifact existence and caching
# ---------------------------------------------------------------------------


def assert_artifact_exists(store, step: str, record_id: str) -> None:
    """Assert that an artifact exists in the store for the given step and record.

    Args:
        store: ArtifactStore instance.
        step: The pipeline step/layer name (e.g., "episodes", "monthly").
        record_id: The artifact ID or a substring to match.
    """
    artifacts = store.list_artifacts(step)
    matching = [a for a in artifacts if record_id in a.artifact_id]
    assert matching, (
        f"No artifact matching '{record_id}' found in step '{step}'. Available: {[a.artifact_id for a in artifacts]}"
    )


def assert_artifact_cached(run_log: dict, step: str, record_id: str) -> None:
    """Assert that a specific artifact was NOT rebuilt (cache hit) during a run.

    Args:
        run_log: Structured run log dict with per-step cache/rebuild info.
        step: The pipeline step name.
        record_id: The artifact ID or substring.
    """
    step_log = _get_step_log(run_log, step)
    rebuilt = step_log.get("rebuilt_ids", [])
    assert record_id not in rebuilt, (
        f"Artifact '{record_id}' was rebuilt in step '{step}' but expected it to be cached. Rebuilt: {rebuilt}"
    )
    cached = step_log.get("cached_ids", [])
    assert any(record_id in cid for cid in cached), (
        f"Artifact '{record_id}' not found in cached list for step '{step}'. Cached: {cached}"
    )


def assert_artifact_rebuilt(run_log: dict, step: str, record_id: str) -> None:
    """Assert that a specific artifact WAS rebuilt during a run.

    Args:
        run_log: Structured run log dict.
        step: The pipeline step name.
        record_id: The artifact ID or substring.
    """
    step_log = _get_step_log(run_log, step)
    rebuilt = step_log.get("rebuilt_ids", [])
    assert any(record_id in rid for rid in rebuilt), (
        f"Artifact '{record_id}' was NOT rebuilt in step '{step}' but expected it to be. Rebuilt: {rebuilt}"
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def assert_provenance_chain(provenance_tracker, artifact_id: str, expected_chain: list[str]) -> None:
    """Assert that the provenance chain for an artifact matches expectations.

    Args:
        provenance_tracker: ProvenanceTracker instance.
        artifact_id: The artifact to trace.
        expected_chain: List of artifact IDs expected in the chain (in any order).
    """
    chain = provenance_tracker.get_chain(artifact_id)
    chain_ids = [r.artifact_id for r in chain]
    for expected_id in expected_chain:
        assert any(expected_id in cid for cid in chain_ids), (
            f"Expected '{expected_id}' in provenance chain for '{artifact_id}'. Actual chain: {chain_ids}"
        )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def assert_search_returns(
    search_fn,
    query: str,
    index_name: str | None,
    expected_ids: list[str],
    top_k: int = 5,
) -> None:
    """Assert that a search query returns expected artifact IDs in top-k results.

    Args:
        search_fn: Callable that takes (query, index_name, top_k) and returns results.
        query: Search query string.
        index_name: Name of the search index (None for default).
        expected_ids: Artifact IDs (or substrings) expected in results.
        top_k: Number of top results to check.
    """
    results = search_fn(query, index_name=index_name, top_k=top_k)
    result_ids = [r.artifact_id for r in results[:top_k]]
    for expected_id in expected_ids:
        assert any(expected_id in rid for rid in result_ids), (
            f"Expected '{expected_id}' in search results for query '{query}' (index: {index_name}). Got: {result_ids}"
        )


def assert_search_indexes_are_independent(
    search_fn,
    index_a: str,
    index_b: str,
    query: str,
) -> None:
    """Assert that two named search indexes return independent results.

    No artifact ID from index_a should appear in index_b results and vice versa.

    Args:
        search_fn: Callable that takes (query, index_name, top_k).
        index_a: First index name.
        index_b: Second index name.
        query: Search query.
    """
    results_a = search_fn(query, index_name=index_a, top_k=20)
    results_b = search_fn(query, index_name=index_b, top_k=20)
    ids_a = {r.artifact_id for r in results_a}
    ids_b = {r.artifact_id for r in results_b}
    overlap = ids_a & ids_b
    assert not overlap, (
        f"Search indexes '{index_a}' and '{index_b}' share artifact IDs: {overlap}. Indexes should be independent."
    )


# ---------------------------------------------------------------------------
# Diffing
# ---------------------------------------------------------------------------


def assert_diff_nonempty(diff_result: Any, artifact_id: str) -> None:
    """Assert that a diff result for an artifact is non-empty (content changed).

    Args:
        diff_result: The diff result object/dict.
        artifact_id: The artifact that was diffed.
    """
    if isinstance(diff_result, dict):
        changes = diff_result.get("changes", [])
    elif hasattr(diff_result, "changes"):
        changes = diff_result.changes
    elif hasattr(diff_result, "has_changes"):
        assert diff_result.has_changes, f"Diff for '{artifact_id}' is empty but expected changes."
        return
    else:
        # Treat as a string diff
        changes = str(diff_result).strip()

    assert changes, f"Diff for artifact '{artifact_id}' is empty but expected changes."


def assert_diff_empty(diff_result: Any, artifact_id: str) -> None:
    """Assert that a diff result for an artifact shows no changes.

    Args:
        diff_result: The diff result object/dict.
        artifact_id: The artifact that was diffed.
    """
    if isinstance(diff_result, dict):
        changes = diff_result.get("changes", [])
    elif hasattr(diff_result, "changes"):
        changes = diff_result.changes
    elif hasattr(diff_result, "has_changes"):
        assert not diff_result.has_changes, f"Diff for '{artifact_id}' has unexpected changes."
        return
    else:
        changes = str(diff_result).strip()

    assert not changes, f"Diff for artifact '{artifact_id}' has unexpected changes: {changes}"


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def assert_verify_passes(verify_result: Any) -> None:
    """Assert that a verification result indicates all checks passed.

    Args:
        verify_result: The verify result object/dict.
    """
    if isinstance(verify_result, dict):
        failures = verify_result.get("failures", [])
        exit_code = verify_result.get("exit_code", 0)
    elif hasattr(verify_result, "failures"):
        failures = verify_result.failures
        exit_code = getattr(verify_result, "exit_code", 0)
    else:
        raise TypeError(f"Unknown verify_result type: {type(verify_result)}")

    assert exit_code == 0, f"Verify returned exit code {exit_code}, expected 0 (clean). Failures: {failures}"
    assert not failures, f"Verify has failures: {failures}"


def assert_verify_fails(verify_result: Any, expected_check: str, expected_count: int) -> None:
    """Assert that verification fails with a specific check and count.

    Args:
        verify_result: The verify result object/dict.
        expected_check: The check name that should have failed (e.g., "merge_integrity").
        expected_count: The expected number of failures for that check.
    """
    if isinstance(verify_result, dict):
        failures = verify_result.get("failures", [])
        exit_code = verify_result.get("exit_code", 0)
    elif hasattr(verify_result, "failures"):
        failures = verify_result.failures
        exit_code = getattr(verify_result, "exit_code", 0)
    else:
        raise TypeError(f"Unknown verify_result type: {type(verify_result)}")

    assert exit_code != 0, (
        f"Verify returned exit code 0 but expected failure. Expected check '{expected_check}' to fail."
    )

    check_failures = [f for f in failures if f.get("check") == expected_check]
    assert len(check_failures) == expected_count, (
        f"Expected {expected_count} failures for check '{expected_check}', "
        f"got {len(check_failures)}. All failures: {failures}"
    )


# ---------------------------------------------------------------------------
# Source sharing and step independence
# ---------------------------------------------------------------------------


def assert_steps_share_sources(store, step_a: str, step_b: str) -> None:
    """Assert that two steps reference the same source artifacts.

    Both steps should list the same source artifact IDs in their provenance.

    Args:
        store: ArtifactStore instance.
        step_a: First step name.
        step_b: Second step name.
    """
    artifacts_a = store.list_artifacts(step_a)
    artifacts_b = store.list_artifacts(step_b)

    # Collect all input hashes referenced by each step
    hashes_a = set()
    for a in artifacts_a:
        hashes_a.update(a.input_hashes)

    hashes_b = set()
    for b in artifacts_b:
        hashes_b.update(b.input_hashes)

    assert hashes_a == hashes_b, (
        f"Steps '{step_a}' and '{step_b}' reference different source hashes. "
        f"Only in {step_a}: {hashes_a - hashes_b}. "
        f"Only in {step_b}: {hashes_b - hashes_a}."
    )


def assert_steps_have_distinct_artifacts(store, step_a: str, step_b: str) -> None:
    """Assert that two steps have no overlapping artifact IDs.

    Args:
        store: ArtifactStore instance.
        step_a: First step name.
        step_b: Second step name.
    """
    ids_a = {a.artifact_id for a in store.list_artifacts(step_a)}
    ids_b = {a.artifact_id for a in store.list_artifacts(step_b)}
    overlap = ids_a & ids_b
    assert not overlap, f"Steps '{step_a}' and '{step_b}' share artifact IDs: {overlap}. Expected distinct artifacts."


# ---------------------------------------------------------------------------
# LLM call and cache counting
# ---------------------------------------------------------------------------


def count_llm_calls(run_log: dict) -> int:
    """Count total LLM calls across all steps in a run log.

    Args:
        run_log: Structured run log dict.

    Returns:
        Total number of LLM calls.
    """
    total = 0
    for step_name, step_data in run_log.get("steps", {}).items():
        total += step_data.get("llm_calls", 0)
    return total


def count_llm_calls_for_step(run_log: dict, step_name: str) -> int:
    """Count LLM calls for a specific step.

    Args:
        run_log: Structured run log dict.
        step_name: The step to count.

    Returns:
        Number of LLM calls for that step.
    """
    step_data = _get_step_log(run_log, step_name)
    return step_data.get("llm_calls", 0)


def count_cache_hits(run_log: dict) -> int:
    """Count total cache hits across all steps in a run log.

    Args:
        run_log: Structured run log dict.

    Returns:
        Total number of cache hits.
    """
    total = 0
    for step_name, step_data in run_log.get("steps", {}).items():
        total += step_data.get("cache_hits", 0)
    return total


def count_cache_hits_for_step(run_log: dict, step_name: str) -> int:
    """Count cache hits for a specific step.

    Args:
        run_log: Structured run log dict.
        step_name: The step to count.

    Returns:
        Number of cache hits for that step.
    """
    step_data = _get_step_log(run_log, step_name)
    return step_data.get("cache_hits", 0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_step_log(run_log: dict, step_name: str) -> dict:
    """Extract per-step data from a run log.

    Run log expected structure:
    {
        "steps": {
            "episodes": {
                "llm_calls": 30,
                "cache_hits": 25,
                "rebuilt_ids": ["ep-conv001", "ep-conv002"],
                "cached_ids": ["ep-conv003", ...],
                "time_seconds": 12.5,
                "tokens_used": 15000,
            },
            ...
        },
        "total_llm_calls": 50,
        "total_cache_hits": 40,
        "total_time": 45.3,
        "total_tokens": 50000,
        "total_cost_estimate": 0.47,
    }
    """
    steps = run_log.get("steps", {})
    if step_name not in steps:
        available = list(steps.keys())
        raise KeyError(f"Step '{step_name}' not found in run log. Available steps: {available}")
    return steps[step_name]
