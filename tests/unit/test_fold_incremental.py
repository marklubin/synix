"""Tests for FoldSynthesis incremental checkpoint resume.

Covers the full decision tree from docs/incremental-fold-design.md:
- First build (no checkpoint)
- Incremental build (new inputs only)
- No new inputs (return previous unchanged)
- Transform changed (full recompute)
- Input removed (full recompute)
- Input replaced (full recompute)
- Checkpoint integrity failure (full recompute)
- Duplicate-content inputs
- Sort order interleave (full recompute)
- Rapid sequential builds
"""

from __future__ import annotations

import hashlib

from synix import Artifact
from synix.transforms import FoldSynthesis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_artifact(label: str, content: str = "content", **metadata) -> Artifact:
    return Artifact(
        label=label,
        artifact_type="test",
        content=content,
        metadata=metadata,
    )


def _make_fold(**kwargs) -> FoldSynthesis:
    defaults = {
        "name": "fold",
        "prompt": "Current: {accumulated}\nNew: {artifact}",
        "initial": "Empty.",
        "label": "out",
        "artifact_type": "summary",
    }
    defaults.update(kwargs)
    return FoldSynthesis(**defaults)


def _make_checkpoint_artifact(
    fold: FoldSynthesis,
    seen_inputs: list[Artifact],
    accumulated: str,
    config: dict | None = None,
) -> Artifact:
    """Build a fake previous artifact with a valid checkpoint.

    ``seen_inputs`` should be the actual Artifact objects so we can derive
    labels and artifact_ids in the correct sorted order.
    """
    fp = fold.compute_fingerprint(config or {})
    sorted_seen = fold._sort_inputs(seen_inputs)
    seen_entries = [
        {"label": a.label, "artifact_id": a.artifact_id}
        for a in sorted_seen
    ]
    content_hash = hashlib.sha256(accumulated.encode()).hexdigest()[:16]
    return Artifact(
        label=fold.label_value,
        artifact_type=fold.artifact_type,
        content=accumulated,
        input_ids=[],
        metadata={
            "_fold_checkpoint": {
                "version": 1,
                "content_hash": content_hash,
                "seen_inputs": seen_entries,
                "transform_fingerprint": fp.to_dict(),
            },
            "input_count": len(seen_entries),
        },
    )


def _seen_labels(checkpoint: dict) -> list[str]:
    """Extract labels from checkpoint in either v0 or v1 format."""
    entries = checkpoint.get("seen_inputs")
    if entries is not None:
        return [e["label"] for e in entries]
    return checkpoint.get("seen_input_labels", [])


def _accumulated(checkpoint: dict, artifact: Artifact) -> str:
    """Get accumulated value — in v1, it's the artifact content itself."""
    return artifact.content


def _make_ctx_with_previous(previous: Artifact | None = None) -> dict:
    """Build a minimal transform context dict with optional previous artifact."""
    ctx: dict = {"llm_config": {}}
    if previous is not None:
        ctx["_previous_artifact"] = previous
    return ctx


# ---------------------------------------------------------------------------
# Test 1: First build — no checkpoint
# ---------------------------------------------------------------------------


class TestFirstBuild:
    def test_folds_all_inputs(self, mock_llm):
        inputs = [_make_artifact(f"ep-{i}", f"event {i}") for i in range(3)]
        t = _make_fold()
        results = t.execute(inputs, _make_ctx_with_previous(None))

        assert len(results) == 1
        assert len(mock_llm) == 3
        assert results[0].label == "out"

    def test_checkpoint_persisted(self, mock_llm):
        inputs = [_make_artifact(f"ep-{i}", f"event {i}") for i in range(3)]
        t = _make_fold()
        results = t.execute(inputs, _make_ctx_with_previous(None))

        checkpoint = results[0].metadata.get("_fold_checkpoint")
        assert checkpoint is not None
        assert _accumulated(checkpoint, results[0]) == results[0].content
        assert len(_seen_labels(checkpoint)) == 3
        assert "transform_fingerprint" in checkpoint

    def test_seen_labels_match_sorted_inputs(self, mock_llm):
        inputs = [_make_artifact(f"ep-{i}", f"event {i}") for i in range(3)]
        t = _make_fold()
        results = t.execute(inputs, _make_ctx_with_previous(None))

        checkpoint = results[0].metadata["_fold_checkpoint"]
        sorted_labels = [a.label for a in t._sort_inputs(inputs)]
        assert _seen_labels(checkpoint) == sorted_labels


# ---------------------------------------------------------------------------
# Test 2: Incremental build — new inputs only
# ---------------------------------------------------------------------------


class TestIncrementalBuild:
    def test_folds_only_new_inputs(self, mock_llm):
        t = _make_fold(sort_by="date")
        seen = [
            _make_artifact("ep-0", "event 0", date="2024-01"),
            _make_artifact("ep-1", "event 1", date="2024-02"),
            _make_artifact("ep-2", "event 2", date="2024-03"),
        ]
        prev = _make_checkpoint_artifact(t, seen, "previous state")

        inputs = seen + [
            _make_artifact("ep-3", "event 3", date="2024-04"),
            _make_artifact("ep-4", "event 4", date="2024-05"),
        ]
        results = t.execute(inputs, _make_ctx_with_previous(prev))

        assert len(mock_llm) == 2  # only ep-3 and ep-4
        assert len(results) == 1

    def test_starts_from_previous_accumulated(self, mock_llm):
        t = _make_fold(sort_by="date")
        seen = [
            _make_artifact("ep-0", "event 0", date="2024-01"),
            _make_artifact("ep-1", "event 1", date="2024-02"),
        ]
        prev = _make_checkpoint_artifact(t, seen, "PREVIOUS STATE")

        inputs = seen + [
            _make_artifact("ep-2", "event 2", date="2024-03"),
        ]
        results = t.execute(inputs, _make_ctx_with_previous(prev))

        # First LLM call should contain the previous accumulated value
        first_prompt = mock_llm[0]["messages"][0]["content"]
        assert "PREVIOUS STATE" in first_prompt

    def test_checkpoint_updated_with_all_labels(self, mock_llm):
        t = _make_fold(sort_by="date")
        seen = [
            _make_artifact("ep-0", "e0", date="2024-01"),
            _make_artifact("ep-1", "e1", date="2024-02"),
        ]
        prev = _make_checkpoint_artifact(t, seen, "prev")

        inputs = seen + [_make_artifact("ep-2", "e2", date="2024-03")]
        results = t.execute(inputs, _make_ctx_with_previous(prev))

        checkpoint = results[0].metadata["_fold_checkpoint"]
        assert len(_seen_labels(checkpoint)) == 3
        assert "ep-2" in _seen_labels(checkpoint)

    def test_step_numbering_continues(self, mock_llm):
        """Step numbers continue from where the checkpoint left off."""
        t = _make_fold(prompt="Step {step}/{total}: {artifact}", sort_by="date")
        seen = [
            _make_artifact("ep-0", "e0", date="2024-01"),
            _make_artifact("ep-1", "e1", date="2024-02"),
        ]
        prev = _make_checkpoint_artifact(t, seen, "prev")

        inputs = seen + [_make_artifact("ep-2", "e2", date="2024-03")]
        results = t.execute(inputs, _make_ctx_with_previous(prev))

        # Should be step 3/3 (continuing from 2 seen)
        prompt = mock_llm[0]["messages"][0]["content"]
        assert "Step 3/3" in prompt


# ---------------------------------------------------------------------------
# Test 3: No new inputs — returns previous unchanged
# ---------------------------------------------------------------------------


class TestNoNewInputs:
    def test_returns_previous_artifact(self, mock_llm):
        t = _make_fold(sort_by="date")
        seen = [
            _make_artifact("ep-0", "e0", date="2024-01"),
            _make_artifact("ep-1", "e1", date="2024-02"),
        ]
        prev = _make_checkpoint_artifact(t, seen, "unchanged")
        inputs = seen
        results = t.execute(inputs, _make_ctx_with_previous(prev))

        assert len(mock_llm) == 0  # no LLM calls
        assert len(results) == 1
        assert results[0] is prev  # same object returned


# ---------------------------------------------------------------------------
# Test 4: Prompt changes — full recompute
# ---------------------------------------------------------------------------


class TestTransformChanged:
    def test_full_recompute_on_prompt_change(self, mock_llm):
        t = _make_fold(prompt="NEW PROMPT: {accumulated} {artifact}", sort_by="date")
        old_fold = _make_fold(prompt="OLD PROMPT: {accumulated} {artifact}", sort_by="date")
        seen = [
            _make_artifact("ep-0", "e0", date="2024-01"),
            _make_artifact("ep-1", "e1", date="2024-02"),
        ]
        prev = _make_checkpoint_artifact(old_fold, seen, "old state")
        inputs = seen
        results = t.execute(inputs, _make_ctx_with_previous(prev))

        assert len(mock_llm) == 2  # full recompute, not 0
        # First call should start from initial, not checkpoint
        first_prompt = mock_llm[0]["messages"][0]["content"]
        assert "old state" not in first_prompt


# ---------------------------------------------------------------------------
# Test 5: Input removed — full recompute
# ---------------------------------------------------------------------------


class TestInputRemoved:
    def test_full_recompute_when_input_missing(self, mock_llm):
        t = _make_fold(sort_by="date")
        seen = [
            _make_artifact("ep-0", "e0", date="2024-01"),
            _make_artifact("ep-1", "e1", date="2024-02"),
            _make_artifact("ep-2", "e2", date="2024-03"),
        ]
        prev = _make_checkpoint_artifact(t, seen, "prev")

        # ep-1 is gone
        inputs = [
            _make_artifact("ep-0", "e0", date="2024-01"),
            _make_artifact("ep-2", "e2", date="2024-03"),
        ]
        results = t.execute(inputs, _make_ctx_with_previous(prev))

        assert len(mock_llm) == 2  # full recompute of remaining inputs
        # Should start from initial
        first_prompt = mock_llm[0]["messages"][0]["content"]
        assert "Empty." in first_prompt


# ---------------------------------------------------------------------------
# Test 6: Input replaced — full recompute
# ---------------------------------------------------------------------------


class TestInputReplaced:
    def test_full_recompute_when_content_changes(self, mock_llm):
        """Same label, different content → artifact_id changes → full recompute."""
        t = _make_fold(sort_by="date")
        seen = [
            _make_artifact("ep-0", "e0", date="2024-01"),
            _make_artifact("ep-1", "e1", date="2024-02"),
        ]
        prev = _make_checkpoint_artifact(t, seen, "prev")

        inputs = [
            _make_artifact("ep-0", "e0", date="2024-01"),
            _make_artifact("ep-1", "CHANGED CONTENT", date="2024-02"),
            _make_artifact("ep-2", "e2", date="2024-03"),
        ]
        results = t.execute(inputs, _make_ctx_with_previous(prev))

        # ep-1's content changed → different artifact_id → checkpoint detects it
        assert len(mock_llm) == 3  # full recompute
        # Should start from initial, not from checkpoint
        first_prompt = mock_llm[0]["messages"][0]["content"]
        assert "Empty." in first_prompt


# ---------------------------------------------------------------------------
# Test 7: Empty inputs
# ---------------------------------------------------------------------------


class TestEmptyInputs:
    def test_returns_initial_content(self, mock_llm):
        t = _make_fold()
        results = t.execute([], _make_ctx_with_previous(None))

        assert len(mock_llm) == 0
        assert len(results) == 1
        assert results[0].content == "Empty."

    def test_checkpoint_on_empty(self, mock_llm):
        t = _make_fold()
        results = t.execute([], _make_ctx_with_previous(None))

        checkpoint = results[0].metadata["_fold_checkpoint"]
        assert _accumulated(checkpoint, results[0]) == "Empty."
        assert _seen_labels(checkpoint) == []


# ---------------------------------------------------------------------------
# Test 8: Checkpoint integrity failure
# ---------------------------------------------------------------------------


class TestCheckpointIntegrity:
    def test_full_recompute_on_mismatch(self, mock_llm):
        t = _make_fold(sort_by="date")
        fp = t.compute_fingerprint({})

        # Create artifact where content hash doesn't match actual content
        prev = Artifact(
            label="out",
            artifact_type="summary",
            content="artifact content",
            metadata={
                "_fold_checkpoint": {
                    "version": 1,
                    "content_hash": "0000000000000000",  # wrong hash
                    "seen_inputs": [{"label": "ep-0", "artifact_id": "sha256:fake"}],
                    "transform_fingerprint": fp.to_dict(),
                },
            },
        )

        inputs = [
            _make_artifact("ep-0", "e0", date="2024-01"),
            _make_artifact("ep-1", "e1", date="2024-02"),
        ]
        results = t.execute(inputs, _make_ctx_with_previous(prev))

        assert len(mock_llm) == 2  # full recompute


# ---------------------------------------------------------------------------
# Test 9: Duplicate-content inputs
# ---------------------------------------------------------------------------


class TestDuplicateContent:
    def test_both_processed(self, mock_llm):
        """Two artifacts with identical content but different labels are both folded."""
        t = _make_fold(sort_by="date")
        inputs = [
            _make_artifact("ep-a", "identical content", date="2024-01"),
            _make_artifact("ep-b", "identical content", date="2024-02"),
        ]
        results = t.execute(inputs, _make_ctx_with_previous(None))

        assert len(mock_llm) == 2  # both processed
        labels = _seen_labels(results[0].metadata["_fold_checkpoint"])
        assert "ep-a" in labels
        assert "ep-b" in labels

    def test_incremental_with_duplicate_content(self, mock_llm):
        """Checkpoint resume works correctly when some artifacts share content."""
        t = _make_fold(sort_by="date")
        seen = [
            _make_artifact("ep-a", "identical content", date="2024-01"),
            _make_artifact("ep-b", "identical content", date="2024-02"),
        ]
        prev = _make_checkpoint_artifact(t, seen, "prev")

        inputs = seen + [
            _make_artifact("ep-c", "identical content", date="2024-03"),
        ]
        results = t.execute(inputs, _make_ctx_with_previous(prev))

        assert len(mock_llm) == 1  # only ep-c is new


# ---------------------------------------------------------------------------
# Test 10: Checkpoint accumulated matches content
# ---------------------------------------------------------------------------


class TestCheckpointConsistency:
    def test_accumulated_equals_content(self, mock_llm):
        t = _make_fold(sort_by="date")
        inputs = [_make_artifact(f"ep-{i}", f"event {i}", date=f"2024-0{i+1}") for i in range(3)]
        results = t.execute(inputs, _make_ctx_with_previous(None))

        checkpoint = results[0].metadata["_fold_checkpoint"]
        assert _accumulated(checkpoint, results[0]) == results[0].content

    def test_accumulated_equals_content_after_incremental(self, mock_llm):
        t = _make_fold(sort_by="date")
        seen = [_make_artifact("ep-0", "e0", date="2024-01")]
        prev = _make_checkpoint_artifact(t, seen, "prev state")

        inputs = seen + [_make_artifact("ep-1", "e1", date="2024-02")]
        results = t.execute(inputs, _make_ctx_with_previous(prev))

        checkpoint = results[0].metadata["_fold_checkpoint"]
        assert _accumulated(checkpoint, results[0]) == results[0].content


# ---------------------------------------------------------------------------
# Tests 11-13: Sort order
# ---------------------------------------------------------------------------


class TestSortOrder:
    def test_new_inputs_after_seen_incremental(self, mock_llm):
        """New inputs sorting after all seen inputs → incremental."""
        t = _make_fold(sort_by="date")
        seen = [
            _make_artifact("ep-a", "a", date="2024-01"),
            _make_artifact("ep-b", "b", date="2024-02"),
        ]
        prev = _make_checkpoint_artifact(t, seen, "prev")

        inputs = seen + [
            _make_artifact("ep-c", "c", date="2024-03"),
        ]
        results = t.execute(inputs, _make_ctx_with_previous(prev))

        assert len(mock_llm) == 1  # only ep-c

    def test_new_inputs_interleave_full_recompute(self, mock_llm):
        """New input sorting between seen inputs → full recompute."""
        t = _make_fold(sort_by="date")
        seen = [
            _make_artifact("ep-a", "a", date="2024-01"),
            _make_artifact("ep-c", "c", date="2024-03"),
        ]
        prev = _make_checkpoint_artifact(t, seen, "prev")

        inputs = [
            _make_artifact("ep-a", "a", date="2024-01"),
            _make_artifact("ep-b", "b", date="2024-02"),  # interleaves
            _make_artifact("ep-c", "c", date="2024-03"),
        ]
        results = t.execute(inputs, _make_ctx_with_previous(prev))

        assert len(mock_llm) == 3  # full recompute

    def test_new_input_before_all_seen_full_recompute(self, mock_llm):
        """New input sorting before all seen inputs → full recompute."""
        t = _make_fold(sort_by="date")
        seen = [
            _make_artifact("ep-b", "b", date="2024-02"),
            _make_artifact("ep-c", "c", date="2024-03"),
        ]
        prev = _make_checkpoint_artifact(t, seen, "prev")

        inputs = [
            _make_artifact("ep-a", "a", date="2024-01"),  # before all seen
            _make_artifact("ep-b", "b", date="2024-02"),
            _make_artifact("ep-c", "c", date="2024-03"),
        ]
        results = t.execute(inputs, _make_ctx_with_previous(prev))

        assert len(mock_llm) == 3  # full recompute


# ---------------------------------------------------------------------------
# Test 17: Concurrent fold layers
# ---------------------------------------------------------------------------


class TestConcurrentFolds:
    def test_independent_checkpoints(self, mock_llm):
        """Two fold transforms checkpoint independently."""
        t1 = _make_fold(name="fold-a", label="out-a")
        t2 = _make_fold(name="fold-b", label="out-b")

        inputs = [_make_artifact("ep-0", "e0")]

        r1 = t1.execute(inputs, _make_ctx_with_previous(None))
        r2 = t2.execute(inputs, _make_ctx_with_previous(None))

        assert _seen_labels(r1[0].metadata["_fold_checkpoint"]) == ["ep-0"]
        assert _seen_labels(r2[0].metadata["_fold_checkpoint"]) == ["ep-0"]
        assert r1[0].label == "out-a"
        assert r2[0].label == "out-b"


# ---------------------------------------------------------------------------
# Test 18: Large checkpoint
# ---------------------------------------------------------------------------


class TestLargeCheckpoint:
    def test_large_accumulated_roundtrips(self, mock_llm):
        t = _make_fold(sort_by="date")
        large_content = "x" * 200_000
        seen = [_make_artifact("ep-0", "e0", date="2024-01")]
        prev = _make_checkpoint_artifact(t, seen, large_content)

        inputs = seen + [_make_artifact("ep-1", "e1", date="2024-02")]
        results = t.execute(inputs, _make_ctx_with_previous(prev))

        # Should resume from large checkpoint
        assert len(mock_llm) == 1  # only ep-1
        first_prompt = mock_llm[0]["messages"][0]["content"]
        assert large_content in first_prompt


# ---------------------------------------------------------------------------
# Test 20: Rapid sequential builds
# ---------------------------------------------------------------------------


class TestRapidSequential:
    def test_three_builds_incremental(self, mock_llm):
        """Build 1 → Build 2 (incremental) → Build 3 (incremental from build 2)."""
        t = _make_fold(sort_by="date")

        # Build 1: fold [ep-0, ep-1, ep-2]
        inputs1 = [_make_artifact(f"ep-{i}", f"event {i}", date=f"2024-0{i+1}") for i in range(3)]
        r1 = t.execute(inputs1, _make_ctx_with_previous(None))
        assert len(mock_llm) == 3
        mock_llm.clear()

        # Build 2: fold [ep-0..ep-3] — incremental from build 1
        inputs2 = inputs1 + [_make_artifact("ep-3", "event 3", date="2024-04")]
        r2 = t.execute(inputs2, _make_ctx_with_previous(r1[0]))
        assert len(mock_llm) == 1  # only ep-3
        mock_llm.clear()

        # Build 3: fold [ep-0..ep-4] — incremental from build 2
        inputs3 = inputs2 + [_make_artifact("ep-4", "event 4", date="2024-05")]
        r3 = t.execute(inputs3, _make_ctx_with_previous(r2[0]))
        assert len(mock_llm) == 1  # only ep-4

        # Final checkpoint should have all 5 labels
        checkpoint = r3[0].metadata["_fold_checkpoint"]
        assert len(_seen_labels(checkpoint)) == 5


# ---------------------------------------------------------------------------
# Test: No previous artifact in context (backward compat)
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_works_without_previous_in_context(self, mock_llm):
        """FoldSynthesis still works when no _previous_artifact is injected."""
        t = _make_fold()
        inputs = [_make_artifact("ep-0", "e0")]
        results = t.execute(inputs, {"llm_config": {}})

        assert len(mock_llm) == 1
        assert results[0].metadata.get("_fold_checkpoint") is not None

    def test_existing_tests_unaffected(self, mock_llm):
        """Original FoldSynthesis behavior is preserved for plain dict context."""
        inputs = [_make_artifact(f"ep-{i}", f"event {i}") for i in range(3)]
        t = FoldSynthesis(
            "progressive",
            prompt="Current: {accumulated}\nNew: {artifact}",
            initial="Empty.",
            label="progressive",
            artifact_type="progressive",
        )
        results = t.execute(inputs, {"llm_config": {}})

        assert len(results) == 1
        assert len(mock_llm) == 3
        assert results[0].label == "progressive"
        assert results[0].metadata["input_count"] == 3
        # Checkpoint is now always present
        assert "_fold_checkpoint" in results[0].metadata
