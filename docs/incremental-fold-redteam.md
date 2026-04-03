<!-- updated: 2026-04-02 -->
<!-- source: codex-cli red-team review of incremental-fold-design.md -->
# Incremental Fold — Red-Team Review

## Critical

### 1. `accept_existing=True` silently drops new fold outputs

A fold artifact like `weekly` already exists → `_save_artifact()` treats the label
as "existing" and returns the cached old artifact instead of the newly computed one.
The new input is absent from the snapshot; every downstream consumer sees stale state.

`accept_existing` is already unsafe for stable-label N:1 outputs. The design does
not address this.

**Refs:** runner.py L207, L240; fold_synthesis.py L144

**Fix required:** `_save_artifact` must always write for N:1 transforms when the
output content has changed, regardless of `accept_existing` mode. Alternatively,
N:1 transforms should be excluded from the `accept_existing` fast path entirely.

### 2. Input removal/replacement not detected

Build 1 folds `[A,B,C]`. Later `B` is edited (new `artifact_id`) or removed. The
pseudocode only computes `new_inputs = [a for a in sorted_inputs if a.artifact_id
not in seen]` — it never verifies that `seen_input_ids` is still a subset of current
inputs. An edited `B` gets folded on top of an accumulator that already includes old
`B`. A removed `B` returns the previous artifact unchanged.

**Fix required:** Before resuming, verify `seen_input_ids ⊆ current_input_ids`. If
not, fall back to full recompute. This is the correct and safe behavior.

```python
current_ids = set(a.artifact_id for a in sorted_inputs)
seen = set(checkpoint["seen_input_ids"])
if not seen.issubset(current_ids):
    # Inputs removed or replaced — full recompute
    new_inputs = sorted_inputs
    accumulated = self.initial
    start_step = 0
```

### 3. Duplicate-content inputs collapse

Two episodes with identical content share the same `artifact_id` (content-hash).
Both `_layer_fully_cached()` and checkpoint `seen = set(...)` collapse them, so the
second one disappears.

**Fix required:** Use `label` (which is unique per artifact) as the tracking key
for seen inputs instead of `artifact_id`. Or use `(label, artifact_id)` tuples.

```python
# Instead of:
seen = set(checkpoint["seen_input_ids"])
# Use:
seen = set(checkpoint["seen_input_labels"])
new_inputs = [a for a in sorted_inputs if a.label not in seen]
```

This also fixes the identity problem for label/metadata changes (issue 4).

## High

### 4. Label/metadata changes invisible to cache

An upstream artifact keeps the same content but its `label` changes or `date`
metadata changes (affecting `sort_by`). Cache keys on `artifact_id` only and reuses
stale fold output. The fold prompt uses `{label}` and ordering can use metadata.

**Fix required:** Track by label, not by artifact_id (see fix for issue 3). This
handles label changes. For metadata-dependent sort order changes, the sort order
validation (issue 5) catches the reordering.

### 5. Sort-order violation: warning-only is wrong

Design recommends logging a warning for sort-order violations and proceeding. If a
late-arriving episode sorts before existing inputs, it gets appended at the end
instead of the correct position. The artifact is wrong by construction.

**Fix required:** If new inputs interleave with seen inputs (i.e., sort position
violates append-only), fall back to full recompute. No warning-and-proceed path.

```python
# After computing new_inputs:
if new_inputs:
    seen_labels = set(checkpoint["seen_input_labels"])
    last_seen_idx = max(
        (i for i, a in enumerate(sorted_inputs) if a.label in seen_labels),
        default=-1
    )
    first_new_idx = min(
        i for i, a in enumerate(sorted_inputs) if a.label not in seen_labels
    )
    if first_new_idx < last_seen_idx:
        # New inputs interleave — full recompute
        new_inputs = sorted_inputs
        accumulated = self.initial
        start_step = 0
```

### 6. Full-refold at scale (operational)

Current implementation re-folds entire corpus on every build. This is the motivating
problem. The design addresses it conceptually but it is not yet implemented.

**Status:** Will be fixed by implementing the design (with the corrections above).

## Medium

### 7. Crash recovery doesn't work as designed

`SnapshotArtifactCache` loads the committed snapshot first and skips same-label
checkpoint artifacts. After a crash, `store.load_artifact(label)` returns stale
committed state, not the recovered checkpoint.

**Fix options:**
- (a) Store fold checkpoint in a separate side-channel (not in artifact metadata)
- (b) Ensure checkpoint layer data takes precedence over committed snapshot for
  same-label artifacts in the cache loading order
- (c) Accept that crash recovery falls back to full recompute (simple, correct)

Recommend (c) for v0. Crash recovery for incremental fold is a nice-to-have, not
a correctness requirement. A full recompute after a crash is safe.

### 8. No integrity validation on checkpoint read

`accumulated`, `artifact.content`, `artifact.input_ids`, and checkpoint
`seen_input_ids` can drift due to bugs or manual repair. Resume trusts blindly.

**Fix required:** On checkpoint read, verify:
```python
if checkpoint["accumulated"] != previous.content:
    # Checkpoint inconsistent — full recompute
    ...
```

## Unenforced Assumptions

| Assumption | Stated? | Enforced? | Fix |
|---|---|---|---|
| Input removed → full recompute | Yes | No | Add subset check |
| New inputs sort after existing | Yes | No | Add interleave check, fall back to full recompute |
| `accept_existing` works for folds | Yes (test plan) | No | Exclude N:1 from accept_existing fast path |
| Crash recovery preserves checkpoint | Yes (test plan) | No | Drop this claim for v0, accept full recompute |
| `artifact_id` is sufficient identity | Implicit | No | Track by label instead |
