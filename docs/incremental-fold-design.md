<!-- updated: 2026-04-02 -->
# Incremental Fold: Design and Verification Plan

## Problem

FoldSynthesis recomputes from scratch whenever inputs change. On each build it:

1. Receives ALL inputs (not just new ones)
2. Starts from `self.initial`
3. Folds every input sequentially via LLM calls
4. Produces a new output artifact

For a weekly rollup over 2,273+ episodes, this means re-folding the entire corpus
on every build — even if only 3 new sessions were ingested. Cost and latency scale
with corpus size, not with delta.

## Goal

Make FoldSynthesis resume from its previous output. When only new inputs have
arrived and the transform config hasn't changed, fold only the delta.

This becomes the default behavior for FoldSynthesis — not a separate class.

## Current State (what exists)

| Infrastructure | Status | Location |
|---|---|---|
| Artifact metadata (arbitrary dict) | Exists | `Artifact.metadata`, persisted in snapshot |
| Previous artifact retrieval | Exists | `store.load_artifact(label)` via SnapshotArtifactCache |
| Transform fingerprinting | Exists | `compute_fingerprint()` — includes prompt, initial, sort_by, metadata_fn |
| Build fingerprinting | Exists | `compute_build_fingerprint(transform_fp, input_ids)` |
| Per-layer checkpoints | Exists | `.synix/checkpoints/{run_id}/{layer_name}.json` |
| Input IDs on artifacts | Exists | `Artifact.input_ids` — list of input artifact_id hashes |
| Crash recovery from checkpoints | Exists | `SnapshotArtifactCache._load_checkpoints()` |

**What's missing:** FoldSynthesis doesn't persist its accumulated state or track
which inputs it has already processed. The runner doesn't expose previous artifacts
to transforms.

## Design

### Checkpoint Structure

After completing a fold, persist in `artifact.metadata["_fold_checkpoint"]`:

```python
{
    "accumulated": str,              # the final accumulated value (= artifact content)
    "seen_input_labels": list[str],  # labels of all inputs processed (ordered)
    "transform_fingerprint": dict,   # fingerprint at time of fold (for validation)
}
```

The `_` prefix signals this is engine-internal metadata, not user-facing.

**Why `labels` not `artifact_ids`:** `artifact_id` is a content hash. Two artifacts
with identical content share the same hash, causing the checkpoint to collapse them.
Labels are unique per artifact and also capture identity changes (renames, metadata
edits) that content hashes miss.

### Resume Decision Tree

On each build, after loading the previous artifact and its checkpoint:

```
Has checkpoint?
  ├─ NO → full compute from self.initial
  └─ YES
      │
      Transform fingerprint matches?
      ├─ NO → full recompute (prompt/config/model changed)
      └─ YES
          │
          Checkpoint integrity valid? (accumulated == previous.content)
          ├─ NO → full recompute (corrupt/repaired checkpoint)
          └─ YES
              │
              All seen_input_labels still present in current inputs?
              ├─ NO → full recompute (input removed or replaced)
              └─ YES
                  │
                  New inputs exist?
                  ├─ NO → return previous artifact unchanged (0 LLM calls)
                  └─ YES
                      │
                      New inputs all sort AFTER seen inputs?
                      ├─ NO → full recompute (interleave detected)
                      └─ YES → incremental fold (Δ LLM calls only)
```

Every branch that can't guarantee correctness falls back to full recompute.
There is no warning-and-proceed path.

### Modified Execute Flow

```python
def execute(self, inputs: list[Artifact], ctx: TransformContext) -> list[Artifact]:
    ctx = self.get_context(ctx)
    client = _get_llm_client(ctx)
    model_config = ctx.llm_config
    prompt_id = self._make_prompt_id()

    sorted_inputs = self._sort_inputs(inputs)
    transform_fp = self.compute_fingerprint(ctx.to_dict() if hasattr(ctx, 'to_dict') else ctx)

    # --- Checkpoint resume logic ---
    previous = ctx.get("_previous_artifact")
    resume = self._try_resume(previous, sorted_inputs, transform_fp)

    if resume is not None:
        new_inputs, accumulated, start_step = resume
        if new_inputs is None:
            # No new inputs — return previous artifact unchanged
            return [previous]
    else:
        # No valid checkpoint — full compute
        new_inputs = sorted_inputs
        accumulated = self.initial
        start_step = 0

    total = len(sorted_inputs)

    # --- Main fold loop (over new_inputs only) ---
    for step, inp in enumerate(new_inputs, start_step + 1):
        rendered = render_template(
            self.prompt,
            accumulated=accumulated,
            artifact=inp.content,
            label=inp.label,
            step=str(step),
            total=str(total),
        )
        response = _logged_complete(
            client, ctx,
            messages=[{"role": "user", "content": rendered}],
            artifact_desc=f"{self.name} step {step}/{total}",
        )
        accumulated = response.content

    # --- Persist checkpoint ---
    all_input_labels = [a.label for a in sorted_inputs]
    output_metadata = {"input_count": len(inputs)}
    if self.metadata_fn is not None:
        output_metadata.update(self.metadata_fn(inputs))
    output_metadata["_fold_checkpoint"] = {
        "accumulated": accumulated,
        "seen_input_labels": all_input_labels,
        "transform_fingerprint": transform_fp.to_dict(),
    }

    return [Artifact(
        label=self.label_value,
        artifact_type=self.artifact_type,
        content=accumulated,
        input_ids=[a.artifact_id for a in inputs],
        prompt_id=prompt_id,
        model_config=model_config,
        metadata=output_metadata,
    )]


def _try_resume(
    self,
    previous: Artifact | None,
    sorted_inputs: list[Artifact],
    transform_fp,
) -> tuple[list[Artifact] | None, str, int] | None:
    """Attempt checkpoint resume. Returns (new_inputs, accumulated, start_step)
    or None if full recompute is needed. new_inputs=None means no change."""

    if previous is None:
        return None

    checkpoint = previous.metadata.get("_fold_checkpoint")
    if checkpoint is None:
        return None

    # 1. Transform fingerprint must match
    stored_fp = Fingerprint.from_dict(checkpoint.get("transform_fingerprint"))
    if stored_fp is None or not transform_fp.matches(stored_fp):
        logger.info("%s: transform changed, full recompute", self.name)
        return None

    # 2. Checkpoint integrity: accumulated must equal artifact content
    if checkpoint.get("accumulated") != previous.content:
        logger.warning("%s: checkpoint inconsistent with artifact content, full recompute", self.name)
        return None

    # 3. All previously-seen inputs must still be present
    seen_labels = checkpoint.get("seen_input_labels", [])
    seen_set = set(seen_labels)
    current_labels = [a.label for a in sorted_inputs]
    current_set = set(current_labels)

    if not seen_set.issubset(current_set):
        removed = seen_set - current_set
        logger.info("%s: %d inputs removed (%s...), full recompute",
                    self.name, len(removed), list(removed)[:3])
        return None

    # 4. Identify new inputs
    new_inputs = [a for a in sorted_inputs if a.label not in seen_set]

    if not new_inputs:
        return (None, previous.content, len(seen_labels))  # no change

    # 5. New inputs must all sort after seen inputs (no interleave)
    last_seen_idx = max(
        (i for i, a in enumerate(sorted_inputs) if a.label in seen_set),
        default=-1,
    )
    first_new_idx = min(
        i for i, a in enumerate(sorted_inputs) if a.label not in seen_set
    )
    if first_new_idx <= last_seen_idx:
        logger.info("%s: new inputs interleave with seen inputs, full recompute", self.name)
        return None

    # All checks passed — safe to resume
    logger.info("%s: resuming from checkpoint, %d new inputs (of %d total)",
                self.name, len(new_inputs), len(sorted_inputs))
    return (new_inputs, checkpoint["accumulated"], len(seen_labels))
```

### Runner Changes

Two changes to `runner.py`:

#### 1. Inject previous artifact into transform context

```python
# In the transform execution block, after gathering inputs and building context:
if isinstance(layer, Transform) and hasattr(layer, 'label_value'):
    previous_artifact = store.load_artifact(layer.label_value)
    if previous_artifact is not None:
        transform_ctx = transform_ctx.with_updates({
            "_previous_artifact": previous_artifact,
        })
```

~5 lines. The `_previous_artifact` key is only set for transforms that declare
a `label_value` (fold-style N:1 transforms).

#### 2. Exclude N:1 stable-label transforms from `accept_existing` skip

The current `_save_artifact` logic with `accept_existing=True` checks if an artifact
with the same label already exists and skips writing if it does. This is correct for
1:1 transforms (same inputs → same outputs) but wrong for N:1 transforms where the
output changes as inputs accumulate.

```python
# In _save_artifact, modify the accept_existing path:
if _accept_existing:
    existing = store.load_artifact(artifact.label)
    # N:1 transforms always write — their output changes with each new input
    is_n_to_1 = hasattr(_layer, 'label_value')
    rebuild = existing is None or is_n_to_1
```

~3 lines changed.

### What Invalidates the Checkpoint

| Change | Behavior | Enforced by |
|---|---|---|
| New inputs arrive (append) | Resume, fold only delta | `_try_resume` new_inputs detection |
| New inputs interleave | Full recompute | `_try_resume` interleave check |
| Prompt changes | Full recompute | Transform fingerprint mismatch |
| Initial value changes | Full recompute | Transform fingerprint mismatch |
| sort_by changes | Full recompute | Transform fingerprint mismatch |
| metadata_fn changes | Full recompute | Transform fingerprint mismatch |
| Model changes | Full recompute | Transform fingerprint mismatch |
| Input removed | Full recompute | `_try_resume` subset check |
| Input replaced (same label, new content) | Full recompute | Label present but sort position/content changed triggers interleave or subset failure |
| Checkpoint corrupted | Full recompute | `_try_resume` integrity check |
| Crash mid-build | Full recompute | No checkpoint in committed snapshot |

Every case that can't guarantee correctness falls back to full recompute.

## Files Changed

| File | Change | Lines |
|---|---|---|
| `src/synix/ext/fold_synthesis.py` | Add `_try_resume()`, modify `execute()` to use it, checkpoint in output metadata | ~60 lines added/modified |
| `src/synix/build/runner.py` | Inject `_previous_artifact` for N:1 transforms; fix `accept_existing` for stable-label N:1 | ~10 lines added/modified |

No changes to: fingerprint.py, snapshots.py, dag.py, models.py, store, or any
other transform type.

## Verification Plan

### Unit Tests (fold_synthesis.py)

1. **First build — no checkpoint exists**
   - Inputs: [A, B, C], no previous artifact
   - Expect: folds all 3, output has `_fold_checkpoint` with all 3 labels
   - Verify: 3 LLM calls made

2. **Incremental build — new inputs only**
   - Inputs: [A, B, C, D, E], previous artifact has checkpoint with [A, B, C]
   - Expect: folds only D and E, starting from previous accumulated value
   - Verify: 2 LLM calls made (not 5)
   - Verify: output content reflects D and E folded onto previous state
   - Verify: output `_fold_checkpoint.seen_input_labels` contains all 5

3. **No new inputs — returns previous artifact unchanged**
   - Inputs: [A, B, C], previous artifact has checkpoint with [A, B, C]
   - Expect: returns previous artifact unchanged (same object)
   - Verify: 0 LLM calls made

4. **Prompt changes — full recompute**
   - Inputs: [A, B, C], previous artifact has checkpoint with [A, B, C] but
     different transform fingerprint
   - Expect: folds all 3 from initial
   - Verify: 3 LLM calls made

5. **Input removed — full recompute**
   - Inputs: [A, C], previous artifact has checkpoint with [A, B, C]
   - Expect: folds A and C from initial (B gone, checkpoint invalid)
   - Verify: 2 LLM calls made from initial, not from checkpoint
   - Verify: seen_input_labels subset check triggered

6. **Input replaced — full recompute**
   - Input B has same label but different content (new artifact_id)
   - Previous checkpoint has [A, B, C] labels
   - Expect: all labels still present, but sort position or content changed
   - Verify: full recompute (interleave check or content-change detection)

7. **Empty inputs — returns initial**
   - Inputs: [], no previous artifact
   - Expect: returns artifact with `self.initial` as content
   - Verify: 0 LLM calls made

8. **Checkpoint integrity failure — full recompute**
   - Previous artifact where `content != checkpoint["accumulated"]`
   - Expect: full recompute
   - Verify: warning logged about inconsistency

9. **Duplicate-content inputs handled correctly**
   - Inputs: [A, B] where A and B have identical content (same artifact_id)
     but different labels
   - Expect: both are folded (tracked by label, not artifact_id)
   - Verify: 2 LLM calls made, both labels in seen_input_labels

10. **Checkpoint accumulated value matches artifact content**
    - After any build (first or incremental), verify:
      `artifact.content == artifact.metadata["_fold_checkpoint"]["accumulated"]`

### Sort Order Tests

11. **New inputs sort after seen — incremental**
    - Sorted: [A, B, C, D, E] where [A, B, C] are seen
    - Expect: incremental fold of [D, E]
    - Verify: 2 LLM calls

12. **New inputs interleave with seen — full recompute**
    - Sorted: [A, D, B, E, C] where [A, B, C] are seen, [D, E] are new
    - D sorts between A and B → interleave detected
    - Expect: full recompute from initial
    - Verify: 5 LLM calls

13. **New input sorts before all seen — full recompute**
    - Sorted: [Z, A, B, C] where [A, B, C] are seen, Z is new but sorts first
    - Expect: full recompute
    - Verify: 4 LLM calls

### Integration Tests (runner.py)

14. **Runner injects _previous_artifact**
    - Define a pipeline with FoldSynthesis
    - Run build twice with additional inputs on second run
    - Verify: second build's transform context contains `_previous_artifact`
    - Verify: second build produces correct incremental result

15. **Full pipeline round-trip**
    - Source → Episodes (Map) → Weekly (Fold) → Core (Fold)
    - Build with 5 sources
    - Add 2 more sources, build again
    - Verify: Episodes builds 2 new (Map is already incremental)
    - Verify: Weekly folds only 2 new episodes
    - Verify: Core folds only the updated Weekly

16. **accept_existing mode — N:1 always writes**
    - Run build with `accept_existing=True`
    - Add new inputs, build again with `accept_existing=True`
    - Verify: fold output is updated (not stale)
    - Verify: new inputs are reflected in the output

17. **Concurrent fold layers**
    - Two independent FoldSynthesis layers in same pipeline
    - Verify: each gets its own `_previous_artifact` and checkpoints independently

### Edge Cases

18. **Very large checkpoint**
    - Fold with accumulated value > 100KB
    - Verify: checkpoint round-trips correctly through metadata serialization

19. **Crash mid-build — clean fallback**
    - Build with checkpoint, simulate interruption after fold completes but
      before snapshot commit
    - Build again with new inputs
    - Verify: falls back to full recompute (committed snapshot has old state)
    - Verify: produces correct output

20. **Rapid sequential builds**
    - Build 1: fold [A, B, C]
    - Build 2: fold [A, B, C, D] (incremental)
    - Build 3: fold [A, B, C, D, E] (incremental from build 2's checkpoint)
    - Verify: build 3 folds only [E], not [D, E]

## Non-Goals

- Checkpointing mid-fold (within the loop). If a fold of 50 new inputs fails at
  input 30, the entire fold retries on next build. Mid-fold checkpointing is a
  future optimization.
- Exposing checkpoint data to users. The `_fold_checkpoint` key is engine-internal.
- Changing any other transform type. MapSynthesis, GroupSynthesis, and
  ReduceSynthesis are unaffected.
- Crash recovery for incremental fold. After a crash, the system falls back to
  full recompute, which is safe and correct. Optimizing crash recovery is future work.

## Red-Team Review

See `docs/incremental-fold-redteam.md` for the full adversarial review. All critical
and high findings have been addressed in this revision:

| Finding | Severity | Resolution |
|---|---|---|
| `accept_existing` drops N:1 outputs | Critical | N:1 transforms excluded from skip path |
| Input removal not detected | Critical | Subset check: `seen_labels ⊆ current_labels` |
| Duplicate-content inputs collapse | Critical | Track by label, not artifact_id |
| Label/metadata changes invisible | High | Track by label — captures identity changes |
| Sort-order violation warning-only | High | Interleave check → full recompute, no warning path |
| Full-refold at scale | High | This design (the whole point) |
| Crash recovery mismatch | Medium | Dropped crash recovery claim; falls back to full recompute |
| No checkpoint integrity validation | Medium | Integrity check: `accumulated == previous.content` |
