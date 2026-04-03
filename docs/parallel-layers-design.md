<!-- updated: 2026-04-03 -->
# Parallel Layer Execution

## Problem

The build runner processes layers sequentially via `for layer in build_order`.
Layers at the same DAG level are independent — their dependencies are already
computed — but they run one after another.

For a pipeline with 5 FoldSynthesis layers at level 2, each taking ~45 minutes
on first build, this means 3.75 hours instead of 45 minutes.

## Current Architecture

```
build_order = resolve_build_order(pipeline)  # topological sort
for layer in build_order:                    # sequential
    # ... execute layer ...
    layer_artifacts[layer.name] = results
```

Within a single layer, work units ARE parallelized via ThreadPoolExecutor
(the `_execute_transform_concurrent` path). But across layers, no parallelism.

## Design

Group layers by level. For each level, run all layers in that level
concurrently.

```
level_groups = group_by(build_order, key=lambda l: l._level)

for level in sorted(level_groups):
    group = level_groups[level]
    if len(group) == 1:
        _run_layer(group[0])            # single layer, run inline
    else:
        with ThreadPoolExecutor(max_workers=len(group)) as pool:
            futures = {pool.submit(_run_layer, layer): layer for layer in group}
            for future in as_completed(futures):
                # collect results, write checkpoints
```

This is safe because:
- Topological sort guarantees all dependencies at lower levels are complete
- `layer_artifacts` dict is written per-layer-name (no cross-layer writes)
- `snapshot_txn.record_artifact()` needs a lock (it writes to shared state)
- `store` (SnapshotArtifactCache) is read-only during execution

## Changes Required

### 1. Extract `_run_single_layer()` function

The body of the current `for layer in build_order:` loop (~230 lines,
runner.py:163-395) needs to become a callable function. It currently
accesses these shared variables:

| Variable | Access | Thread-safe? |
|----------|--------|-------------|
| `layer_artifacts` | Write own key | Yes (dict key per layer) |
| `snapshot_txn` | `record_artifact()`, `record_projection()` | **No — needs lock** |
| `store` | Read-only (`load_artifact`, `list_artifacts`) | Yes |
| `result` | Increment `built`, `cached`, `skipped` | **No — needs lock or atomic** |
| `slogger` | Logging calls | Partially (has internal lock for file writes) |
| `work_dir` | Write to subdirectories | Yes (per-layer paths) |

### 2. Add threading lock for `snapshot_txn`

`BuildTransaction.record_artifact()` writes to `artifact_oids` and
`parent_labels_map` dicts. Under concurrent layers, two threads could
write simultaneously. Add a `threading.Lock`:

```python
snapshot_lock = threading.Lock()

def _record_snapshot_artifact_safe(...):
    with snapshot_lock:
        snapshot_txn.record_artifact(...)
```

### 3. Accumulate stats safely

`result.built += stats.built` is not atomic. Either:
- Accumulate per-layer stats in the return value, sum after join
- Or use a lock (simpler)

Recommend: return `LayerStats` from `_run_single_layer()`, accumulate
after `as_completed`.

### 4. Search surface materialization

`_materialize_layer_search_surfaces()` runs after each layer. For
concurrent layers, surfaces that depend on multiple layers should only
materialize when ALL their source layers are complete. The current code
already checks `search_surface_ready(surface, available_names)` — this
should work correctly if `layer_artifacts` is updated before the check.

## Files Changed

| File | Change |
|------|--------|
| `src/synix/build/runner.py` | Extract `_run_single_layer()`, add level grouping loop, add snapshot lock |
| `src/synix/build/dag.py` | Optional: add `group_by_level()` helper |

No changes to: transforms, fingerprinting, snapshots, SDK, CLI.

## Verification

1. **Existing tests pass** — all 1318 unit tests, 28 e2e server tests
2. **Same build output** — run a pipeline with parallel layers, verify
   artifact content and provenance match sequential build
3. **Concurrent fold layers** — pipeline with 3+ FoldSynthesis at same
   level, verify all run simultaneously (check timestamps in logs)
4. **Single-layer levels** — levels with one layer still run inline
   (no thread overhead)
5. **Snapshot integrity** — verify snapshot transaction is consistent
   after concurrent layer writes
6. **Error handling** — one layer fails, others at same level should
   still complete (or all fail, depending on policy)
7. **Search surface timing** — surface depending on 2 concurrent layers
   only materializes after both complete
