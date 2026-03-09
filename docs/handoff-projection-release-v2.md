# Projection Release v2 — Handoff Report

**Date**: 2026-03-08
**Branch**: `mark/projection-release-v2`
**PR**: https://github.com/marklubin/synix/pull/92
**Closes**: #50, #56

---

## Summary

All 14 phases of the projection-release v2 RFC are implemented and committed. The PR removes `build/` as a platform concept, makes `.synix/` the single source of truth, and adds an explicit release lifecycle with adapter contracts.

**Local status**: `uv run release` passes (1660+ tests, 5 demos green).
**CI status**: Failing on one demo (01-chatbot-export-synthesis) due to Rich terminal width golden mismatch. All unit/integration/e2e tests pass in CI. Demos 02-05 pass in CI.

---

## Blocking CI Issue

Demo-01 release output wraps differently on CI's narrow terminal vs local wide terminal. The golden file has a line break baked in that doesn't match CI.

**Golden (generated locally, wide terminal)**:
```
  ● context-doc  flat_file  →  <CASE_DIR>/.synix/releases/<RELEASE_PATH>
(<N> artifacts)
```

**CI (narrow terminal, no break)**:
```
  ● context-doc  flat_file  →  <CASE_DIR>/.synix/releases/<RELEASE_PATH>  (<N> artifacts)
```

**Fix** — add one normalization in `src/synix/cli/demo_commands.py` `_normalize_output()` after the existing cross-line wrapping fixes (~line 388):

```python
# Join release path with artifact count that wrapped to next line
joined = re.sub(r"(<RELEASE_PATH>)\s*\n\s*(\(<N> artifacts\))", r"\1  \2", joined)
```

Then regenerate demo-01 goldens: `uv run synix demo run templates/01-chatbot-export-synthesis --update-goldens` and sync templates: `bash scripts/sync-templates`.

---

## Commit History (15 commits on PR branch)

| Commit | Description |
|--------|-------------|
| `846c0c4` | Phase 1-3: SnapshotView, manifest schema v2, projection recording |
| `02de55e` | Phase 4-5: Single write path via SnapshotArtifactCache |
| `870bcc4` | Phase 6-7: ReleaseClosure + adapter contract |
| `b78fff6` | Phase 8: Release engine + CLI command |
| `55d0b0d` | Phase 9-11: CLI migration (list/show/lineage/search via snapshots+releases) |
| `058c349` | Phase 12: Remove build/ compatibility files |
| `e9593e6` | Migrate remaining ArtifactStore/ProvenanceTracker imports |
| `85f9316` | Phase 13: Add release step to demo cases |
| `256ff16` | Phase 12 cont: Remove build/ as platform concept |
| `878c1a2` | Phase 13 cont: Update all documentation |
| `037b1e3` | Phase 14: E2E tests for release lifecycle |
| `4207088` | Fix: batch runner snapshot commit, validate/fix .synix checks |
| `1250f59` | Fix: PR review — plan ref safety, verify step, scratch cleanup, deque |
| `c7a6917` | Chore: remove validator/fixer flows from demo templates (experimental) |
| `c95fe02` | Test+docs: release atomicity, adapter filtering, migration guide, receipt schema |

---

## Review Findings — All Addressed

| Finding | Resolution | Commit |
|---------|-----------|--------|
| `_save_plan_artifact` clobbers HEAD | Write to `refs/plans/latest` instead | `1250f59` |
| Scratch realization disk leak | `atexit.register(shutil.rmtree, ...)` | `1250f59` |
| BFS `queue.pop(0)` O(n²) | `collections.deque` with `popleft()` | `1250f59` |
| No `verify()` call in release engine | Added after `apply()`, aborts on failure | `1250f59` |
| `released_at` uses snapshot creation time | Uses actual release time via `_now_iso()` | `1250f59` |
| Missing release atomicity tests | 4 tests: apply failure, verify failure, idempotent, no partial receipts | `c95fe02` |
| Missing SynixSearch source filtering tests | 3 tests: filter, non-empty output, empty edge case | `c95fe02` |
| No receipt JSON schema | `docs/receipt-schema.md` | `c95fe02` |
| No migration guide | `docs/migration-v2.md` | `c95fe02` |
| Validators/fixers broken in demos | Removed from templates, marked experimental | `c7a6917` |
| Projection input resolution bug (SynixSearch vs sources) | **Not a bug** — traced end-to-end, chain is correct | N/A |

---

## Known Deferred Items (acceptable for pre-1.0 single-user CLI)

- **Release file locking**: RFC mentions `.lock` file, not implemented. Low risk for single-user CLI.
- **Partial materialization on adapter failure**: If adapter #2 fails, adapter #1's files remain on disk. `.pending.json` preserved for diagnosis. No automatic rollback.
- **`SnapshotArtifactCache.update_from_build`** doesn't refresh `_parent_labels_map` — stale provenance possible in validators. Validators are experimental.
- **`apply_fix` with SnapshotArtifactCache**: Store is read-only, fix silently vanishes. Fixers are experimental.

---

## New Files Created

### Source
- `src/synix/build/snapshot_view.py` — ref-resolved read API
- `src/synix/build/release.py` — ReleaseClosure, ProjectionDeclaration, ResolvedArtifact
- `src/synix/build/release_engine.py` — execute_release orchestration
- `src/synix/build/adapters.py` — ProjectionAdapter ABC, ReleasePlan, AdapterReceipt, registry
- `src/synix/build/flat_file_adapter.py` — FlatFile adapter
- `src/synix/search/adapter.py` — SynixSearch adapter
- `src/synix/cli/release_commands.py` — `synix release` CLI

### Docs
- `docs/migration-v2.md` — migration guide for existing users
- `docs/receipt-schema.md` — formal versioned receipt JSON schema
- `docs/handoff-projection-release-v2.md` — this file

### Tests
- `tests/unit/test_snapshot_view.py`
- `tests/unit/test_release_closure.py`
- `tests/unit/test_release_engine.py`
- `tests/unit/test_adapters.py`
- `tests/e2e/test_release_flow.py`

---

## Pickup Instructions

```bash
# 1. Get the branch
git fetch origin
git checkout mark/projection-release-v2

# 2. Fix the demo-01 CI issue (see "Blocking CI Issue" above)
#    Edit src/synix/cli/demo_commands.py _normalize_output()
#    Then:
uv run synix demo run templates/01-chatbot-export-synthesis --update-goldens
bash scripts/sync-templates

# 3. Verify everything passes
uv run release

# 4. Commit and push
git add -A
git commit -m "fix: normalize release output line wrapping for CI terminal width"
git push origin mark/projection-release-v2

# 5. CI should go green → merge PR
```

---

## Gate

`uv run release` must pass before any push. This runs: ruff fix → ruff format → sync templates → ruff check → pytest (all tests) → verify-demos (all 5 demos).
