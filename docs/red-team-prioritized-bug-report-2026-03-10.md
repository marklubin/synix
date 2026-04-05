<!-- updated: 2026-03-10 -->

# Synix Prioritized Bug Report

**Date:** 2026-03-10  
**Target repo:** `marklubin/synix`  
**Commit reviewed:** `a03b8d8396587cdc370e7a8df90ee8720ab7a83f`  
**Source report:** red-team sweep on core customer surface plus selected P2 checks

## Triage Summary

This document converts the red-team findings into a fix-oriented bug backlog.

- `P0`: fix before calling the core CLI surface stable
- `P1`: fix next; important trust or operator pain, but not the biggest launch blocker
- `P2`: defer behind core-surface stabilization; still real bugs

## Recommended Fix Order

1. `P0` trust-surface and path-resolution bugs: 3, 4, 5, 8, 9, 11
2. `P1` operator consistency bugs: 1, 10, 12
3. `P2` docs/discoverability and non-core surfaces: 2, 6, 7, 13

## Bug List

### 1. Cold-build planner overestimates downstream artifact counts after cardinality-reducing transforms

- **Priority:** `P1`
- **Severity:** High
- **Surface:** `synix plan`
- **Actual result:** `plan` materially overstates artifact counts, projection sizes, and estimated LLM calls before a first build.
- **Expected result:** `plan` should report realistic downstream counts and call estimates, even when a transform reduces cardinality.
- **Reproduction:**
  1. Create a fresh pipeline with `Source -> ReduceSynthesis -> MapSynthesis`.
  2. Run `synix plan pipeline.py --json`.
  3. Run `synix build pipeline.py`.
  4. Compare planned counts to actual built counts.
  5. Repeat with `Source -> MapSynthesis -> ReduceSynthesis -> SearchSurface + SynixSearch + FlatFile`.
- **Observed examples:**
  - `post.artifact_count = 2` in `plan`, but only `1` artifact actually built.
  - `search.artifact_count = 4` and `context.artifact_count = 2` in `plan`, but release materialized `3` search artifacts and `1` context artifact.
- **Why this matters:** customers use `plan` to judge cost, scope, and whether a prompt or DAG change is safe to run.

### 2. Mesh commands ignore `SYNIX_MESH_ROOT` for real filesystem operations

- **Priority:** `P2`
- **Severity:** High
- **Surface:** `synix mesh`
- **Actual result:** commands report paths under the env-configured mesh root, but real config/token state is written under the default home directory mesh root.
- **Expected result:** all mesh filesystem reads and writes should consistently honor `SYNIX_MESH_ROOT`.
- **Reproduction:**
  1. Export `SYNIX_MESH_ROOT=/tmp/synix-redteam-mesh-root-2`.
  2. Run `synix mesh create --name <name> --pipeline <pipeline.py>`.
  3. Run `synix mesh status --name <name>` with the env var still set.
  4. Observe the mesh is reported missing.
  5. Run the same command without the env var and observe the mesh exists under the default root instead.
- **Why this matters:** operators think they are isolating mesh state in a temp or test root when Synix is actually mutating the default global location.

### 3. Absolute-path pipeline invocation resolves relative `source_dir` against caller cwd

- **Priority:** `P0`
- **Severity:** High
- **Surface:** `synix plan`, `synix build`
- **Actual result:** when a pipeline is invoked by absolute path from another directory, relative `source_dir` values resolve against the shell cwd instead of the pipeline file location.
- **Expected result:** relative pipeline paths should resolve relative to the pipeline file, not the caller's current working directory.
- **Reproduction:**
  1. Create a project whose pipeline uses `pipeline.source_dir = "./sources"`.
  2. From a different directory, run `synix plan /abs/path/to/pipeline.py --json`.
  3. Observe that the plan reports zero inputs or otherwise mis-resolves sources.
  4. Run `synix build /abs/path/to/pipeline.py`.
  5. Compare with running the same commands from inside the project directory, where the pipeline succeeds.
- **Why this matters:** default template-style projects become fragile the moment a user runs them from CI, another repo root, or a script.

### 4. Cross-parent `--build-dir` overrides keep writing snapshots into the original `.synix`

- **Priority:** `P0`
- **Severity:** High
- **Surface:** `synix build`, `synix plan --save`, follow-on `refs` / `runs` / `release`
- **Actual result:** the CLI can announce one build directory while refs and objects are persisted under the pipeline's original `.synix` parent.
- **Expected result:** once `--build-dir` is provided, all writes and subsequent reads should resolve through that override.
- **Reproduction:**
  1. Create a project whose pipeline uses `build-a` by default.
  2. Run `synix build <pipeline.py> --build-dir /tmp/other-parent/build-b --source-dir <real-sources>`.
  3. Observe the CLI reports the override path.
  4. Run `synix refs list --build-dir /tmp/other-parent/build-b`.
  5. Observe `No snapshot store found.`
  6. Run `synix refs list --build-dir <original-build-dir>` and observe the new refs there instead.
  7. Repeat with `synix plan ... --build-dir /tmp/other-parent/build-b --save`.
- **Why this matters:** this breaks trust in one of the most important operator controls and makes downstream commands disagree about where state lives.

### 5. Source loader failures are downgraded into successful empty plans and builds

- **Priority:** `P0`
- **Severity:** High
- **Surface:** custom `Source.load()` extension point, `synix plan`, `synix build`
- **Actual result:** a `Source.load()` exception can become a successful empty plan or build, including exit code `0` and a committed run ref.
- **Expected result:** a source loader failure should stop the run, return a non-zero exit code, and never commit a misleading empty snapshot.
- **Reproduction:**
  1. Implement a custom `Source` subclass whose `load()` raises `RuntimeError`.
  2. Run `synix plan <pipeline.py> --json`.
  3. Observe `status = "cached"` and `artifact_count = 0` instead of an error.
  4. Run `synix build <pipeline.py>`.
  5. Observe traceback output, but process exit `0` and a committed snapshot/ref.
- **Why this matters:** this is a silent correctness failure that makes automation think a run succeeded when it did not.

### 6. `batch-build plan` undercounts source-backed request volume

- **Priority:** `P2`
- **Severity:** Medium
- **Surface:** `synix batch-build plan`
- **Actual result:** the batch planner reports dependency-layer counts rather than the split request count that the real batch run will submit.
- **Expected result:** request estimates should match actual batch requests closely enough to be used for cost and quota planning.
- **Reproduction:**
  1. Create a batch-build project with source-backed transforms that fan out into multiple requests.
  2. Run `synix batch-build plan pipeline.py`.
  3. Run `synix batch-build run pipeline.py`.
  4. Inspect `synix batch-build status <build-id>`.
  5. Compare planned request count to actual submitted request count.
- **Why this matters:** teams can under-budget or under-provision for OpenAI Batch runs.

### 7. Shipped `llms.txt` documents an unsupported ref-to-ref `synix diff` syntax

- **Priority:** `P2`
- **Severity:** Medium
- **Surface:** `llms.txt`
- **Actual result:** the shipped LLM-facing doc advertises `synix diff <ref-a> <ref-b>`, but the CLI only accepts `diff [artifact_id]` with `--old-build-dir`.
- **Expected result:** shipped command docs should match the CLI exactly.
- **Reproduction:**
  1. Read `llms.txt`.
  2. Run `synix diff HEAD refs/releases/local`.
  3. Observe an immediate parser error.
  4. Compare with `synix diff --help`, which documents build-dir-based usage instead.
- **Why this matters:** agents and power users following the repo-shipped command summary hit a dead command path immediately.

### 8. `synix info` crashes on normal projects

- **Priority:** `P0`
- **Severity:** Medium
- **Surface:** `synix info`
- **Actual result:** `synix info` raises `AttributeError: 'Source' object has no attribute 'level'` in ordinary project directories.
- **Expected result:** `synix info` should exit `0` and show project/system details for a normal project.
- **Reproduction:**
  1. Enter a normal Synix project directory with `pipeline.py`.
  2. Run `synix info`.
  3. Observe the `AttributeError` crash before any useful output.
- **Why this matters:** one of the basic operator inspection commands is broken on the happy path.

### 9. `info` and `status` still inspect legacy `build/` paths and miss real snapshot-era state

- **Priority:** `P0`
- **Severity:** Medium
- **Surface:** `synix info`, `synix status`
- **Actual result:** these commands can say there is no build or no projection state even when a valid `.synix` store and released projections exist.
- **Expected result:** both commands should read the real snapshot-era state model and reflect released `search.db` / `context.md` outputs.
- **Reproduction:**
  1. Build and release a project with a real `.synix/releases/local/search.db` and `.synix/releases/local/context.md`.
  2. Run `synix status --build-dir <project>/build`.
  3. Observe a clean build summary that omits real release outputs.
  4. Copy a valid `.synix/` into a directory without `pipeline.py`.
  5. Run `synix info`.
  6. Observe fallback output like `No pipeline.py in current directory` and `No build directory with manifest found` despite the valid store.
- **Why this matters:** operator trust surfaces are reporting stale or false information about the actual build state.

### 10. Invalid ref handling is inconsistent across inspector commands

- **Priority:** `P1`
- **Severity:** Medium
- **Surface:** `list`, `show`, `lineage`, `search --ref`
- **Actual result:** the same bad ref can yield an empty success message in one command, a friendly error in another, and a raw traceback in a third.
- **Expected result:** invalid refs should fail consistently with a single clear non-zero error style across inspector commands.
- **Reproduction:**
  1. Pick any valid project with a snapshot store.
  2. Run `synix list --ref refs/does/not/exist`.
  3. Observe `No artifacts found.`
  4. Run `synix search <query> --ref refs/does/not/exist`.
  5. Observe a traceback.
  6. Run `synix show <artifact> --ref refs/does/not/exist` or `synix lineage <artifact> --ref refs/does/not/exist`.
  7. Observe those commands fail cleanly instead.
- **Why this matters:** users and agents cannot build reliable automation around ref selection when the error contract is inconsistent.

### 11. `synix diff` is not snapshot-accurate on normal snapshot-era projects

- **Priority:** `P0`
- **Severity:** Medium
- **Surface:** `synix diff`
- **Actual result:** single-artifact diff ignores prior run history unless legacy `build/versions/` exists, and full-build diff can claim two different snapshots are identical.
- **Expected result:** `diff` should compare real snapshot history and accurately report changed, added, and removed artifacts.
- **Reproduction:**
  1. Use a project with multiple run refs in the same `.synix/`.
  2. Run `synix diff <artifact-label>`.
  3. Observe it reports no previous version unless legacy `build/versions/<label>` exists.
  4. Copy the project and repoint one copy's `.synix/HEAD` to an older run ref.
  5. Run `synix diff --old-build-dir <old-copy>/build --build-dir <new-copy>/build`.
  6. Observe `No differences between builds` even when `synix list --ref <old-run>` and current `synix list` show different artifact counts.
- **Why this matters:** `diff` is a trust command; false negatives are worse than a loud failure.

### 12. `synix clean` removes release payloads but leaves stale `refs/releases/*`

- **Priority:** `P1`
- **Severity:** Medium
- **Surface:** `synix clean`, `refs list`, `releases show`, `search --release`
- **Actual result:** `clean` deletes release payloads and receipts, but release refs still exist and can be listed afterward.
- **Expected result:** release cleanup should leave the repo in a consistent state, including pruning release refs or clearly marking them invalid.
- **Reproduction:**
  1. Build and release a project to `local`.
  2. Run `synix clean --yes`.
  3. Run `synix refs list`.
  4. Observe `refs/releases/local` still present.
  5. Run `synix releases show local` or `synix search <query> --release local`.
  6. Observe those commands fail because the target was deleted.
- **Why this matters:** after cleanup, one surface still advertises a release that the rest of the CLI cannot use.

### 13. `refs list` hides plan refs created by `synix plan --save`

- **Priority:** `P2`
- **Severity:** Low
- **Surface:** `synix refs list`
- **Actual result:** `refs list` omits `refs/plans/latest` even though `plan --save` creates it and `refs show` can resolve it.
- **Expected result:** `refs list` should either show plan refs or clearly document that it does not.
- **Reproduction:**
  1. Run `synix plan <pipeline.py> --save`.
  2. Confirm `synix refs show refs/plans/latest` returns a real snapshot.
  3. Run `synix refs list`.
  4. Observe the plan ref is omitted.
- **Why this matters:** this is a smaller discoverability issue, but it makes the CLI feel internally inconsistent.

## Summary Table

| # | Bug | Priority | Severity | Surface |
|---|-----|----------|----------|---------|
| 1 | Cold-build planner overestimates downstream counts | P1 | High | `plan` |
| 2 | Mesh ignores `SYNIX_MESH_ROOT` for real FS writes | P2 | High | `mesh` |
| 3 | Absolute-path pipeline invocation mis-resolves `source_dir` | P0 | High | `plan`, `build` |
| 4 | Cross-parent `--build-dir` overrides misroute snapshot state | P0 | High | `build`, `plan --save`, refs/release follow-ons |
| 5 | Source loader failures become successful empty runs | P0 | High | custom sources, `plan`, `build` |
| 6 | `batch-build plan` undercounts request volume | P2 | Medium | `batch-build plan` |
| 7 | `llms.txt` documents unsupported `diff` syntax | P2 | Medium | docs / `llms` |
| 8 | `synix info` crashes on normal projects | P0 | Medium | `info` |
| 9 | `info` / `status` miss real snapshot-era state | P0 | Medium | `info`, `status` |
| 10 | Invalid ref handling is inconsistent | P1 | Medium | `list`, `show`, `lineage`, `search --ref` |
| 11 | `synix diff` is not snapshot-accurate | P0 | Medium | `diff` |
| 12 | `clean` leaves stale release refs | P1 | Medium | `clean`, `refs`, `releases`, `search` |
| 13 | `refs list` hides saved plan refs | P2 | Low | `refs list` |

## Release Readiness Call

- Do **not** mark the core CLI surface as stable until all `P0` bugs are fixed and regression-tested.
- After the `P0` pass, re-run the same core matrix on:
  - `init`
  - `plan`
  - `build`
  - `release`
  - `revert`
  - `search`
  - `list`
  - `show`
  - `lineage`
  - `diff`
  - `verify`
  - `clean`
  - `runs list`
  - `refs list`
  - `refs show`
  - `info`
  - `status`
  - `llms`
