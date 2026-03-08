# RFC: Projection Release v2 — Build, Release, and Portable Closures

**Related issues**: [#34](https://github.com/marklubin/synix/issues/34), [#81](https://github.com/marklubin/synix/issues/81), [#82](https://github.com/marklubin/synix/issues/82)
**Status**: Draft
**Supersedes**: `docs/projection-release-slice-rfc.md`, release-related sections of `docs/snapshots-release-rfc.md`
**Baseline**: `main` after [#91](https://github.com/marklubin/synix/pull/91)

## Thesis

Synix separates build from release absolutely:

- `synix build` produces immutable snapshots in `.synix/`. It writes no user-facing files.
- `synix release` materializes a snapshot to a named release target. It is the only command that creates user-facing outputs.
- The `build/` directory is removed as a platform concept.
- Every release produces a receipt. Receipts are the universal proof of what is live where.
- Projection declarations in the manifest are structured and diffable — not opaque blobs.
- The platform provides a fully resolved artifact closure to every adapter. Adapters are not special; `synix_search` is just the built-in default.

## Current Baseline

What exists on `main` today (after PR #91):

- `.synix/` stores immutable artifact objects, manifests, snapshots, and refs
- `HEAD`, `refs/heads/*`, and `refs/runs/*` exist and work
- `ObjectStore` handles content-addressed storage with schema validation for all five object types
- `RefStore` handles symbolic HEAD, ref resolution, cycle detection, atomic writes
- `BuildTransaction` accumulates artifact state and commits snapshots with optimistic concurrency
- `SearchSurface` is the build-time search capability
- `SynixSearch` is the canonical search output contract

What is still split-brain:

- Every artifact is written **twice**: once to `build/` as `.md` files (via `ArtifactStore`), once to `.synix/objects/` as immutable blobs
- Projections (`search.db`, `context.md`) are written **only** to `build/` — they are not part of the snapshot
- `manifest["projections"]` is always `{}`
- CLI commands (`list`, `show`, `search`, `lineage`) read from `build/`, not from `.synix/`
- Provenance is tracked separately in `build/provenance.json` even though artifact objects already contain `parent_labels` and `input_ids`
- There is no release lifecycle, no release refs, no receipts

## Design Principles

1. **Build is pure.** `synix build` writes immutable state and advances build refs. It creates no user-facing files.
2. **Release is explicit.** `synix release` materializes one snapshot to one named target. It is the only command that writes user-facing outputs.
3. **Projections are structured declarations, not opaque blobs.** The manifest records each projection's adapter, input artifacts, and config. The platform can diff any two manifests generically.
4. **Adapters receive a resolved closure.** The platform walks the artifact graph, resolves content and provenance, and hands a complete `ReleaseClosure` to every adapter. No adapter needs to talk to the object store directly.
5. **Every release has a receipt.** Whether the adapter writes local files or mutates a remote database, a receipt is always written to `.synix/releases/<name>/`. Receipts are the universal answer to "what is live where?"
6. **Adapters are not special.** `synix_search` is the built-in default, but it follows the same contract as any external adapter. The platform does not privilege it.
7. **`build/` does not exist.** This is a breaking change. There is no mutable working directory.

## State Model

Synix distinguishes four kinds of state:

```
artifacts                immutable content (episodes, rollups, core memory)
projection declarations  structured intent (adapter + input artifacts + config)
build refs               pointers to built states (HEAD, refs/heads/*, refs/runs/*)
release state            receipts + materialized outputs (refs/releases/*, .synix/releases/*)
```

The key rules:

- Snapshots own artifacts and projection declarations
- Release refs own deployment intent
- Receipts own deployment proof
- Materialized outputs are adapter-specific and may or may not be local files

## Filesystem Layout

After `synix build` + two releases:

```
project/
  pipeline.py
  exports/                         source conversation files

  .synix/                          canonical store (all platform state)
    HEAD                           "ref: refs/heads/main"
    .lock                          build concurrency lock

    refs/
      heads/
        main                       oid of latest snapshot
      runs/
        20260306T120000Z           oid of run 1 snapshot
        20260307T090000Z           oid of run 2 snapshot
      releases/
        local                      oid of snapshot powering local release
        prod                       oid of snapshot powering prod release

    objects/
      aa/bbccdd...                 artifact objects, content blobs,
      12/345678...                 manifest objects, snapshot objects
      ff/eeddcc...                 (all content-addressed)

    releases/                      release state (one dir per release name)
      local/
        receipt.json               what was released, when, which adapters
        search.db                  materialized by synix_search adapter
        context.md                 materialized by flat_file adapter
      prod/
        receipt.json               receipt only (data lives in Postgres)
```

There is no `build/` directory. The `ArtifactStore`, `ProvenanceTracker`, and `.projection_cache.json` are removed.

## What Build Produces

### Artifact Objects (unchanged from today)

Each artifact is stored as two objects:

```
content blob:  raw text bytes → content_oid
artifact object:
  {
    "type": "artifact",
    "label": "ep-conv-001",
    "artifact_type": "episode",
    "artifact_id": "sha256:...",
    "content_oid": "aabb...",
    "input_ids": ["sha256:..."],
    "parent_labels": ["tx-conv-001"],
    "metadata": {"layer_name": "episodes", "layer_level": 1, ...}
  }
```

### Manifest (changed: projections are now structured)

```json
{
  "type": "manifest",
  "schema_version": 2,
  "pipeline_name": "personal-memory",
  "pipeline_fingerprint": "sha256:...",
  "artifacts": [
    {"label": "tx-conv-001", "oid": "aabb..."},
    {"label": "ep-conv-001", "oid": "ccdd..."},
    {"label": "monthly-2024-03", "oid": "eeff..."},
    {"label": "core", "oid": "1122..."}
  ],
  "projections": {
    "search": {
      "adapter": "synix_search",
      "input_artifacts": ["ep-conv-001", "monthly-2024-03", "core"],
      "config": {
        "modes": ["fulltext"],
        "embedding_config": {}
      },
      "config_fingerprint": "sha256:...",
      "precomputed_oid": null
    },
    "context-doc": {
      "adapter": "flat_file",
      "input_artifacts": ["core"],
      "config": {
        "output_template": "context.md"
      },
      "config_fingerprint": "sha256:...",
      "precomputed_oid": null
    }
  }
}
```

`precomputed_oid` is an optional optimization. If an adapter wants to pre-build an expensive artifact during build (e.g., pre-render a `search.db` image), it stores it in the object store and records the oid here. At release time, the adapter can use the precomputed blob instead of rebuilding from the closure. This is a **cache**, not the source of truth. The structured declaration is always the source of truth.

### Snapshot (unchanged)

```json
{
  "type": "snapshot",
  "schema_version": 1,
  "manifest_oid": "ffgg...",
  "parent_snapshot_oids": ["prev..."],
  "created_at": "2026-03-07T09:00:00Z",
  "pipeline_name": "personal-memory",
  "run_id": "20260307T090000Z"
}
```

## Build Flow

```
synix build

  1. Parse sources → artifact objects → .synix/objects/
  2. Execute transforms → artifact objects → .synix/objects/
  3. Record projection declarations in manifest (structured, no materialization)
  4. Write manifest object → .synix/objects/
  5. Write snapshot object → .synix/objects/
  6. Advance refs/heads/main → snapshot oid
  7. Write refs/runs/<run-id> → snapshot oid

  Output: snapshot oid, run ref. No files on disk for human consumption.

  Single write path. No ArtifactStore. No ProvenanceTracker.
  No build/ directory.
```

```
                 synix build
                      |
     ┌────────────────┼────────────────┐
     v                v                v
  Source.load()   Transform.execute()  Projection declarations
     |                |                |
     v                v                v
  .synix/objects/  .synix/objects/   manifest.projections
  (content blobs)  (content blobs)  (structured, not materialized)
  (artifact objs)  (artifact objs)
                      |
                      v
                   manifest → snapshot → refs
                   (closure over artifacts + projection declarations)
```

## Release Closure

Before dispatching to adapters, the platform resolves a snapshot into a `ReleaseClosure`. This is the universal input that every adapter receives. It is the portable, fully-resolved artifact bundle.

```python
@dataclass
class ResolvedArtifact:
    label: str
    artifact_type: str
    content: str
    artifact_id: str
    layer_name: str
    layer_level: int
    provenance_chain: list[str]   # walked from parent_labels
    metadata: dict

@dataclass
class ProjectionDeclaration:
    name: str
    adapter: str
    input_artifacts: list[str]    # labels
    config: dict
    config_fingerprint: str
    precomputed_oid: str | None

@dataclass
class ReleaseClosure:
    snapshot_oid: str
    manifest_oid: str
    pipeline_name: str
    created_at: str
    artifacts: dict[str, ResolvedArtifact]     # label → resolved
    projections: dict[str, ProjectionDeclaration]
```

Building the closure:

```
  snapshot → manifest → artifact oids
                     → projection declarations

  For each artifact oid:
    load artifact object  → label, type, metadata, parent_labels
    load content blob     → raw text
    walk parent_labels transitively → provenance chain

  Result: ReleaseClosure with everything resolved.
  No adapter needs to touch .synix/objects/ directly.
```

## Release Flow

```
synix release HEAD --to local
synix release HEAD --to prod --target postgres://prod-db/synix_memory

  1. Resolve source ref (HEAD) → snapshot oid
  2. Build ReleaseClosure from snapshot
  3. Resolve destination release name
  4. Resolve target (default: .synix/releases/<name>/)
  5. For each projection in the closure:
     a. Load adapter
     b. adapter.plan(closure, current_receipt) → ReleasePlan
     c. adapter.apply(plan, target) → AdapterReceipt
  6. All adapters succeeded?
     yes → write receipt, advance refs/releases/<name> → snapshot oid
     no  → do NOT advance ref, report failure
```

```
            synix release HEAD --to local
                       |
                       v
              resolve HEAD → S2
                       |
                       v
              build ReleaseClosure
              (all artifacts + provenance resolved)
                       |
              ┌────────┼────────┐
              v                 v
        synix_search       flat_file
        adapter              adapter
              |                 |
              v                 v
        .synix/releases/   .synix/releases/
        local/search.db    local/context.md
              |                 |
              └────────┬────────┘
                       v
              write receipt.json
              advance refs/releases/local → S2
```

## Adapter Contract

```python
class ProjectionAdapter:
    """Target-specific materialization. Every adapter follows this contract."""

    def plan(
        self,
        closure: ReleaseClosure,
        declaration: ProjectionDeclaration,
        current_receipt: AdapterReceipt | None,
    ) -> ReleasePlan:
        """Diff desired state against current state.

        Returns a plan describing what will change. The plan is
        inspectable for dry-run display.
        """
        ...

    def apply(
        self,
        plan: ReleasePlan,
        target: str | Path,
    ) -> AdapterReceipt:
        """Execute the plan. Owns its own internal transaction.

        For files: write to temp, atomic rename.
        For Postgres: BEGIN; ...; COMMIT;
        Returns a receipt describing what was done.
        """
        ...

    def verify(self, receipt: AdapterReceipt, target: str | Path) -> bool:
        """Confirm the release is live and consistent."""
        ...
```

### Adapter Implementations

**synix_search (built-in)**:
- `plan`: diff input_artifacts against receipt's previous artifact list → N new, M removed, config changed?
- `apply`: build FTS5 index from closure artifacts (with provenance baked into a `provenance_chains` table), write to target path via atomic swap. If `precomputed_oid` exists and input artifacts match, copy blob instead of rebuilding.
- `verify`: open search.db, confirm row count matches, confirm FTS5 table exists

**flat_file (built-in)**:
- `plan`: diff input artifact content hashes → changed or unchanged
- `apply`: render markdown from closure artifacts, atomic write to target path
- `verify`: confirm file exists and content hash matches

**postgres (future)**:
- `plan`: diff input_artifacts against receipt's previous list → INSERTs, DELETEs, UPDATEs
- `apply`: `BEGIN; UPSERT ...; DELETE ...; COMMIT;`
- `verify`: query row count + spot-check content hashes

All adapters receive the same `ReleaseClosure`. The platform does not privilege any adapter.

## Release Receipt

Every release writes a receipt to `.synix/releases/<name>/receipt.json`:

```json
{
  "schema_version": 1,
  "release_name": "local",
  "snapshot_oid": "aabb...",
  "manifest_oid": "ccdd...",
  "pipeline_name": "personal-memory",
  "released_at": "2026-03-07T10:00:00Z",
  "source_ref": "HEAD",
  "resolved_ref": "refs/heads/main",
  "adapters": {
    "search": {
      "adapter": "synix_search",
      "target": ".synix/releases/local/search.db",
      "artifacts_applied": 3,
      "status": "success"
    },
    "context-doc": {
      "adapter": "flat_file",
      "target": ".synix/releases/local/context.md",
      "artifacts_applied": 1,
      "status": "success"
    }
  }
}
```

For external targets:

```json
{
  "schema_version": 1,
  "release_name": "prod",
  "snapshot_oid": "aabb...",
  "manifest_oid": "ccdd...",
  "pipeline_name": "personal-memory",
  "released_at": "2026-03-07T10:05:00Z",
  "source_ref": "HEAD",
  "resolved_ref": "refs/heads/main",
  "adapters": {
    "search": {
      "adapter": "postgres",
      "target": "postgres://prod-db/synix_memory",
      "artifacts_applied": 3,
      "artifacts_removed": 0,
      "status": "success"
    }
  }
}
```

The receipt always exists regardless of adapter type. It is the universal proof.

## Self-Contained File Releases

For file-based adapters, the release directory must be self-contained and portable. The `synix_search` adapter bakes provenance into the search.db at release time:

```
search.db (released):
  search_index (FTS5)
    content, label, layer_name, layer_level, metadata

  provenance_chains                    baked at release time
    label TEXT PRIMARY KEY
    chain TEXT (JSON array of ancestor labels)

  citation_edges
    source_label, target_uri, target_label

  release_metadata                     baked at release time
    snapshot_oid, manifest_oid, pipeline_name, released_at

  embeddings (if configured)
    label, vector BLOB
```

A released search.db works standalone. No `.synix/` needed at the destination for runtime queries. The consumer (an agent, a tool) only needs the release directory.

`.synix/` is still needed for:
- Building new snapshots
- Diffing between refs
- Inspecting raw artifact objects
- Running `synix` CLI commands

But the **consumer** of the release only needs the release directory.

## Diff Semantics

Diff operates on manifests, not on release targets or files. It is always a platform operation.

```
synix diff HEAD refs/releases/prod
synix diff refs/runs/20260306T...Z refs/runs/20260307T...Z
```

Resolution:

```
  ref A → snapshot → manifest A
  ref B → snapshot → manifest B

  Compare:
    artifacts:
      + new labels (in A, not in B)
      - removed labels (in B, not in A)
      ~ changed labels (same label, different content_oid)
      = unchanged labels

    projections:
      per projection name:
        input artifact count change
        config fingerprint change
        adapter change
```

This works because projections are structured declarations with explicit `input_artifacts` and `config`. No opaque blobs to compare.

```bash
$ synix diff HEAD refs/releases/prod

  Artifacts:
    + ep-conv-005           episode    (new)
    + ep-conv-006           episode    (new)
    ~ monthly-2024-04       monthly    (content changed)
    ~ core                  core       (content changed)
    = ep-conv-001           episode    (unchanged)
    = ep-conv-002           episode    (unchanged)
    = ep-conv-003           episode    (unchanged)

  Projections:
    search:
      input artifacts: 5 -> 7 (+2)
      config: unchanged
    context-doc:
      input artifacts: 1 (content changed)
      config: unchanged
```

Scoped diff:

```bash
$ synix diff HEAD refs/releases/prod --artifact core
$ synix diff HEAD refs/releases/prod --projection search
```

## Revert Semantics

Revert is release of an older snapshot. There is no special revert machinery.

```
synix revert refs/runs/20260306T...Z --to prod

  Equivalent to:
  synix release refs/runs/20260306T...Z --to prod
```

The adapter sees the desired state (older manifest) and reconciles. For files, it overwrites. For Postgres, it diffs and applies removals/changes.

```
  refs/releases/prod was → S2 (47 artifacts)
  Reverting to S1 (44 artifacts)

  synix_search adapter:
    plan: 44 artifacts desired, 47 currently live → 0 new, 3 to remove
    apply: rebuild search.db from S1 closure, atomic swap

  postgres adapter:
    plan: 44 desired, 47 live → DELETE 3 rows
    apply: BEGIN; DELETE ...; COMMIT;

  Receipt: reverted to S1
  refs/releases/prod → S1
```

## Read Semantics

### Snapshot-Native Reads

These commands read from `.synix/objects/` and default to HEAD:

```bash
synix list [--ref HEAD]
synix show <label> [--ref HEAD]
synix lineage <label> [--ref HEAD]
synix diff <ref-a> <ref-b>
```

They resolve: `ref → snapshot → manifest → artifact oids → content blobs`.

These never require materialized files. They read directly from the object store.

Implementation: `SnapshotView`

```python
view = SnapshotView.open(ref="HEAD")
view.list_artifacts()           # manifest.artifacts
view.get_artifact("core")      # load artifact object + content blob
view.get_provenance("core")    # walk parent_labels
view.get_manifest()            # full manifest
```

### Release-Aware Reads

These commands query materialized release state:

```bash
synix search "pricing"                           # queries default release
synix search "pricing" --release local            # queries named release
synix search "pricing" --ref HEAD                 # scratch realization (see below)
```

Default behavior: `synix search` queries the release named in a pipeline config default (e.g., `local`), or the single release if only one exists.

### Scratch Realizations

`synix search --ref <ref>` needs a search index, but the ref might not be released anywhere. The platform handles this by creating an ephemeral realization:

```
  1. Build ReleaseClosure from ref
  2. synix_search adapter builds search.db in .synix/work/<tmp>/
  3. Query it
  4. Discard .synix/work/<tmp>/

  This never moves a release ref or writes a receipt.
  It is a temporary, read-only operation.
```

## Release Inspection

```bash
$ synix releases list

  Release     Snapshot   Released              Pipeline
  ──────────  ────────   ────────────────────  ────────────────
  local       S3         2026-03-07 10:00:00   personal-memory
  staging     S2         2026-03-07 09:30:00   personal-memory
  prod        S1         2026-03-06 18:00:00   personal-memory

$ synix releases show prod

  Release:   prod
  Snapshot:  S1 (aabb...ccdd)
  Released:  2026-03-06 18:00:00Z
  Pipeline:  personal-memory
  Source:    refs/heads/main

  Adapters:
    search       synix_search  → .synix/releases/prod/search.db    (3 artifacts)
    context-doc  flat_file     → .synix/releases/prod/context.md   (1 artifact)

$ synix releases show prod --json    # machine-readable receipt
```

## Release Transactions

Multi-projection releases need all-or-nothing semantics:

```
  1. Write pending transaction record (.synix/releases/<name>/.pending.json)
  2. For each projection:
     a. adapter.plan()
     b. adapter.apply()
     c. collect adapter receipt
  3. All succeeded?
     yes → write receipt.json, advance release ref, delete .pending.json
     no  → do NOT advance ref, leave .pending.json for diagnosis
```

For file-based adapters, "roll back" means the old files are untouched until the atomic swap in step 3. For external adapters (Postgres), the adapter owns its own internal transaction — if the adapter committed but the overall release fails, the next release attempt is idempotent (the adapter diffs against current state, not against the old receipt).

Adapters must be **idempotent**: re-releasing the same snapshot to the same target converges to the same state.

## Locking

```
  .synix/releases/<name>/.lock    per-release-target lock

  synix release    takes lock
  synix revert     takes lock (same as release)

  Scratch realizations (.synix/work/) use isolated temp dirs
  and never contend with release locks.
```

## Ref Management

```bash
synix refs list                     # all refs (build + release)
synix refs show <ref>               # resolve ref → snapshot details
synix runs list                     # build run history (already exists)
synix releases list                 # release history
synix releases show <name>         # receipt details
```

## Migration: Removing build/

This is a breaking change. The migration path:

1. `ArtifactStore` (build/manifest.json + .md file writer) — removed. ObjectStore is the only write path.
2. `ProvenanceTracker` (build/provenance.json) — removed. Provenance is in artifact objects via `parent_labels` + `input_ids`.
3. `.projection_cache.json` — removed. Content-addressed dedup in the object store replaces it.
4. The dual-write pattern in `runner.py` — replaced with single write to `.synix/objects/`.
5. CLI commands (`list`, `show`, `search`, `lineage`) — rewritten to use `SnapshotView` for reads and release targets for search.
6. `synix clean` — cleans release targets, not `build/`.
7. `synix build` output — reports snapshot oid and run ref. Suggests `synix release` as next step.

## CLI Commands After Migration

```bash
# Build
synix build pipeline.py                          # immutable snapshot only
synix plan pipeline.py                           # dry-run

# Inspect (reads from .synix/)
synix list [--ref HEAD]                          # artifacts in snapshot
synix show <label> [--ref HEAD]                  # render artifact content
synix lineage <label> [--ref HEAD]               # provenance tree
synix diff <ref-a> <ref-b>                       # structured diff

# Release
synix release HEAD --to local                    # materialize to default target
synix release HEAD --to prod --target postgres://...
synix revert <older-ref> --to <release-name>     # release older snapshot

# Search (queries release targets)
synix search "query" [--release local]           # query materialized release
synix search "query" --ref HEAD                  # scratch realization

# Release inspection
synix releases list                              # all releases
synix releases show <name>                       # receipt details
synix refs list                                  # all refs
synix refs show <ref>                            # resolve ref

# Maintenance
synix clean --release local                      # remove release target
```

## Acceptance Criteria

The projection/release redesign is complete when:

- [ ] `synix build` writes only to `.synix/objects/` and `.synix/refs/`. No `build/` directory created.
- [ ] Manifest schema v2 includes structured projection declarations with `input_artifacts` and `config`.
- [ ] `ReleaseClosure` resolves artifacts with content and provenance from the object store.
- [ ] `synix release` dispatches to adapters, writes receipts to `.synix/releases/<name>/`, advances release refs.
- [ ] `synix_search` adapter builds self-contained search.db with provenance baked in.
- [ ] `flat_file` adapter writes self-contained markdown.
- [ ] `synix revert` works as release of an older snapshot.
- [ ] `synix diff` compares two refs via structured manifest comparison.
- [ ] `SnapshotView` provides ref-resolved reads for `list`, `show`, `lineage`.
- [ ] `synix search` queries release targets by default, supports `--ref` for scratch realizations.
- [ ] `synix releases list/show` displays release state and receipts.
- [ ] `synix refs list/show` displays all refs.
- [ ] All release directories are self-contained and portable for file-based adapters.
- [ ] `ArtifactStore`, `ProvenanceTracker`, `.projection_cache.json` are removed.
- [ ] Templates and demos updated to use new CLI flow.

## Implementation Order

1. **SnapshotView** — ref-resolved artifact reads from `.synix/objects/`
2. **Manifest schema v2** — structured projection declarations
3. **Remove ArtifactStore dual-write** — single write path to ObjectStore
4. **Remove ProvenanceTracker** — provenance from artifact `parent_labels`
5. **ReleaseClosure** — resolved artifact bundle with provenance
6. **ProjectionAdapter contract** — plan/apply/verify interface
7. **synix_search adapter** — self-contained search.db with provenance table
8. **flat_file adapter** — self-contained markdown
9. **synix release command** — orchestration, receipts, release refs
10. **synix revert** — thin wrapper over release
11. **Ref-first diff** — structured manifest comparison
12. **SnapshotView CLI integration** — rewrite list/show/lineage to use SnapshotView
13. **Search CLI integration** — search queries release targets, --ref for scratch
14. **synix releases/refs commands** — inspection CLI
15. **Remove build/ directory** — final migration cut
16. **Template and demo updates** — golden files, pipeline examples

## Test Strategy

All tests use `tmp_path`, mocked LLMs, no shared state.

### Unit Tests

- `test_manifest_v2.py` — structured projection declarations, schema validation
- `test_snapshot_view.py` — ref resolution, artifact lookup, provenance walking
- `test_release_closure.py` — closure construction, provenance chain resolution
- `test_adapter_contract.py` — synix_search plan/apply/verify, flat_file plan/apply/verify
- `test_receipt.py` — receipt schema, receipt persistence
- `test_release_lock.py` — lock acquisition, contention

### Integration Tests

- `test_build_no_build_dir.py` — build produces only .synix/ state, no build/ directory
- `test_release_file_based.py` — release produces self-contained directory with receipt
- `test_release_multiple_targets.py` — two releases from different snapshots coexist
- `test_revert_release.py` — release newer, then revert to older
- `test_diff_refs.py` — structured diff between two refs
- `test_search_release.py` — search queries released search.db with provenance
- `test_scratch_realization.py` — search --ref creates and discards temp realization

### E2E Tests

- `test_build_release_flow.py` — build → release → search → verify
- `test_build_release_revert_flow.py` — build → release → build again → diff → revert
- `test_clean_preserves_snapshots.py` — clean release target, snapshots survive

### Failure Tests

- Failed build does not advance build ref (already tested)
- Failed release does not advance release ref
- Failed adapter does not write receipt
- Partial multi-adapter failure leaves pending transaction
- Concurrent release attempts are serialized by lock
- Scratch realization cleanup on error

## Out of Scope

- Remote object storage
- Branching UX beyond refs
- External database adapters (Postgres, Qdrant) — the contract supports them, but only built-in file-based adapters ship in this slice
- Notebook / mutable overlay
- Checkpoint banks (builds on top of this, separate RFC)
- Runtime tool API (builds on top of this, separate RFC)

## Open Questions

1. **Default release name.** Should pipelines declare a default release name (e.g., `local`) so that `synix release HEAD` works without `--to`? Or always require explicit naming?
2. **Auto-release.** Should `synix build` optionally auto-release to a default target for developer convenience? (e.g., `synix build --release local`). This preserves the simple "build and use" workflow while keeping the separation clean.
3. **Precomputed blobs.** Should the `synix_search` adapter always pre-build the search.db during `synix build` (stored as `precomputed_oid`) for instant release? Or always build from the closure at release time? The former is faster for local workflows; the latter is simpler and avoids storing large blobs in the object store.
4. **Receipt history.** Should `.synix/releases/<name>/` keep a history of receipts, or only the latest? History enables audit trails; latest-only is simpler.
