# RFC: Immutable Snapshots, Refs, and Projection Release

**Issue**: [#34](https://github.com/marklubin/synix/issues/34)  
**Status**: Proposed  
**Baseline**: `v0.15.0` (`93b9c6b`)  
**Decision target**: Design approval before implementation

## Summary

Synix should move from a mutable `build/` directory model to a git-like snapshot model:

- `.synix` is the canonical store
- `HEAD` is a first-class ref
- a `manifest` is a closure over artifacts and projections
- a `snapshot` points to one manifest
- `release` materializes projections
- `revert` means releasing an older ref

This design is the generic platform substrate for reproducibility, diffing, release promotion, rollback, and multi-variation deployment. It is required by LENS, but it is not benchmark-specific.

The implementation should land in slices. The first mergeable slice is:

- immutable artifact snapshots
- refs and first-class `HEAD`
- manifest and snapshot objects
- compatibility local `build/` outputs retained for existing commands

Projection build-state capture and `synix release` remain follow-on work. Until that lands, projections are still compatibility outputs under `build/`, not part of the canonical persisted snapshot closure.

## Motivation

At `v0.15.0`, Synix still assumes a single mutable build root:

- artifact payloads and `manifest.json` are rewritten under `build/`
- `provenance.json` is mutable
- `search.db` is mutable
- `.projection_cache.json` is mutable
- CLI commands default to that mutable root

That model is sufficient for one live local build. It is not sufficient for:

- immutable experimental history
- clean diffs between runs
- rollback to known-good states
- multi-target releases
- future projection adapters with incremental release semantics
- checkpointed memory banks for LENS

## Design Goals

1. Every successful build creates an immutable logical snapshot.
2. `HEAD` and refs are first-class platform concepts.
3. A manifest represents a complete closure over artifacts and projections.
4. Projections remain distinct from artifacts because they are build targets.
5. Release is separate from build.
6. Revert is release to an older snapshot, not replay of inverse operations.
7. Projection adapters own target-specific reconciliation semantics.
8. The first implementation can start with file-backed projections and full rebuild release strategies without blocking future incremental adapters.

## Non-Goals

- No generic platform-level per-input apply/revert op model
- No remote object storage in the first implementation
- No branching UX beyond refs in the first implementation
- No requirement that every projection support incremental release on day one
- No notebook or mutable overlay implementation in this RFC

## Terminology

### Object

Any immutable stored item in `.synix/objects`.

### Artifact

Build data such as transcripts, episodes, rollups, summaries, or core-memory blocks.

### Projection

A build target that exposes artifacts through a usable output surface such as:

- search index
- flat-file context doc
- future Postgres or vector database targets

Projections are not “just another artifact.” They are build targets with different lifecycle semantics.

### Manifest

A closure over the exact artifacts and projections for a build state.

### Snapshot

An immutable commit-like object that points to one manifest.

### Ref

A named pointer to a snapshot. Examples:

- `HEAD`
- `refs/heads/main`
- `refs/runs/2026-03-06T12-30-11Z`
- `refs/releases/prod`

### Release

Materialization of projection targets from a ref or snapshot.

### Release Receipt

The provenance record of what was actually materialized, where, and by which adapter.

## Identity Model

Synix should use two different identities for two different jobs:

- `oid`
  - object-store id for any stored object in `.synix/objects`
- `artifact_id`
  - content identity already used inside artifact semantics

The object store should not be keyed directly by the current `artifact_id`, because `artifact_id` is already overloaded with content semantics inside the artifact model.

## Canonical Layout

```text
project/
  pipeline.py
  .synix/
    HEAD
    refs/
      heads/
        main
      runs/
        2026-03-06T12-30-11Z
      tags/
        v0.15.0
      releases/
        prod
        canary
        ab-a
        ab-b
    objects/
      aa/
        bbccddeeff...
      12/
        34567890ab...
    receipts/
      2026-03-06T12-31-02Z.json
```

`HEAD` should be textual and first-class:

```text
ref: refs/heads/main
```

## Object Types

```text
blob
artifact
projection
manifest
snapshot
release_receipt
```

## Object Schemas

### Blob Object

Raw bytes or textual content.

```json
{
  "type": "blob",
  "encoding": "utf-8",
  "bytes": "..."
}
```

### Artifact Object

```json
{
  "type": "artifact",
  "label": "ep-conv-001",
  "artifact_type": "episode",
  "artifact_id": "sha256:...",
  "content_oid": "oid_blob_1",
  "input_artifact_oids": ["oid_art_0"],
  "prompt_id": "episode_summary_v1",
  "model_config": {
    "model": "claude-sonnet-4-20250514"
  },
  "metadata": {
    "source_conversation_id": "conv-001"
  }
}
```

### Projection Object

Projection objects are build targets.

```json
{
  "type": "projection",
  "name": "memory-index",
  "projection_type": "search_index",
  "input_oids": ["oid_art_1", "oid_art_2"],
  "build_state_oid": "oid_blob_or_state",
  "adapter": "search_index",
  "release_mode": "full",
  "metadata": {}
}
```

Future incremental example:

```json
{
  "type": "projection",
  "name": "customer-db",
  "projection_type": "postgres",
  "input_oids": ["oid_art_10", "oid_art_11"],
  "build_state_oid": "oid_delta_plan",
  "adapter": "postgres",
  "release_mode": "incremental",
  "metadata": {
    "schema": "customer_memory"
  }
}
```

### Manifest Object

The manifest is the exact closure over artifacts and projections for one build state.

```json
{
  "type": "manifest",
  "pipeline_name": "monthly-memory",
  "pipeline_fingerprint": "sha256:...",
  "artifacts": {
    "tx-conv-001": "oid_art_1",
    "ep-conv-001": "oid_art_2",
    "core": "oid_art_3"
  },
  "projections": {
    "memory-index": "oid_proj_1",
    "context-doc": "oid_proj_2"
  }
}
```

### Snapshot Object

```json
{
  "type": "snapshot",
  "manifest_oid": "oid_manifest_1",
  "parent_snapshot_oids": ["oid_snapshot_prev"],
  "created_at": "2026-03-06T12:30:11Z",
  "pipeline_name": "monthly-memory"
}
```

### Release Receipt

```json
{
  "type": "release_receipt",
  "ref": "HEAD",
  "resolved_snapshot_oid": "oid_snapshot_1",
  "manifest_oid": "oid_manifest_1",
  "projection_oid": "oid_proj_1",
  "adapter": "search_index",
  "release_mode": "full",
  "target": ".synix/releases/local/current/search.db",
  "created_at": "2026-03-06T12:31:02Z"
}
```

## Snapshot vs Ref

A snapshot and a ref solve different problems:

- `snapshot`
  - immutable build state
- `ref`
  - movable human-meaningful name pointing to a snapshot

The relationship is:

```text
snapshot = what was built
ref      = what Synix should mean by default right now
```

Git analogy:

```text
commit ~= snapshot
branch ~= ref
HEAD   ~= current ref
```

## Refs and Ergonomics

Example:

```text
HEAD -> refs/heads/customer-memory
refs/heads/customer-memory -> snapshot S42
refs/runs/2026-03-06T12:30Z -> snapshot S42
refs/releases/prod -> snapshot S40
refs/releases/canary -> snapshot S42
```

This means:

- `HEAD` is the default build line
- `refs/runs/...` gives immutable run history
- release refs show what is actually materialized

Build and release are intentionally separate:

- `synix build` advances the active build ref
- `synix release` advances a chosen release ref

## Projections As Build Targets

Projections are the build targets that `synix release` materializes.

The hierarchy is:

```text
sources/transforms -> artifacts -> projections -> release
```

Artifacts are build data. Projections are the externally usable outputs.

Examples:

- flat file
- SQLite search index
- future Postgres target
- future Qdrant or Neo4j target

## Projection Lifecycle

Transforms and projections should not share the same lifecycle contract.

Transforms:

- produce artifacts

Projections:

- build projection state
- release projection state
- reconcile current release to target state
- verify release

Recommended split:

```text
ProjectionBuilder
- build(...)
- diff(...)

ProjectionAdapter
- inspect_release(...)
- plan_release(...)
- apply_release(...)
- verify_release(...)
```

Synix may collapse those into fewer types later, but the lifecycle phases must remain explicit.

## Build, Release, and Revert

### Build

`synix build` should:

1. compute changed artifacts incrementally
2. compute projection build state
3. write immutable objects
4. write a manifest object
5. write a snapshot object
6. move a build ref to the new snapshot

### Release

`synix release <ref>` should:

1. resolve a ref to a snapshot
2. load the manifest
3. select projections
4. dispatch to projection adapters
5. materialize usable targets
6. write release receipts
7. optionally advance a release ref

### Revert

`synix revert <ref>` should be a thin wrapper over release of an older snapshot.

Revert should not mean “replay inverse ops.”

Revert means:

- resolve the older target snapshot
- reconcile current released state to that target

## ASCII: Build Model

```text
sources/transforms
        |
        v
    artifacts
        |
        v
projection build state
        |
        v
     manifest
        |
        v
     snapshot
        |
        v
       refs
```

## ASCII: Release Model

```text
          HEAD
           |
           v
   refs/heads/main
           |
           v
       snapshot
           |
           v
       manifest
        /    \
       v      v
 artifacts  projections
               |
               v
      projection adapter plan
               |
               v
           release target
               |
               v
         release receipt
```

## Automated Pipeline Workflow

### Initial Build

```bash
synix build
```

Result:

```text
artifacts built
projections built
manifest M1 written
snapshot S1 written
refs/heads/main -> S1
refs/runs/2026-03-06T12:30Z -> S1
HEAD -> refs/heads/main
```

Then:

```bash
synix release HEAD --to refs/releases/prod
```

That materializes the build targets and updates the chosen release ref.

### Incremental Rebuild

New source data arrives:

```bash
synix build
```

Incremental build behavior:

- only changed source artifacts rebuild
- only affected downstream artifacts rebuild
- unchanged artifacts are reused
- new projection build state is computed
- a new manifest and snapshot are written

Result:

```text
S1 = older snapshot
S2 = new snapshot
refs/heads/main -> S2
refs/runs/2026-03-06T12:45Z -> S2
refs/releases/prod -> S1
```

Then:

```bash
synix diff refs/releases/prod HEAD
synix release HEAD --to refs/releases/prod
```

### Multi-Variation Release

This model supports multiple release targets:

```text
refs/heads/main -> S42
refs/releases/prod -> S40
refs/releases/canary -> S42
refs/releases/ab-a -> S41
refs/releases/ab-b -> S42
```

Example commands:

```bash
synix release HEAD --to refs/releases/canary
synix release refs/runs/2026-03-05T12:30Z --to refs/releases/ab-a
synix release HEAD --to refs/releases/ab-b
```

Clients then choose the release ref they want to read.

## Incremental Build vs Incremental Release

There are two different incremental stories:

### Incremental Build

- only changed artifacts recompute
- this is already broadly aligned with current Synix behavior

### Incremental Release

- only changed projection state is applied to the live target
- this must be owned by the projection adapter

Example:

- one new conversation arrives
- build computes one new summary artifact
- snapshot `S2` records the new logical state
- release adapter decides whether to:
  - rebuild a whole SQLite file and atomically swap it
  - append just one new index entry later
  - apply upserts or migrations in a future Postgres adapter

Synix core should not force one reconciliation strategy across all projections.

## Adapter Reconciliation Contract

Synix core should own:

- refs
- snapshots
- manifests
- release orchestration
- receipts
- diffing old vs target logical state

Projection adapters should own:

- full rebuild vs incremental apply
- shadow swap vs upsert vs migration
- target-specific rollback and verification

Core primitive:

```text
reconcile(current_release_state, target_projection_state) -> plan
apply(plan) -> receipt
```

Not:

```text
apply op 1
revert op 1
apply op 2
revert op 2
...
```

That second model leaks projection internals into the platform and should be avoided.

## Projection Semantics By Type

### Flat File

- build: final bytes
- release: copy or symlink
- revert: re-copy older bytes

### SQLite Search Index

- build: index image or equivalent build state
- release: full atomic swap initially
- later: optional incremental reconcile

### Future Postgres Target

- build: logical desired state, delta plan, or migration plan
- release: adapter applies incremental change set
- revert: adapter reconciles back to older target state

The RFC intentionally does not require incremental release for every projection in the first implementation.

## Streaming / Near-Real-Time Compatibility

This model still works if incremental rebuilds move closer to streaming updates.

The model becomes:

```text
event stream -> incremental builder -> snapshots -> release refs -> clients
```

For higher-rate streaming, Synix may later choose:

1. micro-snapshots
2. streaming build refs plus periodic durable snapshots

That is a policy layer on top of the same snapshot/ref model. The model remains valid as long as build refs, snapshot refs, and release refs remain distinct.

## Notebook / Mutable Overlay

This RFC does not implement notebook semantics, but the design intentionally leaves room for them.

The intended model is:

- immutable build lane
  - artifacts
  - projections
  - snapshots
  - release refs
- mutable runtime lane
  - notebook overlay
  - append-only journal
  - periodic fold-back into immutable snapshots

Notebook writes should bypass full snapshot/projection/release machinery, but not bypass provenance, audit logging, scoping, or checkpoint semantics.

That design should be handled in a separate RFC.

## Compatibility and Migration

The first implementation should prioritize an incremental migration path:

1. Add `.synix` as the canonical store and ref namespace.
2. Keep existing CLI ergonomics where possible by resolving `HEAD` by default.
3. Support file-backed release for the projections Synix already has.
4. Preserve enough compatibility that a user can adopt the new model without destructive migration.

The implementation strategy may choose a temporary bridge layer from the old `build/` layout to the new `.synix` store, but the target mental model should be the one in this RFC.

## Command Semantics

### Build

```bash
synix build
```

- builds artifacts incrementally
- computes projection build state
- writes manifest and snapshot objects
- advances the active build ref
- prints the new snapshot and run ref

### Show / Search / List

By default these should resolve `HEAD`:

```bash
synix list
synix show ep-conv-001
synix search "anthropic"
```

They should also be able to target a specific ref or snapshot:

```bash
synix list --ref refs/runs/2026-03-06T12:30Z
synix search "anthropic" --ref refs/releases/prod
```

### Diff

```bash
synix diff HEAD refs/releases/prod
```

The long-term model should diff refs or snapshots, not mutable build directories.

### Release

```bash
synix release HEAD --to refs/releases/prod
synix release refs/runs/2026-03-06T12:30Z --to refs/releases/canary
```

### Revert

```bash
synix revert refs/runs/2026-03-06T12:30Z --to refs/releases/prod
```

This should be equivalent to release of an older target.

## First Implementation Scope

1. object store under `.synix/objects`
2. first-class refs and `HEAD`
3. immutable manifest and snapshot objects
4. build writing snapshots and moving build refs
5. file-backed projection release
6. release receipts
7. ref/snapshot diffing
8. adapter interface that leaves room for incremental reconcile later

## Out of Scope For First Implementation

- generic notebook overlay
- remote object storage
- true incremental release for every projection type
- branching UX beyond refs
- external database adapters
- projection DAG redesign beyond what is necessary for release

## Test Strategy

The snapshotting feature must match Synix’s existing test discipline:

- `tmp_path` for all filesystem tests
- `CliRunner` for CLI behavior
- mocked LLMs in unit and integration tests
- no shared state
- explicit failure-mode coverage
- every functional behavior change gets e2e coverage

### Unit Tests

Add:

- `tests/unit/test_object_store.py`
  - object write and read roundtrip
  - stable oid computation
  - invalid object rejection

- `tests/unit/test_refs.py`
  - symbolic `HEAD`
  - direct ref resolution
  - missing ref failure
  - atomic ref move

- `tests/unit/test_manifest_snapshot.py`
  - manifest closure serialization
  - snapshot creation
  - parent snapshot linkage

- `tests/unit/test_release_receipts.py`
  - release receipt schema
  - ref and snapshot provenance fields
  - target metadata persistence

- `tests/unit/test_projection_release_contract.py`
  - file-backed adapter planning
  - file-backed adapter apply
  - verify and receipt behavior

- `tests/unit/test_legacy_layout_resolver.py`
  - old mutable layout detection
  - `HEAD` default resolution
  - compatibility lookup during migration

### Integration Tests

Add:

- `tests/integration/test_snapshot_build.py`
  - build writes objects, manifest, snapshot, and refs

- `tests/integration/test_incremental_snapshot_rebuild.py`
  - only affected artifacts rebuild when one source changes
  - unchanged artifacts are reused
  - a new snapshot is created

- `tests/integration/test_projection_release_local.py`
  - release file-backed projections from `HEAD`

- `tests/integration/test_revert_release.py`
  - release newer snapshot, then revert to an older snapshot

- `tests/integration/test_multi_release_refs.py`
  - `prod`, `canary`, and A/B release refs can diverge safely

### End-to-End Tests

Add:

- `tests/e2e/test_snapshot_flow.py`
  - build -> build again -> diff -> release -> revert

Update:

- `tests/e2e/test_demo_flow.py`
  - stop assuming mutable root-level `manifest.json`, `provenance.json`, and `search.db`
  - assert ref and snapshot semantics instead

- other affected demo e2e tests that assume direct mutable build-root layout

### Must-Have Failure Tests

- failed build does not advance the build ref
- failed release does not advance the release ref
- older snapshots remain readable after later builds
- revert to an older ref works after a newer release
- multiple release refs do not interfere with each other
- release receipts are still written or rolled back consistently on partial failures

## Documentation Deliverables

Before closing the implementation:

- update `docs/entity-model.md`
- update CLI docs and examples
- update pipeline API docs if build/release commands change usage semantics
- record demo/template follow-ons for snapshot/release behavior

## Open Questions

These should be answered during implementation design, but do not block approval of the model:

1. Should build automatically materialize file-backed projections into cached build state, or should release handle all final target materialization?
2. How much compatibility should be preserved for legacy `build/` root lookup during the transition?
3. Should `synix diff` accept refs first and legacy build dirs second, or should both remain long term?

## Decision Summary

This RFC recommends:

- first-class `HEAD` and refs
- immutable snapshots
- manifests as closures over artifacts and projections
- projections as build targets with distinct lifecycle semantics
- separate build and release phases
- adapter-owned reconciliation logic
- platform-owned release and revert semantics

That is the cleanest long-term substrate for Synix and the right platform model for LENS.
