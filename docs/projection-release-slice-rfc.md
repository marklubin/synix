# RFC: Snapshot, Projection, and Release Closeout

**Related issues**: [#34](https://github.com/marklubin/synix/issues/34), [#81](https://github.com/marklubin/synix/issues/81), [#82](https://github.com/marklubin/synix/issues/82), [#10](https://github.com/marklubin/synix/issues/10)  
**Status**: Draft  
**Baseline**: `main` after [#91](https://github.com/marklubin/synix/pull/91)  
**Decision target**: Design approval before implementation

## Thesis

Synix should finish the snapshot/projection/release feature by making the separation absolute:

- `synix build` only commits immutable state into `.synix`
- `synix release` is the only command that materializes projections anywhere
- `build/` is not a platform concept
- default reads resolve from immutable refs, not a mutable directory

After this lands, snapshot/projection/release should be considered complete. The next execution-critical work should be sealed checkpoint banks, mounted runtime tools, and built-in artifact families, not another pass on build-vs-release semantics.

## Current Baseline

What exists on `main` today:

- `.synix/` stores immutable artifact objects, manifests, snapshots, and refs
- `HEAD`, `refs/heads/*`, and `refs/runs/*` exist
- `SearchSurface` is the build-time search capability
- `SynixSearch` is the canonical search output contract

What is still missing:

- projection state is not yet part of canonical snapshot closure
- there is no explicit release lifecycle
- there are no release refs or release receipts
- diff is still closer to build-root comparison than ref-first comparison
- revert is not yet a first-class release operation
- most CLI paths still assume mutable local build-root state somewhere in the flow

## Problem

Today Synix still has a split-brain model:

```text
artifacts   -> immutable state in .synix
projections -> mutable realized files
```

That model is not clean enough for:

- reproducible inspection of old pipeline states
- promotion and rollback of search/context outputs
- A/B evaluation over named build states
- CI/CD pipelines that build once and release later
- immutable checkpoint banks

If `build` also implies local materialization, Synix is still mixing:

1. canonical build state
2. release state
3. local convenience state

This RFC removes that ambiguity instead of papering over it.

## Design Principles

1. **Build is pure with respect to releases.**  
   `synix build` writes immutable state and moves build refs. It does not mutate release targets.

2. **Release is explicit.**  
   `synix release` materializes one chosen snapshot into one chosen target and moves a release ref only on success.

3. **Immutable refs are the default inspection surface.**  
   If a user asks to inspect the pipeline state, Synix resolves `HEAD` unless another ref is specified.

4. **Temporary realizations are internal, not releases.**  
   Search or verify may stage scratch state to answer a query, but that must not move a release ref or overwrite a durable target.

5. **`build/` is not special.**  
   If a user wants files in `./build`, that is just one possible `--target-root`, invoked explicitly.

6. **Checkpoint banks build on this model; they do not redefine it.**

## Ergonomics: Before vs After

### Before

```bash
synix build
# writes artifacts and mutable projections somewhere locally

synix list
synix show core
synix search "pricing"
synix verify
```

Mental model:

- local realized files are treated as current truth
- build and release semantics are blurred together
- old states are awkward to inspect unless extra directories are preserved
- promotion and rollback are conventions rather than first-class lifecycle

### After

```bash
synix build
# writes only immutable snapshot state into .synix

synix list
synix show core
synix diff HEAD refs/runs/<older>
synix verify --ref HEAD

synix release HEAD --to refs/releases/local --target-root ./.synix/releases/local/current
synix release HEAD --to refs/releases/prod --target-root ./releases/prod
synix revert refs/runs/<older> --to refs/releases/prod
```

Mental model:

- build creates immutable named states
- inspect, diff, and verify operate over refs
- release is the only thing that materializes outputs
- revert is release of an older snapshot

The user pays a little more explicitness and gets a much cleaner lifecycle in return.

## Pipeline Management Flow

The intended day-to-day operator loop becomes:

1. build immutable state
2. inspect and verify that immutable state
3. release one chosen snapshot to one chosen environment
4. promote or revert by naming refs, not by rebuilding or copying files

Example:

```bash
synix build
synix list --ref HEAD
synix show core --ref HEAD
synix verify --ref HEAD

synix release HEAD --to refs/releases/local --target-root ./.synix/releases/local/current
synix refs show refs/releases/local

synix release HEAD --to refs/releases/staging --target-root /srv/synix/staging
synix revert refs/runs/20260306T230000Z-prev --to refs/releases/staging
```

What changes for the user:

- a build never implies a deployment
- a deployment always points back to one immutable snapshot
- promotion, rollback, and audit all use the same ref vocabulary

## State Model

Synix should distinguish five kinds of state:

```text
artifacts                 immutable canonical records
projection build state    immutable adapter input stored in snapshots
build refs                pointers to built states (HEAD, refs/heads/*, refs/runs/*)
release refs              pointers to materially released states (refs/releases/*)
temporary realizations    scratch materializations used for commands like search
```

The key rule is:

- snapshots own artifacts and projection build state
- release refs own deployment/materialization intent
- temporary realizations are disposable implementation detail

## Object Model

The existing object store already has the right nouns:

- `projection`
- `manifest`
- `snapshot`
- `release_receipt`

This RFC makes them operational.

### Projection Object

Each projection in `pipeline.projections` produces one immutable projection object:

```json
{
  "type": "projection",
  "schema_version": 1,
  "name": "search",
  "projection_type": "synix_search",
  "input_oids": ["oid_art_1", "oid_art_2"],
  "build_state_oid": "oid_proj_state_1",
  "adapter": "synix_search",
  "release_mode": "full",
  "metadata": {
    "config_fingerprint": "sha256:...",
    "source_layers": ["episodes", "core"],
    "target_template": "outputs/search.db"
  }
}
```

`build_state_oid` is the immutable adapter input for release. It may be adapter-specific and opaque, but the projection object must also expose a small stable metadata summary suitable for generic diff and inspection.

`SearchSurface` is not a projection object. It remains a build-time capability.

### Manifest Object

The manifest becomes a complete closure:

```json
{
  "type": "manifest",
  "schema_version": 1,
  "pipeline_name": "personal-memory",
  "pipeline_fingerprint": "sha256:...",
  "artifacts": {
    "core": "oid_art_3"
  },
  "projections": {
    "search": "oid_proj_1",
    "context": "oid_proj_2"
  }
}
```

### Release Receipt

Every successful release writes one receipt per released projection:

```json
{
  "type": "release_receipt",
  "schema_version": 1,
  "ref": "refs/releases/prod",
  "resolved_snapshot_oid": "oid_snapshot_1",
  "manifest_oid": "oid_manifest_1",
  "projection_oid": "oid_proj_1",
  "adapter": "synix_search",
  "release_mode": "full",
  "target_root": "./releases/prod",
  "target": "./releases/prod/outputs/search.db",
  "created_at": "2026-03-07T01:00:00Z"
}
```

The latest receipt for a release ref is the binding record for where that ref materializes by default.

## Build Semantics

`synix build` should do exactly one thing:

1. build or reuse artifacts incrementally
2. compute projection build state
3. write artifact objects
4. write projection objects
5. write the manifest and snapshot
6. advance `HEAD` / `refs/heads/*`
7. mint a new `refs/runs/*`

What `build` does **not** do:

- it does not materialize a local search DB
- it does not write a local context file
- it does not move any release refs

This is the central cleanup.

## Read Semantics

### Snapshot-Native Reads

These commands should read directly from `.synix` and default to `HEAD`:

- `synix list`
- `synix show`
- `synix lineage`
- `synix diff`

All should accept `--ref`.

This requires a snapshot-resolved read abstraction, for example:

```python
view = SnapshotView.open(ref="HEAD")
view.list_artifacts()
view.get_artifact("core")
view.get_manifest()
```

Artifact browsing should not require any temporary materialization.

### Search And Verify Reads

Search and parts of verify operate on projection state rather than artifact state. They should have two allowed execution paths:

1. query canonical immutable projection state directly if the projection adapter supports it
2. otherwise stage an isolated scratch realization under `.synix/work/...`, use it, and discard it

Examples:

```bash
synix search "pricing" --ref HEAD
synix search "pricing" --ref refs/runs/<older>
synix verify --ref HEAD
```

The important rule is that these commands must not silently mutate a durable release target just to answer a query.

### Release-Aware Reads

If a user wants to inspect what is currently live in a named environment, they should do that explicitly:

```bash
synix refs show refs/releases/prod
synix search "pricing" --release refs/releases/prod
```

`--ref` means "inspect immutable build state."  
`--release` means "inspect a named durable materialization."

## Release Semantics

`synix release` should:

1. resolve one source ref
2. load its immutable projection objects
3. resolve one destination release ref
4. resolve one target root
5. compute a release plan
6. apply all selected projection updates
7. write receipts
8. move the destination release ref only after the whole release succeeds

Release is the only command that should create or update durable realized files.

### Target Mapping

The destination release ref binds to a target root, not to one hard-coded file.

Each projection preserves its configured relative target layout under that root.

Example:

- projection metadata says `target_template = "outputs/search.db"`
- `--target-root ./releases/prod`
- effective target becomes `./releases/prod/outputs/search.db`

Nothing in the platform treats `./build` differently from any other explicit target root.

### Release Transactions

Multi-projection releases need an explicit journal:

1. write a pending release transaction record
2. apply projection adapter plans
3. write per-projection receipts
4. move the release ref
5. mark the transaction complete

This is required for operationally credible all-or-nothing release semantics.

### Locking

Any durable release target needs one lock domain across:

- `synix release`
- `synix revert`
- any other command that writes durable target state for that release ref

Scratch realizations used for `search --ref` or `verify --ref` must use isolated temporary roots so they never contend with named durable release targets.

## Ref Management

Refs should be explicit and inspectable:

- `synix refs list`
- `synix refs show <ref>`
- `synix runs list`

The important split is:

- build refs tell the user what immutable states exist
- release refs tell the user what is materially live where

That split is what makes promotion and rollback easy to reason about.

## Diff Semantics

`synix diff` should become ref-first rather than build-dir-first.

Examples:

```bash
synix diff HEAD refs/runs/<older>
synix diff HEAD refs/runs/<older> --artifact core
synix diff HEAD refs/runs/<older> --projection search
```

Projection diff cannot stop at "build_state_oid changed." The generic minimum diff surface should include:

- projection type
- source layers
- input artifact count
- target template
- adapter-defined summary fields

Adapters may add richer diffs later, but the generic diff must already be informative.

## Revert Semantics

`synix revert <older-ref> --to <release-ref>` is just explicit release of an older immutable snapshot to the same destination.

This keeps the model simple:

- `build` creates immutable state
- `release` materializes immutable state
- `revert` re-releases older immutable state

No inverse filesystem operation log is required.

## Major User Stories

### 1. Incremental Builds

Goal:
compile only what changed while keeping every checkpoint inspectable.

Operator flow:

```bash
synix build
synix diff HEAD refs/runs/20260306T230000Z-prev
synix list --ref HEAD
synix show monthly --ref HEAD
```

How it should work:

- incremental build logic reuses prior immutable objects wherever fingerprints match
- the new snapshot records both artifact state and projection build state
- the user can compare the new snapshot against an older run without touching any mutable live directory

Why this matters:

- checkpoint compilation becomes auditable
- resume and replay semantics are simpler because build and release are separate concerns
- cost accounting can key off immutable refs

### 2. CI/CD Promotion

Goal:
build once in CI, then promote that exact snapshot through staging and production.

Operator flow:

```bash
synix build
synix verify --ref HEAD
synix release HEAD --to refs/releases/staging --target-root /srv/synix/staging
synix release HEAD --to refs/releases/prod --target-root /srv/synix/prod
```

How it should work:

- CI creates an immutable run ref and records verification against that ref
- CD promotes the same ref rather than rebuilding
- release receipts tell operators exactly what snapshot each environment is serving

Why this matters:

- promotion becomes reproducible
- failure analysis no longer depends on preserving a mutable workspace
- external systems can cite immutable refs instead of directories

### 3. A/B Testing

Goal:
serve two candidate releases side by side and compare behavior without rebuilding either one.

Operator flow:

```bash
synix release refs/runs/20260306T230000Z-a --to refs/releases/ab-a --target-root /srv/synix/ab/a
synix release refs/runs/20260306T231500Z-b --to refs/releases/ab-b --target-root /srv/synix/ab/b
synix refs show refs/releases/ab-a
synix refs show refs/releases/ab-b
```

How it should work:

- each release ref points to one immutable source snapshot
- each target root is isolated
- diff and verification compare the underlying immutable refs rather than guessing from current files

Why this matters:

- evaluation does not require recompilation
- provenance is preserved for each side of the test
- promotion from canary to prod becomes explicit release, not ad hoc copying

### 4. Immutable Checkpoints

Goal:
treat each checkpoint-scoped memory bank as a sealed, auditable object that can later be mounted by the runtime.

Operator flow:

```bash
synix build --checkpoint ckpt-042
synix refs show HEAD
synix release HEAD --to refs/releases/checkpoints/ckpt-042 --target-root /srv/synix/checkpoints/ckpt-042
```

How it should work:

- the checkpoint build commits an immutable snapshot over the visible prefix only
- release produces sealed projection outputs and receipts for that checkpoint
- later bank manifests can point at that immutable snapshot and its released outputs without ambiguity

Why this matters:

- checkpoint isolation is enforced at build time
- runtime mounting can consume immutable bank manifests instead of mutable directories
- later retrieval and tool contracts have one clear substrate

## Acceptance Criteria

Snapshot/projection/release should be considered complete only when Synix has:

- canonical projection objects recorded in manifests and snapshots
- explicit `synix release`
- release refs plus `synix refs list/show`
- release receipts
- ref-first `diff`
- explicit `revert`
- snapshot-native reads for artifact inspection
- isolated scratch realizations for `search --ref` and `verify --ref`
- no special semantic dependency on `build/`

After that, the next execution-critical path is:

1. sealed checkpoint banks and bank manifests
2. Python-local runtime/tool API, including retrieval over named search surfaces
3. built-in chunk family
4. built-in summary, core-memory, and graph families

That should be treated as new work on top of this substrate, not as unfinished snapshot semantics.
