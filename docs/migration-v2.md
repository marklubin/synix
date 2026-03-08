# Migration Guide: Projection Release v2

This guide covers migrating from the pre-v2 `build/`-based workflow to the new `.synix/`-only architecture.

## What Changed

### `build/` directory removed

Synix no longer creates or reads a `build/` directory. Previously, every `synix build` wrote artifacts twice: once to `build/` as `.md` files (via `ArtifactStore`) and once to `.synix/objects/` as immutable blobs. The dual-write is gone. `.synix/` is the single source of truth.

### `ProvenanceTracker` removed

Provenance was tracked in `build/provenance.json`. That file no longer exists. Provenance is now stored directly on artifact objects via `parent_labels` and `input_ids`, walked transitively at query time.

### `.projection_cache.json` deprecated

This cache file is no longer used. Content-addressed dedup in the object store replaces it.

### Projections are now structured declarations

The manifest records each projection's adapter, input artifacts, and config as structured data. Projections are not materialized during build -- they are materialized during release.

## What Is Safe to Delete

| Path | Status |
|------|--------|
| `build/` | Safe to delete. Synix no longer reads it. |
| `.projection_cache.json` | Safe to delete. No longer used. |

Both can be removed immediately. They have no effect on the new workflow.

## New Workflow

### Build (unchanged command, different output)

```bash
synix build pipeline.py
```

This still runs your pipeline. The difference: it writes only to `.synix/objects/` and `.synix/refs/`. No `build/` directory, no user-facing files. Output is a snapshot OID and run ref.

### Release (new step)

```bash
synix release HEAD --to local
```

This materializes a snapshot into a named release target at `.synix/releases/local/`. The release directory contains the search index (`search.db`), flat file outputs (`context.md`), and a receipt (`receipt.json`).

### Search

**Querying a release** (standard):

```bash
synix search "query"
```

Queries the default release. Requires a release to exist first.

**Querying a specific release**:

```bash
synix search "query" --release local
```

**Ad-hoc search without releasing** (scratch realization):

```bash
synix search "query" --ref HEAD
```

This builds a temporary search index in `.synix/work/`, runs the query, and discards the index. No release ref is moved, no receipt is written.

### List, Show, Lineage

These commands read directly from `.synix/` snapshots. No `build/` directory needed.

```bash
synix list [--ref HEAD]
synix show <label> [--ref HEAD]
synix lineage <label> [--ref HEAD]
```

They resolve: `ref -> snapshot -> manifest -> artifact objects -> content blobs`.

## Step-by-Step Migration

1. **Build as before:**

   ```bash
   synix build pipeline.py
   ```

2. **Release to materialize outputs:**

   ```bash
   synix release HEAD --to local
   ```

3. **Use search, list, show as before:**

   ```bash
   synix search "query"
   synix list
   synix show core
   synix lineage core
   ```

4. **Clean up old files (optional):**

   ```bash
   rm -rf build/
   rm -f .projection_cache.json
   ```

## Summary of Command Changes

| Before | After | Notes |
|--------|-------|-------|
| `synix build pipeline.py` | `synix build pipeline.py` | Same command, but no `build/` created |
| (implicit) | `synix release HEAD --to local` | New required step to materialize outputs |
| `synix search "query"` | `synix search "query"` | Now queries a release target |
| `synix list` | `synix list` | Reads from `.synix/` snapshots |
| `synix show <label>` | `synix show <label>` | Reads from `.synix/` snapshots |
| `synix lineage <label>` | `synix lineage <label>` | Reads from `.synix/` snapshots |
