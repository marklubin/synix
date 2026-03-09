# Receipt JSON Schema

Every `synix release` writes a receipt to `.synix/releases/<name>/receipt.json`. This document specifies the schema.

## Schema Version

Current version: **1**

Readers must check `schema_version` before parsing. Breaking changes to the receipt format will bump the schema version. Non-breaking additions (new optional fields) will not bump the version.

## ReleaseReceipt

Top-level receipt written by the release engine after all adapters succeed.

```json
{
  "schema_version": 1,
  "release_name": "local",
  "snapshot_oid": "aabb1234...",
  "manifest_oid": "ccdd5678...",
  "pipeline_name": "personal-memory",
  "released_at": "2026-03-07T10:00:00+00:00",
  "source_ref": "HEAD",
  "adapters": {
    "<projection-name>": { ... }
  }
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | `int` | Receipt schema version. Currently `1`. |
| `release_name` | `string` | Name of the release target (e.g., `"local"`, `"prod"`). |
| `snapshot_oid` | `string` | OID of the snapshot that was released. |
| `manifest_oid` | `string` | OID of the manifest within the snapshot. |
| `pipeline_name` | `string` | Name of the pipeline that produced the snapshot. |
| `released_at` | `string` | ISO 8601 timestamp (UTC) of when the release completed. |
| `source_ref` | `string` | The ref that was resolved to obtain the snapshot (e.g., `"HEAD"`, `"refs/heads/main"`). |
| `adapters` | `object` | Map of projection name to `AdapterReceipt`. One entry per projection in the manifest. |

## AdapterReceipt

Per-projection receipt written by each adapter after `apply()` succeeds.

```json
{
  "adapter": "synix_search",
  "projection_name": "search",
  "target": ".synix/releases/local/search.db",
  "artifacts_applied": 3,
  "status": "success",
  "applied_at": "2026-03-07T10:00:00+00:00",
  "details": {}
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `adapter` | `string` | Adapter type that executed the projection (e.g., `"synix_search"`, `"flat_file"`). |
| `projection_name` | `string` | Name of the projection as declared in the manifest. |
| `target` | `string` | Where the adapter wrote its output. For file-based adapters, a file path. For remote adapters, a connection URI. |
| `artifacts_applied` | `int` | Number of artifacts materialized by this adapter. |
| `status` | `string` | `"success"` or `"failed"`. A receipt is only persisted if all adapters succeed. |
| `applied_at` | `string` | ISO 8601 timestamp (UTC) of when this adapter completed. |
| `details` | `object` | Adapter-specific metadata. Contents vary by adapter type. |

### Built-in Adapter Details

**`synix_search`**: The `details` object may include FTS5 row counts, embedding status, and verification checksums.

**`flat_file`**: The `details` object may include content hash of the rendered output.

## Directory Structure

```
.synix/releases/
  <release-name>/
    receipt.json                    # Current (latest) receipt
    .pending.json                   # Present only during an in-progress release
    search.db                       # Materialized by synix_search adapter
    context.md                      # Materialized by flat_file adapter
    history/
      20260307T100000p0000.json     # Historical receipt (timestamp-named)
      20260308T140000p0000.json     # Another historical receipt
```

### `receipt.json`

The current receipt. Overwritten on each successful release to this target.

### `.pending.json`

Written at the start of a release, deleted on success. If present after a release command exits, the release failed mid-flight. Contains the snapshot OID and start timestamp for diagnosis.

### `history/`

Append-only directory of historical receipts. Each file is a full `ReleaseReceipt` snapshot named by its `released_at` timestamp (with colons removed and `+` replaced by `p` for filesystem safety). History is never pruned automatically.

The history filename format: `<released_at_sanitized>.json`

Example: `2026-03-07T10:00:00+00:00` becomes `20260307T100000p0000.json`.

## Compatibility Policy

- `schema_version` will be incremented for breaking changes (removed fields, changed semantics).
- New optional fields may be added without a version bump.
- Readers should check `schema_version` and reject or warn on unknown versions.
- The `from_dict` classmethod on `ReleaseReceipt` uses `.get()` with defaults for forward compatibility.
