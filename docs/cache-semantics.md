# Cache Semantics

Synix uses a fingerprint-based caching system to determine when artifacts need rebuilding. Every build artifact stores a **build fingerprint** — a self-describing, versioned hash that captures everything that went into producing it.

## Rebuild Trigger Matrix

| Change | Triggers Rebuild? | Fingerprint Component | Notes |
|--------|-------------------|-----------------------|-------|
| Source file content | Yes | `inputs` | Upstream artifact_id changes |
| Prompt template file | Yes | `prompt` | Part of transform fingerprint |
| LLM config (model, temp) | Yes | `model` | Part of transform fingerprint |
| Transform config (topics, budget) | Yes | `config` | Part of transform fingerprint |
| Transform source code | Yes | `source` | Part of transform fingerprint |
| Projection config | Yes | `config` | Part of projection fingerprint |
| New source files added | Yes | `inputs` | New artifacts created |
| Source files removed | No | — | Orphans remain until clean |
| Fingerprint scheme version | Yes | (scheme mismatch) | Auto-invalidates on upgrade |

## Fingerprint Anatomy

A fingerprint has three parts:

```
{
  "scheme": "synix:transform:v1",    // How this fingerprint was generated
  "digest": "a1b2c3d4...",           // SHA256 of all components combined
  "components": {                     // What went into the digest
    "source": "e5f6...",             // Hash of transform source code
    "prompt": "g7h8...",             // Hash of prompt template
    "config": "i9j0...",            // Hash of transform-specific config
    "model": "k1l2..."              // Hash of LLM config
  }
}
```

### Scheme Conventions

| Scheme | Entity | Components |
|--------|--------|------------|
| `synix:transform:v1` | Transform identity | `source`, `prompt`, `config`, `model` |
| `synix:build:v1` | Per-artifact build context | `transform` (digest), `inputs` (sorted input IDs) |
| `synix:projection:v1` | Projection identity | `sources` (artifact IDs), `config` |

### Scheme Versioning

If we add, remove, or change a component in a fingerprint scheme, we bump the version (e.g., `v1` → `v2`). A scheme mismatch is an automatic cache miss — the `Fingerprint.matches()` method enforces this.

## Cache Decision Flow

For each artifact, the runner evaluates:

1. **Artifact doesn't exist** → rebuild (reason: "new artifact")
2. **Build fingerprint provided AND stored fingerprint exists** → compare via `Fingerprint.matches()`:
   - Match → cached
   - Mismatch → rebuild with `explain_diff()` reasons
3. **No stored fingerprint** (pre-upgrade artifact) → rebuild once to populate fingerprint
4. **No build fingerprint provided** (caller not updated) → legacy field comparison fallback

## Using `--explain-cache`

The `synix plan --explain-cache` flag shows a per-layer breakdown of cache decisions:

```bash
uvx synix plan pipeline.py --explain-cache
```

This adds a "Cache Decision Breakdown" table after the normal plan output showing:
- Which layers are cached vs need rebuilding
- Which fingerprint components caused a cache miss
- The fingerprint scheme version and component hashes

## Backward Compatibility

Artifacts built before the fingerprint system was introduced will be automatically rebuilt once to populate their fingerprints. After this one-time rebuild, fingerprint-based caching takes over and provides more precise invalidation (including transform source code changes, which were previously invisible to the cache system).
