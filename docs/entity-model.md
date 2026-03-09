# Entity Model

## Artifact (base concept)

Every output of the build system is an **Artifact**. An artifact is an immutable, versioned build output.

```python
@dataclass
class Artifact:
    label: str                # human-readable semantic name (e.g., "ep-conv-123")
    artifact_type: str        # "transcript", "episode", "rollup", "core_memory", "search_index"
    artifact_id: str          # SHA256 content hash — the true identity
    input_ids: list[str]      # artifact IDs (hashes) of all inputs that produced this
    prompt_id: str | None     # prompt template version (for LLM-derived artifacts)
    model_config: dict | None # model, temperature, etc (for LLM-derived artifacts)
    created_at: datetime
    content: str              # the actual text content
    metadata: dict            # flexible metadata (source file, date range, etc)
```

**Two subtypes exist conceptually (weekend: only derived):**

- **DerivedArtifact** — pure function of inputs. Rebuild from scratch when inputs change. Episode summaries, rollups, core memory. This is what we build this weekend.
- **StatefulArtifact** — append/update semantics. Agent scratchpad, external backends (Mem0, etc). Synix tracks versions but doesn't rebuild. **v0.2 — do NOT implement this weekend.**

## ProvenanceRecord

Every artifact knows its lineage.

```python
@dataclass
class ProvenanceRecord:
    artifact_id: str              # the artifact this record describes (SHA256 hash)
    parent_labels: list[str]      # labels of artifacts that were inputs
    prompt_id: str | None         # prompt used to produce it
    model_config: dict | None     # LLM config used
    created_at: datetime
```

## Layer

A named level in the memory hierarchy. Layers form a DAG.

```python
@dataclass
class Layer:
    name: str                     # "transcripts", "episodes", "monthly", "core"
    level: int                    # 0 = raw, 1 = first derivation, etc.
    depends_on: list[str]         # names of layers this depends on
    transform: str                # name of the transform function/prompt
    grouping: str | None          # how to group inputs ("by_conversation", "by_month", "single")
```

## Pipeline

The full declared memory architecture.

```python
@dataclass
class Pipeline:
    name: str
    layers: list[Layer]
    # The DAG is implicit in Layer.depends_on
```

## Projection

A **Projection** materializes build artifacts into a usable output surface — the "packaging/deploy" step.

```python
@dataclass
class Projection:
    name: str                     # "memory-index", "context-doc"
    projection_type: str          # "search_index", "flat_file", (future: "qdrant", "neo4j", etc.)
    sources: list[ProjectionSource]
    config: dict
```

```python
@dataclass
class ProjectionSource:
    layer: str                    # layer name
    search_modes: list[str]       # ["fulltext", "semantic"]
```

**Two projections for the demo:**

1. **Search Index** (`search_index` type) — SQLite FTS5 with hierarchical search. Every result carries full provenance chain (always on).
2. **Context Document** (`flat_file` type) — renders core memory as a system prompt insert.

The fundamental output of Synix is: **system prompt + RAG.** Context doc = system prompt. Search index = RAG.

---

## Artifact Storage (`.synix/`)

All build state lives under `.synix/`. There is no `build/` directory. The `ObjectStore` is the single write path for all content-addressed objects; the legacy `ArtifactStore` and `ProvenanceTracker` have been removed.

```
.synix/
├── HEAD                    # symbolic ref → refs/heads/main
├── .lock                   # build concurrency lock
├── objects/                # content-addressed blobs (artifacts, manifests, snapshots)
│   ├── aa/bbccdd...
│   ├── 12/345678...
│   └── ff/eeddcc...
├── refs/                   # git-like refs
│   ├── heads/
│   │   └── main            # oid of latest snapshot
│   ├── runs/
│   │   ├── 20260306T...Z   # oid of run 1 snapshot
│   │   └── 20260307T...Z   # oid of run 2 snapshot
│   └── releases/
│       └── local           # oid of snapshot powering local release
├── work/                   # build-time transient data
│   ├── surfaces/           # FTS5 search surfaces for build-time transform queries
│   └── logs/               # build logs (JSONL)
└── releases/               # materialized projection outputs (one dir per release name)
    └── local/
        ├── receipt.json    # what was released, when, which adapters
        ├── search.db       # materialized by synix_search adapter
        └── context.md      # materialized by flat_file adapter
```

Each artifact is stored as two objects in `.synix/objects/`:

**Content blob** — raw text bytes, content-addressed.

**Artifact object:**
```json
{
  "type": "artifact",
  "label": "ep-abc123",
  "artifact_type": "episode",
  "artifact_id": "sha256:...",
  "content_oid": "aabb...",
  "input_ids": ["sha256:..."],
  "parent_labels": ["tx-abc123"],
  "metadata": {
    "layer_name": "episodes",
    "layer_level": 1,
    "source_conversation_id": "abc123",
    "date": "2025-01-15",
    "message_count": 24
  }
}
```

Provenance is embedded in each artifact object via `parent_labels` and `input_ids`. There is no separate `provenance.json` file — provenance chains are walked by resolving `parent_labels` transitively through the object store.

## Cache/Rebuild Logic

Cache decisions use the `SnapshotArtifactCache`, which reads from `.synix/objects/` via `SnapshotView`. The logic compares build fingerprints (inputs, prompt, model config, transform source) against what is stored in the current snapshot.

```python
def needs_rebuild(label, current_input_ids, current_prompt_id):
    existing = cache.load_artifact(label)
    if existing is None:
        return True
    if existing.input_ids != current_input_ids:
        return True
    if existing.prompt_id != current_prompt_id:
        return True
    return False
```

Hash comparison. If any input changed or the prompt changed, rebuild. Otherwise skip.

## Search Index (SQLite FTS5)

```sql
CREATE VIRTUAL TABLE search_index USING fts5(
    content,
    label,
    layer_name,
    layer_level,
    metadata  -- JSON string with dates, source info, etc.
);
```

Query with layer filtering:
```sql
SELECT *, rank FROM search_index
WHERE search_index MATCH ?
AND layer_level >= ?
ORDER BY rank;
```

Provenance drill-down: Given a search result's `label`, walk `parent_labels` transitively through artifact objects in `.synix/objects/` to get the full chain back to raw transcript. Released search.db files also include a `provenance_chains` table with pre-baked lineage for standalone queries.

---

## Architecture North Star

> Note: The projection release v2 design ([docs/projection-release-v2-rfc.md](../docs/projection-release-v2-rfc.md)) is now the current architecture. Build and release are fully separated: `synix build` produces immutable snapshots, `synix release` materializes projections to named targets via adapters.

**Projections as build infrastructure.** `SearchSurface` declarations enable build-time retrieval. Transforms declare `uses=[surface]` to access search during execution. At release time, the `synix_search` adapter materializes a self-contained search.db from the snapshot's `ReleaseClosure`.

**The adapter contract:** Every projection adapter receives a fully resolved `ReleaseClosure` — artifacts with content and provenance already walked. Adapters implement `plan/apply/verify`. The platform does not privilege any adapter; `synix_search` follows the same contract as any external adapter.

**Future work:**
- External database adapters (Postgres, Qdrant) following the same adapter contract
- Checkpoint banks (builds on top of release receipts)
- Runtime tool API (builds on top of portable release directories)
