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

## Artifact Storage (Filesystem)

```
build/
├── manifest.json           # maps label → file path + metadata
├── provenance.json         # all provenance records
├── layer0-transcripts/
│   ├── abc123.json
│   └── def456.json
├── layer1-episodes/
│   ├── ep-abc123.json
│   └── ep-def456.json
├── layer2-monthly/
│   ├── 2025-01.json
│   └── 2025-02.json
├── layer3-core/
│   └── core-memory.json
└── search.db               # SQLite FTS5 database
```

Each artifact JSON file:
```json
{
  "label": "ep-abc123",
  "artifact_type": "episode",
  "artifact_id": "sha256:...",
  "input_ids": ["sha256:..."],
  "prompt_id": "episode_summary_v1",
  "model_config": {"model": "claude-sonnet-4-20250514", "temperature": 0.3},
  "created_at": "2025-02-07T10:30:00Z",
  "content": "In this conversation, Mark discussed...",
  "metadata": {
    "source_conversation_id": "abc123",
    "date": "2025-01-15",
    "message_count": 24
  }
}
```

## Cache/Rebuild Logic

```python
def needs_rebuild(label, current_input_ids, current_prompt_id):
    existing = store.load_artifact(label)
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

Provenance drill-down: Given a search result's `label`, walk `provenance.json` recursively to get the full chain back to raw transcript.

---

## Architecture North Star (post-demo, do NOT build)

**Projections as build infrastructure.** The same search index that serves user queries can also serve downstream transforms. A topical rollup queries the episode search index for relevant episodes. At scale, this is essential.

**The full vision:** Layers can depend on projections. Projections can compose. The build DAG includes both artifacts and projections as nodes.

**For this weekend:** The runner builds layers in order. By the time topical rollup runs, the episode search index already exists. The transform just queries it. No new DAG concepts needed.

**Future work:**
- Projections as first-class DAG nodes
- Composable projections (merge multiple search indexes)
- Intermediate projections as build acceleration
- Proprietary clustering for auto-topic discovery
