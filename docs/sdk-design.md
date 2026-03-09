# Synix SDK Design

## Overview

A Python SDK that provides programmatic access to synix as an infrastructure layer. The "boto3 for synix" — covers both the data plane (consume releases, search, retrieve artifacts) and the control plane (manage sources, build pipelines, create releases).

This is **not** a new projection type. It's an ergonomic layer over existing primitives (`SnapshotView`, release engine, search retriever, object store) that makes synix directly embeddable in agent runtimes, benchmarks, and any system that uses synix as a processing substrate.

## Primary Use Cases

### 1. unified-memory (Agent Runtime)

An agent runtime that ingests ChatGPT/Claude exports and live session logs, processes them through a synix pipeline, and exposes two outputs:

- **Core memory artifact** — injected into every system prompt (Letta model)
- **Search index** — RAG tool for the agent

```python
import synix

mem = synix.open_project("~/unified-memory").release("local")

# System prompt: inject core memory directly from artifact
system_prompt = f"""You are Mark's assistant.

## Core Memory
{mem.artifact("core").content}

## Current Status
{mem.artifact("work-status").content}
"""

# RAG tool
def search_memory(query: str) -> list[dict]:
    return [
        {"content": r.content, "source": r.label, "score": r.score}
        for r in mem.search(query, limit=5)
    ]
```

### 2. LENS (Benchmark Harness)

A benchmark that evaluates memory systems by streaming episodes and scoring retrieval at checkpoints. Needs programmatic control over the full lifecycle.

```python
import synix

# Generate benchmark dataset
project = synix.open_project(f"datasets/scopes/{scope_id}")
project.build(pipeline="pipeline.py", concurrency=8)

# Evaluate a synix-backed memory adapter
project = synix.open_project(adapter_workspace)
release = project.release("eval")
results = release.search(question, limit=budget)
core = release.artifact("core-memory").content
```

## SDK v1 Scope

### Entry Point

```python
project = synix.open_project(path)
```

Finds `.synix/` in or above `path`. Returns a `Project` handle. Lazy — nothing loads until accessed.

### Data Plane (Release Consumption)

```python
release = project.release("local")          # named release (materialized)
release = project.release("HEAD")           # scratch realization (ephemeral)

# Artifact access — the core primitive
artifact = release.artifact("core")         # → Artifact object
artifact.content                            # → str (the text)
artifact.metadata                           # → dict
artifact.provenance                         # → list[str] (parent labels)
artifact.layer                              # → str (layer name)
artifact.artifact_type                      # → str

# Search
results = release.search("query")
results = release.search("query", mode="semantic", limit=10)

# Named index (when multiple search surfaces exist)
idx = release.index("memory-search")
idx.search("query", mode="hybrid")

# Browse
for layer in release.layers():
    layer.name                              # "episodes"
    layer.level                             # 1
    layer.count                             # 1719
    for a in layer.artifacts():             # → iterator[Artifact]
        ...

# Lineage
chain = release.lineage("monthly-2025-06")  # → list[Artifact]

# Materialized flat files (when you need the file, not the artifact)
path = release.flat_file_path("context-doc")  # → Path to .md on disk
content = release.flat_file("context-doc")    # → str (convenience)
```

**Artifact vs flat file**: Artifacts are the core primitive. `release.artifact("core").content` gives you the text directly — no projection needed. Flat files exist for external consumers that need a file on disk. The SDK treats artifacts as first-class; flat files are a convenience accessor over materialized projections.

### Control Plane (Pipeline Management)

```python
# Source management — abstract away the filesystem
source = project.source("exports")          # named source from pipeline
source.add("./new-claude-export.json")      # copy/link into source dir
source.add_text("raw notes", label="meeting-2026-03-08")  # create from string
source.list()                               # → list of source files

# Build
result = project.build()                    # full pipeline build
result = project.build(dry_run=True)        # plan only
result.built                                # 7
result.cached                               # 1712
result.total_time                           # 42.3 (seconds)
result.snapshot_oid                          # "abc123..." (None if dry_run)

# Release
receipt = project.release_to("local")       # materialize HEAD → named release (returns dict)
receipt["adapters"]                         # per-projection adapter receipts
receipt["snapshot_oid"]                     # released snapshot

# Inspect
project.releases()                          # → ["local", "staging"]
project.refs()                              # → dict of ref → snapshot OID
release.receipt()                           # → release receipt dict
```

### Design Principles

1. **Artifacts are the core primitive** — everything flows through artifacts. Flat files and search indices are projections of artifacts, not separate concepts.

2. **Lazy everything** — `synix.open_project()` finds the path. `project.release()` resolves the ref. Nothing loads into memory until you access `.content`, `.search()`, etc.

3. **Sources are managed, not filesystem ops** — `source.add()` abstracts file placement. You hand the SDK a document, it puts it in the right place. Whether that's a directory, a database, or a remote store is an implementation detail.

4. **Releases are the read boundary** — all data plane reads go through a release. This ensures consistency: you're always reading from a known-good snapshot.

5. **HEAD = scratch** — `project.release("HEAD")` creates an ephemeral realization for dev/eval. No receipt, no ref advancement. Discarded after use.

6. **Extensible** — new projection adapters automatically become accessible via the SDK. A future Postgres adapter would surface as `release.index("pg-search").search(...)`.

## SDK v2 Scope (Stateful Artifacts / Buffer)

Deferred but designed-for. The fast write path for live agent runtimes.

### Concept

Two memory tiers with different write semantics:

- **Derived artifacts** (v1) — pure function of inputs. Rebuild through pipeline. Slow, high quality. Core memory, episodes, rollups.
- **Stateful artifacts** (v2) — append/update semantics. Instant writes, immediately queryable. Agent scratchpad, working memory, observations. Synix tracks versions but doesn't rebuild through the pipeline.

### Sketch

```python
# Fast write — no pipeline, no LLM, instant
project.buffer.write(
    "User prefers window seats on flights",
    tags=["preference", "travel"],
)
project.buffer.write(
    "Booked flight to SFO for March 15",
    tags=["event", "travel"],
)

# Immediately queryable
project.buffer.search("travel preferences")

# Buffer state
project.buffer.list(tags=["preference"])    # → filtered entries
project.buffer.count()                      # → int

# Pipeline can optionally fold buffer into next build
# Buffer entries become source artifacts → flow through DAG
project.build(include_buffer=True)
```

### Storage

Buffer lives in `.synix/buffer/` — append-only JSONL or SQLite. Indexed for instant search (FTS5). Separate from the immutable object store. On `build(include_buffer=True)`, entries are snapshot into sources and flow through the normal pipeline.

### Letta Model Mapping

| Letta Concept | Synix Equivalent |
|--------------|-----------------|
| Core memory (system prompt block) | `release.artifact("core").content` |
| Archival memory (long-term search) | `release.search(query)` |
| Recall memory (conversation buffer) | `project.buffer.search(query)` |
| Archival insert | `project.buffer.write(text, tags)` |
| Core memory update | Pipeline rebuild → new core artifact |

## Implementation Notes

### Module Location

`src/synix/sdk.py` — single module in the monorepo. Imports from existing internals:
- `synix.build.snapshot_view.SnapshotView` — artifact reads
- `synix.build.refs.RefStore` — ref resolution
- `synix.build.release_engine.execute_release` — release materialization
- `synix.search.retriever` — search queries
- `synix.build.runner` — pipeline execution
- `synix.build.object_store.ObjectStore` — content-addressed reads

### Public API Surface

```python
# Top-level
synix.open_project(path: str | Path) -> Project

# Project
Project.release(name: str) -> Release
Project.source(name: str) -> Source
Project.build(**kwargs) -> BuildResult
Project.release_to(name: str, ref: str = "HEAD") -> ReleaseReceipt
Project.releases() -> list[str]
Project.refs() -> dict[str, str]

# Release
Release.artifact(label: str) -> Artifact
Release.artifacts(layer: str | None = None) -> Iterator[Artifact]
Release.search(query: str, **kwargs) -> list[SearchResult]
Release.index(name: str) -> SearchIndex
Release.layers() -> list[Layer]
Release.lineage(label: str) -> list[Artifact]
Release.flat_file(name: str) -> str
Release.flat_file_path(name: str) -> Path
Release.receipt() -> dict

# Source
Source.add(path: str | Path) -> None
Source.add_text(content: str, label: str) -> None
Source.list() -> list[str]

# Artifact (dataclass, read-only)
Artifact.content: str
Artifact.metadata: dict
Artifact.provenance: list[str]
Artifact.layer: str
Artifact.artifact_type: str
Artifact.label: str
Artifact.artifact_id: str
```
