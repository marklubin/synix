# Synix Architecture

## Core Concepts

- **Artifact** — Immutable, versioned build output. Content-addressed via SHA256.
- **Layer** — Typed Python object in the build DAG.
- **Pipeline** — Declared in Python. Sources → Transforms → Projections.
- **Release** — Materialized projections from a snapshot.

## Build Pipeline

```
Sources → EpisodeSummary → FoldSynthesis (5 layers) → CoreSynthesis → Search + FlatFile
```

## Search

Hybrid search: FTS5 keyword + fastembed semantic + RRF fusion.
