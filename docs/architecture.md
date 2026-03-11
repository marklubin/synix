# Architecture

Programmable memory for AI agents — but why "programmable"? Because not all memory is the same, and the rules should be different for different kinds.

## The insight

Every other agent memory system gives you one flat namespace. A fact learned 30 seconds ago and a preference built over 50 sessions get the same storage, the same retrieval rules, the same lifecycle. That's like running an operating system with no memory hierarchy — everything in one giant heap.

Real memory has tiers. Different kinds of information move at different speeds, change at different rates, and need different management. A CPU cache, L1/L2/L3, and main memory aren't just "faster and slower" — they have different physics. Agent memory is the same.

## The four tiers

```
┌─────────────────────────────────────────────────┐
│  Tier 0 — Execution    (ms → min)               │
│  What the agent is doing right now.              │
│  Volatile. Attention-routed. Token-budgeted.     │
├─────────────────────────────────────────────────┤
│  Tier 1 — Session      (min → hrs)               │
│  The working conversation.                        │
│  Warm. Updated every turn. LRU eviction.          │
├─────────────────────────────────────────────────┤
│  Tier 2 — Experience   (hrs → wks)               │
│  Patterns across sessions.                        │
│  Cool. Search-based retrieval. Batch processing.  │
│  ← CURRENT SYNIX PRODUCT                         │
├─────────────────────────────────────────────────┤
│  Tier 3 — Identity     (wks → permanent)          │
│  Who the agent is. Who the user is.               │
│  Cold. Slowest-changing. Most consequential.      │
└─────────────────────────────────────────────────┘
```

Each tier has its own physics:
- **Tier 0** is the live inference context — stack frames, tool outputs, the current turn. Volatile by design.
- **Tier 1** holds the working conversation. Updated every turn, evicts cold entries under budget pressure.
- **Tier 2** consolidates patterns across sessions — episode summaries, monthly rollups, thematic clusters. This is where Synix operates today.
- **Tier 3** is identity — stable preferences, relational maps, core beliefs. Formed through compression of lived experience, not stored directly.

## How data moves

Four flows connect the tiers:

1. **Consolidation** (↓): Hot state cools and compresses. Execution artifacts become session state, session state consolidates into episodes, episodes compress into identity traits.

2. **Retrieval** (↑): Like a CPU cache hierarchy. Tier 0 misses go to Tier 1, Tier 1 misses go to Tier 2, and so on. Each level is slower but deeper.

3. **Invalidation** (↑): When a source of truth changes (user correction, new information), stale entries are flushed from higher tiers — like write-through cache semantics.

4. **Ingestion** (←): Data enters at the physics-appropriate tier. Live conversation enters at Tier 0. Bulk document imports enter at Tier 2. User profile seeds enter at Tier 3.

## What's built today

| Tier | Status | What exists |
|------|--------|-------------|
| Tier 0 | Validated | Stack+heap execution protocol. 65% token reduction in tested workloads. |
| Tier 1 | Designed | Session actor specified but not built. |
| **Tier 2** | **Production** | **The current Synix pipeline.** Sources → transforms → artifacts → search. Incremental rebuilds, provenance tracking, fingerprint-based caching. |
| Tier 3 | Designed | Identity formation from episodic compression specified but not built. |

The Synix you use today is the **Tier 2 experience layer** — a programmable pipeline that processes raw sources (conversations, documents, reports, any data) into structured memory with full provenance and incremental rebuilds.

## The current system: pipelines

A pipeline is a directed acyclic graph (DAG) of layers:

```
Sources → Transforms → Artifacts → Projections
   │           │            │            │
   │     MapSynthesis       │      SearchSurface
   │     GroupSynthesis      │      FlatFile
   │     ReduceSynthesis     │      SynixSearch
   │     FoldSynthesis       │
   │     Chunk               │
   │                         │
   └── raw source files       └── immutable, content-addressed
```

**Sources** read raw files — conversation exports, documents, reports, photos, sensor data, git logs, API responses, anything your pipeline knows how to process. The architecture doesn't assume text.

**Transforms** process artifacts through LLM calls. Five patterns: 1:1 (Map), N:M (Group), N:1 (Reduce), sequential N:1 (Fold), and 1:N (Chunk, no LLM).

**Artifacts** are immutable and content-addressed. Each one records its input IDs (provenance), prompt ID, and model config. You can trace any artifact back to the sources that produced it.

**Projections** materialize artifacts into queryable outputs at release time — a SQLite FTS5 search index (`search.db`), a flat markdown file for agent context (`context.md`), or both.

### Incremental rebuilds

Every transform has a fingerprint: a hash of its source code, prompt template, config, and model settings. When you rebuild:

1. The runner computes fingerprints for each transform
2. For each artifact, it compares the current build fingerprint (transform + inputs) against the stored one
3. Matches → cached (skip). Mismatches → rebuild.
4. Downstream transforms that depend on rebuilt artifacts are also invalidated.

Change an episode summarization prompt → episodes rebuild → monthly rollups rebuild → core memory rebuilds. But sources stay cached. Add new source files → only new episodes process; existing ones stay cached.

### Provenance

Every artifact records its `input_ids` — the artifact IDs it was built from. Walking backwards through these chains takes you from any output all the way to the raw source data.

```bash
uvx synix lineage core-memory-2026-03
# core-memory-2026-03
#   ← monthly-2026-01, monthly-2026-02, monthly-2026-03
#     ← episode-2026-01-05, episode-2026-01-12, ...
#       ← conversation-2026-01-05.json
```

## Where it's going

**Today:** Programmable Tier 2 pipelines. Define your memory architecture in Python, build it, search it, trace it.

**Next:** Tier 1 session actor + Tier 0↔1 interface. Your agent's working conversation becomes managed state with eviction to the experience layer.

**Eventually:** Agents program their own memory. The pipeline definition becomes something the agent itself evolves — adding layers, changing prompts, restructuring its own memory architecture based on what it learns.

The architecture is designed so that each tier can be built independently. The Tier 2 pipeline you use today will become one actor in a multi-tier system — no migration required.

## Learn more

- [Getting Started](getting-started.md) — build your first pipeline in 5 minutes
- [Pipeline API](pipeline-api.md) — full Python API for transforms, projections, and search
- [Integration Guide](integration.md) — how your agent reads Synix output
- [Cache Semantics](cache-semantics.md) — the fingerprint and rebuild system in detail
