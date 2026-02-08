# Synix Backlog

Items identified during v0.9 development and PR review that are deferred to future releases.

---

## Adapters & Parsing

### Time-gap episode chunking
Long conversations (hours/days of elapsed time) should be splittable into multiple episodes based on time gaps between messages. Now that adapters capture per-message timestamps (`last_message_date`), a chunking transform or adapter option could split a single conversation into multiple transcript artifacts when the gap between consecutive messages exceeds a threshold (e.g., 4 hours).

**Depends on:** per-message timestamps (done in v0.9)

### ChatGPT full tree export (`include_branches=True`)
Currently the ChatGPT adapter linearizes the conversation tree by following the active branch via `current_node`. An optional `include_branches=True` flag could export all branches as separate transcripts, useful for analyzing regeneration patterns or preserving the full conversation history.

### Richer message metadata
Individual message timestamps, token counts, and model identifiers per-message (ChatGPT exports include these). Currently only conversation-level metadata is captured.

---

## Provenance

### Document retrospective-only limitation
Synix provenance is retrospective: it tracks which inputs and prompts produced each artifact, but it cannot answer forward-looking questions like "if I change this transcript, what downstream artifacts would be affected?" without re-running `synix plan`. This is a known limitation of the content-addressed caching model.

Add a "Known Limitations" section to README documenting this, with a pointer to `synix plan` as the workaround.

### Inference-aware provenance
Future provenance could track not just "what inputs were used" but "what parts of the input influenced the output" via attention or citation extraction. This would enable more precise cache invalidation (change a transcript, but if the changed part wasn't used by the episode summary, skip rebuild). Requires LLM-level integration.

---

## Search & Retrieval

### Streaming search results
Large result sets could be streamed rather than fully materialized before display.

### Search result explanations
Show why a result matched (which terms, which embedding dimensions were closest) for debugging and trust.

---

## Build System

### Projections as first-class DAG nodes
Layers can declare dependency on a projection (not just other layers). This enables patterns like: build episode search index, then have topical rollup query it. Currently this works via convention (runner materializes intermediate projections), but making it explicit in the DAG would be cleaner.

### Composable projections
Merge multiple search indexes into a unified query surface. Currently each projection is independent.

### Intermediate projection acceleration
Auto-index any layer for downstream use, not just explicitly declared projections.

### Pipeline branching / A/B testing
Run two pipeline variants side-by-side on the same source data, compare outputs. Useful for evaluating prompt changes, model swaps, or architecture experiments.

### Cost estimation improvements
`synix plan` currently estimates costs based on token counts and model pricing. Could be improved with historical cost tracking (actual vs estimated) and budget alerts.

---

## Infrastructure

### StatefulArtifact
Append/update semantics for agent-writable memory (scratchpad, working memory). Synix tracks versions but doesn't rebuild. Deferred from v0.1 design doc.

### Pluggable projection backends
Postgres, Neo4j, Qdrant, Mem0 as projection targets. Currently SQLite + filesystem only.

### Eval harness
Systematic evaluation of memory quality across pipeline configurations. DSPy integration for prompt optimization.

### Hosted platform
Team collaboration, memory governance, shared pipelines.
