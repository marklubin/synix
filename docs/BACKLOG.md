# Synix Backlog

Items identified during v0.9 development and PR review that are deferred to future releases.

---

## Adapters & Parsing

### Time-gap episode chunking
Long conversations (hours/days of elapsed time) should be splittable into multiple episodes based on time gaps between messages. Now that adapters capture per-message timestamps (`last_message_date`), a chunking transform or adapter option could split a single conversation into multiple transcript artifacts when the gap between consecutive messages exceeds a threshold (e.g., 4 hours).

**Depends on:** per-message timestamps (done in v0.9)

### ChatGPT full tree export (`include_branches=True`)
Currently the ChatGPT adapter linearizes the conversation tree by following the active branch via `current_node`. An optional `include_branches=True` flag could export all branches as separate transcripts, useful for analyzing regeneration patterns or preserving the full conversation history.

### Voice mode / audio transcription support
ChatGPT voice mode conversations store message content as `dict` parts (`content_type: "audio_transcription"` with a `"text"` field, plus `audio_asset_pointer` and `real_time_user_audio_video_asset_pointer` types) instead of plain strings. The adapter currently only extracts `isinstance(p, str)` parts, so voice conversations are silently dropped (0 text parts → skipped). Fix: extract `.text` from `audio_transcription` dict parts. This recovers 2 of 10 conversations in the current test export.

### Rich message metadata and non-text content
ChatGPT exports contain rich structured content beyond plain text that we currently ignore: tool/function calls and results (`tool` role messages), code interpreter executions, DALL-E image generations, file uploads, web browsing results, and per-message metadata (timestamps, token counts, model identifiers per message). Claude exports similarly contain tool use blocks and artifact content. Extracting this into transcript metadata would enable richer episode summaries (e.g., "user asked the model to generate an image of X", "model ran Python code to analyze Y") and better provenance for tool-heavy conversations.

---

## Provenance

### Document retrospective-only limitation
Synix provenance is retrospective: it tracks which inputs and prompts produced each artifact, but it cannot answer forward-looking questions like "if I change this transcript, what downstream artifacts would be affected?" without re-running `synix plan`. This is a known limitation of the content-addressed caching model.

Add a "Known Limitations" section to README documenting this, with a pointer to `synix plan` as the workaround.

### Inference-aware provenance
Future provenance could track not just "what inputs were used" but "what parts of the input influenced the output" via attention or citation extraction. This would enable more precise cache invalidation (change a transcript, but if the changed part wasn't used by the episode summary, skip rebuild). Requires LLM-level integration.

---

## Search & Retrieval

### Semantic policy lookup for downstream transforms
Transforms like `enrich_cs_brief` currently receive all policy index artifacts as direct inputs (full concatenation into the prompt). At scale this won't fit in context. Instead, the transform should query a search index over the policy layer to find only the relevant policies for each product — e.g., "find return/warranty policies that apply to electronics over $500". Requires projections as first-class DAG nodes (so a layer can declare a dependency on a search projection) and a retrieval API the transform can call at execution time.

### Streaming search results
Large result sets could be streamed rather than fully materialized before display.

### Search result explanations
Show why a result matched (which terms, which embedding dimensions were closest) for debugging and trust.

### Provenance drill-down from search results
Search results show provenance chains (artifact IDs), but there's no interactive way to drill into them. A "drill-down" workflow would let users pick a search result, then navigate its provenance tree — showing the content at each level (e.g., core → monthly → episode → transcript) inline. Could be an interactive CLI mode or a `synix trace <artifact-id>` command that walks the chain and renders each ancestor's content with layer context. Currently `synix lineage` shows the tree structure but not the content; `synix show` displays a single artifact but doesn't chain them.

---

## Build System

### Smarter `synix clean`
Currently `synix clean` removes the entire build directory. Add granular options: `--layer <name>` to clean a single layer (and downstream), `--projections` to clean only projections, `--stale` to remove orphaned artifacts not referenced in the current pipeline. Could also add `--dry-run` to preview what would be deleted.

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

### Backend batch API support
Leverage provider batch APIs (OpenAI `/v1/batches`, Anthropic Message Batches, DeepSeek equivalent) for bulk rebuilds where latency isn't critical. Submit N requests as a batch, poll for results. Typically 50% cheaper than synchronous calls. Good fit for full pipeline rebuilds overnight or large corpus ingestion. Would require an async runner mode that submits the batch, persists the batch ID, and resumes when results are ready.

### Request-level batching (multi-input prompts)
Pack multiple inputs into a single LLM request and ask for multiple labeled outputs. e.g., send 5 transcripts in one prompt, get 5 episode summaries back as a JSON array keyed by conversation ID. Benefits: fewer API calls (lower latency, fewer rate limit hits), amortizes system prompt tokens across N inputs. Tradeoff: need reliable structured output parsing to split responses back into individual artifacts. Natural fit for 1:1 transforms like episode summaries. Would add a configurable `batch_size` to layer config, with single-input as fallback.

### Reachability pruning: only build layers needed by projections
Currently the runner builds every layer in the DAG unconditionally. Real build systems (make, bazel) trace backwards from targets to find transitive deps and only build those. This would let users define a large pipeline but only materialize a subset via projection selection.

- Wire up the `skipped` counter in `LayerStats`/`RunResult` (currently dead code — never incremented).
- Implementation: walk backwards from projection source layers through `depends_on`, mark reachable layers, skip unreachable ones.
- Also consider: should "projection" be aliased as "target" in the CLI/docs for build-system familiarity?

### High-speed inference provider for volume layers
Episode summaries are the highest-volume LLM calls in a typical pipeline (1 per conversation) but don't require frontier-level reasoning. Currently running on DeepSeek (~14s avg per call), which is cost-optimized but slow. Add a fast provider for volume layers:

- **Groq** (Llama 3.3 70B): ~500 tok/s, OpenAI-compatible API, free tier available
- **Gemini 2.0 Flash**: fast, 1M context window, OpenAI-compatible endpoint
- **GPT-4o-mini**: fast, cheap, solid for summarization

The multi-provider LLM infrastructure already supports per-layer `llm_config` overrides, so this is just a config change in example pipelines + documenting the recommended provider-per-layer strategy (fast/cheap for episodes, balanced for rollups, high-quality for core synthesis).

---

## Validation & Fixing

### Guided remediation for unresolvable conflicts
When the LLM can't auto-resolve a contradiction, let the user type free-text instructions for a guided retry. Requires: `BaseFixer.guided_fix()` method, `semantic_guided_fix.txt` prompt, CLI free-text input. Single retry, no infinite loop.

### PII auto-redaction fixer
PIIValidator detects only; a `PIIRedactionFixer` would auto-replace patterns with redacted versions (e.g., `4111****1111`). Non-interactive.

### Rules-based post-resolution validation
After a semantic fix is applied, re-run the conflict prompt to verify the fix actually resolved the contradiction. Warn if the conflict persists.

### Trace toggle
`--trace=off|on` flag to skip trace artifact storage during validation/fixing. Currently always-on.

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
