# SYNIX v0.9 BUILD PLAN

**Milestone:** Public Accessibility Release
**Date:** February 2026
**Status:** Planning
**Document Type:** Engineering Plan + Handoff Specification

This document captures the complete engineering plan for the Synix v0.9 release, including user stories, functional requirements, design decisions from prior discussions, implementation guidance, testing strategy, and acceptance criteria. It is intended as a self-contained handoff to any engineer or agent who needs to pick up and execute this work.

---

## 1. Project Context

### 1.1 What is Synix?

Synix is a build system for agent memory. It takes a corpus of historical conversations, processes them through a configurable DAG of LLM-backed transforms and aggregates, and produces memory artifacts at multiple levels of abstraction — with full provenance, content-addressed caching, and incremental rebuilds.

**One-liner:** "Synix is a continuous build system for agent memory. Sources are raw conversations, build rules are transforms and aggregates, targets are the memory artifacts your agent uses. Change a rule, rebuild incrementally, trace any artifact back to its sources."

**Competitive position:** No existing tool does this. LlamaIndex has sequential ingestion pipelines (flat, no DAG, retrieval-only). Mem0 and Letta manage runtime memory (not offline compilation). Academic work (A-MEM, MemTree) does hierarchical summarization on single streams, not multi-layer DAG compilation of corpora with build-system semantics. The build system metaphor — DAGs, cache keys, incremental rebuilds, provenance — is the core differentiator.

### 1.2 Architecture Mental Model

Synix has two conceptual modules that must be reflected in CLI UX and code organization starting with v0.9:

**synix-build** — The core build system. DAG execution, transform primitives (transform/aggregate/fold/merge), content-addressed caching, incremental rebuilds, provenance tracking. This is a standalone build system that could target any output.

**synix-search** — The built-in search plane. The default output surface for build artifacts. Fulltext + semantic (hybrid) search with altitude-aware querying and provenance drill-down. Ships as the default, but the build system is the core — a user could theoretically point output at their own vector store.

These do not need to be separate services or processes for v0.9, but the CLI namespaces, module boundaries, and user-facing documentation should already reflect this separation. Design decisions about chunking, indexing strategy, and retrieval tuning belong to the search module. Design decisions about DAG resolution, caching, and transform execution belong to the build module.

### 1.3 v0.9 Goal

A stranger can install Synix, point it at conversation exports (ChatGPT JSON, Claude JSON, or plain text/markdown files), run `synix plan` to see the build graph, run `synix build` and watch progress with real-time logging, then run `synix search` and get semantically-relevant results with provenance drill-down. The experience should feel polished, fast enough to be usable on a corpus of 500+ conversations, and demonstrate the core thesis: this is a build system for memory, not a retrieval pipeline.

### 1.4 Key Design Decisions from Prior Discussions

**"Offline" vs "Continuous Deployment":** Synix is not a batch ETL system. The end-state is continuous deployment — new conversations trigger incremental rebuilds of affected subgraph only, asynchronously from the agent's request path. The agent always reads from the last successful build. For v0.9, builds are explicitly triggered via CLI, but the architecture (shadow index swap, incremental rebuild) should already support the CD model.

**Chunking lives in the search plane, not the build plane.** Build artifacts should be whole, coherent documents (episode summaries, monthly syntheses, topical clusters). Chunking for retrieval granularity happens at index time in synix-search. This reinforces the module separation and avoids polluting provenance (a chunk's provenance relationship to its parent artifact is a search concern, not a build concern).

**Batch API is the primary LLM execution strategy for large corpora.** Direct concurrent calls for interactive/small runs, batch API submission for large runs. Batch is 50% cheaper on OpenAI and avoids rate limit management complexity. The pipeline doesn't care — it submits transforms and gets results.

---

## 2. User Stories & Work Items

Stories are ordered by implementation sequence. Each story includes context from design discussions, functional requirements, acceptance criteria, and testing approach.

---

### S01: Logging & Observability

| Field | Value |
|---|---|
| Priority | P0 — Unblocks all other work |
| Estimate | 1–2 days |
| Dependencies | None |
| CLI Surface | `synix build --verbose`, `synix build -v`, `synix build -vv` |

**Design Context:**
Currently Synix produces zero visible output during long LLM calls. A user running a build over 100+ conversations will see a frozen terminal for minutes. This is the single highest-ROI item because it unblocks debugging for every other story and determines whether a first-time user trusts the tool or kills the process.

**Functional Requirements:**

- **FR-1.1:** Default output (no flags): Progress bar or step counter showing current step name, records processed/total, elapsed time. Minimal but not silent.
- **FR-1.2:** `-v` (verbose): Step-level detail — which step is executing, input/output record counts, cache hit/miss per record, timing per step, token counts per LLM call.
- **FR-1.3:** `-vv` (debug): Full LLM prompts and responses (truncated to configurable max length), internal DAG resolution decisions, cache key computations.
- **FR-1.4:** Per-run file-based logging. Every run writes a structured log file to `.synix/logs/{run_id}.jsonl` regardless of verbosity flags. Log file includes all debug-level information.
- **FR-1.5:** Run summary on completion: total time, total LLM calls, total tokens, total cost estimate, cache hit rate, artifacts created/updated/unchanged.

**Acceptance Criteria:**

- Running `synix build` on a 10-conversation corpus shows progress updates at least every 5 seconds.
- Log file is written and contains structured entries for every LLM call, cache decision, and step transition.
- Run summary is printed on completion with accurate counts.
- `-v` shows per-step timing and cache hit/miss breakdown.

**Testing:**

- Unit: Log formatter produces valid JSONL. Verbosity levels filter correctly.
- Integration: Run a build, assert log file exists and contains expected entry types.
- Manual: Visual inspection of terminal output at each verbosity level.

---

### S02: Build/Search UX Separation

| Field | Value |
|---|---|
| Priority | P0 — Architectural foundation |
| Estimate | 1–2 days |
| Dependencies | None |
| CLI Surface | `synix build`, `synix search`, `synix plan`, `synix verify` |

**Design Context:**
The CLI and code should reflect two distinct modules from the start. This is a UX and organizational concern, not a microservices concern. The user should already think of "building memory" and "searching memory" as distinct operations, even though they share a process and database. This sets up the long-term story: "Synix ships with a built-in search plane, but the build system is the core."

**Functional Requirements:**

- **FR-2.1:** CLI commands are namespaced: `synix build [subcommands]`, `synix search [query]`, `synix plan`, `synix verify`. No ambiguity about which module a command belongs to.
- **FR-2.2:** Code is organized into distinct modules: `synix/build/` (pipeline, transforms, DAG, cache, artifact store) and `synix/search/` (indexing, retrieval, ranking, embeddings). Shared primitives (artifact model, config, logging) live in `synix/core/`.
- **FR-2.3:** The build module has no import dependencies on the search module. The search module depends on the artifact store (read-only) from the build module.
- **FR-2.4:** Configuration reflects the separation: build config (LLM provider, pipeline definition, cache settings) is distinct from search config (embedding model, index settings, retrieval parameters).

**Acceptance Criteria:**

- `synix build` and `synix search` are distinct CLI entry points.
- Removing the search module entirely does not break the build module (verified by import test).
- `synix --help` shows clear grouping of build vs search commands.

**Testing:**

- Unit: Import `synix.build` without `synix.search` on the Python path — no ImportError.
- Integration: Run a build with search module mocked/removed, verify artifacts are produced.

---

### S03: Multi-Provider LLM Support

| Field | Value |
|---|---|
| Priority | P0 — Required for public release |
| Estimate | 0.5–1 day |
| Dependencies | None |
| CLI Surface | Config file / env vars |

**Design Context:**
Cannot ship a public tool locked to one LLM provider. The minimum viable version is OpenAI native + `base_url` override for anything OpenAI-compatible (DeepSeek, Ollama, vLLM, Together, Groq, Fireworks). Anthropic support via their SDK is a nice-to-have but not required for v0.9 since Anthropic's API is not OpenAI-compatible.

**Functional Requirements:**

- **FR-3.1:** Configuration accepts `provider`, `model`, `api_key`, and `base_url` fields. Default provider is OpenAI.
- **FR-3.2:** Any OpenAI-compatible API works by setting `base_url`. This covers Ollama (`http://localhost:11434/v1`), vLLM, DeepSeek, Together, Groq, etc.
- **FR-3.3:** Separate LLM config for build transforms vs search embeddings. A user might use GPT-4o for transforms but a local model for embeddings.
- **FR-3.4:** Batch API support is provider-aware. If the provider supports batch (OpenAI, Anthropic), use it for large runs. If not (Ollama, vLLM), fall back to concurrent direct calls.
- **FR-3.5:** Configuration can be set via config file (`synix.toml` or `synix.yaml`), environment variables (`SYNIX_LLM_MODEL`, `SYNIX_LLM_BASE_URL`, etc.), or CLI flags. Precedence: CLI > env > config file.

**Acceptance Criteria:**

- A build completes successfully against OpenAI, and against Ollama with `base_url` override.
- Embedding generation works with a different provider than the transform LLM.
- Missing or invalid API key produces a clear error before any LLM calls are made.

**Testing:**

- Unit: Config resolution precedence (CLI > env > file). Provider detection from base_url.
- Integration: Mock LLM server that implements OpenAI-compatible API. Full build against mock server.
- E2E: Build against real OpenAI (gated behind API key, CI-optional).

---

### S04: Concurrent Execution & Batch API

| Field | Value |
|---|---|
| Priority | P0 — Unusable without this at scale |
| Estimate | 2–3 days |
| Dependencies | S01 (logging), S03 (multi-provider) |
| CLI Surface | `synix build --concurrency 10`, `synix build --batch` |

**Design Context:**
Sequential LLM calls over 1000 records is a non-starter. Testing with 10 conversations was already noticeably slow sequentially. Two execution strategies are needed:

1. **Concurrent direct calls** (default): Configurable parallelism (default 10). Good for interactive use, small-to-medium corpora, local models. Requires rate limit awareness.
2. **Batch API** (`--batch` flag): Submit all transforms for a step as a batch job, poll for completion. 50% cheaper on OpenAI. Higher latency per-request but better throughput for large corpora. Fits the "submit a build, come back when it's done" model.

The pipeline should not care which execution strategy is used — it produces a set of (input, prompt) pairs for each step, hands them to the executor, and gets results back. The executor is the strategy boundary.

Independent transforms within a step should execute concurrently. Transforms across steps must respect DAG ordering (a monthly summary can't start until its input episode summaries are complete). Within a step, all records at that step are independent and can be parallelized.

**Functional Requirements:**

- **FR-4.1:** `LLMExecutor` interface with two implementations: `ConcurrentExecutor` (async, configurable concurrency limit) and `BatchExecutor` (OpenAI batch API).
- **FR-4.2:** `ConcurrentExecutor` respects rate limits. Exponential backoff on 429s. Configurable max concurrency (default 10).
- **FR-4.3:** `BatchExecutor` submits all pending transforms for a step as a single batch, polls for completion with configurable interval, and maps results back to records.
- **FR-4.4:** `BatchExecutor` handles partial failures — if some records in a batch fail, those specific records are retried or reported, not the entire batch.
- **FR-4.5:** `--batch` flag activates batch mode. Without it, concurrent direct calls are used.
- **FR-4.6:** Progress logging integrates with executor — concurrent mode shows per-record progress, batch mode shows batch submission/polling/completion status.
- **FR-4.7:** If the provider does not support batch API (detected from config), `--batch` flag falls back to concurrent with a warning.

**Acceptance Criteria:**

- A 100-record transform step completes in roughly 1/10th the time with concurrency=10 vs sequential.
- Batch mode submits a single batch to OpenAI, polls, and correctly maps results to records.
- A rate-limited response (429) triggers backoff, not a crash.
- Partial batch failure reports failed records without losing successful ones.

**Testing:**

- Unit: `ConcurrentExecutor` respects concurrency limit (mock LLM, assert max N in-flight). Backoff logic on 429. `BatchExecutor` request/response mapping.
- Integration: Mock LLM server with configurable latency. Verify concurrent execution is actually parallel (timing-based assertion). Mock batch API endpoint.
- E2E: Real batch submission to OpenAI (CI-optional, costs money).

---

### S05: Embeddings & Hybrid Search

| Field | Value |
|---|---|
| Priority | P0 — Table stakes for any memory product |
| Estimate | 2–3 days |
| Dependencies | S02 (search module separation), S03 (multi-provider) |
| CLI Surface | `synix search "query"` |

**Design Context:**
Every LLM memory product ships with semantic search. Keyword-only search will feel broken to anyone who tries Synix. The search plane needs hybrid retrieval: BM25/FTS (already exists) + vector similarity (new). Score fusion combines results.

Embeddings are generated at index time in the search module, not during the build. Build artifacts are whole documents. The search indexer may optionally chunk large artifacts for retrieval granularity, but this is a search concern.

For v0.9, keep everything in SQLite to avoid adding external dependencies. Use `sqlite-vec` or similar for vector storage, or store embeddings as blobs and do brute-force cosine similarity (fine for <100K vectors). If neither is viable, use a simple in-memory FAISS index persisted to disk alongside the SQLite FTS index.

**Functional Requirements:**

- **FR-5.1:** Embedding generation at index time. When artifacts are indexed for search, embeddings are generated using the configured embedding model.
- **FR-5.2:** Hybrid retrieval: queries run against both FTS index and vector index. Results are combined using Reciprocal Rank Fusion (RRF) or simple weighted score combination.
- **FR-5.3:** Search results include: content snippet, artifact ID, step name (altitude), relevance score, and provenance references.
- **FR-5.4:** Altitude-aware search: `synix search "rust" --step monthly` restricts to monthly summaries. `synix search "rust"` searches all altitudes.
- **FR-5.5:** Provenance drill-down from search results: `synix search "rust" --trace` shows the source chain for each result (monthly → episodes → raw turns).
- **FR-5.6:** Embedding model is configurable separately from the transform LLM (FR-3.3). Default: OpenAI `text-embedding-3-small`.
- **FR-5.7:** Search configuration: number of results (`--top-k`, default 10), altitude filter (`--step`), retrieval mode (`--mode hybrid|semantic|keyword`).

**Acceptance Criteria:**

- `synix search "machine learning"` returns semantically relevant results even if the exact phrase doesn't appear in artifacts (e.g., artifacts about "neural networks" or "model training" surface).
- Search with `--step monthly` returns only monthly-level artifacts.
- `--trace` shows at least 2 levels of provenance for each result.
- Hybrid mode outperforms keyword-only on a manually-curated test set of 10 queries.

**Testing:**

- Unit: RRF score fusion produces correct rankings given mock FTS + vector scores. Altitude filtering. Embedding dimension validation.
- Integration: Index a known set of artifacts, run queries, assert expected results are in top-5. Compare hybrid vs keyword-only on test queries.
- E2E: Full pipeline — build from test corpus, index, search, verify results are relevant and provenance is valid.

---

### S06: `synix plan` & BuildPlan Artifact

| Field | Value |
|---|---|
| Priority | P1 — The "this is a build system" demo moment |
| Estimate | 1–2 days |
| Dependencies | S01 (logging) |
| CLI Surface | `synix plan` |

**Design Context:**
This is the moment someone goes "oh, this is a build system." They define a pipeline, run `synix plan`, and see: "12 conversations changed → 12 episode re-summarizations → 3 monthly rebuilds → 1 core update. Estimated cost: $0.47. Estimated time: 3 min." This makes the Terraform/dbt analogy concrete. It's also the natural CLI surface for the `BuildPlan` intermediate artifact — the parsed pipeline + topo sort stored as a queryable object.

**Functional Requirements:**

- **FR-6.1:** `synix plan` parses the pipeline definition, resolves the DAG, compares against cached artifacts, and outputs a plan showing: what would be built, what's cached (skip), what's invalidated (rebuild), and what's new.
- **FR-6.2:** Plan output includes per-step breakdown: step name, record count (new/changed/cached), estimated LLM calls, estimated tokens, estimated cost (based on model pricing), estimated time (based on average latency).
- **FR-6.3:** Plan output includes a total summary: total LLM calls, total estimated cost, total estimated time, cache hit rate.
- **FR-6.4:** Plan output is human-readable by default (formatted table) and machine-readable with `--json` flag.
- **FR-6.5:** `BuildPlan` is stored as an artifact that can be inspected. `synix plan --save` persists the plan. `synix build --plan {plan_id}` executes a saved plan.
- **FR-6.6:** Plan correctly identifies invalidated downstream artifacts when a build rule (prompt) changes. Changing the episode summarization prompt should show all episodes + all monthly summaries + core as needing rebuild, even if source conversations haven't changed.

**Acceptance Criteria:**

- `synix plan` on a fresh corpus shows all records as "new."
- `synix plan` after a successful build with no changes shows all records as "cached."
- Changing a transform prompt and running `synix plan` shows the correct downstream invalidation cascade.
- Cost estimates are within 2x of actual cost for a subsequent `synix build`.

**Testing:**

- Unit: DAG resolution produces correct topo sort. Cache comparison correctly identifies new/changed/cached records. Cost estimation from token counts + model pricing.
- Integration: Build a corpus, modify one source, run plan, verify only affected subgraph is marked for rebuild. Change a prompt, verify full downstream cascade.
- E2E: `synix plan` → `synix build` → compare plan estimates to actual run metrics (from S01 logging).

---

### S07: Search Index Swap Mode

| Field | Value |
|---|---|
| Priority | P1 — Required for continuous deployment model |
| Estimate | 1–2 days |
| Dependencies | S02 (search module), S05 (embeddings) |
| CLI Surface | Transparent — no user-facing flags needed |

**Design Context:**
Currently the search index is DROP TABLE + recreate on every build. This means during a rebuild, search is unavailable. For the continuous deployment model, the agent should always be able to read from the last successful build while a new build compiles. The implementation is shadow build + atomic swap: build new index alongside old, swap on completion.

**Functional Requirements:**

- **FR-7.1:** Build writes to a shadow index (new table/file) while the active index remains queryable.
- **FR-7.2:** On successful build completion, the shadow index atomically replaces the active index. On failure, the active index is unchanged.
- **FR-7.3:** At most 2 copies of the index exist at any time (active + shadow). Previous shadow is cleaned up before new build starts.
- **FR-7.4:** `synix search` always reads from the active index, never the in-progress shadow.

**Acceptance Criteria:**

- Start a build, run `synix search` concurrently — search returns results from the previous build, not partial results from the in-progress build.
- Build failure leaves the active index intact.
- Successful build swaps cleanly — next search returns new results.

**Testing:**

- Unit: Shadow table creation and atomic swap logic.
- Integration: Concurrent build + search. Build intentionally fails midway, verify active index unchanged.

---

### S08: Artifact Diffing

| Field | Value |
|---|---|
| Priority | P1 — "Holy shit" demo moment |
| Estimate | 1 day |
| Dependencies | S01 (logging, run IDs) |
| CLI Surface | `synix diff {artifact_id}`, `synix diff --run {run_a} {run_b}` |

**Design Context:**
"I changed my summarization prompt, rebuilt, and here's exactly how the world model changed." This is the moment someone realizes Synix is a workbench, not just a pipeline. It connects directly to the experimentation thesis — you can see what a build rule change actually did to your knowledge artifacts.

**Functional Requirements:**

- **FR-8.1:** `synix diff {artifact_id}` shows the text diff between the current version of an artifact and its previous version.
- **FR-8.2:** `synix diff --run {run_a} {run_b}` shows all artifacts that changed between two runs, with diffs for each.
- **FR-8.3:** Diff output is human-readable (inline diff with +/- markers, colored if terminal supports it) and machine-readable with `--json`.
- **FR-8.4:** Diff includes metadata: which build rule changed (if any), which sources changed (if any), timestamp of each version.
- **FR-8.5:** Artifact versions are retained across runs. At minimum, the current and previous version of each artifact are stored. Configurable retention depth (default 3).

**Acceptance Criteria:**

- Change a prompt, rebuild, `synix diff {artifact_id}` shows meaningful differences in the artifact content.
- Add a new source conversation, rebuild, `synix diff --run` shows which artifacts were affected and how.
- Diffing an unchanged artifact shows "no changes."

**Testing:**

- Unit: Diff algorithm produces correct output for known text pairs. Version storage and retrieval.
- Integration: Build → modify prompt → rebuild → diff shows changes. Build → add source → rebuild → diff shows affected artifacts.

---

### S09: `synix verify`

| Field | Value |
|---|---|
| Priority | P1 — Confidence builder for new users |
| Estimate | 1 day |
| Dependencies | S05 (search index), S02 (module separation) |
| CLI Surface | `synix verify` |

**Design Context:**
If Synix is pitched as a build system with provenance, "verify the build" is expected. This is the equivalent of `terraform validate` or a build system's integrity check. It should catch corruption, orphaned artifacts, broken provenance chains, and inconsistencies between the build output and the search index.

**Functional Requirements:**

- **FR-9.1:** Provenance integrity: every artifact has valid source references. No broken links in the DAG.
- **FR-9.2:** Search index consistency: every artifact that should be indexed is indexed. No orphaned index entries pointing to deleted artifacts.
- **FR-9.3:** Cache consistency: materialization keys match stored artifacts. No stale cache entries.
- **FR-9.4:** Pipeline consistency: stored artifacts match the current pipeline definition. Flag artifacts that were produced by a different pipeline version (stale build rules).
- **FR-9.5:** Output is a verification report: checks passed, checks failed, warnings. Exit code 0 for clean, 1 for failures, 2 for warnings only.
- **FR-9.6:** `synix verify --fix` attempts to auto-repair recoverable issues (rebuild stale artifacts, remove orphans, reindex missing entries). Destructive fixes require `--fix --confirm`.

**Acceptance Criteria:**

- Clean build followed by `synix verify` returns exit code 0.
- Manually deleting an artifact file and running `synix verify` detects the broken provenance.
- Manually corrupting the search index and running `synix verify` detects the inconsistency.
- `--fix` recovers from a missing index entry by reindexing.

**Testing:**

- Unit: Each verification check against known-good and known-bad states.
- Integration: Build, corrupt various components, verify detects each class of issue. Verify `--fix` recovers.

---

### S10: Plain Text / Markdown Source Adapter

| Field | Value |
|---|---|
| Priority | P1 — Universal escape hatch |
| Estimate | 0.5–1 day |
| Dependencies | None |
| CLI Surface | Pipeline config: `source: { type: "text", path: "./exports/" }` |

**Design Context:**
ChatGPT JSON and Claude JSON adapters already exist. Adding plain text / markdown as a third adapter means anyone can use Synix regardless of where their data comes from. "Got Slack exports? Dump them as text files. Got Notion exports? They're already markdown." We don't actually parse markdown structure — we treat it as plain text. The adapter contract is: give me text + metadata (timestamp, source identifier, optional conversation structure).

**Functional Requirements:**

- **FR-10.1:** Text/markdown adapter reads files from a directory. Each file becomes one source record.
- **FR-10.2:** Metadata extraction: filename is used as source identifier. If the file has YAML frontmatter (standard markdown convention), parse it for `title`, `date`, `source`, `tags`, or any other fields. Otherwise, metadata is inferred from filename and file modification date.
- **FR-10.3:** Optional conversation structure detection: if the text contains obvious turn markers (e.g., "Human:", "Assistant:", "User:", "AI:", or similar patterns), the adapter can optionally parse these into a structured conversation format matching the ChatGPT/Claude adapter output.
- **FR-10.4:** Glob/filter support: `path: "./exports/*.md"` or `exclude: ["*.tmp"]`.
- **FR-10.5:** Adapter registry is pluggable: adding a new adapter means implementing a defined interface (`Adapter.load(config) -> List[SourceRecord]`) and registering it. Third-party adapters should be possible.

**Acceptance Criteria:**

- A directory of 10 markdown files is ingested and processed through the full pipeline.
- YAML frontmatter is correctly parsed into record metadata.
- Files without frontmatter use filename and modification date as metadata.
- A file with "Human:/Assistant:" turns is optionally parsed into conversation structure.

**Testing:**

- Unit: Frontmatter parsing (with and without). Filename metadata inference. Turn detection regex.
- Integration: Full build from a directory of markdown files, verify artifacts are produced with correct provenance back to source files.

---

## 3. Testing Strategy

### 3.1 Test Pyramid

```
                    ┌─────────┐
                    │  E2E    │  2-3 scenarios, real LLM calls (CI-optional)
                   ─┤         ├─
                  / └─────────┘ \
                 /                \
                ┌──────────────────┐
                │  Integration     │  Per-story, mock LLM, real SQLite
               ─┤                  ├─
              / └──────────────────┘ \
             /                        \
            ┌──────────────────────────┐
            │  Unit                    │  Per-module, no I/O, fast
            └──────────────────────────┘
```

**Unit tests** — Fast, no I/O, no LLM calls. Test individual functions: DAG resolution, cache key computation, config parsing, diff algorithm, log formatting, score fusion, adapter parsing, etc. Target: every module has unit tests. Run on every commit.

**Integration tests** — Mock LLM server (returns deterministic responses), real SQLite, real file I/O. Test story-level flows: build pipeline with mock LLM, verify artifacts and provenance. Search after build, verify results. Plan after build, verify cache detection. Target: one integration test per story. Run on every commit.

**E2E tests** — Real LLM calls against OpenAI. Full pipeline: ingest real conversation exports → build → search → verify. Expensive, slow, non-deterministic. Target: 2–3 golden-path scenarios. Run manually or on release branches, gated behind API key.

### 3.2 Test Infrastructure

**Mock LLM Server:**

A lightweight HTTP server that implements the OpenAI-compatible API (`/v1/chat/completions`, `/v1/embeddings`, `/v1/batches`). Returns deterministic responses based on input hashing or fixed fixtures. This is the single most important test infrastructure investment — it unblocks all integration tests without LLM costs.

Requirements:
- Implements `/v1/chat/completions` — returns configurable responses (fixture-based or echo-based).
- Implements `/v1/embeddings` — returns deterministic embedding vectors (hash-based, so same input always gets same vector).
- Implements `/v1/batches` — simulates batch API flow (create → poll → complete).
- Configurable latency for testing concurrency behavior.
- Configurable error injection (429 rate limit, 500 server error, timeout) for testing retry/backoff logic.

**Test Corpus:**

A small, committed test corpus (not generated, handwritten) of 10–15 synthetic conversations covering:
- Single-topic conversations (easy case)
- Multi-topic conversations (tests episode parsing)
- Short conversations (1–3 turns)
- Long conversations (50+ turns)
- Conversations with overlapping topics across different months (tests aggregation)
- Conversations with minimal content (edge case — should produce minimal artifacts, not crash)

This corpus is version-controlled and never changes. It's the "golden input" for all integration and E2E tests.

**Golden Output Snapshots:**

For deterministic tests (mock LLM with fixed responses), store expected outputs as snapshot files. Integration tests compare actual output to snapshots. When a legitimate change is made (new field added to artifact, format change), snapshots are explicitly updated via `synix test --update-snapshots`.

### 3.3 E2E Test Scenarios

**E2E-1: Fresh Build + Search (Golden Path)**

```
Given: Test corpus of 15 conversations (committed fixtures), clean state
When:  synix build (full pipeline: episodes → summaries → monthly → core)
Then:  - Exit code 0
       - Artifacts exist at every step in the DAG
       - Every artifact has valid provenance
       - synix verify returns clean
       - synix search "topic from corpus" returns relevant results
       - synix search "topic" --trace shows valid provenance chain
       - Run log exists and contains expected entry types
```

**E2E-2: Incremental Rebuild**

```
Given: Completed build from E2E-1
When:  Add 2 new conversations to the corpus, run synix build
Then:  - Only affected artifacts are rebuilt (new episodes + affected monthly + core)
       - Previously-cached artifacts are untouched (verify by checking artifact timestamps or version IDs)
       - synix plan (run before build) correctly predicted what would be rebuilt
       - Total LLM calls << full rebuild LLM calls
       - synix verify returns clean
```

**E2E-3: Prompt Change Cascade**

```
Given: Completed build from E2E-1
When:  Change the episode summarization prompt, run synix plan, then synix build
Then:  - Plan shows all episodes + all downstream (monthly, core) as needing rebuild
       - Plan shows source records as cached (prompt change doesn't affect source parsing)
       - Build re-runs all episode transforms with new prompt
       - synix diff shows changes between old and new artifacts
       - synix verify returns clean
```

### 3.4 CI Pipeline

```
on push:
  - lint (ruff/flake8)
  - type check (mypy/pyright)
  - unit tests (fast, no I/O)
  - integration tests (mock LLM server, real SQLite)
  - coverage report (target: 80%+ on build module, 70%+ on search module)

on release branch:
  - all of the above
  - E2E tests (real OpenAI, gated behind SYNIX_TEST_API_KEY secret)
  - performance benchmark: build time for test corpus (track regression)
```

### 3.5 Performance Benchmarks

Track the following metrics across releases on the standard test corpus:

- Full build time (15 conversations, mock LLM with 100ms latency)
- Incremental build time (add 2 conversations to existing build)
- Search latency (p50, p95 for 10 queries against full index)
- Cache hit rate on no-change rebuild (should be 100%)
- Memory usage peak during build

Not pass/fail — tracked for regression detection. Alert if any metric degrades by >20% between releases.

---

## 4. Implementation Sequence

```
Week 1:
  S01  Logging & Observability         ████████
  S02  Build/Search UX Separation       ████████
  S03  Multi-Provider LLM Support           ████

Week 2:
  S04  Concurrent Execution & Batch API ████████████
  S05  Embeddings & Hybrid Search       ████████████
       Test infra: Mock LLM Server      ████████

Week 3:
  S06  synix plan & BuildPlan           ████████
  S07  Search Index Swap Mode           ████████
  S08  Artifact Diffing                     ████

Week 4:
  S09  synix verify                     ████████
  S10  Text/Markdown Source Adapter         ████
       E2E test scenarios               ████████
       Polish, docs, README             ████████
```

**Critical path:** S01 → S04 (can't test concurrency without logging) → S05 (search needs executor) → S06 (plan needs accurate cache/cost info). Everything else can parallelize.

**Risk items:**
- S04 (concurrency/batch) is the highest-complexity story and the most likely to surface unexpected issues (rate limits, batch API quirks, error handling edge cases).
- S05 (embeddings) has a dependency decision: sqlite-vec vs FAISS vs brute-force. Spike this early in week 2.
- Mock LLM server is infrastructure that blocks integration tests for S04–S10. Build it at the start of week 2.

---

## 5. Out of Scope for v0.9

The following items were discussed and explicitly deferred:

| Item | Reason | Target |
|---|---|---|
| Semantic citations / source attribution | Design TBD. Provenance chain (`.sources()` → `.leaves()`) is sufficient for v0.9. | v1.0 |
| Incremental agent-managed memory (StatefulArtifact) | Bridge from batch to live agent memory. Design TBD. Hardest item on the full backlog. | v1.0 |
| Postgres migration | SQLite is fine for single-user. Postgres is for hosted/multi-tenant. | v1.0+ |
| Universal source adapters (Slack, Notion, email, etc.) | Plain text/markdown is the escape hatch. Specific adapters are feature work, not architecture. | v1.0+ |
| Branch experiments / promote | Already designed in the Synix spec. Deferred to keep v0.9 scope tight. | v1.0 |
| Chunking configuration | Chunking lives in the search plane. For v0.9, whole artifacts are indexed. Chunking is an optimization for large artifacts. | v0.9.x or v1.0 |

---

## 6. Definition of Done — v0.9 Release

- [ ] All 10 stories meet their acceptance criteria
- [ ] All 3 E2E test scenarios pass against real OpenAI
- [ ] `synix verify` returns clean on the E2E test corpus
- [ ] CI pipeline is green (unit + integration)
- [ ] README documents: installation, quickstart (3 commands to first search result), configuration, CLI reference
- [ ] `pip install synix` works (package published to PyPI or installable from GitHub)
- [ ] A new user with a ChatGPT export can go from zero to search results in under 5 minutes (excluding LLM processing time)
- [ ] Build on a 500-conversation corpus completes in under 30 minutes with concurrency=10

---

## Appendix A: Glossary

| Term | Definition |
|---|---|
| Artifact | A record produced by a build step. Has content, metadata, provenance references, and a materialization key. |
| Altitude | How far an artifact is from raw sources in the DAG. Raw conversations are altitude 0, episode summaries are altitude 1, monthly syntheses are altitude 2, etc. |
| Build rule | A configured transform, aggregate, fold, or merge step in the pipeline. Analogous to a Bazel rule or dbt model. |
| DAG | Directed acyclic graph. The dependency graph of build steps. Determines execution order and invalidation cascade. |
| Fold | A sequential build rule with accumulator state. Processes ordered inputs one at a time, carrying state forward. Used for rolling world models. |
| Materialization key | Content-addressed cache key for an artifact. Composed of: input content hash + prompt/config hash + step version. If the key matches, the artifact is cached and doesn't need rebuilding. Analogous to a build system cache key. |
| Provenance | The chain of sources that produced an artifact. Every artifact can trace back to the raw conversation turns that influenced it. |
| Shadow index | A search index built alongside the active index during a rebuild. Swapped atomically on success. |
| Step | A named stage in the pipeline (e.g., "episodes", "monthly", "core"). Each step has a build rule type and configuration. |
| Transform | A 1:1 build rule. One source record produces one artifact. |
| Aggregate | An N:1 build rule with grouping. Multiple source records grouped by key produce one artifact per group. |
| Merge | A deduplication build rule. Multiple sources potentially describing the same thing produce one canonical artifact. |

## Appendix B: File/Module Structure (Target)

```
synix/
├── cli/
│   ├── __init__.py
│   ├── main.py              # CLI entry point, arg parsing
│   ├── build_commands.py     # synix build, synix plan
│   ├── search_commands.py    # synix search
│   └── verify_commands.py    # synix verify, synix diff
├── core/
│   ├── __init__.py
│   ├── config.py             # Config resolution (file > env > CLI)
│   ├── logging.py            # Structured logging, verbosity levels
│   ├── models.py             # Artifact, SourceRecord, BuildPlan models
│   └── errors.py             # Error types
├── build/
│   ├── __init__.py
│   ├── pipeline.py           # Pipeline definition, DAG construction
│   ├── dag.py                # Topo sort, dependency resolution, invalidation
│   ├── transforms.py         # Transform, Aggregate, Fold, Merge primitives
│   ├── cache.py              # Materialization keys, cache store
│   ├── executor.py           # LLMExecutor interface
│   ├── concurrent.py         # ConcurrentExecutor implementation
│   ├── batch.py              # BatchExecutor implementation
│   ├── artifacts.py          # ArtifactStore (read/write/version)
│   ├── plan.py               # BuildPlan generation, cost estimation
│   └── verify.py             # Build verification checks
├── search/
│   ├── __init__.py
│   ├── indexer.py            # Index builder (FTS + embeddings)
│   ├── retriever.py          # Hybrid retrieval, RRF fusion
│   ├── embeddings.py         # Embedding generation, provider abstraction
│   └── results.py            # SearchResult model, provenance drill-down
├── adapters/
│   ├── __init__.py
│   ├── registry.py           # Adapter registry, pluggable interface
│   ├── chatgpt.py            # ChatGPT JSON export adapter
│   ├── claude.py             # Claude JSON export adapter
│   └── text.py               # Plain text / markdown adapter
└── tests/
    ├── unit/
    │   ├── test_dag.py
    │   ├── test_cache.py
    │   ├── test_transforms.py
    │   ├── test_plan.py
    │   ├── test_config.py
    │   ├── test_adapters.py
    │   ├── test_retriever.py
    │   └── test_diff.py
    ├── integration/
    │   ├── conftest.py        # Mock LLM server fixture
    │   ├── test_build.py      # Full build with mock LLM
    │   ├── test_incremental.py
    │   ├── test_search.py
    │   ├── test_plan.py
    │   └── test_verify.py
    ├── e2e/
    │   ├── test_golden_path.py
    │   ├── test_incremental.py
    │   └── test_prompt_change.py
    └── fixtures/
        ├── corpus/            # 15 synthetic conversations
        ├── mock_responses/    # Deterministic LLM responses
        └── snapshots/         # Golden output snapshots
```
