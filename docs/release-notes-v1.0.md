# Synix v1.0 Release Notes

## What Is Synix

A build system for agent memory. Declare your memory architecture in Python — conversations are sources, prompts are build rules, summaries and world models are artifacts. Change a config, only affected layers rebuild. Trace any artifact back through the dependency graph to its source conversation.

```bash
uvx synix init my-project && cd my-project
uvx synix build pipeline.py
uvx synix release HEAD --to local
uvx synix search "return policy" --release local
```

---

## What's New Since v0.15

v1.0 represents the culmination of a red-team audit and hardening cycle. No new features were added — this release is about making every existing feature trustworthy.

### Trust & Correctness (P0 fixes)

**Source errors cascade through the DAG.** A broken source now propagates `status="error"` to every downstream transform. Before: a source that couldn't load files would silently produce zero artifacts, and downstream layers would show "cached" or "new" depending on prior state. Now: the entire downstream chain shows `error: upstream error: <source_name>`. The plan summary counts errors separately from rebuild/cached totals. The JSON plan output reflects error status faithfully.

**`synix diff` is snapshot-accurate.** `diff_builds()` and `diff_artifact_by_label()` now use `SnapshotView` (the ref-resolved read path) instead of `SnapshotArtifactCache`. This fixes false-negative diffs where added/removed artifacts between snapshots were missed.

**Path resolution is pipeline-relative.** Relative `source_dir` and `build_dir` in pipeline files now resolve against the pipeline file's location, not the caller's working directory. `Source(dir="./custom")` overrides also resolve correctly. `--build-dir` CLI overrides force recomputation of `.synix` location.

**`release --target` is hardened.** The target directory is created before adapter dispatch. `FlatFileAdapter` always writes under the target directory using the configured filename (removed an ambiguous `is_dir()` branch). Multi-adapter releases (search + flat-file) to external directories work predictably.

**`synix info` and `synix status` tell the truth.** Both commands read adapter targets from `receipt.json` instead of guessing filenames in `.synix/releases/<name>/`. External release targets are reported with full paths. Both commands now accept `--build-dir` and `--synix-dir` options for non-default project layouts. A shared `_discover_synix_dir()` function provides consistent discovery: explicit option → `pipeline.py` in cwd → `cwd/build` convention → `cwd/.synix`.

**Source-load failures are loud.** The build runner re-raises source load exceptions as `RuntimeError` instead of swallowing them. The planner shows `status="error"` with the exception message. DLQ does not rescue source failures — sources are the DAG foundation.

**Invalid ref handling is consistent.** `synix list --ref`, `synix search --ref`, and other ref-consuming commands now exit non-zero with a clear error when the ref doesn't resolve, instead of silently succeeding with empty output.

### Operator Consistency (P1 fixes)

**Planner estimates are accurate.** `_plan_transform_layer()` stores placeholder artifacts of the estimated count for downstream planning, instead of forwarding raw inputs. This prevents overestimation in deep DAGs.

**`synix clean` preserves release refs.** Clean removes release payloads (`search.db`, `context.md`) and work files, but release refs (`.synix/refs/releases/<name>`) are preserved. Refs have their own lifecycle — use `refs` commands to manage them. Clean warns about external release targets (created via `release --target`) but does not delete them.

**`synix refs list` shows all ref types.** Now includes `refs/plans` in addition to `refs/heads`, `refs/runs`, and `refs/releases`.

### CLI & Docs (P2 fixes)

**Mesh respects `SYNIX_MESH_ROOT`.** The mesh subsystem reads the environment variable for root directory configuration.

**`batch-build plan` counts are accurate.** The batch planner now tracks cardinality through the DAG instead of calling `estimate_output_count(1)` for every layer.

**`llms.txt` removed.** The auto-generated documentation file and its CI workflow were removed — documentation is getting a complete overhaul.

---

## Stable Surface Area (v1.0 Contract)

Everything below is the supported public API. Breaking changes to these interfaces will follow semver.

### CLI Commands

| Command | Description |
|---------|-------------|
| `synix build [PIPELINE]` | Execute pipeline, produce immutable snapshot in `.synix/` |
| `synix plan [PIPELINE]` | Dry-run — per-artifact rebuild/cached/error counts + cost estimate |
| `synix release [REF] --to NAME` | Materialize projections to a named release |
| `synix revert [REF] --to NAME` | Re-release an older snapshot |
| `synix search QUERY --release NAME` | Hybrid search (FTS5 + semantic + provenance) |
| `synix list` | All artifacts in current snapshot |
| `synix show [ID]` | Render artifact content |
| `synix lineage [ID]` | Provenance chain for an artifact |
| `synix diff` | Compare snapshots or artifacts across builds |
| `synix info` | System info + pipeline config + build status |
| `synix status` | Build layers + releases + stale artifacts + violations |
| `synix init [PATH]` | Create new project from template |
| `synix clean` | Remove releases and work files (preserves snapshots) |
| `synix refs list` | All refs (heads, runs, releases, plans) |
| `synix refs show [REF]` | Resolve ref to snapshot OID |
| `synix releases list` | All named releases |
| `synix releases show [NAME]` | Release receipt details |
| `synix runs list` | Build run history |
| `synix runs show [ID]` | Run metadata |

### Pipeline API

```python
from synix import Pipeline, Source, SynixSearch, FlatFile
from synix.ext import EpisodeSummary, MonthlyRollup, CoreSynthesis

pipeline = Pipeline("my-agent")
pipeline.source_dir = "./conversations"
pipeline.build_dir = "./build"
pipeline.llm_config = {"model": "claude-sonnet-4-20250514"}

transcripts = Source("transcripts")
episodes = EpisodeSummary("episodes", depends_on=[transcripts])
rollup = MonthlyRollup("monthly", depends_on=[episodes])
core = CoreSynthesis("core", depends_on=[rollup])

pipeline.add(transcripts, episodes, rollup, core)
pipeline.add(SynixSearch("search", sources=[episodes, rollup, core]))
pipeline.add(FlatFile("context-doc", sources=[core]))
```

**Layer types:**
- `Source(name, dir=None)` — Root layer, reads files from source_dir
- `Transform(name, depends_on=[...])` — Abstract base for LLM transforms
- `SearchSurface(name, sources=[], modes=[])` — Build-time search capability
- `SynixSearch(name, sources=[], modes=[])` — Search projection (materialized at release)
- `FlatFile(name, sources=[], config={})` — Markdown context doc projection

**Bundled transforms (`synix.ext`):**
- `EpisodeSummary` — 1:1 conversation → episode summary
- `MonthlyRollup` — N:M time-grouped rollup
- `TopicalRollup` — N:M topic-grouped rollup
- `CoreSynthesis` — N:1 distilled core memory
- `Chunk` — 1:N text splitting (no LLM)

**Generic transforms (`synix.transforms`):**
- `MapSynthesis` — 1:1 with custom prompt
- `GroupSynthesis` — N:M with custom grouping
- `ReduceSynthesis` — N:1 with custom prompt
- `FoldSynthesis` — Sequential fold
- `Merge` — Combine inputs

### SDK

```python
import synix

project = synix.open_project(".")
project.build()
project.release_to("local")

release = project.release("local")
results = release.search("return policy", limit=5)
for r in results:
    print(r.content, r.provenance)

artifact = release.artifact("ep-001")
print(artifact.content, artifact.layer, artifact.metadata)
```

**Entry points:** `synix.open_project(path)`, `synix.init(path, pipeline=...)`

**Core classes:** `Project`, `Release`, `SdkSource`, `SdkArtifact`, `SdkSearchResult`, `BuildResult`, `SearchHandle`

**Error hierarchy:** `SdkError` → `SynixNotFoundError`, `ReleaseNotFoundError`, `ArtifactNotFoundError`, `SearchNotAvailableError`, `EmbeddingRequiredError`, `ProjectionNotFoundError`, `PipelineRequiredError`

### MCP Server

20 tools exposed via FastMCP over stdio. Full SDK surface available to AI agents:

`open_project`, `init_project`, `load_pipeline`, `build`, `release`, `search`, `get_artifact`, `list_artifacts`, `list_layers`, `lineage`, `list_releases`, `show_release`, `get_flat_file`, `list_refs`, `source_list`, `source_add_text`, `source_add_file`, `source_remove`, `source_clear`, `clean`

### Projection Adapters

Two built-in adapters with a stable contract (`plan` / `apply` / `verify`):

- **`synix_search`** — FTS5 index + provenance chains + citation edges + optional embeddings
- **`flat_file`** — Markdown context document with configurable output filename

### Storage Format (`.synix/`)

```
.synix/
├── objects/           # Content-addressed (SHA256), immutable
├── refs/              # Git-like: HEAD → refs/heads/main → OID
│   ├── heads/
│   ├── runs/
│   ├── releases/
│   └── plans/
├── releases/<name>/   # Materialized projections + receipt.json
└── work/              # Transient (cleaned by synix clean)
```

### Demo Templates

7 bundled templates (available via `synix init` and `synix demo`):

1. `01-chatbot-export-synthesis` — ChatGPT/Claude exports → episodes → rollup → search
2. `02-tv-returns` — Product return data → policy memory
3. `03-team-report` — Meeting notes → team synthesis
4. `04-sales-deal-room` — Sales pipeline → deal room memory
5. `05-batch-build` — Batch API example (OpenAI)
6. `06-claude-sessions` — Claude conversation exports
7. `07-chunked-search` — Chunk transform + hybrid search

---

## Experimental (Not Covered by Semver)

These features ship in v1.0 but their APIs may change without a major version bump. Use at your own risk in production.

### Validators & Fixers

`synix validate` / `synix fix` — LLM-backed quality checks and auto-remediation.

**Validators:** `Citation`, `SemanticConflict`, `PII`, `MutualExclusion`, `RequiredField`
**Fixers:** `SemanticEnrichment`, `CitationEnrichment`

Status: Removed from demo templates. Not exercised by the red-team audit. APIs (`BaseValidator`, `BaseFixer`, violation state format) are subject to change.

### Batch Build

`synix batch-build` — OpenAI Batch API support for high-volume layers.

Status: Works for OpenAI models. Lacks schema versioning, batch chunking for large layers, concurrent-writer guards, and Anthropic batch support. See [backlog](BACKLOG.md) for hardening items.

### Mesh

`synix mesh` — Multi-project orchestration with subsession processing and dashboard.

Status: Functional but not audited. Configuration via `SYNIX_MESH_ROOT` environment variable. APIs subject to change.

---

## Deprioritized (Not Shipping)

These are tracked as open issues but are explicitly out of scope for v1.0. They represent future directions, not commitments.

### Infrastructure (Post-v1.0)

| Issue | Title | Notes |
|-------|-------|-------|
| #31 | Hosted platform | Team collaboration, governance |
| #30 | Eval harness | DSPy integration, quality evaluation |
| #29 | Pluggable projection backends | Postgres, Neo4j, Qdrant, Mem0 |
| #28 | StatefulArtifact | Append/update semantics, agent-writable |

### Build System Enhancements

| Issue | Title | Notes |
|-------|-------|-------|
| #97 | Snapshot-addressed releases with named symlinks | Release layout redesign |
| #95 | DLQ: formalize partial-build semantics | Downstream/release interaction |
| #88 | LENS: checkpointed artifact banks + runtime tools | Runtime artifact access |
| #82 | Python-local runtime/tool API | Snapshot-scoped banks |
| #81 | Checkpoint projections + sealed banks | Projection checkpointing |
| #80 | Oversized transcripts crash build | Context window overflow |
| #73 | Unbounded prompt construction | Reduce/Group context limits |
| #37 | Buildfile programming model | CDK-style constructs |
| #36 | Constructs library | Reusable pipeline patterns |
| #22 | Reachability pruning | Only build what projections need |
| #21 | Request-level batching | Multi-input LLM prompts |
| #20 | Backend batch API hardening | Anthropic batches, chunking |
| #19 | Cost estimation improvements | Historical tracking, alerts |
| #18 | Pipeline branching / A/B testing | Side-by-side comparison |
| #17 | Intermediate projection acceleration | Auto-index layers |
| #16 | Composable projections | Merge search indexes |
| #15 | Search surfaces enhancements | Build-time query capabilities |
| #14 | Smarter clean (--layer, --stale, --dry-run) | Granular cleanup |
| #23 | High-speed inference provider | Groq, Gemini Flash for volume |

### Search & Retrieval

| Issue | Title | Notes |
|-------|-------|-------|
| #87 | Mesh/API parity for hybrid retrieval | Layered ranking |
| #13 | Interactive provenance drill-down | From search results |
| #12 | Search result explanations | Why a result matched |
| #11 | Streaming search results | Large result sets |
| #10 | Retrieval API over named surfaces | Programmatic retrieval |

### Adapters & Parsing

| Issue | Title | Notes |
|-------|-------|-------|
| #86 | Built-in graph artifact family | Graph retrieval |
| #85 | Built-in core-memory artifact family | Structured core memory |
| #84 | Built-in summary artifact family | Typed summaries |
| #79 | Claude export bare-list format | Auto-detection fix |
| #53 | Parser metadata passthrough | Frontmatter fields |
| #7 | Rich message metadata | Tool calls, DALL-E, code |
| #6 | Voice mode / audio transcription | ChatGPT voice |
| #5 | ChatGPT branch export | Full conversation tree |
| #4 | Time-gap episode chunking | Multi-episode splits |

### Validation & Fixing

| Issue | Title | Notes |
|-------|-------|-------|
| #52 | Validate/verify interaction with traces | Trace artifact compat |
| #27 | Trace toggle (--trace=off\|on) | Skip trace storage |
| #26 | Rules-based post-resolution validation | Re-verify after fix |
| #25 | PII auto-redaction fixer | Auto-redact patterns |
| #24 | Guided remediation | User-guided conflict resolution |

### Templates

| Issue | Title | Notes |
|-------|-------|-------|
| #49 | life-archive | Personal knowledge synthesizer |
| #48 | example-evolver | Few-shot promotion pipeline |
| #47 | memory-diff | See what changed in your cognition |
| #46 | golden-questions | Retrieval test harness |
| #45 | rebuild-not-migrate | Memory is rebuildable |
| #44 | tenant-safe-memory | Entity-isolated, no data leaks |
| #43 | policy-brain | Agent cites rules |
| #42 | episode-builder | Time-aware chunking |
| #41 | decision-log | Agent cites rationale |
| #40 | fact-extraction-memory | Conversations → structured facts |
| #39 | compress-then-embed | Better retrieval, lower cost |
| #38 | persistent-user-memory | Agent stops forgetting |

### Other

| Issue | Title | Notes |
|-------|-------|-------|
| #61 | Built-in decision logging | Decision primitive |
| #60 | Typed artifact schemas | Per-layer typing |
| #58 | A/B experimentation guide | Template + docs |
| #34 | First-class build snapshots + artifact diff | Partially done |
| #33 | Embedding generation hard failure | Partially done |
| #9 | Inference-aware provenance | Attention tracking |
| #8 | Document retrospective provenance limitation | Docs |

---

## Closed Issues (Resolved in v1.0)

| Issue | Title | Resolution |
|-------|-------|------------|
| #83 | Built-in chunk artifact family | `Chunk` transform shipped in #96 |
| #62 | Fail-fast on empty source layers | Source error propagation in red-team fixes |
| #59 | Cache semantics documentation | Docs overhaul in progress |
| #57 | Rich search output | Shipped in search CLI |
| #56 | Provenance lineage summarization | Shipped in projection release v2 |
| #55 | Pipeline-relative imports | Path resolution fixed in red-team audit |
| #54 | Non-interactive automation mode | `--plain` flag on build/plan |
| #51 | Transform code change invalidation | Fingerprint-based cache invalidation |
| #50 | Content-addressed artifact cache | `.synix/objects/` store |

---

## Stats

- **520 files changed** since v0.9.0, **65,616 insertions**, 8,489 deletions
- **1,954 tests** (unit + integration + e2e), all passing
- **7 demo templates** with golden file verification
- **18 version tags** from v0.9.0 to v1.0
- **Python 3.11+**, SQLite + filesystem only, no external databases
- **Zero external services required** beyond an LLM API key
