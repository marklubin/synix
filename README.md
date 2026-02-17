<p align="center">
  <img src="./assets/logo.svg" alt="Synix logo" width="120">
</p>

<pre align="center">
 ███████╗██╗   ██╗███╗   ██╗██╗██╗  ██╗
 ██╔════╝╚██╗ ██╔╝████╗  ██║██║╚██╗██╔╝
 ███████╗ ╚████╔╝ ██╔██╗ ██║██║ ╚███╔╝
 ╚════██║  ╚██╔╝  ██║╚██╗██║██║ ██╔██╗
 ███████║   ██║   ██║ ╚████║██║██╔╝ ██╗
 ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝
</pre>

<h3 align="center">A build system for agent memory.</h3>

<p align="center">
  <video src="./templates/02-tv-returns/tv_returns.mp4" width="720" controls></video>
</p>

## The Problem

Agent memory hasn't converged. Mem0, Letta, Zep, LangMem — each bakes in a different architecture because the right one depends on your domain and changes as your agent evolves. Most systems force you to commit to a schema early. Changing your approach means migrations or starting over.

## What Synix Does

Conversations are sources. Prompts are build rules. Summaries and world models are artifacts. Declare your memory architecture in Python, build it, then change it — only affected layers rebuild. Trace any artifact back through the dependency graph to its source conversation.

```bash
uvx synix build pipeline.py
uvx synix search "return policy"
uvx synix validate                # experimental
```

## Quick Start

`uvx synix init` scaffolds a working project with source files, a multi-layer pipeline, and a validator.

```bash
uvx synix init my-project
cd my-project
```

Build the pipeline (requires an LLM API key — see pipeline.py for config):

```bash
uvx synix build
```

Browse what was built:

```bash
uvx synix list                    # all artifacts, grouped by layer
uvx synix show final-report       # render an artifact as markdown
uvx synix show final-report --raw # full JSON with metadata and artifact IDs
```

Search and validate:

```bash
uvx synix search "hiking"         # full-text search across all indexed layers
uvx synix validate                # run declared validators (experimental)
```

## Using Your Build Output

After a build, Synix gives you two things: a search index and flat artifact files.

**Search via CLI:**
```bash
uvx synix search "return policy"
uvx synix search "warranty terms" --top-k 5 --trace
```

**Use the artifacts directly:**

Build output lives in `./build/` — JSON files per artifact, a `manifest.json` index, and a SQLite FTS5 database. Read them, copy them, or point any tool that speaks SQLite at `search.db`.

```bash
ls build/layer2-cs_product_brief/
sqlite3 build/search.db "SELECT label, layer_name FROM search_index LIMIT 5"
```

## Entity Model

Synix has two kinds of identity. Understanding the difference explains how caching, provenance, and search work.

**Label** — a human-readable semantic name like `ep-conv-123` or `monthly-2024-03`. Labels are stable across rebuilds; they identify *what* an artifact represents. You use labels in `synix show`, `synix lineage`, and search results.

**Artifact ID** — a SHA256 content hash like `sha256:a1b2c3...`. Artifact IDs change whenever the content changes. They are the *true identity* for caching and provenance — if the hash matches, the artifact hasn't changed.

```
Artifact
├── label           "ep-conv-123"          # semantic name (stable)
├── artifact_id     "sha256:a1b2c3..."     # content hash (changes on rebuild)
├── artifact_type   "episode"              # transcript, episode, rollup, core_memory
├── content         "Summary of..."        # the actual text
├── input_ids       ["sha256:x...", ...]   # artifact IDs of inputs that produced this
├── prompt_id       "episode_summary_v3"   # prompt template version (LLM-derived only)
└── metadata        {date, title, ...}     # flexible key-value metadata
```

**Provenance** traces every artifact back to its inputs. Each provenance record stores the artifact ID (hash), the labels of its parent artifacts, the prompt used, and the model config. This is how `synix lineage` reconstructs the full dependency chain from a core memory document back to the original conversations.

**Layers** are typed Python objects in a DAG — `Source("transcripts") → EpisodeSummary("episodes") → MonthlyRollup("monthly") → CoreSynthesis("core")`. Dependencies are expressed as object references via `depends_on`. The pipeline is the full declared architecture: layers, projections, validators.

## Defining a Pipeline

A pipeline is a Python file that declares your memory architecture. Layers are real Python objects — `Source` for inputs, transform classes for LLM steps, `SearchIndex` and `FlatFile` for outputs.

```python
# pipeline.py
from synix import Pipeline, Source, SearchIndex, FlatFile
from synix.transforms import EpisodeSummary, MonthlyRollup, CoreSynthesis

pipeline = Pipeline("my-memory")
pipeline.source_dir = "./sources"
pipeline.build_dir = "./build"
pipeline.llm_config = {
    "model": "claude-sonnet-4-20250514",
    "temperature": 0.3,
    "max_tokens": 1024,
}

# Layer 0: auto-detect and parse source files
transcripts = Source("transcripts")

# Layer 1: one summary per conversation
episodes = EpisodeSummary("episodes", depends_on=[transcripts])

# Layer 2: group episodes by month
monthly = MonthlyRollup("monthly", depends_on=[episodes])

# Layer 3: synthesize everything into core memory
core = CoreSynthesis("core", depends_on=[monthly], context_budget=10000)

pipeline.add(transcripts, episodes, monthly, core)

# Projections — how artifacts become usable
pipeline.add(
    SearchIndex(
        "memory-index",
        sources=[episodes, monthly, core],
        search=["fulltext", "semantic"],
        embedding_config={
            "provider": "fastembed",
            "model": "BAAI/bge-small-en-v1.5",
        },
    )
)
pipeline.add(
    FlatFile("context-doc", sources=[core], output_path="./build/context.md")
)

# Optional: validators and fixers (experimental — APIs may change)
from synix.validators import PII, SemanticConflict
from synix.fixers import SemanticEnrichment

pipeline.add_validator(PII(severity="warning"))
pipeline.add_validator(SemanticConflict())
pipeline.add_fixer(SemanticEnrichment())
```

Because pipelines are Python, you can generate layers dynamically:

```python
from synix.transforms import TopicalRollup

for topic in ["career", "projects", "health"]:
    pipeline.add(TopicalRollup(
        f"topic-{topic}", depends_on=[episodes],
        config={"topics": [topic]},
    ))
```

## Built-in Components

### Sources

Drop files into `source_dir` — the `parse` transform auto-detects format by file structure.

| Format | Extensions | Notes |
|--------|-----------|-------|
| **ChatGPT** | `.json` | `conversations.json` exports. Handles regeneration branches via `current_node`. |
| **Claude** | `.json` | Claude conversation exports with `chat_messages` arrays. |
| **Text / Markdown** | `.txt`, `.md` | YAML frontmatter support. Auto-detects conversation turns (`User:` / `Assistant:` prefixes). |

### Transforms

Import from `synix.transforms`:

| Class | What it does |
|-------|-------------|
| `EpisodeSummary` | 1 transcript → 1 episode summary via LLM. |
| `MonthlyRollup` | Groups episodes by calendar month, synthesizes each via LLM. |
| `TopicalRollup` | Groups episodes by user-declared topics. Requires `config={"topics": [...]}`. |
| `CoreSynthesis` | All rollups → single core memory document. Respects `context_budget`. |
| `Merge` | Groups artifacts by content similarity (Jaccard), merges above threshold. |

### Projections

Import from `synix`:

| Class | Output | Purpose |
|-------|--------|---------|
| `SearchIndex` | `build/search.db` | SQLite FTS5 index across selected layers. Optional embedding support for semantic/hybrid search. |
| `FlatFile` | `build/context.md` | Renders artifacts as markdown. Ready to paste into an LLM system prompt. |

### Validators (Experimental)

> **Note:** The validate/fix workflow is experimental. APIs and output formats may change in future releases.

Import from `synix.validators`:

| Class | What it checks |
|-------|---------------|
| `MutualExclusion` | Merged artifacts don't mix values of a metadata field (e.g., `customer_id`). |
| `RequiredField` | Artifacts in specified layers have a required metadata field. |
| `PII` | Detects credit cards, SSNs, emails, phone numbers in content. |
| `SemanticConflict` | LLM-based detection of contradictions across synthesized artifacts. |
| `Citation` | Verifies artifacts cite their source artifacts with valid URIs. |

### Fixers (Experimental)

Import from `synix.fixers`:

| Class | What it fixes |
|-------|--------------|
| `SemanticEnrichment` | Resolves semantic conflicts by rewriting with source episode context. Interactive approval. |
| `CitationEnrichment` | Adds missing citation references to artifacts. |

## CLI Reference

| Command | What it does |
|---------|-------------|
| `uvx synix init <name>` | Scaffold a new project with sources, pipeline, and README. |
| `uvx synix build` | Run the pipeline. Only rebuilds what changed. |
| `uvx synix plan` | Dry-run — show what would build without running transforms. |
| `uvx synix plan --explain-cache` | Plan with inline cache decision reasons per artifact. |
| `uvx synix list [layer]` | List all artifacts with short artifact IDs, optionally filtered by layer. |
| `uvx synix show <id>` | Display an artifact's content. Resolves by label or artifact ID prefix. `--raw` for JSON. |
| `uvx synix search <query>` | Full-text search across indexed layers. `--mode hybrid` for semantic. |
| `uvx synix validate` | *(Experimental)* Run declared validators against build artifacts. |
| `uvx synix fix` | *(Experimental)* LLM-assisted repair of validation violations. |
| `uvx synix verify` | Check build integrity (hashes, provenance). |
| `uvx synix lineage <id>` | Show the full provenance chain for an artifact. |
| `uvx synix clean` | Delete the build directory. |
| `uvx synix batch-build plan` | *(Experimental)* Dry-run showing which layers would batch vs sync. |
| `uvx synix batch-build run` | *(Experimental)* Submit a batch build via OpenAI Batch API. `--poll` to wait. |
| `uvx synix batch-build resume <id>` | *(Experimental)* Resume a previously submitted batch build. |
| `uvx synix batch-build list` | *(Experimental)* Show all batch build instances and their status. |
| `uvx synix batch-build status <id>` | *(Experimental)* Detailed status for a specific batch build. `--latest` for most recent. |

Commands that take a pipeline path (`build`, `plan`, `validate`, `fix`, `clean`) default to `./pipeline.py` in the current directory.

## Batch Build (Experimental)

> **Warning:** Batch build is experimental. Commands, state formats, and behavior may change in future releases.

The OpenAI Batch API processes LLM requests asynchronously at **50% cost** with a 24-hour SLA. Synix wraps this into `batch-build` — submit your pipeline, disconnect, come back when it's done.

### Quick Example

```python
# pipeline.py — mixed-provider pipeline
pipeline.llm_config = {
    "provider": "openai",           # OpenAI layers → batch mode (automatic)
    "model": "gpt-4o",
}

episodes = EpisodeSummary("episodes", depends_on=[transcripts])
monthly = MonthlyRollup("monthly", depends_on=[episodes])

# Force this layer to run synchronously via Anthropic
core = CoreSynthesis("core", depends_on=[monthly], batch=False)
core.config = {"llm_config": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"}}
```

```bash
# Submit and wait for completion
uvx synix batch-build run pipeline.py --poll
```

### Poll vs Resume

**Poll workflow** — submit and wait in a single session:

```bash
uvx synix batch-build run pipeline.py --poll --poll-interval 120
```

**Resume workflow** — submit, disconnect, come back later:

```bash
# Submit (exits after first batch is submitted)
uvx synix batch-build run pipeline.py
#   Build ID: batch-a1b2c3d4
#   Resume with: synix batch-build resume batch-a1b2c3d4 pipeline.py --poll

# Check on it later
uvx synix batch-build status --latest

# Resume and poll to completion
uvx synix batch-build resume batch-a1b2c3d4 pipeline.py --poll
```

### The `batch` Parameter

Each transform accepts an optional `batch` parameter controlling whether it uses the Batch API:

| Value | Behavior |
|-------|----------|
| `None` (default) | Auto-detect: batch if the layer's provider is native OpenAI, sync otherwise. |
| `True` | Force batch mode. Raises an error if the provider is not native OpenAI. |
| `False` | Force synchronous execution, even if the provider supports batch. |

```python
episodes = EpisodeSummary("episodes", depends_on=[transcripts])              # auto
monthly = MonthlyRollup("monthly", depends_on=[episodes], batch=True)        # force batch
core = CoreSynthesis("core", depends_on=[monthly], batch=False)              # force sync
```

### Provider Restrictions

Batch mode **only works with native OpenAI** (`provider="openai"` with no custom `base_url`). Transforms using Anthropic, DeepSeek, or OpenAI-compatible endpoints via `base_url` always run synchronously. Setting `batch=True` on a non-OpenAI layer is a hard error.

### Transform Requirements

Transforms used in batch builds must be **stateless** — their `execute()` method must be idempotent and produce deterministic prompts from the same inputs. All built-in transforms (`EpisodeSummary`, `MonthlyRollup`, `TopicalRollup`, `CoreSynthesis`) meet this requirement.

See [docs/batch-build.md](docs/batch-build.md) for the full specification including state management, error handling, and the request collection protocol.

## Key Capabilities

**Incremental rebuilds** — Change a prompt or add new conversations. Only downstream artifacts reprocess.

**Fingerprint-based caching** — Every artifact stores a build fingerprint capturing inputs, prompt, model config, transform config, and transform source code. Change any component and only affected artifacts rebuild. See [docs/cache-semantics.md](docs/cache-semantics.md) for the full rebuild trigger matrix.

**Cache explainability** — `uvx synix plan --explain-cache` shows inline reasons for every cache hit or miss directly in the plan tree, so you can see exactly which fingerprint component caused a rebuild.

**Altitude-aware search** — Query episode summaries, monthly rollups, or core memory. Drill into provenance from any result.

**Full provenance** — Every artifact chains back to the source conversations that produced it, through every transform in between.

**Git-like artifact resolution** — `uvx synix show` resolves artifacts by unique prefix of label or artifact ID, just like `git show` resolves commits.

**Validation and repair** — Detect semantic contradictions and PII leaks across artifacts, then fix them with LLM-assisted rewrites.

**Architecture evolution** — Swap monthly rollups for topic-based clustering. Transcripts and episodes stay cached. No migration scripts.

## Where Synix Fits

| | Mem0 | Letta | Zep | LangMem | **Synix** |
|---|---|---|---|---|---|
| **Approach** | API-first memory store | Agent-managed memory | Temporal knowledge graph | Taxonomy-driven memory | Build system with pipelines |
| **Incremental rebuilds** | — | — | — | — | Yes |
| **Provenance tracking** | — | — | — | — | Full chain to source |
| **Architecture changes** | Migration | Migration | Migration | Migration | Rebuild |
| **Schema** | Fixed | Fixed | Fixed | Fixed | You define it |

Synix is not a memory store. It's the build system that produces one.

## Known Limitations

These are the highest-priority open issues. See the [issue tracker](https://github.com/marklubin/synix/issues) for the full backlog.

| Issue | Priority | Description |
|-------|----------|-------------|
| [#53](https://github.com/marklubin/synix/issues/53) | P0 | **Parser metadata passthrough** — YAML frontmatter fields in source files are not propagated to artifact metadata. Custom fields like `author` or `date` are silently dropped. |
| [#52](https://github.com/marklubin/synix/issues/52) | P0 | **Validate/verify and trace artifacts** — Trace artifacts from provenance tracking can trigger false positives in validators that expect only content artifacts. |
| [#57](https://github.com/marklubin/synix/issues/57) | P1 | **Rich search output** — Search results show artifact labels but not inline content snippets or provenance context. Requires multiple commands to get the full picture. |
| [#56](https://github.com/marklubin/synix/issues/56) | P1 | **Provenance summarization** — Lineage output is raw dependency chains. No summarized view or filtering for large graphs. |
| [#55](https://github.com/marklubin/synix/issues/55) | P1 | **Pipeline-relative imports** — Custom transforms using relative imports fail when the pipeline file is outside the project root. |
| [#54](https://github.com/marklubin/synix/issues/54) | P1 | **Non-interactive automation mode** — No `--quiet` / `--json` output mode for CI or scripted usage. Rich formatting assumes a TTY. |
| [#33](https://github.com/marklubin/synix/issues/33) | — | **Embedding failures are silent** — If embedding generation fails, search indexing silently falls back to keyword-only instead of erroring. |

**Removed source files are not cleaned up** — Deleting a source file does not remove its downstream artifacts. Run `uvx synix clean` and rebuild to purge orphans.

## Links

- [synix.dev](https://synix.dev)
- [GitHub](https://github.com/marklubin/synix)
- [llms.txt](./llms.txt) — machine-readable project summary for LLMs
- MIT License
