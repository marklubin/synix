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
uvx synix validate
uvx synix search "return policy"
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

Validate and search:

```bash
uvx synix validate                # run declared validators
uvx synix search "hiking"         # full-text search across all indexed layers
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

## Defining a Pipeline

A pipeline is a Python file that declares your memory architecture: sources, transforms, projections, and validators.

```python
# pipeline.py
from synix import Pipeline, Layer, Projection, ValidatorDecl, FixerDecl

pipeline = Pipeline("my-memory")
pipeline.source_dir = "./exports"
pipeline.build_dir = "./build"
pipeline.llm_config = {
    "model": "claude-sonnet-4-20250514",
    "temperature": 0.3,
    "max_tokens": 1024,
}

# Layer 0: auto-detect and parse source files
pipeline.add_layer(Layer(name="transcripts", level=0, transform="parse"))

# Layer 1: one summary per conversation
pipeline.add_layer(Layer(
    name="episodes", level=1, depends_on=["transcripts"],
    transform="episode_summary", grouping="by_conversation",
))

# Layer 2: group episodes by month
pipeline.add_layer(Layer(
    name="monthly", level=2, depends_on=["episodes"],
    transform="monthly_rollup", grouping="by_month",
))

# Layer 3: synthesize everything into core memory
pipeline.add_layer(Layer(
    name="core", level=3, depends_on=["monthly"],
    transform="core_synthesis", grouping="single",
    context_budget=10000,
))

# Projections — how artifacts become usable
pipeline.add_projection(Projection(
    name="memory-index", projection_type="search_index",
    sources=[
        {"layer": "episodes", "search": ["fulltext"]},
        {"layer": "monthly", "search": ["fulltext"]},
        {"layer": "core", "search": ["fulltext"]},
    ],
))
pipeline.add_projection(Projection(
    name="context-doc", projection_type="flat_file",
    sources=[{"layer": "core"}],
    config={"output_path": "./build/context.md"},
))

# Optional: validators and fixers
pipeline.add_validator(ValidatorDecl(name="pii", config={"severity": "warning"}))
pipeline.add_validator(ValidatorDecl(name="semantic_conflict", config={
    "llm_config": pipeline.llm_config,
}))
pipeline.add_fixer(FixerDecl(name="semantic_enrichment"))
```

Because pipelines are Python, you can generate layers dynamically:

```python
for topic in ["career", "projects", "health"]:
    pipeline.add_layer(Layer(
        name=f"topic-{topic}", level=2, depends_on=["episodes"],
        transform="topical_rollup", grouping="by_topic",
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

| Name | Grouping | What it does |
|------|----------|-------------|
| `parse` | — | Auto-discovers and parses all source files into transcript artifacts. |
| `episode_summary` | `by_conversation` | 1 transcript → 1 episode summary via LLM. |
| `monthly_rollup` | `by_month` | Groups episodes by calendar month, synthesizes each via LLM. |
| `topical_rollup` | `by_topic` | Groups episodes by user-declared topics. Requires `config={"topics": [...]}`. |
| `core_synthesis` | `single` | All rollups → single core memory document. Respects `context_budget`. |
| `merge` | — | Groups artifacts by content similarity (Jaccard), merges above threshold. |

### Projections

| Type | Output | Purpose |
|------|--------|---------|
| `search_index` | `build/search.db` | SQLite FTS5 index across selected layers. Optional embedding support for semantic/hybrid search. |
| `flat_file` | `build/context.md` | Renders artifacts as markdown. Ready to paste into an LLM system prompt. |

### Validators

| Name | What it checks |
|------|---------------|
| `mutual_exclusion` | Merged artifacts don't mix values of a metadata field (e.g., `customer_id`). |
| `required_field` | Artifacts in specified layers have a required metadata field. |
| `pii` | Detects credit cards, SSNs, emails, phone numbers in content. |
| `semantic_conflict` | LLM-based detection of contradictions across synthesized artifacts. |

### Fixers

| Name | What it fixes |
|------|--------------|
| `semantic_enrichment` | Resolves semantic conflicts by rewriting with source episode context. Interactive approval. |

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
| `uvx synix validate` | Run declared validators against build artifacts. |
| `uvx synix fix` | LLM-assisted repair of validation violations. |
| `uvx synix verify` | Check build integrity (hashes, provenance). |
| `uvx synix lineage <id>` | Show the full provenance chain for an artifact. |
| `uvx synix clean` | Delete the build directory. |

Commands that take a pipeline path (`build`, `plan`, `validate`, `fix`, `clean`) default to `./pipeline.py` in the current directory.

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
