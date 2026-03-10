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
uvx synix release HEAD --to local
uvx synix search "return policy" --release local
```

## Quick Start

```bash
uvx synix init my-project
cd my-project
```

Add your API key (see `pipeline.py` for provider config), then build and release:

```bash
uvx synix build                   # produce immutable snapshot in .synix/
uvx synix release HEAD --to local # materialize projections (search.db, context.md)
```

Browse, search, and inspect:

```bash
uvx synix list                    # all artifacts, grouped by layer
uvx synix show final-report       # render an artifact
uvx synix search "hiking" --release local  # search a released projection
uvx synix runs list               # immutable build snapshots
uvx synix releases list           # all named releases and their snapshots
uvx synix validate                # run declared validators (experimental)
```

All build state lives under `.synix/` — content-addressed objects, snapshots, refs, and releases. There is no `build/` directory. `uvx synix clean` removes release targets and transient work state; it does not delete snapshot history.

> **Note:** The `.synix` on-disk format may evolve before `v1.0`. Objects are schema-versioned, and future changes will preserve a compatibility path rather than silently reusing incompatible state.

## Defining a Pipeline

A pipeline is a Python file. Layers are real objects with dependencies expressed as object references.

```python
# pipeline.py
from synix import Pipeline, SearchSurface, Source, SynixSearch
from synix.transforms import MapSynthesis, ReduceSynthesis

pipeline = Pipeline("my-pipeline")
pipeline.source_dir = "./sources"
pipeline.llm_config = {
    "provider": "anthropic",
    "model": "claude-haiku-4-5-20251001",
    "temperature": 0.3,
    "max_tokens": 1024,
}

# Parse source files
bios = Source("bios", dir="./sources/bios")

# 1:1 — apply a prompt to each input
work_styles = MapSynthesis(
    "work_styles",
    depends_on=[bios],
    prompt="Infer this person's work style in 2-3 sentences:\n\n{artifact}",
    artifact_type="work_style",
)

# N:1 — combine all inputs into one output
report = ReduceSynthesis(
    "report",
    depends_on=[work_styles],
    prompt="Write a team analysis from these profiles:\n\n{artifacts}",
    label="team-report",
    artifact_type="report",
)

report_search = SearchSurface(
    "report-search",
    sources=[work_styles, report],
    modes=["fulltext"],
)

pipeline.add(bios, work_styles, report, report_search)
pipeline.add(SynixSearch("search", surface=report_search))
```

This is a complete, working pipeline. `uvx synix build pipeline.py` runs it.

`SearchSurface` is the build-time search capability. `SynixSearch` is the canonical local search output. `SearchIndex` still works as a compatibility API, but new pipelines should use surfaces plus `SynixSearch`.

Compatibility migration during `v0.x`:

```python
from synix import SearchIndex

pipeline.add(SearchIndex("search", sources=[report], search=["fulltext"]))
```

Existing `SearchIndex` pipelines remain supported during the current `v0.x` migration window. New templates and docs use `SearchSurface + SynixSearch`, and any future deprecation will ship with an explicit migration note instead of a silent break.

Projections are not materialized at build time. `synix build` records projection declarations in the manifest; `synix release HEAD --to <name>` materializes them into `.synix/releases/<name>/` (search.db, context.md). Search queries target a named release: `uvx synix search "query" --release local`.

For the full pipeline API, built-in transforms, validators, and advanced patterns, see [docs/pipeline-api.md](docs/pipeline-api.md).

## Platform Transforms (`synix.transforms`)

The `synix.transforms` module provides generic transform shapes — no custom subclass needed for common flows.

```python
from synix.transforms import MapSynthesis, GroupSynthesis, ReduceSynthesis, FoldSynthesis, Chunk
```

| Transform | Pattern | Use when... |
|-----------|---------|-------------|
| `MapSynthesis` | 1:1 | Each input gets its own LLM call |
| `GroupSynthesis` | N:M | Group inputs by a metadata key, one output per group |
| `ReduceSynthesis` | N:1 | All inputs become a single output |
| `FoldSynthesis` | N:1 sequential | Accumulate through inputs one at a time |
| `Chunk` | 1:N | Split each input into smaller pieces (no LLM) |

LLM-backed transforms take a `prompt` string with placeholders like `{artifact}`, `{artifacts}`, `{group_key}`, `{accumulated}`. Changing the prompt automatically invalidates the cache. `Chunk` is pure text processing — configure with `chunk_size`, `separator`, or a custom `chunker` callable.

For full parameter reference and examples of each, see [docs/pipeline-api.md#generic-transforms-synixtransforms](docs/pipeline-api.md#generic-transforms-synixtransforms).

When you need logic beyond prompt templating — filtering, conditional branching, multi-step chains — write a [custom Transform subclass](docs/pipeline-api.md#custom-transforms).

## Bundled Ext Transforms (`synix.ext`)

Synix also ships a small set of opinionated memory-oriented transforms in `synix.ext`:

| Class | What it does |
|-------|-------------|
| `EpisodeSummary` | 1 transcript → 1 episode summary |
| `MonthlyRollup` | Group episodes by month, synthesize each |
| `TopicalRollup` | Group episodes by user-defined topics |
| `CoreSynthesis` | All rollups → single core memory document |

These are bundled convenience transforms, not the generic platform primitives.

## Other Built-ins

Import from `synix.transforms`:

| Class | What it does |
|-------|-------------|
| `Merge` | Group artifacts by content similarity (Jaccard) |

## CLI Reference

| Command | What it does |
|---------|-------------|
| `uvx synix init <name>` | Scaffold a new project with sources, pipeline, and README |
| `uvx synix build` | Run the pipeline. Only rebuilds what changed. Produces immutable snapshot in `.synix/` |
| `uvx synix plan` | Dry-run — show what would build without running transforms |
| `uvx synix plan --explain-cache` | Plan with inline cache decision reasons |
| `uvx synix release HEAD --to <name>` | Materialize projections from a snapshot to a named release target |
| `uvx synix revert <ref> --to <name>` | Release an older snapshot to a release target |
| `uvx synix releases list` | List all named releases with their snapshots |
| `uvx synix releases show <name>` | Display release receipt details. `--json` for machine-readable |
| `uvx synix refs list` | List all refs (build runs + releases) |
| `uvx synix refs show <ref>` | Resolve a ref to snapshot details |
| `uvx synix runs list` | List immutable build snapshots recorded under `.synix` |
| `uvx synix list [layer]` | List all artifacts, optionally filtered by layer. Reads from `.synix/` via SnapshotView |
| `uvx synix show <id>` | Display an artifact. Resolves by label or ID prefix. `--raw` for JSON |
| `uvx synix search <query>` | Search a release target. `--release <name>` to pick target, `--mode hybrid` for semantic |
| `uvx synix validate` | *(Experimental)* Run validators against build artifacts |
| `uvx synix fix` | *(Experimental)* LLM-assisted repair of violations |
| `uvx synix lineage <id>` | Show the full provenance chain for an artifact. Reads from `.synix/` |
| `uvx synix clean` | Remove release targets and transient work state from `.synix/` |
| `uvx synix batch-build plan` | *(Experimental)* Dry-run showing which layers would batch vs sync |
| `uvx synix batch-build run` | *(Experimental)* Submit a batch build via OpenAI Batch API. `--poll` to wait |
| `uvx synix batch-build resume <id>` | *(Experimental)* Resume a previously submitted batch build |
| `uvx synix batch-build list` | *(Experimental)* Show all batch build instances and their status |
| `uvx synix batch-build status <id>` | *(Experimental)* Detailed status for a specific batch build. `--latest` for most recent |
| `uvx 'synix[mesh]' mesh create` | *(Experimental)* Create a new mesh with config and token |
| `uvx 'synix[mesh]' mesh provision` | *(Experimental)* Join this machine to a mesh as server or client |
| `uvx 'synix[mesh]' mesh status` | *(Experimental)* Show mesh health, members, and last build |
| `uvx 'synix[mesh]' mesh list` | *(Experimental)* List all meshes on this machine |

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

## Mesh — Distributed Builds (Experimental)

> **Warning:** Mesh is experimental. Commands, configuration, and behavior may change in future releases.

Synix Mesh distributes pipeline builds across machines over a private network (Tailscale). A central server receives source files from clients, runs builds, and distributes artifact bundles back. Clients automatically watch local directories, submit new files, and pull results.

```bash
# Mesh needs the [mesh] extra for its dependencies
uvx 'synix[mesh]' mesh create --name my-mesh --pipeline ./pipeline.py
uvx 'synix[mesh]' mesh provision --name my-mesh --role server
uvx 'synix[mesh]' mesh provision --name my-mesh --role client --server server-host:7433

# Check status
uvx 'synix[mesh]' mesh status --name my-mesh
```

All mesh state persists in `~/.synix-mesh/` on disk. Features: debounced build scheduling, ETag-based artifact distribution, shared-token auth, automatic leader election with term-based fencing, deploy hooks, webhook notifications.

See [docs/mesh.md](docs/mesh.md) for the full guide — configuration, server API, failover protocol, security model, and data layout.

## MCP Server — Agent Integration

Synix exposes its full SDK as an [MCP](https://modelcontextprotocol.io) server, so AI agents can manage pipelines, build memory, and search — all via structured tool calls over stdin/stdout.

```bash
# Start directly
uvx 'synix[mcp]' python -m synix.mcp

# Or configure in your MCP client (Claude Desktop, etc.)
```

```json
{
    "mcpServers": {
        "synix": {
            "command": "uvx",
            "args": ["--from", "synix[mcp]", "python", "-m", "synix.mcp"],
            "env": {"SYNIX_PROJECT": "/path/to/project"}
        }
    }
}
```

20 tools covering the full lifecycle: `init_project`, `open_project`, `load_pipeline`, `build`, `release`, `search`, `source_add_text`, `source_add_file`, `get_artifact`, `lineage`, and more.

See [docs/mcp.md](docs/mcp.md) for the full tool reference and agent workflow examples.

## Key Capabilities

**Incremental rebuilds** — Change a prompt or add new sources. Only downstream artifacts reprocess.

**Full provenance** — Every artifact chains back to the source conversations that produced it. `uvx synix lineage <id>` shows the full tree.

**Fingerprint-based caching** — Build fingerprints capture inputs, prompts, model config, and transform source code. Change any component and only affected artifacts rebuild. See [docs/cache-semantics.md](docs/cache-semantics.md).

**Altitude-aware search** — Query across episode summaries, rollups, or core memory via named release targets. Drill into provenance from any result.

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

## Learn More

| Doc | Contents |
|-----|----------|
| [Pipeline API](docs/pipeline-api.md) | Full Python API — ext transforms, built-in transforms, projections, validators, custom transforms |
| [Projection Release v2](docs/projection-release-v2-rfc.md) | Build/release separation, `.synix/` as single source of truth, adapter contract, release receipts |
| [Search Surface RFC](docs/search-surface-rfc.md) | Build-time search capabilities and search surface declarations |
| [Entity Model](docs/entity-model.md) | Artifact identity, storage format, cache logic |
| [Cache Semantics](docs/cache-semantics.md) | Rebuild trigger matrix, fingerprint scheme |
| [Batch Build](docs/batch-build.md) | *(Experimental)* OpenAI Batch API for 50% cost reduction |
| [Mesh](docs/mesh.md) | *(Experimental)* Distributed builds across machines via Tailscale |
| [Python SDK](docs/sdk.md) | Programmatic access — init, build, release, search from Python |
| [MCP Server](docs/mcp.md) | Agent integration — 20 MCP tools, client configuration, workflows |
| [CLI UX](docs/cli-ux.md) | Output formatting, color scheme |

## Links

- [synix.dev](https://synix.dev)
- [GitHub](https://github.com/marklubin/synix)
- [llms.txt](./llms.txt) — machine-readable project summary for LLMs
- [Issue tracker](https://github.com/marklubin/synix/issues) — known limitations and roadmap
- MIT License
