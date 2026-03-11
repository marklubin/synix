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

<h3 align="center">Programmable memory for AI agents.</h3>

## Get started in 60 seconds

```bash
uvx synix init my-project
cd my-project
cp .env.example .env              # add your API key
# add your source data to ./sources/
uvx synix build
uvx synix release HEAD --to local
uvx synix search "your query" --release local
```

That's it. You get episode summaries, monthly rollups, a core memory document, and full-text search — with every insight traced back to its source.

## What just happened

Synix processed your sources through a **pipeline** — a directed graph of transforms you define in Python:

1. **Sources** — raw data was parsed from `./sources/`
2. **Episodes** — each source got an LLM-generated summary (1:1)
3. **Monthly rollups** — episodes were grouped by month and synthesized (N:M)
4. **Core memory** — all rollups were compressed into a single document (N:1)
5. **Search index** — everything was indexed for full-text search

The template gave you a working pipeline. When you need to change it — different prompts, different grouping, different layers — you edit `pipeline.py`. Same tool, no migration.

## The problem Synix solves

Every agent memory tool — Mem0, Letta, Zep, LangMem — gives you one flat bucket. Same storage, same rules, same lifecycle for everything your agent knows. A fact learned 30 seconds ago and a preference built over 50 sessions get the same treatment.

When memory breaks, it breaks silently — contradictions, stale context, hallucinated recall. And when you want to change how memory works, you're looking at a migration or starting over.

Synix lets you **program** how memory works — define the layers, write the prompts, control the lifecycle. Change your memory architecture and only affected layers rebuild. Trace any output back to the source that produced it.

## How your agent uses the output

After `synix release`, your agent queries memory at inference time:

```python
import synix

project = synix.open_project("./my-project")
mem = project.release("local")

# Search memory
results = mem.search("return policy", limit=5)
for r in results:
    print(f"[{r.layer}] {r.label} ({r.score:.2f})")

# Or load core memory as flat context
context = mem.flat_file("context-doc")
# → inject into your agent's system prompt
```

Synix runs offline. Your agent reads the output at runtime. They're decoupled — Synix doesn't need to be running while your agent serves requests.

**Other integration options:** [MCP server](docs/mcp.md) for Claude/Cursor, [CLI](docs/integration.md#option-3-cli-from-automation) for automation, or [direct SQLite access](docs/integration.md#option-4-direct-file-access) to `search.db`. See the [Integration Guide](docs/integration.md).

## Customize everything

The template gave you a default pipeline. Open `pipeline.py` to see what's inside:

```python
from synix import FlatFile, Pipeline, SearchSurface, Source, SynixSearch
from synix.ext import CoreSynthesis, EpisodeSummary, MonthlyRollup

pipeline = Pipeline("agent-memory")
pipeline.source_dir = "./sources"
pipeline.llm_config = {
    "provider": "anthropic",
    "model": "claude-haiku-4-5-20251001",
}

transcripts = Source("transcripts")
episodes = EpisodeSummary("episodes", depends_on=[transcripts])
monthly = MonthlyRollup("monthly", depends_on=[episodes])
core = CoreSynthesis("core", depends_on=[monthly])

memory_search = SearchSurface(
    "memory-search",
    sources=[episodes, monthly, core],
    modes=["fulltext"],
)

pipeline.add(transcripts, episodes, monthly, core, memory_search)
pipeline.add(SynixSearch("search", surface=memory_search))
pipeline.add(FlatFile("context-doc", sources=[core]))
```

Change a prompt → only downstream artifacts rebuild. Add new sources → only new episodes process. Swap `MonthlyRollup` for `TopicalRollup` → source parsing and episodes stay cached. No migrations.

For custom pipelines beyond the built-in transforms, use the generic transform shapes:

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

See [Pipeline API](docs/pipeline-api.md) for the full reference.

## Where Synix fits

|                          | Mem0    | Letta   | Graphiti | LangMem | **Synix**       |
|--------------------------|---------|---------|----------|---------|-----------------|
| **Approach**             | Memory API | Agent-managed blocks | Temporal knowledge graph | Taxonomy extraction | Programmable architecture |
| **You define the rules** | No      | No      | No       | No      | Yes — in Python |
| **Change architecture**  | Migration | Migration | Migration | Migration | Incremental rebuild |
| **Provenance**           | —       | —       | —        | —       | Full chain to source |
| **Memory lifecycle**     | —       | —       | —        | —       | Per-layer rules |
| **Schema**               | Fixed   | Fixed   | Fixed    | Fixed   | You define it   |

Synix is not a memory store. Mem0/Letta/Graphiti store and retrieve memories. Synix is the system that **produces** structured memory from raw sources — conversations, documents, reports, transactions, anything — with full provenance, incremental rebuilds, and an architecture you control.

**When Synix is the right choice:** You want to control how memory is structured, not just store things. You need provenance. You expect your memory architecture to evolve. You want different layers with different rules.

**When something else is better:** You just need a key-value memory store (→ Mem0). You need a knowledge graph (→ Graphiti). You need real-time memory management during inference (→ not yet — planned).

## Current status

- **Stable:** Pipeline definition, build, release, search, provenance, incremental rebuilds, CLI, Python SDK
- **Works but early:** Templates, MCP server for agent integration
- **Experimental:** Validation/repair, batch builds (OpenAI Batch API), distributed builds (mesh)
- **Not built yet:** Real-time runtime, agent-driven memory evolution, multi-tenancy

Synix is a working tool used in production for personal and project memory pipelines. It is pre-1.0 — on-disk formats may evolve with a compatibility path. Solo-maintainer project.

## Templates

```bash
uvx synix init --list    # see all templates
```

| Template | What it does | Best for |
|----------|-------------|----------|
| `08-agent-memory` | Session logs → episodes → rollups → core memory + search | **Agent memory across sessions (default)** |
| `01-chatbot-export-synthesis` | Chat exports → episodes → rollups → core memory | Personal AI, journal apps |
| `07-chunked-search` | Documents → chunks → search index | RAG, document Q&A |
| `03-team-report` | Bios + briefs → analysis → searchable report | Team knowledge bases |
| `02-tv-returns` | Customer calls → episodes → trends → report | Support analysis |
| `04-sales-deal-room` | Deal docs → analysis → search | Sales intelligence |

## CLI Reference

| Command | What it does |
|---------|-------------|
| `uvx synix init <name>` | Scaffold a new project from a template |
| `uvx synix build` | Run the pipeline. Only rebuilds what changed. |
| `uvx synix plan` | Dry-run — show what would build. `--explain-cache` for details |
| `uvx synix release HEAD --to <name>` | Materialize search index + flat files from a build |
| `uvx synix search <query>` | Search a release. `--release <name>`, `--mode hybrid`, `--trace` |
| `uvx synix list [layer]` | List all artifacts, optionally filtered by layer |
| `uvx synix show <id>` | Display an artifact. `--raw` for JSON |
| `uvx synix lineage <id>` | Full provenance chain for an artifact |
| `uvx synix releases list` | List all named releases |
| `uvx synix runs list` | List immutable build snapshots |
| `uvx synix clean` | Remove release targets and work state |
| `uvx synix validate` | *(Experimental)* Run validators |
| `uvx synix batch-build run` | *(Experimental)* Batch build via OpenAI Batch API |
| `uvx 'synix[mesh]' mesh create` | *(Experimental)* Distributed builds via Tailscale |

## Architecture direction

Synix is designed around the idea that different kinds of memory need different management — four tiers with distinct physics, from execution context (milliseconds) to identity (permanent). Today, Synix manages the experience tier via programmable pipelines. The architecture is designed to expand across all four tiers — and eventually, agents will program their own memory.

See [docs/architecture.md](docs/architecture.md) for the full picture.

## Learn More

| Doc | Contents |
|-----|----------|
| [Getting Started](docs/getting-started.md) | Build your first pipeline in 5 minutes |
| [Architecture](docs/architecture.md) | The four-tier memory model and where Synix fits |
| [Integration Guide](docs/integration.md) | How your agent uses Synix output — SDK, MCP, CLI, direct access |
| [Pipeline API](docs/pipeline-api.md) | Full Python API — transforms, projections, validators, custom transforms |
| [Python SDK](docs/sdk.md) | Programmatic access — init, build, release, search |
| [MCP Server](docs/mcp.md) | Agent integration — 20 MCP tools, client configuration |
| [Cache Semantics](docs/cache-semantics.md) | Fingerprint scheme, rebuild triggers |
| [Batch Build](docs/batch-build.md) | *(Experimental)* OpenAI Batch API — 50% cost reduction |
| [Mesh](docs/mesh.md) | *(Experimental)* Distributed builds across machines |

## Links

- [synix.dev](https://synix.dev)
- [GitHub](https://github.com/marklubin/synix)
- [Issue tracker](https://github.com/marklubin/synix/issues)
- MIT License
