# Getting Started

Build your first memory pipeline in 5 minutes.

## What you'll build

A pipeline that takes source documents, produces episode summaries, rolls them up by month, and creates a searchable core memory document. Every insight traces back to its source.

## Prerequisites

- Python 3.11+
- An API key: [Anthropic](https://console.anthropic.com/) or [OpenAI](https://platform.openai.com/)

## Step 1: Create a project

```bash
uvx synix init my-project --template 01-chatbot-export-synthesis
cd my-project
```

This creates:
- `sources/` — where your source data goes
- `pipeline_monthly.py` — a working pipeline definition
- `.env.example` — template for your API key

Copy the env template and add your key:

```bash
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=sk-ant-...
```

## Step 2: Add source data

Drop your data into `sources/`. The template includes a sample session transcript to start with.

Sources can be anything — conversation exports (ChatGPT, Claude JSON), documents, reports, notes, markdown, plain text. Synix detects common formats automatically.

## Step 3: Build

```bash
uvx synix build pipeline_monthly.py
```

What happens:
1. **Sources parsed** — files are read from `./sources/`
2. **Episodes generated** — each source gets an episode summary (1:1, via `EpisodeSummary`)
3. **Monthly rollups** — episodes are grouped by month and synthesized (N:M, via `MonthlyRollup`)
4. **Core memory** — all rollups are compressed into a single core memory document (N:1, via `CoreSynthesis`)

Each step only runs if its inputs or prompts changed. Re-run `synix build` after adding new sources — only the new episodes process; everything else is cached.

## Step 4: Explore

```bash
uvx synix list                          # all artifacts, grouped by layer
uvx synix show episode-2024-03-15       # render one episode
uvx synix lineage episode-2024-03-15    # trace it back to its source
```

The lineage command is the provenance chain — you can see exactly which source produced each episode, which episodes produced each monthly rollup, and which rollups produced the core memory.

## Step 5: Release and search

Build produces immutable artifacts. Release materializes them into a searchable index:

```bash
uvx synix release HEAD --to local
uvx synix search "how did I feel about the move" --release local
```

Results show matching artifacts with scores. Add `--trace` to see provenance for each result:

```bash
uvx synix search "how did I feel about the move" --release local --trace
```

## Step 6: Change something

Open `pipeline_monthly.py` and edit the episode summarization prompt, or change `MonthlyRollup` to `TopicalRollup` to cluster by theme instead of calendar month.

```bash
uvx synix build pipeline_monthly.py
```

Only downstream artifacts rebuild. If you changed the episode prompt, episodes regenerate but the source parsing is cached. If you added a new source file, only the new episode processes — existing episodes and their downstream rollups stay cached.

## Step 7: Use it from your agent

After releasing, your agent can query memory at inference time:

```python
import synix

project = synix.open_project("./my-project")
mem = project.release("local")

# Search memory
results = mem.search("return policy", mode="hybrid", limit=5)
for r in results:
    print(f"[{r.layer}] {r.label} ({r.score:.2f})")
    print(f"  {r.content[:200]}...")

# Or load the core memory as flat context
context = mem.flat_file("context-doc")
# → inject into your agent's system prompt
```

Synix runs offline — it processes sources into structured memory. Your agent reads the output at runtime. They're decoupled.

## What's next

- **Customize the pipeline:** [Pipeline API](pipeline-api.md) — all transform types, projections, validators
- **Integrate with your agent:** [Integration Guide](integration.md) — SDK, MCP server, CLI, direct file access
- **Understand the architecture:** [Architecture](architecture.md) — the four-tier memory model
- **Try other templates:** `uvx synix init --list-templates` to see all available templates
- **Learn about caching:** [Cache Semantics](cache-semantics.md) — how Synix decides what to rebuild
