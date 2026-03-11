# Agent Memory Template

Give your agent memory that persists across sessions.

## What it does

Session transcripts → episode summaries → monthly rollups → core memory document + search index.

Every insight traces back to the session that produced it. Add new sessions and rebuild — only new episodes process; everything else is cached.

## Quick start

```bash
uvx synix init my-agent --template 08-agent-memory
cd my-agent
```

Add your API key:
```bash
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=sk-ant-...
```

Save a session transcript to `sources/`:
```bash
echo "User: What's the deployment process?\nAssistant: First you..." > sources/session-001.md
```

Build, release, search:
```bash
uvx synix build
uvx synix release HEAD --to local
uvx synix search "deployment process" --release local
```

## Use from your agent

```python
import synix

project = synix.open_project("./my-agent")
mem = project.release("local")

# Search memory at inference time
results = mem.search("deployment process", limit=5)
for r in results:
    print(f"[{r.layer}] {r.label}: {r.score:.2f}")

# Or load core memory as flat context
context = mem.flat_file("context-doc")
# → inject into system prompt
```

## What's in the pipeline

| Layer | Type | What it does |
|-------|------|-------------|
| `transcripts` | Source | Reads session files from `./sources/` |
| `episodes` | EpisodeSummary (1:1) | One structured summary per session |
| `monthly` | MonthlyRollup (N:M) | Groups episodes by month, synthesizes each |
| `core` | CoreSynthesis (N:1) | All rollups → single core memory document |
| `memory-search` | SearchSurface | Full-text search across all layers |
| `context-doc` | FlatFile | Core memory rendered as markdown |

## Customize

**Change the rollup strategy:** Replace `MonthlyRollup` with `TopicalRollup` for theme-based clustering (see comment in `pipeline.py`).

**Add semantic search:** Uncomment the `embedding_config` in `pipeline.py` to enable hybrid keyword+semantic search.

**Use a different model:** Change `pipeline.llm_config` to use OpenAI, or override per-layer for the core synthesis step.

**Add more layers:** Add a `ReduceSynthesis` or `FoldSynthesis` between rollups and core for multi-stage consolidation. See [Pipeline API](../../docs/pipeline-api.md).
