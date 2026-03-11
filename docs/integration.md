# Integration Guide

How to use Synix memory from your agent.

## The model

Synix runs offline — it processes conversations into structured memory artifacts. Your agent reads the output at inference time. They're decoupled.

```
conversations ──→ synix build ──→ artifacts (immutable)
                                       │
                              synix release
                                       │
                                       ▼
                              search.db + context.md
                                       │
                         your agent reads at runtime
```

You rebuild when new conversations arrive. Your agent always reads the latest release. They don't need to be running at the same time.

## Option 1: Python SDK

The most direct integration. Import `synix`, open your project, and query the release.

### Search memory

```python
import synix

project = synix.open_project("/path/to/my-project")
mem = project.release("local")

# Keyword search (fastest, no embeddings needed)
results = mem.search("return policy", mode="keyword", limit=5)

# Hybrid search (keyword + semantic, requires embedding_config in pipeline)
results = mem.search("return policy", mode="hybrid", limit=5)

for r in results:
    print(f"[{r.layer}] {r.label}: {r.score:.2f}")
    print(f"  Source: {' → '.join(r.provenance)}")
    print(f"  {r.content[:300]}")
```

Each result includes:
- `content` — the matching artifact text
- `label` — artifact identifier
- `layer` — which pipeline layer produced it (e.g., "episodes", "monthly")
- `score` — relevance score
- `provenance` — list of ancestor labels back to the source
- `metadata` — arbitrary metadata from the pipeline

### Filter by layer

Search specific layers to control the level of detail:

```python
# Only search episode summaries (granular)
results = mem.search("return policy", layers=["episodes"], limit=5)

# Only search the core memory (high-level)
results = mem.search("return policy", layers=["core"], limit=5)
```

### Load flat context

If your pipeline includes a `FlatFile` projection, you can load it directly into your agent's system prompt:

```python
context = mem.flat_file("context-doc")
# context is a string — the rendered markdown of your core memory
# inject it into your agent's system prompt
```

### Read specific artifacts

```python
# Get a specific artifact by label
core = mem.artifact("core-memory")
print(core.content)

# Walk provenance
lineage = mem.lineage("core-memory")
for ancestor in lineage:
    print(f"← {ancestor.label} ({ancestor.layer})")
```

### Full lifecycle from Python

```python
import synix

# Open project and load pipeline
project = synix.open_project("./my-project")
project.load_pipeline()

# Add a new conversation
src = project.source("transcripts")
src.add_text(conversation_text, label="session-2026-03-10")

# Rebuild (incremental — only new artifacts process)
result = project.build()
print(f"Built: {result.built}, Cached: {result.cached}")

# Release and search
project.release_to("local")
mem = project.release("local")
results = mem.search("what happened today")
```

### Error handling

```python
from synix.sdk import (
    SynixNotFoundError,        # no .synix/ directory found
    ReleaseNotFoundError,      # named release doesn't exist
    ArtifactNotFoundError,     # label not in snapshot
    EmbeddingRequiredError,    # semantic search requested but no embeddings
    PipelineRequiredError,     # operation needs a loaded pipeline
)
```

## Option 2: MCP Server

For agents running in Claude Desktop, Cursor, or any MCP-compatible host. Synix exposes its full SDK as MCP tools over stdio.

### Configure your MCP client

```json
{
    "mcpServers": {
        "synix": {
            "command": "uvx",
            "args": ["--from", "synix[mcp]", "python", "-m", "synix.mcp"],
            "env": {
                "SYNIX_PROJECT": "/path/to/my-project"
            }
        }
    }
}
```

### Available tools

| Tool | What it does |
|------|-------------|
| `open_project` | Open a synix project directory |
| `build` | Run the pipeline (incremental) |
| `release` | Materialize search index from latest build |
| `search` | Query the search index |
| `get_artifact` | Read a specific artifact by label |
| `lineage` | Walk provenance chain for an artifact |
| `list_artifacts` | List all artifacts, optionally filtered by layer |
| `list_layers` | List layers with artifact counts |
| `get_flat_file` | Read a flat file projection (e.g., context.md) |
| `source_add_text` | Add a conversation as text |
| `source_add_file` | Add a conversation from a file path |
| `source_list` | List source files |
| `list_releases` | List all named releases |

### Agent workflow

A typical agent session:

1. `open_project` → connect to the memory project
2. `search` → retrieve relevant memory for the current query
3. Use results in the response
4. `source_add_text` → save the current conversation as a new source
5. `build` → rebuild memory (incremental)
6. `release` → update the search index

See [docs/mcp.md](mcp.md) for the full tool reference.

## Option 3: CLI from automation

Pipe Synix CLI output into your automation scripts:

```bash
# Search with JSON output
uvx synix search "return policy" --release local --mode keyword --limit 5

# Get a specific artifact
uvx synix show core-memory

# List all artifacts in a layer
uvx synix list episodes
```

Useful for cron jobs, CI/CD pipelines, or shell-based agent frameworks.

## Option 4: Direct file access

The release directory contains standard files you can read directly:

```
.synix/releases/local/
├── search.db       # SQLite FTS5 database — query with any SQLite client
├── context.md      # Flat markdown file — load as agent context
└── receipt.json    # Release metadata (snapshot ref, timestamps, adapter status)
```

### Query search.db directly

```python
import sqlite3

conn = sqlite3.connect(".synix/releases/local/search.db")
cursor = conn.execute(
    "SELECT label, layer, content, rank FROM search WHERE search MATCH ? ORDER BY rank LIMIT 5",
    ("return policy",)
)
for row in cursor:
    print(f"[{row[1]}] {row[0]}: {row[3]:.2f}")
```

### Load context.md

```python
with open(".synix/releases/local/context.md") as f:
    context = f.read()
# inject into system prompt
```

## When to rebuild

Rebuild when new conversations arrive. Synix handles the rest:

- **New sources:** Only new episodes process. Existing artifacts stay cached.
- **Changed prompts:** Only downstream artifacts rebuild.
- **Changed model config:** Affected layers rebuild (fingerprint includes model settings).
- **Nothing changed:** Build completes instantly (everything cached).

Automate rebuilds with cron, a file watcher, or trigger from your application when new conversations are saved.

```bash
# Example cron: rebuild every hour
0 * * * * cd /path/to/project && uvx synix build && uvx synix release HEAD --to local
```

## Search modes

| Mode | What it does | Requirements |
|------|-------------|--------------|
| `keyword` | BM25 full-text search | Default. No extra config. |
| `semantic` | Cosine similarity on embeddings | Requires `embedding_config` on `SearchSurface` |
| `hybrid` | Keyword + semantic with rank fusion | Requires `embedding_config` |
| `layered` | Search with layer-level weighting | Requires `embedding_config` |

If your pipeline only declares `modes=["fulltext"]`, use `keyword` mode. If it declares `modes=["fulltext", "semantic"]` with an `embedding_config`, all modes are available.

## Learn more

- [SDK Reference](sdk.md) — full Python API with all methods and types
- [MCP Server](mcp.md) — complete tool reference and configuration
- [Architecture](architecture.md) — why memory needs tiers
- [Getting Started](getting-started.md) — build your first pipeline
