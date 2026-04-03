# MCP Server (Full SDK)

Synix exposes its full SDK as an [MCP](https://modelcontextprotocol.io) server. AI agents can manage pipelines, build memory, and search via structured tool calls over stdin/stdout.

> **Two MCP surfaces exist:**
> - **This server** (`synix.mcp`) — full pipeline control (28 tools). For power users and local development. Runs via stdio.
> - **Knowledge server** (`synix serve`) — simplified agent interface (4 tools: `ingest`, `search`, `get_context`, `list_buckets`). For always-on deployment. Runs over HTTP. See [Knowledge Server](knowledge-server-architecture.md).

## Setup

Install with the `mcp` extra:

```bash
pip install synix[mcp]
# or
uvx --from 'synix[mcp]' python -m synix.mcp
```

### MCP Client Configuration

Add to your MCP client settings (Claude Desktop, Cursor, etc.):

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

Set `SYNIX_PROJECT` to auto-open a project on server start. Without it, the agent must call `init_project` or `open_project` first.

## Tool Reference

### Project Lifecycle

| Tool | Args | Description |
|------|------|-------------|
| `init_project` | `path` | Create a new synix project at path |
| `open_project` | `path="."` | Open an existing project (walks upward to find `.synix/`) |
| `load_pipeline` | `path=None` | Load a pipeline definition from a Python file |

### Build & Release

| Tool | Args | Description |
|------|------|-------------|
| `build` | `pipeline_path`, `dry_run`, `concurrency`, `timeout` | Run transforms, produce a snapshot |
| `release` | `name="local"`, `ref="HEAD"` | Materialize projections to a named release |

### Source Management

| Tool | Args | Description |
|------|------|-------------|
| `source_list` | `source_name` | List files in a source directory |
| `source_add_text` | `source_name`, `content`, `filename` | Create a text file in the source directory |
| `source_add_file` | `source_name`, `file_path` | Copy an existing file into the source directory |
| `source_remove` | `source_name`, `filename` | Remove a file from the source directory |
| `source_clear` | `source_name` | Remove all files from the source directory |

### Search

| Tool | Args | Description |
|------|------|-------------|
| `search` | `query`, `release_name`, `mode`, `limit`, `layers`, `surface` | Search memory in a named release |

Modes: `keyword` (FTS5), `semantic` (embeddings), `hybrid` (both + RRF), `layered`.

### Inspection

| Tool | Args | Description |
|------|------|-------------|
| `get_artifact` | `label`, `release_name` | Read a specific artifact by label |
| `list_artifacts` | `release_name`, `layer` | List artifacts (summary only) |
| `list_layers` | `release_name` | List all layers with artifact counts |
| `lineage` | `label`, `release_name` | Walk the provenance chain for an artifact |
| `get_flat_file` | `name`, `release_name` | Read a flat file projection's content |
| `list_releases` | — | List all named releases |
| `show_release` | `name` | Show release receipt (snapshot OID, timing, counts) |
| `list_refs` | — | List all refs (build snapshots + release pointers) |

### Admin

| Tool | Args | Description |
|------|------|-------------|
| `clean` | — | Remove releases and work directories |

## Typical Agent Workflow

```
1. open_project (or init_project for new)
2. load_pipeline (from a pipeline.py file)
3. source_add_text / source_add_file (feed data)
4. build (run transforms, produce snapshot)
5. release (materialize search indexes)
6. search / get_artifact / lineage (query memory)
```

## Error Handling

All tools propagate SDK exceptions as MCP error responses:

| Error | When |
|-------|------|
| `ValueError("No project open")` | Any tool called before `open_project`/`init_project` |
| `SynixNotFoundError` | `open_project` — no `.synix/` found |
| `PipelineRequiredError` | `build` — no pipeline loaded |
| `ReleaseNotFoundError` | Any release accessor with invalid release name |
| `ArtifactNotFoundError` | `get_artifact`/`lineage` with invalid label |
| `ProjectionNotFoundError` | `get_flat_file` with nonexistent projection |
| `SdkError` | `source_*` with undeclared source name or path traversal |

## Architecture

- **Single-tenant**: one project per server process, stored in `_state["project"]`
- **Transport**: stdio (stdin/stdout) via [FastMCP](https://github.com/jlowin/fastmcp)
- **No LLM calls**: the server itself makes no LLM calls — transforms do during `build()`
- **Path validation**: filenames are validated against path traversal at the SDK layer

## Testing

```bash
# Unit tests — direct function calls (no MCP transport)
uv run pytest tests/unit/test_mcp_tools.py -v

# E2E functional — full MCP protocol lifecycle (no API key)
uv run pytest tests/e2e/test_mcp_agent.py -k "not live_agent" -v

# Live agent — real LLM + MCP tools (requires OPENAI_API_KEY)
uv run pytest tests/e2e/test_mcp_agent.py -k live_agent -v --log-cli-level=INFO

# Override backend/model
AGENT_BACKEND=anthropic AGENT_MODEL=claude-sonnet-4-20250514 uv run pytest -k live_agent
```
