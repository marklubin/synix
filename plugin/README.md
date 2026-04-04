# Synix Plugin for Claude Code

Persistent cross-session memory for Claude Code via a Synix knowledge server.

## What It Does

- **Session start hook** — fetches the synthesized context document from the Synix server and injects it into the conversation, so Claude starts every session with full memory
- **Session end hook** — pushes the session transcript (`.jsonl`) to the Synix server for ingestion into the next build cycle
- **MCP server** — connects to the Synix knowledge server's MCP endpoint, exposing search, ingest, context retrieval, and bucket listing tools
- **Memory skill** — teaches Claude when and how to use the memory tools (search prior work, ingest notes, retrieve context)

## Prerequisites

- A running Synix knowledge server (see [Synix docs](https://github.com/marklubin/synix))
- Network access to the server from your machine (e.g. same LAN, Tailscale, SSH tunnel)
- `bash`, `curl`, `python3` (standard on macOS and Linux)
- Claude Code CLI installed (`npm install -g @anthropic-ai/claude-code`)

## Installation

### Quick start (development / testing)

Run Claude Code with the `--plugin-dir` flag pointing to this directory:

```bash
claude --plugin-dir /path/to/synix/plugin
```

### Persistent install

1. **Add the synix repo as a plugin marketplace** (one-time):

   In Claude Code, run:
   ```
   /plugin marketplace add https://github.com/marklubin/synix
   ```

2. **Install the plugin**:

   ```
   /plugin install synix@synix --scope user
   ```

   Scope options:
   - `user` — available in all projects on this machine
   - `project` — available only in the current project (stored in `.claude/settings.json`)
   - `local` — available only for you in this project (stored in `.claude/settings.local.json`)

3. **Reload plugins** in your current session:

   ```
   /reload-plugins
   ```

### Verify installation

Run `/plugin` in Claude Code and check the **Installed** tab — you should see "synix" listed and enabled.

Then test the MCP connection:

```
Ask Claude: "Can you list the available memory buckets?"
```

If the server is reachable, Claude will call `list_buckets()` and return the configured buckets.

## Configuration

The plugin connects to a Synix knowledge server. The server URL defaults to `http://salinas:8200`.

To use a different server, set the `SYNIX_SERVER_URL` environment variable before launching Claude Code:

```bash
export SYNIX_SERVER_URL="http://your-server:8200"
claude
```

Or for a one-off:

```bash
SYNIX_SERVER_URL="http://localhost:8200" claude
```

### MCP endpoint

The `.mcp.json` file configures the MCP server connection. If your server is at a different address, edit `.mcp.json`:

```json
{
  "synix": {
    "type": "streamable-http",
    "url": "http://your-server:8200/mcp/"
  }
}
```

**Note**: Both the hooks and the MCP server need to reach the same Synix server. The hooks use `SYNIX_SERVER_URL` (with fallback), while the MCP connection uses the URL in `.mcp.json`. Make sure these point to the same server.

## How It Works

### Session start (`hooks/session_start.sh`)

On every new session, the hook calls `GET /api/v1/flat-file/context-doc` to fetch the current synthesized context document. This is injected into the conversation as `additionalContext`, so Claude immediately has full memory context — recent activity, open threads, research state, user model, and work status.

If the server is unreachable, the hook silently succeeds with no context injected. Sessions are never blocked by a down server.

### Session end (`hooks/session_end.sh`)

When a session ends, the hook:
1. Reads the `sessionId` from the hook input JSON
2. Locates the session transcript at `~/.claude/projects/*/{sessionId}.jsonl`
3. POSTs the transcript content to `POST /api/v1/ingest/sessions`

This feeds the session into the Synix build pipeline. On the next auto-build cycle, the server will process it through episode summarization, fold synthesis, and context generation.

If the server is unreachable or the session file can't be found, the hook silently succeeds.

### MCP tools

The MCP server exposes the full Synix tool surface. Key tools for memory:

| Tool | Purpose |
|------|---------|
| `search(query, layers?, limit?)` | Hybrid search across memory (fulltext + semantic) |
| `ingest(bucket, content, filename)` | Write content to a source bucket |
| `get_context(name?)` | Retrieve a synthesized context document |
| `list_buckets()` | List available ingestion buckets |

### Memory skill (`skills/memory/SKILL.md`)

The skill teaches Claude when to proactively use memory tools:
- **Search** when the user references prior work, decisions, or context
- **Ingest** after calls, meetings, or when the user says "remember this"
- **Get context** when starting a complex task where historical context helps

## Structure

```
plugin/
  .claude-plugin/
    plugin.json              Plugin manifest (name, version, pointers to hooks/mcp)
  .mcp.json                  MCP server configuration (streamable-http to synix)
  hooks/
    hooks.json               Hook registration (SessionStart, Stop)
    session_start.sh         Fetches context on session start
    session_end.sh           Pushes transcript on session end
  skills/
    memory/
      SKILL.md               Memory skill definition
  README.md                  This file
```

## Troubleshooting

### "Cannot connect to MCP server"

- Verify the Synix server is running: `curl -sf http://your-server:8200/api/v1/buckets`
- Check network access (Tailscale, firewall, etc.)
- Verify the URL in `.mcp.json` matches your server

### Sessions not appearing in memory

- Check that the session end hook is registered: run `/plugin` and look for hook errors
- Verify the server is reachable from the hook: `curl -sf http://your-server:8200/api/v1/buckets`
- Check server logs for ingest errors
- Auto-build has a cooldown (default 300s) — the session won't appear in search until the next build completes

### No context injected at session start

- The `context-doc` projection must exist — run a build + release on the server first
- Check that `SYNIX_SERVER_URL` is set correctly (or that the default `http://salinas:8200` resolves on your network)
