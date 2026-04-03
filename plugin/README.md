# Synix Plugin for Claude Code

Persistent cross-session memory for Claude Code via a Synix knowledge server.

## What It Does

- **Session start hook** — fetches synthesized context from the Synix server and injects it into the conversation automatically
- **Session end hook** — pushes the session transcript to the Synix server for future retrieval and synthesis
- **MCP server** — exposes `search`, `ingest`, `get_context`, and `list_buckets` tools for on-demand memory access
- **Memory skill** — teaches Claude when and how to use the memory tools

## Installation

Copy or symlink the `plugin/` directory into your Claude Code plugins location, or point your Claude Code configuration to this directory.

## Configuration

The plugin connects to a Synix knowledge server. By default it uses `http://salinas:8200`.

To use a different server, set the `SYNIX_SERVER_URL` environment variable:

```bash
export SYNIX_SERVER_URL="http://localhost:8200"
```

## Requirements

- `bash`, `curl`, `python3` (standard on macOS and Linux)
- A running Synix knowledge server

## Structure

```
plugin/
  .claude-plugin/plugin.json   Plugin manifest
  .mcp.json                    MCP server configuration
  hooks/
    hooks.json                 Hook registration
    session_start.sh           Fetches context on session start
    session_end.sh             Pushes transcript on session end
  skills/
    memory/
      SKILL.md                 Memory skill definition
  README.md                    This file
```

## How It Works

**On session start**, `session_start.sh` calls `GET /api/v1/flat-file/context-doc` to retrieve the current synthesized context document and injects it via `hookSpecificOutput.additionalContext`. If the server is unreachable, the hook silently succeeds with no context injected.

**On session end**, `session_end.sh` reads the session ID from hook input, locates the `.jsonl` transcript file, and POSTs it to `POST /api/v1/ingest/sessions`. If the server is unreachable or the session file cannot be found, the hook silently succeeds.

**During the session**, the MCP server at `/mcp/` provides tools for searching memory, ingesting documents, and retrieving the full context document.
