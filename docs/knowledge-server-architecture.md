<!-- updated: 2026-04-02 -->
# Synix Knowledge Server Architecture

## Problem

Knowledge is currently managed in a git repo (`agent-memory-disco`) synced across
multiple machines. Git's push/pull/merge model adds friction for what is
fundamentally additive, append-only content. The synix-agent-mesh solved part of
this but introduced distributed coordination complexity (term fencing, member
tracking, heartbeats) that isn't justified when one physical server has all the
compute.

## Design

One Synix server on salinas. One optional auth gateway on oxnard. A Claude Code
plugin for ingest and retrieval. Cron outputs feed in as documents.

```
┌──────────────────────────────────────────────────────┐
│  ANY CLIENT (Claude Code, Cursor, scripts, crons)    │
│                                                      │
│  MCP tools:                                          │
│    source_add_text / source_add_file  → ingest       │
│    search                             → query        │
│    get_artifact / get_flat_file       → read         │
│    build (manual trigger)             → rebuild      │
│                                                      │
│  Plugin hooks (Claude Code):                         │
│    on_startup   → pull context-doc, inject prompt    │
│    on_session_end → push session transcript          │
└──────────────┬───────────────────────┬───────────────┘
               │ Tailscale             │ Public
               ▼                       ▼
┌────────────────────────┐  ┌──────────────────────────┐
│  SALINAS               │  │  OXNARD                  │
│  (physical server)     │  │  (cloud VPS)             │
│                        │◄─│  Caddy reverse proxy     │
│  synix-server          │  │    auth (API key)        │
│    ├ MCP HTTP :8200    │  │    mcp → sal:8200        │
│    ├ Viewer   :9471    │  │    view → sal:9471       │
│    ├ Auto-builder      │  └──────────────────────────┘
│    └ Pipeline engine   │
│                        │
│  .synix/               │
│    ├ objects/           │
│    ├ refs/              │
│    ├ releases/          │
│    └ sources/           │
│        ├ sessions/      │  ← Claude Code transcripts
│        ├ documents/     │  ← manual ingest (md, text, pdf)
│        └ reports/       │  ← cron outputs
└────────────────────────┘
```

## Sources (format-based, not domain-based)

Three source types distinguished by format, not by content domain:

| Source | What | Adapter |
|--------|------|---------|
| `sessions` | Claude Code session logs (.jsonl.gz) | Decompress, merge subsessions, sanitize JSON |
| `documents` | Markdown, text, PDF — anything | Passthrough |
| `reports` | Cron outputs (structured text) | Passthrough |

Bulk conversation exports (ChatGPT JSON, Claude JSON) are a one-time backfill
operation, not a live pipeline source.

## Pipeline

All Level 2+ layers use incremental fold semantics: `prev_state + Δ(new episodes) → new_state`.
Cost scales with delta, not corpus size. See `docs/incremental-fold-design.md` for the
checkpoint mechanism.

```
SOURCES
  ├─ sessions/
  ├─ documents/
  └─ reports/


LEVEL 1 — EPISODES (1:1, MapSynthesis)

  episodes              one summary per source document
                        existing artifacts untouched; only new sources summarize


LEVEL 2 — ROLLUPS (all IncrementalFold: prev + Δ → new)

  weekly                one per ISO week, 9-week rolling window
  research              one per active research thread
  hypotheses            one per hypothesis under test (ACTIVE/SUPPORTED/REFUTED/DORMANT)
  open-threads          single doc — unresolved work, blockers, pending actions

  (additional rollup layers added without recomputing upstream)


LEVEL 3 — SYNTHESIS (IncrementalFold, reads from level 2)

  core                  persistent identity + context
  work-status           current priorities, blockers, next actions


PROJECTIONS

  search                FTS5 + semantic over all layers
  context-doc.md        core + work-status (injected into agent prompts on startup)
  weekly-brief.md       latest weekly + open-threads
```

Pipeline definitions are Python files, dynamically reloadable via `load_pipeline()`.
Adding a new Level 2 rollup reads from the existing episode pool — nothing upstream
recomputes.

## Claude Code Plugin

Domain-agnostic. Ships with Synix. Three components:

### 1. MCP Server Config
Points to the remote Synix server. Gives agents search, ingest, and browse tools.

```json
{
  "mcpServers": {
    "synix": {
      "url": "http://salinas:8200/mcp"
    }
  }
}
```

### 2. Startup Hook
Pulls the `context-doc` flat file from the server and injects it into the session
context. This replaces manually-maintained CONTEXT.md files.

```
synix context pull --server http://salinas:8200 --format claude-md
```

### 3. Session-End Hook
Pushes the completed session transcript to the server. Idempotent — deduplicates
by session ID.

```
synix session push --server http://salinas:8200
```

## Cron Integration

Existing crons on salinas (email-triage, daily-intel, social tracking) redirect
their output to `sources/reports/` instead of (or in addition to) the notify
system. Reports become documents that flow through the pipeline like everything
else.

Downstream pipeline layers (e.g. `relationships`, `engagement`) can be defined
over the report episodes to maintain rolling context for social tracking and
outreach work.

## Synix Server Mode

A simplification of synix-agent-mesh. Drops: mesh coordination, term fencing,
member tracking, cluster protocol. Keeps: MCP HTTP transport, viewer, auto-builder,
pipeline engine.

Runs as a systemd service on salinas. Four async tasks:

1. **MCP HTTP** (:8200) — streamable HTTP, all Synix tools
2. **Viewer** (:9471) — Flask web UI for browsing and search
3. **Auto-builder** — watches source dirs, triggers pipeline on changes
4. **Pipeline engine** — runs whatever `pipeline.py` is loaded

## Platform vs Domain Separation

Everything described above as shipping with Synix is domain-agnostic:
- The server, plugin, CLI commands, and transport are platform
- Source types are format-based (sessions, documents, reports), not domain-based
- IncrementalFold is a generic transform primitive

Domain-specific configuration lives in `pipeline.py`:
- Which rollup layers exist and their prompts
- Which flat files get projected
- Which layers feed into core synthesis
- Cron report format expectations

## Migration Path

1. Build IncrementalFold into Synix core (see `docs/incremental-fold-design.md`)
2. Extract synix-server from synix-agent-mesh (drop mesh, keep server/viewer/builder)
3. Build `synix context pull` and `synix session push` CLI commands
4. Build `synix setup claude-code` for plugin installation
5. Deploy synix-server on salinas as systemd service
6. Import existing .synix/objects/ (4,574 artifacts) to salinas
7. Bulk ingest key documents from agent-memory-disco
8. Point all MCP configs to salinas:8200
9. Redirect cron outputs to sources/reports/
10. Retire synix-agent-mesh
