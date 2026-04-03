<!-- updated: 2026-04-03 -->
# Synix Knowledge Server Architecture

## Problem

Knowledge is currently managed in a git repo (`agent-memory-disco`) synced across
multiple machines. Git's push/pull/merge model adds friction for what is
fundamentally additive, append-only content. The synix-agent-mesh solved part of
this but introduced distributed coordination complexity (term fencing, member
tracking, heartbeats) that isn't justified when one physical server has all the
compute.

## Design

One Synix server on salinas. A real Claude Code plugin for ingest and retrieval.
Cron outputs feed in as documents. Everything goes through MCP or REST.

```
┌──────────────────────────────────────────────────────┐
│  CLAUDE CODE (any machine)                           │
│                                                      │
│  synix plugin (real Claude Code plugin):             │
│    ├─ MCP server → synix on salinas                  │
│    │   tools: ingest, search, get_context            │
│    ├─ SessionStart hook → inject context doc          │
│    ├─ Stop hook → push session transcript             │
│    └─ skills: memory (model-invoked)                 │
└──────────────┬───────────────────────────────────────┘
               │ Tailscale / HTTP
               ▼
┌──────────────────────────────────────────────────────┐
│  SALINAS (synix serve)                               │
│                                                      │
│  MCP HTTP :8200                                      │
│    tools:                                            │
│      ingest(bucket, content, filename)               │
│      search(query, layers?, limit?)                  │
│      get_context(name?)                              │
│      list_buckets()                                  │
│                                                      │
│  REST API :8200 (alongside MCP)                      │
│    GET  /api/v1/health                               │
│    GET  /api/v1/flat-file/{name}                     │
│    POST /api/v1/ingest/{bucket}                      │
│                                                      │
│  Viewer :9471                                        │
│  Auto-builder (watches buckets, triggers pipeline)   │
│                                                      │
│  Buckets (source endpoints):                         │
│    sessions/   → .jsonl.gz adapter (decompress,      │
│                   merge subsessions, sanitize)        │
│    documents/  → passthrough (md, txt, pdf)           │
│    reports/    → passthrough (cron outputs)            │
│                                                      │
│  Pipeline:                                           │
│    sources → episodes → weekly/research/hypotheses/  │
│    open-threads → core/work-status → search + flats  │
└──────────────────────────────────────────────────────┘
```

## Buckets

Pre-declared ingestion endpoints, like S3 buckets. Each bucket maps to a source
directory with format-specific conventions:

| Bucket | Dir | Adapter | What goes in |
|--------|-----|---------|-------------|
| `sessions` | `sources/sessions/` | Decompress, merge subsessions, sanitize JSON | Claude Code session transcripts (.jsonl.gz) |
| `documents` | `sources/documents/` | Passthrough | Notes, specs, memos, research (md, txt, pdf) |
| `reports` | `sources/reports/` | Passthrough | Cron outputs, automated reports |

Buckets are declared in `synix-server.toml`. Adding a new bucket is a config change.

## MCP Surface

Four tools. Simple and agent-friendly:

- **`ingest(bucket, content, filename)`** — put a document into a named bucket
- **`search(query, layers?, limit?)`** — search the knowledge base
- **`get_context(name?)`** — fetch a synthesized flat file (default: context-doc)
- **`list_buckets()`** — list available buckets and descriptions

This replaces the 28-tool MCP surface from the existing `synix.mcp.server`. The
existing server stays for power-user/local-dev pipeline manipulation. The knowledge
server exposes the simplified surface for agents.

## Pipeline

All Level 2+ layers use incremental fold semantics: `prev_state + Δ(new episodes) → new_state`.
Cost scales with delta, not corpus size. See `docs/incremental-fold-design.md`.

```
SOURCES
  ├─ sessions/
  ├─ documents/
  └─ reports/

LEVEL 1 — EPISODES (1:1, MapSynthesis)
  episodes              one summary per source document

LEVEL 2 — ROLLUPS (all FoldSynthesis with incremental checkpoint)
  weekly                one per ISO week, 9-week rolling window
  research              one per active research thread
  hypotheses            one per hypothesis under test
  open-threads          single doc — unresolved work, blockers, pending

LEVEL 3 — SYNTHESIS (FoldSynthesis, reads from level 2)
  core                  persistent identity + context
  work-status           current priorities, blockers, next actions

PROJECTIONS
  search                FTS5 + semantic over all layers
  context-doc.md        core + work-status (injected into agent prompts)
  weekly-brief.md       latest weekly + open-threads
```

Pipeline definitions are Python files. Adding a new rollup layer reads from the
existing episode pool — nothing upstream recomputes. The server loads the pipeline
on startup from the configured `pipeline_path`.

## Claude Code Plugin

Real Claude Code plugin at `plugin/` in the synix repo.

```
plugin/
├── .claude-plugin/plugin.json    # manifest
├── .mcp.json                     # MCP server → synix on salinas
├── hooks/
│   ├── hooks.json                # SessionStart + Stop hooks
│   ├── session_start.sh          # fetch context, inject into session
│   └── session_end.sh            # push session transcript to server
└── skills/
    └── memory/
        └── SKILL.md              # model-invoked memory skill
```

**SessionStart hook**: fetches context-doc via REST, outputs JSON to inject
it as additional context into the session.

**Stop hook**: reads session ID from stdin, locates the `.jsonl` file, POSTs
it to the server's ingest endpoint.

**Memory skill**: model-invoked — Claude autonomously searches memory when it
needs context and ingests documents when the user mentions calls or decisions.

## Server Mode

`synix serve` — a CLI command that runs three async services:

1. **MCP HTTP** (:8200) — simplified tools + REST endpoints on one Starlette app
2. **Viewer** (:9471) — Flask web UI for browsing and search
3. **Auto-builder** — watches bucket dirs, triggers pipeline on changes

Config via `synix-server.toml`. Runs as a systemd service on salinas.

## Backfill

Three data sources migrate into the new server:

1. **Existing artifact store** (3,554 objects from synix-agent-mesh) — copy `.synix/`
   directly. Preserves episodes, rollups, fold checkpoints.

2. **Session transcripts** (591 files from `~/unified-memory/sessions/`) — copy into
   `sessions/` bucket. Matching episodes are already built, no recompute.

3. **agent-memory-disco corpus** (359 .md files) — copy into `documents/` bucket.
   These are new content — will generate ~359 new episode summaries on first build.
   Rollup layers fully recompute (new prompts differ from agent-mesh prompts).

After initial backfill, new content enters through:
- **Sessions**: Stop hook pushes each session transcript automatically
- **Documents**: `ingest("documents", content, filename)` via MCP during sessions
- **Reports**: Crons write directly to `sources/reports/` on salinas

## Platform vs Domain Separation

**Platform (ships with synix):**
- Server module, MCP tools, REST API, auto-builder, viewer
- Bucket config system
- Claude Code plugin structure
- IncrementalFold transform

**Domain-specific (user's `pipeline.py`):**
- Which rollup layers exist and their prompts
- Which flat files get projected
- Which layers feed into core synthesis
