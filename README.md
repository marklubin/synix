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
  <video src="./examples/02-tv-returns/tv_returns.mp4" width="720" controls></video>
</p>

## The Problem

Agent memory hasn't converged. Mem0, Letta, Zep, LangMem — each bakes in a different architecture because the right one depends on your domain and changes as your agent evolves. Most systems force you to commit to a schema early. Changing your approach means migrations or starting over.

## What Synix Does

Conversations are sources. Prompts are build rules. Summaries and world models are artifacts. Declare your memory architecture in Python, build it, then change it — only affected layers rebuild. Trace any artifact back through the dependency graph to its source conversation.

```bash
uvx synix build pipeline.py
uvx synix validate
uvx synix search "return policy"
```

## Using Your Build Output

After a build, Synix gives you two things: a search index and flat artifact files.

**Search via CLI:**
```bash
synix search "return policy"
synix search "warranty terms" --top-k 5 --trace
```

**Use the artifacts directly:**

Build output lives in `./build/` — JSON files per artifact, a `manifest.json` index, and a SQLite FTS5 database. Read them, copy them, or point any tool that speaks SQLite at `search.db`.

```bash
ls build/layer2-cs_product_brief/
sqlite3 build/search.db "SELECT artifact_id, layer_name FROM search_index LIMIT 5"
```

## Key Capabilities

**Incremental rebuilds** — Change a prompt or add new conversations. Only downstream artifacts reprocess.

**Altitude-aware search** — Query episode summaries, monthly rollups, or core memory. Drill into provenance from any result.

**Full provenance** — Every artifact chains back to the source conversations that produced it, through every transform in between.

**Validation and repair** — Detect semantic contradictions and PII leaks across artifacts, then fix them with LLM-assisted rewrites.

**Architecture evolution** — Swap monthly rollups for topic-based clustering. Transcripts and episodes stay cached. No migration scripts.

## Known Limitations

Synix provenance is retrospective: it tracks which inputs and prompts
produced each artifact. However, it cannot answer forward-looking
questions such as:

> "If I change this transcript, what downstream artifacts would be affected?"

Answering this requires re-running `synix plan`.

This is a known limitation of the content-addressed caching model.


## Where Synix Fits

| | Mem0 | Letta | Zep | LangMem | **Synix** |
|---|---|---|---|---|---|
| **Approach** | API-first memory store | Agent-managed memory | Temporal knowledge graph | Taxonomy-driven memory | Build system with pipelines |
| **Incremental rebuilds** | — | — | — | — | Yes |
| **Provenance tracking** | — | — | — | — | Full chain to source |
| **Architecture changes** | Migration | Migration | Migration | Migration | Rebuild |
| **Schema** | Fixed | Fixed | Fixed | Fixed | You define it |

Synix is not a memory store. It's the build system that produces one.

## Links

- [synix.dev](https://synix.dev)
- [GitHub](https://github.com/marklubin/synix)
- MIT License
