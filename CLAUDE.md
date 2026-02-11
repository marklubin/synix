# CLAUDE.md — Synix

## What Is Synix

Synix is **a build system for agent memory**. Declarative pipelines define how raw conversations become searchable, hierarchical memory with full provenance tracking. Change a config, only affected layers rebuild. Think `make` or `dbt`, but for AI agent memory.

The fundamental output: **system prompt + RAG**, built from raw conversations with full lineage tracking.

## Core Concepts

- **Artifact** — immutable, versioned build output (transcript, episode, rollup, core memory). Content-addressed via SHA256.
- **Layer** — named level in the memory hierarchy. Layers form a DAG (transcripts → episodes → rollups → core).
- **Pipeline** — declared in Python. Defines layers, transforms, grouping strategies, and projections.
- **Projection** — materializes artifacts into usable outputs (search index via SQLite FTS5, context doc as markdown).
- **Provenance** — every artifact traces back to its inputs. Always included in search results.
- **Cache/Rebuild** — hash comparison: if inputs or prompt changed, rebuild. Otherwise skip.

Full entity model, storage format, and dataclass definitions: [docs/entity-model.md](docs/entity-model.md)
Pipeline Python API and examples: [docs/pipeline-api.md](docs/pipeline-api.md)

## Module Structure

```
src/synix/
├── __init__.py
├── cli.py              # Click CLI — synix run, synix search, synix lineage
├── pipeline/
│   ├── config.py       # Parse pipeline Python module into Pipeline/Layer objects
│   ├── dag.py          # DAG resolution — build order, rebuild detection
│   └── runner.py       # Execute pipeline — walk DAG, run transforms, cache artifacts
├── artifacts/
│   ├── store.py        # Artifact storage — save/load/query (filesystem-backed)
│   └── provenance.py   # Provenance tracking — record and query lineage chains
├── transforms/
│   ├── base.py         # Base transform interface
│   ├── parse.py        # Source parsers — ChatGPT/Claude JSON → transcript artifacts
│   ├── summarize.py    # LLM transforms — episode, rollup, core synthesis
│   └── prompts/        # Prompt templates as text files
├── projections/
│   ├── base.py         # Base projection interface
│   ├── search_index.py # SQLite FTS5 — materialize and query
│   └── flat_file.py    # Render core memory as context document
└── sources/
    ├── chatgpt.py      # ChatGPT export parser
    └── claude.py       # Claude export parser
```

## Key Module Interfaces

**pipeline/runner.py** calls:
- `artifacts.store.{save,load}_artifact()`, `get_content_hash()` — cache checking
- `transforms.*.execute(inputs, config) -> Artifact`
- `projections.*.materialize(artifacts, config)` — after build
- `artifacts.provenance.record(artifact_id, parent_ids, prompt_id, model_config)`

**cli.py** calls:
- `pipeline.config.load(path) -> Pipeline`
- `pipeline.runner.run(pipeline, source_dir) -> RunResult`
- `projections.search_index.query(query, layers) -> list[SearchResult]`
- `artifacts.provenance.get_chain(artifact_id) -> list[ProvenanceRecord]`

## CLI Commands

```bash
synix run pipeline.py [--source-dir ./exports]   # Build pipeline + materialize projections
synix search "query" [--layers episodes,core]     # Search with provenance chains
synix lineage <artifact-id>                       # Provenance tree view
synix status                                      # Build summary table
```

CLI UX requirements (Rich formatting, colors, progress): [docs/cli-ux.md](docs/cli-ux.md)

## Environment & Build

- Python 3.11+, SQLite (stdlib), `ANTHROPIC_API_KEY` in env
- No external databases, no Docker, no web server
- UV-native: `uv sync`, `uv run synix`, `uv run pytest`

## Critical Rules

- **DO NOT** refactor core engine or abstract prematurely
- **DO NOT** implement StatefulArtifact, branching, eval harness, or any v0.2 feature
- **DO NOT** add Postgres, Neo4j, or any external database — SQLite + filesystem only
- **DO NOT** build a web UI
- **Every module must have at least basic tests**
- Write tests BEFORE or ALONGSIDE the module, never after
- Mock the LLM for unit and integration tests — only E2E hits real API
- Use `tmp_path` for all filesystem tests — no shared state

## Reference Docs

| Doc | Contents |
|-----|----------|
| [docs/entity-model.md](docs/entity-model.md) | Dataclass definitions, storage format, FTS5 schema, cache logic, architecture north star |
| [docs/pipeline-api.md](docs/pipeline-api.md) | Python pipeline definition, examples, config change demo |
| [docs/cli-ux.md](docs/cli-ux.md) | CLI UX requirements, color scheme, per-command formatting |
| [docs/prompt-templates.md](docs/prompt-templates.md) | All prompt template contents (episode, rollup, core) |
| [docs/test-plan.md](docs/test-plan.md) | Test structure, fixtures, all unit/integration/E2E test specs |
| [docs/build-phases.md](docs/build-phases.md) | Phase 1-5 implementation breakdown |
| [docs/DESIGN.md](docs/DESIGN.md) | Vision, origin story, full design narrative |
| [docs/BACKLOG.md](docs/BACKLOG.md) | Deferred items from v0.9 development |
