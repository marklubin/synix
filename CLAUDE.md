# CLAUDE.md — Synix

## What Is Synix

Synix is **a build system for agent memory**. Declarative pipelines define how raw conversations become searchable, hierarchical memory with full provenance tracking. Change a config, only affected layers rebuild. Think `make` or `dbt`, but for AI agent memory.

The fundamental output: **system prompt + RAG**, built from raw conversations with full lineage tracking.

## Core Concepts

- **Artifact** — immutable, versioned build output (transcript, episode, rollup, core memory). Content-addressed via SHA256.
- **Layer** — typed Python object in the build DAG. `Source` for inputs, `Transform` subclasses for LLM steps, `SearchSurface`/`SynixSearch`/`FlatFile` for projections. Dependencies are object references via `depends_on`.
- **Pipeline** — declared in Python. `Pipeline.add(*layers)` routes Source/Transform to layers, projection declarations to the manifest.
- **Projection** — structured declaration in the manifest. Materialized at release time by adapters (`synix_search`, `flat_file`). Not built during `synix build`.
- **Release** — `synix release HEAD --to <name>` materializes projections from a snapshot into `.synix/releases/<name>/`. Produces a receipt.
- **Provenance** — every artifact traces back to its inputs via `parent_labels`. Always included in search results. No separate `provenance.json`.
- **Cache/Rebuild** — hash comparison via `SnapshotArtifactCache`: if inputs or prompt changed, rebuild. Otherwise skip.

Full entity model, storage format, and dataclass definitions: [docs/entity-model.md](docs/entity-model.md)
Pipeline Python API and examples: [docs/pipeline-api.md](docs/pipeline-api.md)

## Module Structure

```
src/synix/
├── __init__.py            # Public API: Pipeline, Source, Transform, SearchIndex, FlatFile, Artifact
├── core/
│   └── models.py          # Layer hierarchy (Source, Transform, SearchIndex, FlatFile, Pipeline)
├── sdk.py                 # SDK — programmatic access (Project, Release, search, init/open_project)
├── build/
│   ├── runner.py          # Execute pipeline — walk DAG, run transforms, cache artifacts
│   ├── plan.py            # Dry-run planner — per-artifact rebuild/cached decisions
│   ├── dag.py             # DAG resolution — build order from depends_on references
│   ├── pipeline.py        # Pipeline loader — import Python module, extract Pipeline object
│   ├── object_store.py    # ObjectStore — single content-addressed write path (.synix/objects/)
│   ├── refs.py            # RefStore — git-like refs (heads/main, runs/*, releases/*)
│   ├── snapshots.py       # BuildTransaction + commit_build_snapshot
│   ├── snapshot_view.py   # SnapshotView — ref-resolved reads from .synix/objects/
│   ├── error_classifier.py # Error classification — DLQ vs fatal, DeadLetterQueue
│   ├── fingerprint.py     # Build fingerprints — synix:transform:v2 scheme
│   ├── llm_transforms.py  # Bundled memory transforms + shared LLM helper functions
│   ├── parse_transform.py # Source parser — ChatGPT/Claude JSON → transcript artifacts
│   ├── merge_transform.py # Merge transform — Jaccard similarity grouping
│   ├── transforms.py      # Transform base + registry (string dispatch fallback)
│   ├── validators.py      # Built-in validators (PII, SemanticConflict, Citation, etc.)
│   ├── fixers.py          # Built-in fixers (SemanticEnrichment, CitationEnrichment)
│   ├── projections.py     # Projection dispatch
│   └── cassette.py        # Record/replay for LLM + embedding calls
├── transforms/
│   ├── __init__.py        # Re-export: MapSynthesis, GroupSynthesis, ReduceSynthesis, FoldSynthesis, Merge
│   └── base.py            # BaseTransform (legacy compat)
├── ext/
│   ├── __init__.py        # Re-export: bundled memory transforms + migration compatibility exports
│   ├── map_synthesis.py   # Generic 1:1 synthesis transform implementation
│   ├── group_synthesis.py # Generic N:M grouping synthesis transform implementation
│   ├── reduce_synthesis.py# Generic N:1 synthesis transform implementation
│   ├── fold_synthesis.py  # Generic sequential fold synthesis transform implementation
│   └── chunk.py           # Generic 1:N text chunking transform (no LLM)
├── validators/
│   └── __init__.py        # Re-export: MutualExclusion, RequiredField, PII, SemanticConflict, Citation
├── fixers/
│   └── __init__.py        # Re-export: SemanticEnrichment, CitationEnrichment
├── projections/
│   └── __init__.py        # Re-export: SearchIndexProjection, FlatFileProjection
├── search/
│   ├── indexer.py         # SQLite FTS5 — build, query, shadow swap
│   ├── embeddings.py      # Embedding provider — fastembed, OpenAI, cached
│   └── retriever.py       # Hybrid search — keyword + semantic + RRF fusion
├── mcp/
│   ├── __init__.py        # Package exports (main, mcp)
│   ├── __main__.py        # Entry point for `python -m synix.mcp`
│   └── server.py          # FastMCP server — 20 tools exposing full SDK surface
├── cli/                   # Click CLI commands
│   ├── main.py
│   ├── build_commands.py
│   ├── artifact_commands.py
│   └── ...
└── templates/             # Bundled demo pipelines (synix init, synix demo)
```

## Key Module Interfaces

**Pipeline model** (`core/models.py`):
- `Source(name)` — root layer, loads files from source_dir
- `Transform(name, depends_on=[...])` — abstract, subclass with `execute()` + `split()`
- `SearchSurface(name, sources=[...], modes=[...])` — build-time search capability declaration
- `SynixSearch(name, surface=...)` — canonical search output, materialized at release time
- `FlatFile(name, sources=[...])` — markdown context doc projection, materialized at release time
- `Pipeline.add(*layers)` — routes Source/Transform to layers, projection declarations to manifest

**build/runner.py** calls:
- `isinstance(layer, Source)` → `layer.load(config)` for parsing
- `isinstance(layer, Transform)` → `layer.compute_fingerprint()`, `layer.split()`, `layer.execute()`
- Projection declarations recorded in manifest (not materialized at build time)
- Single write path via `ObjectStore` → `.synix/objects/`

## CLI Commands

```bash
synix build pipeline.py                          # Produce immutable snapshot in .synix/
synix build pipeline.py --dlq                    # Build with dead letter queue (skip content-filter errors)
synix plan pipeline.py                           # Dry-run — per-artifact rebuild/cached counts
synix plan pipeline.py --explain-cache           # Plan with cache decision reasons
synix release HEAD --to local                    # Materialize projections to a named release
synix revert <ref> --to <name>                   # Release an older snapshot
synix search "query" --release local             # Search a release target
synix search "query" --ref HEAD                  # Scratch realization (ephemeral)
synix lineage <artifact-id>                      # Provenance tree (reads from .synix/)
synix list                                       # All artifacts via SnapshotView
synix show <id>                                  # Render artifact (markdown)
synix releases list                              # All named releases
synix releases show <name>                       # Release receipt details
synix refs list                                  # All refs (build + release)
synix refs show <ref>                            # Resolve ref to snapshot
synix clean                                      # Remove .synix/releases/ and .synix/work/
```

CLI UX requirements (Rich formatting, colors, progress): [docs/cli-ux.md](docs/cli-ux.md)

## Environment & Build

- Python 3.11+, SQLite (stdlib), `ANTHROPIC_API_KEY` in env
- No external databases, no Docker, no web server
- UV-native: `uv sync`, `uv run synix`, `uv run pytest`

## Contributing

**Before pushing any changes**, run the full release check:

```bash
uv run release
```

This runs: sync templates → ruff fix → ruff check → pytest → verify all demos. All must pass before pushing. CI runs the same checks — if release passes locally, CI will pass.

To run just the demo verifications standalone (faster feedback during UX work):

```bash
uv run verify-demos
```

**Workflow changes** (`.github/workflows/`): test locally with [`act`](https://github.com/nektos/act) before pushing:

```bash
act push -W .github/workflows/<workflow>.yml --secret-file .secrets
```

Store local secrets in `.secrets` (gitignored):
```
ANTHROPIC_API_KEY=sk-ant-...
```

## PR & Issue Workflow

Every PR must link to the GitHub issues it addresses:

1. **Before creating a PR**, check open issues: `gh issue list --state open --json number,title`
2. **Review each issue** against the PR's changes — map features to issues
3. **Include `Closes #N`** in the PR body for each issue fully resolved by the PR
4. **Reference without closing** (`#N`) for issues only partially addressed
5. **PR body format**: `Closes #N` directives at the top, then `## Summary` with bullet points linking each feature back to its issue number

## Critical Rules

- **Customer-facing docs** (READMEs, templates, `synix init` output) must use `uvx synix` for all CLI commands. Internal dev docs (CLAUDE.md, test files) use `uv run synix`.
- **DO NOT** refactor core engine or abstract prematurely
- **DO NOT** implement StatefulArtifact, branching, eval harness, or any v0.2 feature
- **DO NOT** add Postgres, Neo4j, or any external database — SQLite + filesystem only
- **DO NOT** build a web UI
- **Every module must have at least basic tests**
- Write tests BEFORE or ALONGSIDE the module, never after
- **Fail fast and loud — never eat errors silently.** This is a core design principle across the entire codebase:
  - **No bare `except: pass/continue/return []`** in build logic, validators, fixers, or transforms. Every `except` block must either re-raise, log a warning with `exc_info=True`, or return a result that explicitly communicates the failure (e.g., `action="skip"` with a description of why).
  - **Validators and fixers fail closed by default** — if the component cannot do its job (no LLM client, missing prompt file, unparseable LLM response), it must raise `RuntimeError`/`ValueError`, not silently return empty results. Use `fail_open=True` in config to opt into graceful degradation.
  - **Never make "best effort corrections" on behalf of the user** unless the correction is trivially obvious, well-documented, and the user explicitly opted in. If data is ambiguous or missing, surface the error — don't guess.
  - **Acceptable exceptions**: cache files (corrupted cache → rebuild), type-probing patterns (try int/float/date parsing), plan-mode estimation (read-only speculation), and CLI display helpers. These may degrade silently because they have no correctness impact.
- **Tests must cover failure modes, not just happy paths** — for every new validator, fixer, or LLM-backed component: test infra failures (missing prompt files, bad LLM config), test fail-closed behavior (default should raise, not silently pass), test edge cases in input parsing (special characters, empty inputs, malformed responses). Happy-path-only tests are incomplete.
- Mock the LLM for unit and integration tests — only E2E hits real API
- Use `tmp_path` for all filesystem tests — no shared state
- **Every functional behavior change must have e2e test coverage** — unit tests alone are insufficient. If a change affects CLI output, plan display, cache detection, or artifact metadata, write an e2e test that exercises the full path.
- **Always consider template/demo impact** — changes to transforms, plan output, artifact metadata, or CLI formatting will affect golden files in `templates/*/golden/`. Regenerate goldens (`uv run synix demo run <template> --update-goldens`) and verify normalization rules in `demo_commands.py._normalize_output()` still cover new output patterns.

## Reference Docs

| Doc | Contents |
|-----|----------|
| [docs/sdk.md](docs/sdk.md) | SDK API guide — Project, Release, search, pipeline interop, error hierarchy |
| [docs/entity-model.md](docs/entity-model.md) | Dataclass definitions, storage format, FTS5 schema, cache logic, architecture north star |
| [docs/pipeline-api.md](docs/pipeline-api.md) | Python pipeline definition, examples, config change demo |
| [docs/cli-ux.md](docs/cli-ux.md) | CLI UX requirements, color scheme, per-command formatting |
| [docs/prompt-templates.md](docs/prompt-templates.md) | All prompt template contents (episode, rollup, core) |
| [docs/test-plan.md](docs/test-plan.md) | Test structure, fixtures, all unit/integration/E2E test specs |
| [docs/projection-release-v2-rfc.md](docs/projection-release-v2-rfc.md) | Build/release separation, `.synix/` layout, adapter contract, release receipts |
| [docs/build-phases.md](docs/build-phases.md) | Phase 1-5 implementation breakdown (historical) |
| [docs/DESIGN.md](docs/DESIGN.md) | Vision, origin story, full design narrative |
| [docs/mcp.md](docs/mcp.md) | MCP server setup, tool reference, agent integration |
| [docs/BACKLOG.md](docs/BACKLOG.md) | Deferred items from v0.9 development |
