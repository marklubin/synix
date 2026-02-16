# CLAUDE.md — Synix

## What Is Synix

Synix is **a build system for agent memory**. Declarative pipelines define how raw conversations become searchable, hierarchical memory with full provenance tracking. Change a config, only affected layers rebuild. Think `make` or `dbt`, but for AI agent memory.

The fundamental output: **system prompt + RAG**, built from raw conversations with full lineage tracking.

## Core Concepts

- **Artifact** — immutable, versioned build output (transcript, episode, rollup, core memory). Content-addressed via SHA256.
- **Layer** — typed Python object in the build DAG. `Source` for inputs, `Transform` subclasses for LLM steps, `SearchIndex`/`FlatFile` for projections. Dependencies are object references via `depends_on`.
- **Pipeline** — declared in Python. `Pipeline.add(*layers)` routes Source/Transform to layers, SearchIndex/FlatFile to projections automatically.
- **Projection** — materializes artifacts into usable outputs. `SearchIndex` (SQLite FTS5 + optional embeddings), `FlatFile` (markdown context doc).
- **Provenance** — every artifact traces back to its inputs. Always included in search results.
- **Cache/Rebuild** — hash comparison: if inputs or prompt changed, rebuild. Otherwise skip.

Full entity model, storage format, and dataclass definitions: [docs/entity-model.md](docs/entity-model.md)
Pipeline Python API and examples: [docs/pipeline-api.md](docs/pipeline-api.md)

## Module Structure

```
src/synix/
├── __init__.py            # Public API: Pipeline, Source, Transform, SearchIndex, FlatFile, Artifact
├── core/
│   └── models.py          # Layer hierarchy (Source, Transform, SearchIndex, FlatFile, Pipeline)
├── build/
│   ├── runner.py          # Execute pipeline — walk DAG, run transforms, cache artifacts
│   ├── plan.py            # Dry-run planner — per-artifact rebuild/cached decisions
│   ├── dag.py             # DAG resolution — build order from depends_on references
│   ├── pipeline.py        # Pipeline loader — import Python module, extract Pipeline object
│   ├── artifacts.py       # Artifact storage — save/load/query (filesystem-backed)
│   ├── provenance.py      # Provenance tracking — record and query lineage chains
│   ├── fingerprint.py     # Build fingerprints — synix:transform:v2 scheme
│   ├── llm_transforms.py  # Built-in LLM transforms (EpisodeSummary, MonthlyRollup, etc.)
│   ├── parse_transform.py # Source parser — ChatGPT/Claude JSON → transcript artifacts
│   ├── merge_transform.py # Merge transform — Jaccard similarity grouping
│   ├── transforms.py      # Transform base + registry (string dispatch fallback)
│   ├── validators.py      # Built-in validators (PII, SemanticConflict, Citation, etc.)
│   ├── fixers.py          # Built-in fixers (SemanticEnrichment, CitationEnrichment)
│   ├── projections.py     # Projection dispatch
│   └── cassette.py        # Record/replay for LLM + embedding calls
├── transforms/
│   ├── __init__.py        # Re-export: EpisodeSummary, MonthlyRollup, TopicalRollup, CoreSynthesis, Merge
│   └── base.py            # BaseTransform (legacy compat)
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
- `SearchIndex(name, sources=[...])` — FTS5 + optional embeddings projection
- `FlatFile(name, sources=[...], output_path=...)` — markdown context doc projection
- `Pipeline.add(*layers)` — routes Source/Transform to layers, SearchIndex/FlatFile to projections

**build/runner.py** calls:
- `isinstance(layer, Source)` → `layer.load(config)` for parsing
- `isinstance(layer, Transform)` → `layer.compute_fingerprint()`, `layer.split()`, `layer.execute()`
- Projection materialization via `SearchIndexProjection` / `FlatFileProjection`

## CLI Commands

```bash
synix build pipeline.py                          # Build pipeline + materialize projections
synix plan pipeline.py                           # Dry-run — per-artifact rebuild/cached counts
synix plan pipeline.py --explain-cache           # Plan with cache decision reasons
synix search "query" [--layers episodes,core]    # Search with provenance chains
synix lineage <artifact-id>                      # Provenance tree view
synix list                                       # All artifacts, grouped by layer
synix show <id>                                  # Render artifact (markdown)
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
| [docs/entity-model.md](docs/entity-model.md) | Dataclass definitions, storage format, FTS5 schema, cache logic, architecture north star |
| [docs/pipeline-api.md](docs/pipeline-api.md) | Python pipeline definition, examples, config change demo |
| [docs/cli-ux.md](docs/cli-ux.md) | CLI UX requirements, color scheme, per-command formatting |
| [docs/prompt-templates.md](docs/prompt-templates.md) | All prompt template contents (episode, rollup, core) |
| [docs/test-plan.md](docs/test-plan.md) | Test structure, fixtures, all unit/integration/E2E test specs |
| [docs/build-phases.md](docs/build-phases.md) | Phase 1-5 implementation breakdown |
| [docs/DESIGN.md](docs/DESIGN.md) | Vision, origin story, full design narrative |
| [docs/BACKLOG.md](docs/BACKLOG.md) | Deferred items from v0.9 development |
