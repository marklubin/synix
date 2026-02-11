# Build Phases

## Phase 1: Foundation

**1a. Project scaffolding**
- pyproject.toml with dependencies (click, rich, anthropic, sqlite3)
- Directory structure, empty __init__.py files, pytest structure

**1b. Artifact store** (`artifacts/store.py`)
- `save_artifact(artifact)` / `load_artifact(artifact_id)` / `list_artifacts(layer)` / `get_content_hash(artifact_id)`
- Manifest management (manifest.json)
- Tests: save/load roundtrip, list by layer, hash checking

**1c. Provenance** (`artifacts/provenance.py`)
- `record(artifact_id, parent_ids, prompt_id, model_config)`
- `get_parents(artifact_id)` / `get_chain(artifact_id)` — recursive walk to roots
- Backed by provenance.json
- Tests: record and retrieve, chain walking

## Phase 2: Parsers & Transforms

**2a. Source parsers** (`sources/chatgpt.py`, `sources/claude.py`)
- Parse exports → list of transcript Artifacts (conversation ID, title, date, full text, message count)
- Tests: parse fixtures, verify metadata

**2b. Transform base** (`transforms/base.py`)
- Abstract base: `execute(inputs: list[Artifact], config: dict) -> Artifact`
- Prompt template loading from transforms/prompts/

**2c. LLM transforms** (`transforms/summarize.py`)
- `EpisodeSummaryTransform` — one conversation → one episode summary
- `MonthlyRollupTransform` — group by month → monthly synthesis
- `TopicalRollupTransform` — queries episode search index per topic, synthesizes
- `CoreSynthesisTransform` — all rollups → core memory
- Use Anthropic SDK (claude-sonnet-4-20250514, temperature 0.3)
- Tests: mock LLM, verify prompt construction and artifact output

## Phase 3: Pipeline Engine

**3a. Config parser** (`pipeline/config.py`)
- Load pipeline.py → Pipeline object with Layer list
- Validate: DAG acyclic, depends_on references exist, exactly one level-0 layer

**3b. DAG resolver** (`pipeline/dag.py`)
- `resolve_build_order(pipeline)` — topological sort
- `get_rebuild_set(pipeline, store)` — which layers need work via hash comparison

**3c. Pipeline runner** (`pipeline/runner.py`)
- Walk layers in build order, gather inputs, group, check rebuild, run transform, save artifact, record provenance
- After each layer: materialize that layer's projection (if any) so downstream transforms can use it
- After all layers: materialize final projections
- Return summary: built/cached/skipped counts, timing

## Phase 4: Projections & CLI

**4a-b. Projections** (`projections/`)
- Search index: FTS5 materialize + query with layer filtering, provenance chain always included
- Flat file: render core memory as markdown context document

**4c. CLI** (`cli.py`)
- `synix run` / `synix search` / `synix lineage` / `synix status`
- Rich output formatting (see [CLI UX](cli-ux.md))

## Phase 5: Integration, E2E & Demo

- Run E2E tests with 50-conversation subset, then full 1000+ set
- Demo rehearsal: full build → search → config change → partial rebuild → search again
- Screen record
