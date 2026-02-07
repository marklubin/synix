# CLAUDE.md — Synix Weekend Build Sprint

## What Is Synix

Synix is **a build system for agent memory**. It treats memory as a build artifact — declarative pipelines define how raw conversations become searchable, hierarchical memory with full provenance tracking. Change a config, only affected layers rebuild.

Think `make` or `dbt`, but for AI agent memory.

## Demo Target (Sunday Night)

A 90-second screen recording showing:

1. **Drop in exports** — ChatGPT and Claude conversation exports land in a directory
2. **`synix run`** — pipeline processes: transcripts → episode summaries → monthly rollups → core memory. Progress visible.
3. **`synix search "what do I think about anthropic?"`** — returns synthesized results with provenance
4. **Config change** — edit pipeline config to change rollup categories (e.g., swap monthly rollups for topic-based rollups)
5. **`synix run` again** — only upper layers rebuild (transcripts and episodes cached). Fast.
6. **`synix search` again** — different results, same data, new provenance chains

Three runs, three behaviors, one system. That's the demo.

---

## Entity Model

### Artifact (base concept)

Every output of the build system is an **Artifact**. An artifact is an immutable, versioned build output.

```python
@dataclass
class Artifact:
    artifact_id: str          # unique identifier
    artifact_type: str        # "transcript", "episode", "rollup", "core_memory", "search_index"
    content_hash: str         # SHA256 of content
    input_hashes: list[str]   # hashes of all inputs that produced this
    prompt_id: str | None     # prompt template version (for LLM-derived artifacts)
    model_config: dict | None # model, temperature, etc (for LLM-derived artifacts)
    created_at: datetime
    content: str              # the actual text content
    metadata: dict            # flexible metadata (source file, date range, etc)
```

**Two subtypes exist conceptually (weekend: only derived):**

- **DerivedArtifact** — pure function of inputs. Rebuild from scratch when inputs change. Episode summaries, rollups, core memory. This is what we build this weekend.
- **StatefulArtifact** — append/update semantics. Agent scratchpad, external backends (Mem0, etc). Synix tracks versions but doesn't rebuild. **v0.2 — do NOT implement this weekend.**

### ProvenanceRecord

Every artifact knows its lineage.

```python
@dataclass
class ProvenanceRecord:
    artifact_id: str              # the artifact this record describes
    parent_artifact_ids: list[str] # artifacts that were inputs
    prompt_id: str | None         # prompt used to produce it
    model_config: dict | None     # LLM config used
    created_at: datetime
```

### Layer

A named level in the memory hierarchy. Layers form a DAG.

```python
@dataclass
class Layer:
    name: str                     # "transcripts", "episodes", "monthly", "core"
    level: int                    # 0 = raw, 1 = first derivation, etc.
    depends_on: list[str]         # names of layers this depends on
    transform: str                # name of the transform function/prompt
    grouping: str | None          # how to group inputs ("by_conversation", "by_month", "single")
```

### Pipeline

The full declared memory architecture. A pipeline is a YAML/TOML config that defines layers and their relationships.

```python
@dataclass
class Pipeline:
    name: str
    layers: list[Layer]
    # The DAG is implicit in Layer.depends_on
```

### Projection

A **Projection** materializes build artifacts into a usable output surface. This is the "packaging/deploy" step — the build system produces artifacts, projections turn them into something an agent can actually use.

```python
@dataclass
class Projection:
    name: str                     # "memory-index", "context-doc"
    projection_type: str          # "search_index", "flat_file", (future: "qdrant", "neo4j", etc.)
    sources: list[ProjectionSource]  # which layers feed into this projection
    config: dict                  # projection-type-specific config
```

```python
@dataclass
class ProjectionSource:
    layer: str                    # layer name
    search_modes: list[str]       # ["fulltext", "semantic"] — how this layer is searchable
```

**Two projections for the demo:**

1. **Search Index** (`search_index` type) — SQLite FTS5 with hierarchical search across layers. Every result carries its full provenance chain (always on, not optional).

```
SearchIndex
├── Layer 1: episode summaries (fulltext)
├── Layer 2: rollups - monthly/topical (fulltext + semantic)
└── Layer 3: core memory (fulltext + semantic)

Each search result includes:
- content: the matching text
- layer: which altitude it came from
- artifact_id: which artifact it belongs to
- provenance_chain: list of artifact_ids tracing back to raw source (ALWAYS included)
```

2. **Context Document** (`flat_file` type) — renders the core memory artifact as a ready-to-use system prompt / context window insert. This is the actual deliverable for an agent: "here is your memory, formatted for your context window."

The fundamental output of Synix is: **system prompt + RAG.** The context document is the system prompt. The search index is the RAG. Projections are how you declare both from the same build artifacts.

---

## Module Structure

```
src/synix/
├── __init__.py
├── cli.py              # Click CLI — synix run, synix search, synix lineage
├── pipeline/
│   ├── __init__.py
│   ├── config.py       # Parse pipeline YAML config into Pipeline/Layer objects
│   ├── dag.py          # DAG resolution — determine build order, detect what needs rebuild
│   └── runner.py       # Execute the pipeline — walk DAG, run transforms, cache artifacts
├── artifacts/
│   ├── __init__.py
│   ├── store.py        # Artifact storage — save/load/query artifacts (filesystem-backed)
│   └── provenance.py   # Provenance tracking — record and query lineage chains
├── transforms/
│   ├── __init__.py
│   ├── base.py         # Base transform interface
│   ├── parse.py        # Source parsers — ChatGPT JSON, Claude JSON → transcript artifacts
│   ├── summarize.py    # LLM transforms — episode summary, rollup, core synthesis
│   └── prompts/        # Prompt templates as text files
│       ├── episode_summary.txt
│       ├── monthly_rollup.txt
│       ├── topical_rollup.txt
│       └── core_memory.txt
├── search/
│   ├── __init__.py
│   ├── index.py        # SQLite FTS5 search index — build and query
│   └── results.py      # Search result formatting with provenance
├── projections/
│   ├── __init__.py
│   ├── base.py         # Base projection interface
│   ├── search_index.py # SQLite FTS5 projection — materializes artifacts into searchable index
│   └── flat_file.py    # Flat file projection — renders core memory as context document
└── sources/
    ├── __init__.py
    ├── chatgpt.py      # ChatGPT export parser (conversations.json)
    └── claude.py       # Claude export parser (JSON format)
```

### Module Interfaces (contracts between modules)

**pipeline/runner.py** calls:
- `artifacts.store.load_artifact(artifact_id)` / `save_artifact(artifact)`
- `artifacts.store.get_content_hash(artifact_id)` — to check if rebuild needed
- `transforms.*.execute(inputs: list[Artifact]) -> Artifact`
- `projections.*.materialize(artifacts: list[Artifact])` — after build, project into output surfaces
- `artifacts.provenance.record(artifact_id, parent_ids, prompt_id, model_config)`

**cli.py** calls:
- `pipeline.config.load(path) -> Pipeline`
- `pipeline.runner.run(pipeline, source_dir) -> RunResult`
- `projections.search_index.query(query, layers=None) -> list[SearchResult]`
- `artifacts.provenance.get_chain(artifact_id) -> list[ProvenanceRecord]`

**projections/search_index.py:**
- `materialize(artifacts: list[Artifact], config) -> None` — build FTS5 index from artifacts
- `query(q: str, layers: list[str] | None) -> list[SearchResult]` — search with layer filtering
- Every SearchResult always includes full provenance chain

**projections/flat_file.py:**
- `materialize(artifacts: list[Artifact], config) -> None` — render to markdown file
- Output is a ready-to-use context document (system prompt insert)

---

## Pipeline Definition (Python API)

The pipeline is defined in Python, not YAML. More flexible, more expressive, better for dynamic layer generation.

```python
# pipeline.py — Personal Memory Pipeline
from synix import Pipeline, Layer, Projection

pipeline = Pipeline("personal-memory")

# Sources
pipeline.source_dir = "./exports"
pipeline.build_dir = "./build"

# LLM defaults
pipeline.llm_config = {
    "model": "claude-sonnet-4-20250514",
    "temperature": 0.3,
    "max_tokens": 1024,
}

# Layers — the memory hierarchy
pipeline.add_layer(Layer(
    name="transcripts",
    level=0,
    transform="parse",
))

pipeline.add_layer(Layer(
    name="episodes",
    level=1,
    depends_on=["transcripts"],
    transform="episode_summary",
    grouping="by_conversation",
))

pipeline.add_layer(Layer(
    name="monthly",
    level=2,
    depends_on=["episodes"],
    transform="monthly_rollup",
    grouping="by_month",
))

pipeline.add_layer(Layer(
    name="core",
    level=3,
    depends_on=["monthly"],
    transform="core_synthesis",
    grouping="single",
    context_budget=10000,
))

# Projections — how artifacts become usable
pipeline.add_projection(Projection(
    name="memory-index",
    projection_type="search_index",
    sources=[
        {"layer": "episodes", "search": ["fulltext"]},
        {"layer": "monthly", "search": ["fulltext"]},
        {"layer": "core", "search": ["fulltext"]},
    ],
))

pipeline.add_projection(Projection(
    name="context-doc",
    projection_type="flat_file",
    sources=[
        {"layer": "core"},
    ],
    output_path="./build/context.md",
))
```

**Demo config change:** swap monthly for topical rollups:

```python
# pipeline_topical.py — just change level 2
# ... same as above, but replace monthly layer with:

pipeline.add_layer(Layer(
    name="topics",
    level=2,
    depends_on=["episodes"],
    transform="topical_rollup",
    grouping="by_topic",
    config={"topics": ["career", "technical-projects", "san-francisco", "personal-growth", "ai-and-agents"]},
    # Transform queries the episode search index (already built) to find relevant episodes per topic
))

pipeline.add_layer(Layer(
    name="core",
    level=3,
    depends_on=["topics"],  # changed dependency
    transform="core_synthesis",
    grouping="single",
    context_budget=10000,
))
```

Because it's Python, you can also do things like:
```python
# Generate a layer per topic dynamically
for topic in ["work", "health", "projects", "relationships"]:
    pipeline.add_layer(Layer(
        name=f"topic-{topic}",
        level=2,
        depends_on=["episodes"],
        transform="topical_rollup",
        grouping="by_topic",
        config={"topic": topic},
    ))
```

This is why code > config. You can't do that in YAML.

---

## Artifact Storage (Filesystem)

```
build/
├── manifest.json           # maps artifact_id → file path + metadata
├── provenance.json         # all provenance records
├── layer0-transcripts/
│   ├── abc123.json         # artifact file (content + metadata)
│   └── def456.json
├── layer1-episodes/
│   ├── ep-abc123.json      # one per conversation
│   └── ep-def456.json
├── layer2-monthly/
│   ├── 2025-01.json
│   └── 2025-02.json
├── layer3-core/
│   └── core-memory.json
└── search.db               # SQLite FTS5 database
```

Each artifact JSON file:
```json
{
  "artifact_id": "ep-abc123",
  "artifact_type": "episode",
  "content_hash": "sha256:...",
  "input_hashes": ["sha256:..."],
  "prompt_id": "episode_summary_v1",
  "model_config": {"model": "claude-sonnet-4-20250514", "temperature": 0.3},
  "created_at": "2025-02-07T10:30:00Z",
  "content": "In this conversation, Mark discussed...",
  "metadata": {
    "source_conversation_id": "abc123",
    "date": "2025-01-15",
    "message_count": 24
  }
}
```

### Cache/Rebuild Logic

```python
def needs_rebuild(artifact_id, current_input_hashes, current_prompt_id):
    existing = store.load_artifact(artifact_id)
    if existing is None:
        return True
    if existing.input_hashes != current_input_hashes:
        return True
    if existing.prompt_id != current_prompt_id:
        return True
    return False
```

That's it. Hash comparison. No fancy cache invalidation. If any input changed or the prompt changed, rebuild. Otherwise skip.

---

## Search Index (SQLite FTS5)

```sql
CREATE VIRTUAL TABLE search_index USING fts5(
    content,
    artifact_id,
    layer_name,
    layer_level,
    metadata  -- JSON string with dates, source info, etc.
);
```

**Query with layer filtering:**
```sql
SELECT *, rank FROM search_index
WHERE search_index MATCH ?
AND layer_level >= ?  -- optional: search only at certain altitudes
ORDER BY rank;
```

**Provenance drill-down:** Given a search result's `artifact_id`, walk `provenance.json` recursively to get the full chain back to raw transcript. Return as a list.

---

## CLI Commands

```bash
# Process all exports through the pipeline
synix run pipeline.py [--source-dir ./exports]
# Output: progress bar per layer, summary of artifacts built/cached/skipped
# Then: materializes all declared projections (search index, context doc)

# Search across all layers (queries the search_index projection)
synix search "what do I think about anthropic?" [--layers episodes,monthly,core]
# Output: ranked results with layer label, snippet, artifact ID, and provenance chain

# Show provenance chain for an artifact
synix lineage <artifact-id>
# Output: tree view showing inputs at each level down to raw transcript

# Show build status
synix status
# Output: layer summary — artifact counts, last build time, cache hit rate, projection status
```

### CLI Output Style

Use `rich` library for:
- Progress bars during pipeline run
- Colored layer labels in search results
- Tree view for lineage display
- Tables for status

---

## Prompt Templates

### episode_summary.txt
```
You are summarizing a conversation between a user and an AI assistant.

<conversation>
{transcript}
</conversation>

Write a concise episode summary (200-400 words) that captures:
1. The main topics discussed
2. Key decisions, insights, or conclusions reached
3. Any action items or commitments made
4. The emotional tone and context of the conversation

Write in third person. Focus on information that would be useful for long-term memory.
```

### monthly_rollup.txt
```
You are synthesizing multiple conversation summaries from {month} {year} into a monthly overview.

<episode_summaries>
{episodes}
</episode_summaries>

Write a monthly rollup (300-600 words) that captures:
1. Major themes and recurring topics this month
2. How the user's thinking evolved across conversations
3. Key decisions and their context
4. Important facts, preferences, or relationships mentioned

Prioritize information that builds a coherent picture of this period. Deduplicate across episodes.
```

### topical_rollup.txt
```
You are synthesizing conversation summaries related to the topic: {topic}

<episode_summaries>
{episodes}
</episode_summaries>

Write a topical synthesis (300-600 words) that captures:
1. Everything the user has discussed about {topic}
2. How their views or understanding evolved over time
3. Key facts, opinions, and decisions related to {topic}
4. Connections to other topics or areas of their life

Write as a coherent narrative, not a list of conversations.
```

### core_memory.txt
```
You are synthesizing all available information into a core memory document for an AI agent.

Budget: {context_budget} tokens maximum.

<rollup_summaries>
{rollups}
</rollup_summaries>

Create a structured core memory that includes:
1. **Identity**: Who is this person? Background, profession, key relationships.
2. **Current focus**: What are they working on right now? What matters most?
3. **Preferences & style**: Communication preferences, technical opinions, values.
4. **Key history**: Major life events, career arc, important decisions.
5. **Active threads**: Ongoing projects, unresolved questions, things to follow up on.

This will be placed directly in an AI agent's context window. Be specific and factual. Every sentence should be useful. Stay within the token budget.
```

## Architecture North Star (post-demo, do NOT build this weekend)

**Projections as build infrastructure.** The same search index that serves user queries can also serve downstream transforms. A topical rollup doesn't need all episodes in context — it queries the episode search index for relevant ones. At scale, this is essential.

**The full vision:** Layers can depend on projections. Projections can compose (multiple search indexes merged into a unified interface). The build DAG includes both artifacts and projections as nodes. This keeps everything clean and declarative even at massive scale.

**For this weekend:** The runner builds layers in order. By the time topical rollup runs, the episode search index already exists. The transform just queries it. No new DAG concepts needed — just pass the index to the transform.

**Future work:**
- Projections as first-class DAG nodes (layers can declare dependency on a projection)
- Composable projections (merge multiple search indexes into unified query surface)
- Intermediate projections as build acceleration (auto-index any layer for downstream use)
- Proprietary clustering method for auto-topic discovery using intermediate search

---

### CRITICAL RULES
- **DO NOT** refactor the core engine or abstract prematurely
- **DO NOT** implement StatefulArtifact, branching, eval harness, or any v0.2 feature
- **DO NOT** add Postgres, Neo4j, or any external database. SQLite + filesystem only.
- **DO NOT** build a web UI
- **DO NOT** spend more than 30 minutes on any single module before moving to the next
- **Every module must have at least basic tests**
- **Target: working E2E demo by Sunday 6pm**

### Phase 1: Foundation (Friday night / Saturday morning)

**1a. Project scaffolding**
- pyproject.toml with dependencies (click, rich, anthropic, sqlite3)
- Directory structure as specified above
- Empty __init__.py files
- Basic test structure with pytest

**1b. Artifact store** (`artifacts/store.py`)
- `save_artifact(artifact: Artifact) -> None` — write JSON to build dir
- `load_artifact(artifact_id: str) -> Artifact | None` — read from build dir
- `list_artifacts(layer: str) -> list[Artifact]` — list all artifacts in a layer
- `get_content_hash(artifact_id: str) -> str | None`
- Manifest management (manifest.json)
- Tests: save/load roundtrip, list by layer, hash checking

**1c. Provenance** (`artifacts/provenance.py`)
- `record(artifact_id, parent_ids, prompt_id, model_config) -> None`
- `get_parents(artifact_id) -> list[str]`
- `get_chain(artifact_id) -> list[ProvenanceRecord]` — recursive walk to roots
- Backed by provenance.json
- Tests: record and retrieve, chain walking

### Phase 2: Parsers & Transforms (Saturday morning)

**2a. Source parsers** (`sources/chatgpt.py`, `sources/claude.py`)
- Parse ChatGPT `conversations.json` → list of transcript Artifacts
- Parse Claude export JSON → list of transcript Artifacts
- Each transcript artifact has: conversation ID, title, date, full text, message count
- Tests: parse sample fixtures, verify metadata

**2b. Transform base** (`transforms/base.py`)
- Abstract base: `execute(inputs: list[Artifact], config: dict) -> Artifact`
- Prompt template loading from transforms/prompts/

**2c. LLM transforms** (`transforms/summarize.py`)
- `EpisodeSummaryTransform` — one conversation → one episode summary
- `MonthlyRollupTransform` — group of episodes by month → monthly synthesis
- `TopicalRollupTransform` — queries episode search index for relevant episodes per topic, then synthesizes. User declares topics in pipeline config. At runtime, the search index is already built from earlier layers — the transform just queries it.
- `CoreSynthesisTransform` — all rollups → core memory
- All read from prompt template files
- Use Anthropic SDK (claude-sonnet-4-20250514, temperature 0.3)
- Tests: mock LLM, verify prompt construction and artifact output

### Phase 3: Pipeline Engine (Saturday afternoon)

**3a. Config parser** (`pipeline/config.py`)
- Load synix.yaml → Pipeline object with Layer list
- Validate: DAG is acyclic, all depends_on references exist, exactly one level-0 layer
- Tests: parse valid config, reject invalid configs

**3b. DAG resolver** (`pipeline/dag.py`)
- `resolve_build_order(pipeline) -> list[Layer]` — topological sort
- `get_rebuild_set(pipeline, store) -> list[Layer]` — which layers need work
- For each layer, check if artifacts exist with matching input hashes
- Tests: correct ordering, correct rebuild detection

**3c. Pipeline runner** (`pipeline/runner.py`)
- `run(pipeline, source_dir, build_dir) -> RunResult`
- Walk layers in build order
- For each layer:
  - Gather inputs from dependent layers
  - Group inputs according to layer's grouping strategy
  - For each group, check if rebuild needed (hash comparison)
  - If needed: run transform, save artifact, record provenance
  - If not: skip (log cache hit)
- **After each layer completes:** materialize that layer's projection (if any) so downstream transforms can use it. Specifically: the episode search index is built after episodes complete, so the topical rollup transform can query it.
- After all layers: materialize final projections (full search index, context doc)
- Return summary: built/cached/skipped counts, timing
- Tests: full pipeline run with mock transforms

### Phase 4: Projections & CLI (Saturday evening)

**4a. Projection base** (`projections/base.py`)
- Abstract: `materialize(artifacts: list[Artifact], config: dict) -> None`

**4b. Search index projection** (`projections/search_index.py`)
- `materialize(artifacts, config)` — populate FTS5 from artifacts across specified layers
- `query(q: str, layers: list[str] | None) -> list[SearchResult]` — ranked search
- SearchResult includes: content snippet, artifact_id, layer_name, layer_level, score, provenance_chain (always)
- Tests: index and query roundtrip, layer filtering, provenance chain present

**4c. Flat file projection** (`projections/flat_file.py`)
- `materialize(artifacts, config)` — render core memory artifact as markdown
- Output: ready-to-use context document
- Tests: output file exists, content matches core artifact

**4d. CLI** (`cli.py`)
- `synix run <pipeline.py>` — load pipeline module, run build, materialize all projections, show progress
- `synix search <query>` — query search index projection, display results with provenance
- `synix lineage <artifact-id>` — show provenance tree
- `synix status` — show build summary (artifacts per layer, cache hit rate, projection status)
- Use rich for all output formatting
- Tests: CLI smoke tests (commands don't crash)

### CLI UX Requirements

The CLI is a selling point. Use Click + Rich to make it feel polished.

- **`synix run`**: Rich progress bars per layer (not just a spinner). Show: layer name, artifact count, built/cached/skipped. Final summary table with timing.
- **`synix search`**: Results displayed as Rich panels — layer label colored by level, content snippet, artifact ID. Provenance chain shown as indented tree below each result.
- **`synix lineage`**: Rich Tree widget showing full dependency graph from artifact down to raw transcript.
- **`synix status`**: Rich table — layer name, artifact count, last build time, cache hit ratio.
- **Colors**: Use a consistent color scheme per layer level. Level 0 = dim/grey, Level 1 = blue, Level 2 = green, Level 3 = gold/yellow.
- **Error handling**: All errors through Rich console with clear messages. Never a raw Python traceback in normal operation.
- **Help text**: Every command and option has clear `--help` text.

---

## Test Plan

### Philosophy
Tests are not optional or Phase 5. **Every module gets tests as it's built.** No module is "done" until its tests pass. This is critical because:
1. Claude Code agents will be writing code — tests verify correctness without manual inspection
2. Incremental builds depend on hash comparison being exactly right — one bug here corrupts everything
3. The demo must work flawlessly on recording day — no time for debugging

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures: temp dirs, sample artifacts, mock LLM
├── fixtures/
│   ├── chatgpt_export.json  # Real (anonymized) ChatGPT export subset
│   ├── claude_export.json   # Real (anonymized) Claude export subset
│   └── sample_pipeline.py   # Test pipeline config
├── unit/
│   ├── test_artifact_store.py
│   ├── test_provenance.py
│   ├── test_dag.py
│   ├── test_config.py
│   ├── test_parsers.py
│   ├── test_transforms.py
│   ├── test_search_index.py
│   ├── test_flat_file.py
│   └── test_cli.py
├── integration/
│   ├── test_pipeline_run.py       # Full pipeline with mock LLM
│   ├── test_incremental_rebuild.py # Cache hit/miss scenarios
│   ├── test_config_change.py      # Swap config, verify partial rebuild
│   └── test_projections.py        # End-to-end projection materialization
└── e2e/
    ├── test_demo_flow.py          # The exact demo sequence, automated
    └── test_real_data.py          # Run against real exports (slow, optional)
```

### conftest.py Fixtures

```python
@pytest.fixture
def tmp_build_dir(tmp_path):
    """Clean build directory for each test."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    return build_dir

@pytest.fixture
def sample_artifacts():
    """Pre-built artifacts for testing downstream modules."""
    return [
        Artifact(artifact_id="t-001", artifact_type="transcript", content="...", ...),
        Artifact(artifact_id="t-002", artifact_type="transcript", content="...", ...),
        Artifact(artifact_id="ep-001", artifact_type="episode", content="...", ...),
    ]

@pytest.fixture
def mock_llm(monkeypatch):
    """Mock Anthropic API — returns deterministic responses based on prompt."""
    def mock_create(*args, **kwargs):
        # Return predictable content based on the transform type
        prompt = kwargs.get("messages", [{}])[0].get("content", "")
        if "episode summary" in prompt.lower():
            return MockResponse("This is a summary of the conversation about...")
        elif "monthly rollup" in prompt.lower():
            return MockResponse("In January 2025, the main themes were...")
        elif "core memory" in prompt.lower():
            return MockResponse("## Identity\nMark is a software engineer...")
        return MockResponse("Mock response")
    monkeypatch.setattr("anthropic.Anthropic.messages.create", mock_create)

@pytest.fixture
def sample_pipeline(tmp_build_dir):
    """A complete test pipeline with all layers and projections."""
    # Returns a Pipeline object ready to run with mock data
```

### Unit Tests (per module)

**test_artifact_store.py**
- `test_save_and_load_roundtrip` — save artifact, load by ID, content matches
- `test_load_nonexistent_returns_none` — missing ID returns None
- `test_list_by_layer` — save 5 artifacts across 3 layers, list each layer correctly
- `test_content_hash_computed` — save artifact, hash is SHA256 of content
- `test_manifest_persistence` — save artifacts, reload store from disk, manifest intact
- `test_overwrite_artifact` — save same ID twice, latest wins, manifest updated

**test_provenance.py**
- `test_record_and_retrieve` — record provenance, get_parents returns correct IDs
- `test_chain_walking` — 3-level chain (core → monthly → episode → transcript), get_chain returns full path
- `test_chain_multiple_parents` — monthly rollup with 5 episode inputs, all in chain
- `test_persistence` — record provenance, reload from disk, data intact

**test_dag.py**
- `test_topological_sort_simple` — 4 layers, correct build order
- `test_topological_sort_diamond` — diamond dependency, no duplicate processing
- `test_cycle_detection` — circular dependency raises error
- `test_rebuild_detection_all_new` — empty build dir, everything needs rebuild
- `test_rebuild_detection_all_cached` — matching hashes, nothing needs rebuild
- `test_rebuild_detection_partial` — change prompt at level 2, levels 2+3 rebuild, 0+1 cached
- `test_rebuild_cascades` — changing level 1 forces rebuild of 2 and 3

**test_config.py**
- `test_load_pipeline_module` — import pipeline.py, get Pipeline object with correct layers
- `test_validate_acyclic` — valid DAG passes
- `test_validate_cyclic_rejected` — circular depends_on raises error
- `test_validate_missing_dependency` — depends_on references nonexistent layer, raises error
- `test_validate_single_root` — exactly one level-0 layer required

**test_parsers.py**
- `test_chatgpt_parse_basic` — parse fixture, correct number of transcripts
- `test_chatgpt_metadata` — conversation ID, title, date, message count all populated
- `test_chatgpt_message_ordering` — messages in chronological order
- `test_chatgpt_empty_conversation` — gracefully skip or handle
- `test_claude_parse_basic` — same as chatgpt tests for Claude format
- `test_claude_metadata` — correct metadata extraction
- `test_mixed_sources` — parse both formats, no collisions

**test_transforms.py** (with mock LLM)
- `test_episode_summary_prompt_construction` — verify prompt includes transcript, correct template
- `test_episode_summary_output_artifact` — output is valid Artifact with correct type, metadata
- `test_monthly_rollup_groups_by_month` — 10 episodes across 3 months, 3 rollup calls
- `test_topical_rollup_clusters` — episodes get clustered, each cluster produces rollup
- `test_core_synthesis_respects_budget` — context_budget passed through, output within limit
- `test_prompt_template_loading` — templates load from prompts/ directory
- `test_prompt_id_versioning` — prompt changes produce different prompt_id

**test_search_index.py**
- `test_materialize_and_query` — index 10 artifacts, query returns relevant results
- `test_layer_filtering` — query with layer filter, only matching layers returned
- `test_provenance_always_included` — every search result has provenance_chain field
- `test_ranking` — more relevant results rank higher
- `test_empty_query` — graceful handling
- `test_rebuild_replaces_index` — materialize twice, second run replaces first cleanly

**test_flat_file.py**
- `test_materialize_creates_file` — output file exists at specified path
- `test_content_matches_core` — file content matches core memory artifact
- `test_markdown_formatting` — output is valid markdown

**test_cli.py**
- `test_run_command_exists` — `synix run --help` succeeds
- `test_search_command_exists` — `synix search --help` succeeds
- `test_lineage_command_exists` — `synix lineage --help` succeeds
- `test_status_command_exists` — `synix status --help` succeeds
- `test_run_missing_pipeline_errors` — `synix run nonexistent.py` gives clear error
- `test_search_no_index_errors` — `synix search` before any run gives clear error

### Integration Tests

**test_pipeline_run.py** (mock LLM, real everything else)
- `test_full_pipeline_mock_llm` — run complete pipeline with mock LLM, all layers built, all projections materialized
- `test_artifact_count_matches_expectations` — correct number of artifacts per layer
- `test_all_artifacts_have_provenance` — every non-root artifact has provenance records
- `test_search_returns_results_after_run` — pipeline run → search works
- `test_context_doc_exists_after_run` — pipeline run → context.md exists

**test_incremental_rebuild.py** (the critical one)
- `test_second_run_all_cached` — run twice, second run builds 0 artifacts
- `test_new_source_partial_rebuild` — add one conversation, only its chain rebuilds
- `test_prompt_change_cascades` — change episode prompt, episodes + rollups + core rebuild, transcripts cached
- `test_cache_metrics_accurate` — RunResult reports correct built/cached/skipped counts

**test_config_change.py** (the demo moment)
- `test_swap_monthly_to_topical` — run pipeline.py then pipeline_topical.py:
  - Transcripts: 0 rebuilt (cached)
  - Episodes: 0 rebuilt (cached)
  - Topics: N rebuilt (new layer)
  - Core: 1 rebuilt (dependency changed)
  - Projections: re-materialized
- `test_search_results_differ` — same query returns different results after config change
- `test_context_doc_differs` — context.md content changes after config change

**test_projections.py**
- `test_search_index_reflects_all_layers` — results come from multiple layers
- `test_provenance_chain_depth` — core result traces back to transcript through all layers
- `test_flat_file_is_ready_to_use` — context.md could be pasted into a system prompt

### E2E Tests

**test_demo_flow.py** (the exact recording sequence, automated)
```python
def test_demo_sequence(real_exports_dir, tmp_build_dir):
    """Run the exact demo — this test passing means the demo will work."""
    # 1. First run — full build
    result1 = run("pipeline.py", real_exports_dir, tmp_build_dir)
    assert result1.built > 0
    assert result1.cached == 0
    
    # 2. Search
    results = search("anthropic", tmp_build_dir)
    assert len(results) > 0
    assert all(r.provenance_chain for r in results)
    
    # 3. Context doc exists
    assert (tmp_build_dir / "context.md").exists()
    
    # 4. Second run — all cached
    result2 = run("pipeline.py", real_exports_dir, tmp_build_dir)
    assert result2.built == 0
    assert result2.cached > 0
    
    # 5. Config change — partial rebuild
    result3 = run("pipeline_topical.py", real_exports_dir, tmp_build_dir)
    assert result3.cached > 0  # transcripts + episodes
    assert result3.built > 0   # topics + core
    
    # 6. Search again — different results
    results2 = search("anthropic", tmp_build_dir)
    assert results2 != results  # different provenance at minimum
```

**test_real_data.py** (slow, run manually before recording)
- Run on full conversation set (1000+ conversations)
- Spot check: search results make sense
- Spot check: provenance chains are accurate
- Spot check: context.md is coherent
- Timing: full build < 30 minutes, cached rebuild < 10 seconds

### Test Rules for Agents

1. **Write tests BEFORE or ALONGSIDE the module, never after.** If a module doesn't have tests, it's not done.
2. **All tests must pass before moving to the next phase.** Run `uv run pytest` at each phase gate.
3. **Mock the LLM for unit and integration tests.** Only E2E tests hit the real API.
4. **Use tmp_path for all filesystem tests.** No shared state between tests.
5. **Test the failure cases, not just the happy path.** Missing files, corrupt data, empty inputs.
6. **Integration tests are the most important.** The incremental rebuild tests are what make or break the demo.

### Phase 5: Integration, E2E & Demo (Sunday)

**5a. Run E2E tests**
- Run `test_demo_flow.py` with a 50-conversation subset first (fast validation)
- Run `test_real_data.py` on full 1000+ conversation set
- Fix any integration bugs found
- All tests green before recording

**5b. Demo rehearsal**
- Run the exact demo sequence:
  1. `synix run pipeline.py` (full build — all layers + projections)
  2. `synix search "anthropic"` (show results + provenance)
  3. `cat build/context.md` (show the flat context doc projection — this is what goes in an agent's system prompt)
  4. Edit pipeline_topical.py (monthly → topical rollups)
  5. `synix run pipeline_topical.py` (partial rebuild — transcripts + episodes cached, rollups + core rebuilt, projections re-materialized)
  6. `synix search "anthropic"` (different results, same data, new provenance)
- Polish output formatting for screen recording

**5c. Screen record**
- Record the demo
- Upload

---

## Dependencies & Build

UV-native build. No hatchling, no setuptools.

```toml
[project]
name = "synix"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "click>=8.0",
    "rich>=13.0",
    "anthropic>=0.40.0",
    "pydantic>=2.0",
]

[project.scripts]
synix = "synix.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

All commands run through `uv`:
```bash
uv sync          # install deps
uv run synix     # run CLI
uv run pytest    # run tests
```

---

## README Outline (for open source + YC app)

```markdown
# Synix — A Build System for Agent Memory

Memory is a build artifact.

## The Problem
AI agents need memory, but there's no consensus on the right architecture.
Mem0 does flat key-value. Cognee builds knowledge graphs. Zep does session summaries.
Each approach works for some use cases. None is universal.
Worse: switching architectures means starting over.

## The Insight
The right question isn't "what's the best memory architecture?"
It's "how do I discover the right architecture for my agent, fast?"

## What Synix Does
Synix is `make` for agent memory. You declare a memory pipeline in Python:
sources → transforms → artifacts → projections. Synix handles:

- **Incremental rebuilds** — change a prompt, only affected artifacts reprocess
- **Full provenance** — trace any memory back through the dependency graph to source
- **Projections** — same build artifacts, multiple output surfaces (search index, context documents, knowledge graphs)
- **Hierarchical search** — query at any altitude (episode summaries → synthesized core memory)
- **Code-driven pipelines** — define memory architecture in Python, not config files

The fundamental output: **system prompt + RAG**, built from raw conversations with full lineage tracking.

## Quick Start
(demo commands here)

## How It Works
Sources → Transforms → Artifacts → Projections
(architecture diagram — the layer cake with projection outputs)

## Roadmap
- [ ] v0.1: Derived artifacts, search index + context doc projections, CLI
- [ ] v0.2: Stateful artifacts (scratchpad, agent-writable memory), unified artifact interface
- [ ] v0.3: Pipeline branching, A/B testing memory architectures
- [ ] v0.4: Pluggable projection backends (Postgres, Neo4j, Qdrant, Mem0)
- [ ] v0.5: Eval harness, cost estimation, DSPy integration for prompt optimization
- [ ] v1.0: Hosted platform, team collaboration, memory governance
```

---

## Environment

- Python 3.11+
- SQLite (stdlib)
- Anthropic API key in environment: ANTHROPIC_API_KEY
- No external databases
- No Docker
- No web server

## What Success Looks Like

By Sunday 6pm:
- `synix run pipeline.py` processes 1000+ real conversations through the full pipeline
- Search index projection populated — `synix search` returns relevant results with provenance chains
- Context doc projection outputs a clean markdown file usable as agent system prompt
- Config change + `synix run pipeline_topical.py` demonstrates cache-aware partial rebuild (fast — only rollups + core rebuild)
- Output looks professional enough to screen record
- All tests pass: unit, integration, and E2E
- Total API cost under $30
