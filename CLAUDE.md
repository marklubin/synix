# CLAUDE.md - Synix Project Guide

## Project Overview

Synix is a **build system for agent memory**. It provides declarative pipelines for processing conversation exports with full provenance tracking, incremental processing, and branching for experimentation.

**Key analogy:** Chat logs are source files. Prompts are build rules. Summaries are build artifacts. Change a rule → incremental rebuild. Trace any artifact back to its sources.

## Development Commands

```bash
# Install dependencies
cd ~/synix
uv sync

# Run tests
uv run pytest

# Lint and format
uv run ruff check --fix
uv run ruff format

# Type check
uv run pyright

# CLI
uv run synix --help
uv run synix init <name> --from <export-file>
uv run synix run
uv run synix search <query>
uv run synix status
uv run synix export <step> --format markdown
```

## Code Style

- Python 3.12+, fully typed (pyright strict mode)
- Ruff for linting and formatting (line length: 100)
- SQLAlchemy 2.0 with sync sessions
- Pydantic for schemas and validation

## Architecture

```
src/synix/
├── cli.py              # Click CLI commands
├── config.py           # Pydantic settings
├── pipeline.py         # Pipeline orchestration
├── db/
│   ├── engine.py       # Two-layer DB (control + artifacts)
│   ├── control.py      # Pipeline, Run models
│   └── artifacts.py    # Record, RecordSource models + FTS
├── services/
│   ├── records.py      # Record CRUD + provenance
│   └── search.py       # FTS5 search
├── steps/
│   ├── base.py         # Step ABC + version hashing
│   ├── transform.py    # 1:1 transforms
│   ├── aggregate.py    # N:1 by period
│   ├── fold.py         # N:1 sequential with state
│   └── merge.py        # Multi-source combination
├── sources/
│   ├── base.py         # Source ABC
│   ├── chatgpt.py      # ChatGPT export parser
│   └── claude.py       # Claude export parser
└── surfaces/
    ├── base.py         # Surface ABC
    └── file.py         # Filesystem publishing
```

## Two-Layer Storage

- **control.db**: Pipeline definitions, run history (control plane)
- **artifacts.db**: Records, provenance links, FTS index (data plane)

---

# Implementation Status

## Completed Phases

### Phase 1a: Core Pipeline (COMPLETE)
- [x] Two-layer storage (control.db + artifacts.db)
- [x] TransformStep (1:1) and AggregateStep (N:1 with grouping)
- [x] Claude and ChatGPT source parsers
- [x] FTS5 search with auto-sync triggers
- [x] Incremental processing via materialization keys
- [x] CLI: init, run, status, search, plan, runs

### Phase 1b: Extended Pipeline (COMPLETE - PR #1)
- [x] FoldStep for sequential state accumulation
- [x] MergeStep for multi-source combination
- [x] Artifact publishing (FileSurface)
- [x] CLI: export command
- [x] 115 tests passing (58 unit + 57 E2E)

---

# Backlog

## Priority 1: Search Drill-Down API (DESIGN.md §4.6)

**Status:** NOT IMPLEMENTED

The DESIGN.md specifies a drill-down API for provenance navigation that is not yet implemented.

### Missing Features

#### 1. `SearchHit.sources()`
Returns direct parents only (one hop up the DAG).
```python
for hit in results:
    sources = hit.sources()  # → list[Record]
```
**Implementation notes:**
- SearchHit needs a reference to session or lazy-loading capability
- Query RecordSource table where record_id = hit.record_id
- Return ordered by source_order

#### 2. `SearchHit.leaves()`
Returns deduplicated leaf records (records with empty sources[]).
```python
leaves = hit.leaves(max_depth=10, max_count=100)
```
**Implementation notes:**
- Breadth-first traversal up the provenance DAG
- Stop when sources[] is empty (leaf node)
- Deduplicate by record_id
- Respect max_depth and max_count limits

#### 3. `Record.lineage()`
Returns full provenance tree.
```python
record = pipeline.get("record-uuid")
lineage = record.lineage()  # → ProvenanceTree
```
**Implementation notes:**
- Recursive traversal of RecordSource links
- Return tree structure (not flat list)
- Consider cycle detection (should not happen but defensive)

#### 4. Unscoped Search with Altitude Deduplication
When searching without `step=` filter, return highest-altitude hits with drill-down to lower altitudes.
```python
results = pipeline.search("Rust")  # Highest-altitude dedup
```
**Implementation notes:**
- Define altitude based on DAG depth (leaves = 0, derived = depth from leaves)
- Group hits by content similarity or provenance chain
- Return highest-altitude representative with drill-down path

### Files to Modify
- `src/synix/services/search.py` - SearchHit class enhancements
- `src/synix/db/artifacts.py` - Record.lineage() method
- `src/synix/pipeline.py` - Unscoped search logic

### Tests Needed
- `tests/synix/unit/test_search_drilldown.py`
- `tests/synix/e2e/test_provenance_navigation.py`

---

## Priority 2: Semantic Search (Phase 2)

**Status:** NOT STARTED

- [ ] Add `embedding` column to Record model
- [ ] OpenAI embedding generation processor
- [ ] Hybrid search: FTS + semantic + recency via RRF
- [ ] Pipeline search modes: `pipeline.search(query, mode="hybrid")`

---

## Priority 3: Benchmark Harness (Phase 3)

**Status:** NOT STARTED

- [ ] LongMemEval benchmark integration
- [ ] LoCoMo benchmark integration
- [ ] Benchmark runner infrastructure
- [ ] Baseline measurements with current pipeline

---

## Priority 4: Branching (Phase 4)

**Status:** NOT STARTED

- [ ] Branch creation (`pipeline.branch("experiment")`)
- [ ] Branch isolation (separate records per branch)
- [ ] Branch promotion with upsert semantics
- [ ] Branch comparison and evaluation

---

## Future Features (Unscheduled)

- MergeStep deduplication strategies (content_hash, metadata_match, fuzzy)
- MergeStep conflict resolution (prefer_latest, prefer_first, keep_all)
- Additional surfaces (API, MCP publishing)
- Smart memory operations (ADD/UPDATE/DELETE/NOOP classifier)
- Entity extraction and relationship graph
- CLI: `synix lineage <record-id>` command
- Web UI with drill-down visualization

---

# Session Journal

## Protocol

Each session working on Synix should be journaled here. This provides continuity across context window resets and tracks decision-making rationale.

### Entry Format
```
## Session: YYYY-MM-DD - <summary>

**Context:** What state was the project in?
**Goal:** What was attempted?
**Outcome:** What was accomplished?
**Decisions:** Key choices made and rationale
**Blockers:** What prevented progress?
**Next:** Recommended next steps
```

---

## Session: 2026-02-06 - Phase 1b Implementation

**Context:** Phase 1a complete with 58 tests. Plan document specified FoldStep, MergeStep, Surfaces, and CLI export.

**Goal:** Implement all Phase 1b features per plan.

**Outcome:**
- Implemented FoldStep with sequential state accumulation
- Implemented MergeStep with multi-source combination
- Implemented Surface ABC and FileSurface for artifact publishing
- Added CLI export command
- All 115 tests passing
- Created PR #1: https://github.com/marklubin/synix/pull/1

**Decisions:**
1. FoldStep sorts by `meta.time.created_at` for deterministic ordering
2. MergeStep uses `sources: list[str]` parameter instead of `from_`
3. FileSurface supports template variables: {step_name}, {run_id}, {timestamp}, {date}
4. Surface.name given default value to avoid dataclass inheritance issues

**Blockers:**
- Dataclass field ordering with inheritance required default values
- Test Record instances needed explicit metadata_ initialization

**Next:**
- Merge PR #1
- Implement Search Drill-Down API (Priority 1 backlog item)
- This is required by DESIGN.md §4.6 but was not in Phase 1b plan

---

# Execution Tracking Protocol

## Purpose

Track implementation progress against DESIGN.md to ensure no features are missed.

## Checklist: DESIGN.md Feature Coverage

### §3.2 Runtime Entities - Record Model
- [x] id, content, step_name, branch, run_id, created_at
- [x] content_fingerprint (materialization)
- [x] metadata_ (JSON column)
- [x] audit (JSON column)
- [x] sources (via RecordSource join table)
- [ ] embedding (Phase 2)

### §3.3 Materialization Keys
- [x] Transform: (branch, step_name, input_record_id, step_version_hash)
- [x] Aggregate: (branch, step_name, group_key, combined_fingerprint, step_version_hash)
- [x] Fold: (branch, step_name, count, combined_fingerprint, step_version_hash)
- [x] Merge: (branch, step_name, sources_fingerprint, step_version_hash)

### §3.5 Build Rule Types
- [x] transform (1:1)
- [x] aggregate (N:1 by period)
- [x] fold (N:1 sequential with state)
- [x] merge (N:N from multiple sources) - basic implementation
- [ ] merge deduplication strategies (content_hash, metadata_match, fuzzy)
- [ ] merge conflict resolution (prefer_latest, prefer_first, keep_all)

### §4.3 Branching
- [ ] branch creation
- [ ] branch isolation
- [ ] branch.promote() with upsert semantics
- [ ] stale detection after promotion

### §4.6 Querying
- [x] pipeline.search(query, step=) - scoped FTS search
- [ ] pipeline.search(query) - unscoped with altitude deduplication
- [ ] SearchHit.sources() - direct parents
- [ ] SearchHit.leaves() - leaf records with limits
- [ ] Record.lineage() - full provenance tree

### §4.7 Hello World Experience
- [x] synix init
- [x] synix run
- [x] synix search
- [ ] synix init --from <file> (auto-create default pipeline)

### §4.8 Cost Estimation
- [x] pipeline.plan() shows steps and record counts
- [ ] Token estimation per step
- [ ] Cost calculation with model pricing

### §6.1 Sources (v0.1)
- [x] Claude export parser
- [x] ChatGPT export parser

### §6.2 Outputs (v0.1)
- [x] FTS search surface
- [x] File artifact publishing
- [ ] Projection surface (for agent context)

## How to Use This Checklist

1. Before starting work, review unchecked items to identify gaps
2. After implementing a feature, check it off and note the session
3. If a feature deviates from DESIGN.md, document the rationale
4. Periodically review DESIGN.md for newly relevant sections

---

# Quick Reference

## Key Files
- `DESIGN.md` - Full specification and rationale
- `README.md` - User-facing documentation
- `pyproject.toml` - Dependencies and tooling config
- `tests/synix/` - Test suite (unit/ and e2e/)

## Test Fixtures
- `tests/fixtures/claude_export.json` - Sample Claude export
- `tests/fixtures/chatgpt_export.json` - Sample ChatGPT export

## Common Patterns

### Creating a Step
```python
@dataclass
class MyStep(Step):
    step_type: str = field(init=False, default="mystep")
    # Add step-specific fields with defaults

    def compute_materialization_key(self, inputs, branch) -> str:
        # Return unique cache key

    def execute(self, inputs, llm, run_id) -> Record:
        # Process inputs, return output record
```

### Running Tests
```bash
# All tests
uv run pytest

# Specific test file
uv run pytest tests/synix/unit/test_fold.py -v

# With coverage
uv run pytest --cov=synix
```
