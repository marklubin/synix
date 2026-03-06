# RFC: Search Surfaces, Default Synix Search, and Release Targets

**Related issues**: [#15](https://github.com/marklubin/synix/issues/15), [#10](https://github.com/marklubin/synix/issues/10), [#81](https://github.com/marklubin/synix/issues/81)
**Status**: Proposed
**Baseline**: `main` after [#89](https://github.com/marklubin/synix/pull/89)
**Decision target**: Design approval before implementation

## Summary

Synix should stop treating search as primarily a projection target and instead model it as a build-time capability with a clean release boundary:

- `SearchSurface` is the logical searchable capability built from artifacts
- `SynixSearch` is the default Synix-provided search realization and tooling bundle
- explicit release targets are optional advanced adapters layered on top
- transforms consume search through tools or a build context, not through implicit file paths
- checkpoint or bank semantics are intentionally deferred to a separate RFC

This design solves the immediate need for semantic lookup inside a single consistent build closure without making projections recursive live DAG inputs.

## Motivation

Current Synix behavior works, but the abstraction is wrong:

- [SearchIndex](/home/mark/synix/src/synix/core/models.py) is modeled as a projection
- the runner progressively materializes `build/search.db`
- transforms like [TopicalRollup](/home/mark/synix/src/synix/build/llm_transforms.py) find search by reading `search_db_path`
- the build contract depends on mutable side effects in `build/`

That is convenient, but it mixes three separate concepts:

1. logical searchable state
2. local build-time query machinery
3. published or released output targets

That makes several important cases awkward:

- a transform needs semantic lookup over prior artifacts during the same build
- a user wants the default local search experience and does not care about backend details
- a user wants to release the same search surface into a backend with different schema or embedding requirements
- a benchmark or checkpoint system wants to certify a whole closure rather than treat one search DB as independently released truth

The goal is to give Synix one clean model that covers all of those without forcing every build to become a multi-stage mini-deployment.

## Design Goals

1. Search inside a build should work without implicit `search_db_path` conventions.
2. The logical searchable state should be backend-neutral.
3. The default Synix search experience should remain easy and opaque.
4. Advanced release targets should be explicit and opt-in.
5. The whole closure should still live or die together during a build.
6. The design should not require checkpoint or release semantics to be recursive build inputs.
7. Existing `SearchIndex` pipelines should remain supportable during migration.

## Non-Goals

- This RFC does not define checkpoint or bank sealing semantics.
- This RFC does not require every search backend to share one physical schema.
- This RFC does not require incremental release for every target on day one.
- This RFC does not redesign the full runtime or tool API for all artifact families.
- This RFC does not force an immediate breaking removal of `SearchIndex`.

## Problem Statement

The current model says:

```text
artifacts -> projection -> build/search.db
```

and then treats `build/search.db` as both:

- the internal semantic search capability for later transforms
- the effective released search output

This creates several problems:

- The semantic dependency is hidden in config, not the pipeline model.
- The build contract depends on mutable side effects.
- A search DB looks more "published" than it really is.
- If a downstream layer reveals that an earlier layer is semantically wrong, the internal search state should not have been treated as independently valid.
- Future backends like Postgres or Qdrant should not force the logical surface to become backend-specific.

## Options Considered

### Option A: Refs Everywhere

Every semantic dependency would consume a released projection through a ref.

```text
episodes -> seal checkpoint ref -> release search -> topical rollup reads ref
```

Pros:

- very clean semantics
- easy to audit
- easy to explain formally

Cons:

- too awkward for ordinary pipelines
- every build-time semantic lookup becomes a staged subpipeline
- bad ergonomics for common authoring

This is a good model for checkpointed benchmark banks, but not for normal build-time search.

### Option B: Projection Nodes As Live DAG Inputs

Treat projections as ordinary DAG nodes and let transforms depend on partially materialized projection state.

Pros:

- closest to current implementation
- least invasive at first glance

Cons:

- mixes logical state with mutable realization state
- effectively invents hidden refs inside the runner
- hard to reason about correctness
- likely to frustrate users once they need more than one search target

This RFC rejects that model.

### Option C: Internal Search Surfaces Plus Separate Release Targets

Treat search as an internal logical capability, and treat release as a separate publishing concern.

```text
artifacts -> SearchSurface -> local Synix search tooling
                            -> explicit release targets
```

Pros:

- clean build semantics
- good ergonomics
- supports both default behavior and advanced backends
- keeps the whole closure internally consistent

Cons:

- requires a new public concept
- requires migration from the current `SearchIndex` model

This RFC chooses Option C.

## Core Definitions

### SearchSurface

A logical searchable view over a set of source artifacts.

It defines:

- which artifacts participate
- which query modes are supported
- what stable row ids and source-backed refs exist
- what canonical surface state Synix should build

It does not commit to one concrete backend image as the primary truth.

### SynixSearch

The default Synix-provided realization and tool bundle for a `SearchSurface`.

This is intentionally opaque in public semantics:

- users should not depend on a SQLite schema
- users should not depend on `search.db` internals
- Synix may change the implementation later

### Release Target

An explicit publication target for a surface, such as:

- default local Synix search bundle
- Postgres-backed search
- Qdrant-backed search
- future HTTP or runtime service integrations

### Canonical Surface State

The immutable, backend-neutral build result of a `SearchSurface`.

This is what Synix stores canonically in `.synix` and what realizers consume.

### Local Realization

A disposable build-time realization of the surface used by transforms or local tools during the build.

The local realization is not the canonical truth.

## Recommended Public Model

The model should be:

```text
artifacts -> SearchSurface -> canonical surface state
                               |              |
                               |              +-> explicit release targets
                               |
                               +-> default Synix local search tooling
```

This means:

- builds produce immutable surface state
- transforms can use search without pretending that one backend image is independently released truth
- users get a default search artifact and tool experience "for free"
- advanced users can declare explicit releases when they need backend-specific behavior

## Public API Shape

### New Pipelines

New pipelines should use:

- `SearchSurface(...)`
- `SynixSearch(...)` if they want the default Synix search output/tooling

Illustrative shape:

```python
from synix import Pipeline, SearchSurface, Source, SynixSearch
from synix.ext import CoreSynthesis, EpisodeSummary, MonthlyRollup

pipeline = Pipeline("personal-memory")

transcripts = Source("transcripts")
episodes = EpisodeSummary("episodes", depends_on=[transcripts])
monthly = MonthlyRollup("monthly", depends_on=[episodes])
core = CoreSynthesis("core", depends_on=[monthly], context_budget=10000)

memory_search = SearchSurface(
    "memory-search",
    sources=[episodes, monthly, core],
    modes=["fulltext", "semantic"],
    embedding_config={
        "provider": "fastembed",
        "model": "BAAI/bge-small-en-v1.5",
    },
)

pipeline.add(transcripts, episodes, monthly, core, memory_search)
pipeline.release(
    SynixSearch(
        "default-search",
        surface=memory_search,
        output_path="./build/search.db",
    )
)
```

### Build-Time Search Use

Transforms that need search should consume a search handle through an execution context, not a file path:

```python
topics = TopicalRollup(
    "topics",
    depends_on=[episodes],
    uses=[episode_search],
)
```

The contract should be:

- transforms declare which surfaces they use
- Synix realizes those surfaces locally before the transform runs
- transforms access those surfaces through a context object
- transforms do not open `build/search.db` directly

### Advanced Release Targets

If a user needs something specific, they declare it explicitly:

```python
pipeline.release(
    PostgresSearchRelease(
        "prod-search",
        surface=memory_search,
        target="postgres://...",
        mode="incremental",
        adapter_config={
            "embedding_model": "text-embedding-3-large",
        },
    )
)
```

The public guidance should be:

- use `SynixSearch` unless you have specific backend needs
- once you need backend-specific behavior, declare it explicitly

## Transform Contract

This RFC recommends a clean transform API break rather than another layer of path or config hacks.

### Recommended Shape

Today transforms receive:

```python
def split(self, inputs: list[Artifact], config: dict) -> list[tuple[list[Artifact], dict]]: ...
def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]: ...
```

That contract is too overloaded. It currently mixes:

- user config
- resolved LLM config
- workspace paths
- implicit service handles like `search_db_path`

The recommended replacement is:

```python
def split(self, inputs: list[Artifact], ctx: TransformContext) -> list[WorkUnit]: ...
def execute(self, inputs: list[Artifact], ctx: TransformContext) -> list[Artifact]: ...
```

Where `TransformContext` is a first-class object supplied by the runner.

### Why A Context Object

A context object keeps the contract clean:

- config remains structured instead of becoming a service bag
- build-time capabilities like search are explicit
- Synix can evolve the underlying realization without changing transform code
- the same pattern can later cover other build-time capabilities besides search

### Declaring Surface Usage

Transforms that need internal search should declare that need in the pipeline model.

Illustrative shape:

```python
topics = TopicalRollup(
    "topics",
    depends_on=[episodes],
    uses=[episode_search],
    config={
        "topics": [
            "programming-and-tools",
            "ai-and-llms",
            "systems-and-infrastructure",
        ],
    },
)
```

`uses=[...]` should mean:

- this transform needs those search surfaces available during `split()` and `execute()`
- the planner can show that dependency explicitly
- the transform fingerprint can include the relevant surface identity

This is intentionally not modeled as `depends_on`.

`depends_on` remains artifact flow.
`uses` is capability access inside the build.

### TransformContext Fields

The exact class can evolve, but the public contract should include at least:

- `ctx.config`
  - user-declared transform config for this layer
- `ctx.llm_config`
  - resolved LLM config for this invocation
- `ctx.workspace`
  - stable build-scoped workspace metadata when needed
- `ctx.surface(name)`
  - return a handle for a declared surface
- `ctx.logger`
  - optional structured logging hook

Illustrative usage:

```python
def split(self, inputs: list[Artifact], ctx: TransformContext):
    topics = ctx.config["topics"]
    search = ctx.surface("episode-search")

    units = []
    for topic in topics:
        hits = search.search(topic.replace("-", " "), mode="semantic", limit=25)
        relevant_labels = {hit.artifact_label for hit in hits}
        relevant = [ep for ep in inputs if ep.label in relevant_labels] or inputs
        units.append(WorkUnit(inputs=relevant, extras={"topic": topic}))
    return units
```

### Search Handle Contract

The search handle should be backend-agnostic.

Illustrative shape:

```python
search = ctx.surface("episode-search")
hits = search.search(
    "docker compose alternatives",
    mode="hybrid",
    layers=["episodes"],
    limit=10,
)
```

Search results should be stable source-backed records, not raw DB rows or file paths.

Minimum useful fields:

- `row_id`
- `artifact_label`
- `artifact_oid`
- `score`
- `content`
- `metadata`
- `source_refs`

### Local Realization Semantics

When a transform declares `uses=[episode_search]`, the runner should:

1. ensure the canonical surface state exists
2. create or reuse a local realization for that surface
3. construct a search handle over that realization
4. pass it through `TransformContext`

The transform must not know whether that local realization is backed by:

- SQLite
- an in-memory structure
- a future optimized default implementation

That is the implementation detail that `SynixSearch` and the local search tooling should keep opaque.

### Compatibility Bridge

The first implementation can keep a temporary compatibility path:

- if a transform still uses the old `config` signature, Synix adapts the new context down to the old call shape
- `search_db_path` can remain as a migration shim

But the new public contract should be:

- declare `uses=[surface]`
- consume `ctx.surface("...")`

### Why Not Just Pass Tools In Config

Passing search or tool handles through `config` would preserve the current ambiguity and make the API harder to reason about long-term.

The context split is cleaner because:

- config is declarative input
- context is execution environment

That distinction is worth making explicitly.

## Internal Build Model

### What SearchSurface Builds

`SearchSurface` should create canonical immutable state, not just one database file.

That canonical state should include:

1. normalized search rows
2. stable row ids
3. source-backed evidence refs
4. optional embedding records
5. one surface-state manifest object

The first implementation can keep this minimal. The important part is the separation of logical state from realization.

### Surface Rows

Each searchable record should normalize one source artifact into one row-like object.

Illustrative shape:

```json
{
  "row_id": "episodes:ep-123",
  "artifact_oid": "oid_artifact_1",
  "artifact_id": "sha256:...",
  "layer": "episodes",
  "content": "Conversation about Docker Compose alternatives...",
  "metadata": {
    "title": "Docker Compose Alternatives",
    "date": "2025-02-14"
  },
  "source_refs": [
    {
      "artifact_label": "t-chat-42",
      "offsets": []
    }
  ]
}
```

### Embedding Records

Embeddings should be separate canonical records keyed by `row_id`.

This allows:

- reuse across realizers
- cleaner diffing later
- adapter-specific embedding overrides when needed

### Surface State Manifest

One immutable object should represent the surface closure.

Illustrative shape:

```json
{
  "type": "surface_state",
  "schema_version": 1,
  "surface_name": "memory-search",
  "surface_type": "search",
  "modes": ["fulltext", "semantic"],
  "row_oids": ["..."],
  "embedding_oids": ["..."],
  "source_artifact_oids": ["..."],
  "fingerprint": "sha256:..."
}
```

This object becomes the canonical handoff point to local realizers and release adapters.

## Realization Model

### Default Local Realization

Synix should automatically create a local disposable realization for build-time use.

This is where the current SQLite implementation naturally fits.

Illustrative location:

```text
.synix/work/<run_id>/surfaces/memory-search/default/
```

Important semantics:

- local realization is disposable
- local realization is rebuildable from `surface_state`
- local realization is not the canonical truth

### Default Synix Search

`SynixSearch` should expose:

- the default local search artifact for compatibility
- standard Synix search tooling and client access

Internally that may still be SQLite today, but the public contract should stay opaque.

### Explicit Release Adapters

Release adapters consume canonical surface state, not one intermediate backend image.

That means:

- Postgres release is built from `surface_state`
- Qdrant release is built from `surface_state`
- the SQLite local realization is not the canonical source for those releases

This is critical for flexibility.

## Backend-Specific Derived State

The canonical surface state should be stable and reusable, but it should not try to capture every possible backend-specific requirement.

So the layering should be:

1. canonical shared surface state
2. adapter-specific derived release state
3. release receipt that records what was used

That allows:

- different embedding models per backend
- different table or payload schemas per backend
- multiple release targets from one logical surface

Example:

```python
memory_search = SearchSurface(
    "memory-search",
    sources=[episodes, monthly, core],
    modes=["fulltext", "semantic", "hybrid"],
    embedding_config={
        "provider": "fastembed",
        "model": "BAAI/bge-small-en-v1.5",
    },
)

pipeline.release(
    SynixSearch(
        "default-search",
        surface=memory_search,
    )
)

pipeline.release(
    PostgresSearchRelease(
        "prod-search",
        surface=memory_search,
        adapter_config={
            "embedding_model": "text-embedding-3-large",
            "schema_mode": "normalized_v2",
        },
    )
)
```

This should be explicit and fingerprinted in receipts.

## Template Rewrites

These examples are illustrative API targets, not implementation-ready snippets.

### Template 01: Chatbot Export Synthesis (Monthly)

Current model:

- `SearchIndex` as projection
- `FlatFile` as projection

New model:

- `memory-search` is a `SearchSurface`
- `SynixSearch` is the default search release/tooling bundle
- `ContextDoc` remains a release target

```python
memory_search = SearchSurface(
    "memory-search",
    sources=[episodes, monthly, core],
    modes=["fulltext", "semantic"],
)

pipeline.add(transcripts, episodes, monthly, core, memory_search)
pipeline.release(SynixSearch("default-search", surface=memory_search))
pipeline.release(ContextDoc("context-doc", sources=[core], output_path="./build/context.md"))
```

### Template 01: Chatbot Export Synthesis (Topical)

This is the template that most clearly benefits from the new model.

```python
episode_search = SearchSurface(
    "episode-search",
    sources=[episodes],
    modes=["fulltext", "semantic"],
)

topics = TopicalRollup(
    "topics",
    depends_on=[episodes],
    uses=[episode_search],
    config={"topics": [...]},
)

memory_search = SearchSurface(
    "memory-search",
    sources=[episodes, topics, core],
    modes=["fulltext", "semantic"],
)

pipeline.add(transcripts, episodes, episode_search, topics, core, memory_search)
pipeline.release(SynixSearch("default-search", surface=memory_search))
```

This removes `search_db_path` from the semantic contract.

### Template 02: TV Returns

The main need here is policy lookup during enrichment.

```python
policy_search = SearchSurface(
    "policy-search",
    sources=[policy_index],
    modes=["fulltext", "semantic", "hybrid"],
)

cs_product_brief = DemoEnrichCSBriefTransform(
    "cs_product_brief",
    depends_on=[product_offers],
    uses=[policy_search],
)

cs_search = SearchSurface(
    "cs-search",
    sources=[cs_product_brief],
    modes=["fulltext", "semantic"],
)

pipeline.add(product_offers, policies, policy_index, policy_search, cs_product_brief, cs_search)
pipeline.release(SynixSearch("default-search", surface=cs_search))
```

### Template 03: Team Report

This pipeline likely needs only a final search surface:

```python
search = SearchSurface(
    "search",
    sources=[bios, project_brief, work_styles, team_dynamics, final_report],
    modes=["fulltext"],
)

pipeline.add(..., search)
pipeline.release(SynixSearch("default-search", surface=search))
```

### Template 04: Sales Deal Room

If citation or fixer workflows need build-time retrieval later, they can use a dedicated internal surface. Otherwise the final search surface is sufficient.

```python
deal_search = SearchSurface(
    "deal-search",
    sources=[
        competitor_docs,
        product_specs,
        deal_context,
        win_reports,
        competitive_intel,
        strategy,
        call_prep,
    ],
    modes=["fulltext"],
)

pipeline.add(..., deal_search)
pipeline.release(SynixSearch("default-search", surface=deal_search))
```

### Template 05: Batch Build

This pipeline does not need internal semantic lookup. It just gets the final surface plus default release:

```python
search = SearchSurface(
    "search",
    sources=[bios, work_styles, team_summary],
    modes=["fulltext"],
)

pipeline.add(bios, work_styles, team_summary, search)
pipeline.release(SynixSearch("default-search", surface=search))
```

### Template 06: Claude Sessions

This looks like the monthly personal-memory pipeline:

```python
session_search = SearchSurface(
    "session-search",
    sources=[summaries, rollups, core],
    modes=["fulltext"],
)

pipeline.add(source, summaries, rollups, core, session_search)
pipeline.release(SynixSearch("default-search", surface=session_search))
pipeline.release(ContextDoc("context", sources=[core], output_path="./build/context.md"))
```

## LENS Shape Under This Model

Checkpoint semantics are intentionally deferred, but the LENS-side build shape becomes much cleaner:

```python
chunks = ChunkFamily("chunks", depends_on=[episodes])

base_search = SearchSurface(
    "base-search",
    sources=[chunks],
    modes=["fulltext", "semantic", "hybrid", "layered"],
)

summaries = SummaryFamily(
    "summaries",
    depends_on=[chunks],
    uses=[base_search],
)

core = CoreMemoryFamily(
    "core",
    depends_on=[chunks, summaries],
)

graph = GraphFamily(
    "graph",
    depends_on=[chunks],
)
```

Later, a separate checkpoint or bank RFC can define how Synix certifies prefix-valid child snapshots over closures that include:

- chunk artifacts
- summary or core-memory artifacts
- graph artifacts
- search-surface state

This RFC deliberately does not force that design now.

## Migration Strategy

### Public Migration

Do not hard-break the old `SearchIndex` API immediately.

Instead:

1. introduce `SearchSurface`
2. introduce `SynixSearch`
3. document those as the preferred new API
4. keep `SearchIndex` as compatibility sugar

### Mechanical Lowering

`SearchIndex(...)` can be interpreted internally as:

```text
SearchSurface(...) + SynixSearch(...)
```

That means existing pipelines can keep working while new pipelines use the correct model.

### Legacy Compatibility

For a migration window:

- `SearchIndex` still works
- Synix can still emit `./build/search.db`
- `search_db_path` can remain as a compatibility shim

But the new design target should be:

- transforms consume search tools or a build context
- not a path string

## Recommended Issue Reframing

### Issue #15

The current title, "Projections as first-class DAG nodes", is too close to the rejected Option B model.

The real design target is closer to:

- search surfaces and build-time query capabilities
- explicit semantic lookup access inside the build
- without treating release projections as recursive live build dependencies

The issue can stay open, but its implementation target should be rewritten before coding.

### Issue #10

`#10` remains the natural place for:

- a consistent search or retrieval tool API
- stable source-backed refs in responses
- mode support like `keyword`, `semantic`, `hybrid`, and `layered`

### Issue #81

Checkpoint or bank certification should be designed separately on top of:

- snapshots
- search surfaces
- other artifact families

This RFC intentionally leaves that separate.

## Implementation Phases

### Phase 1: Internal Split

- add `SearchSurface` model
- add canonical surface-state object type(s)
- refactor current SQLite search build into a local realizer
- keep current `SearchIndex` behavior as compatibility lowering

### Phase 2: Build-Time Consumption

- add build context or tool access for transforms
- migrate `TopicalRollup` off `search_db_path`
- add explicit semantic capability declaration in the pipeline model

### Phase 3: Default Synix Search Release

- add `SynixSearch`
- keep implementation opaque
- preserve current default `search.db` behavior as the initial default release

### Phase 4: Explicit Release Adapters

- allow backend-specific search targets
- add receipts and target-specific adapter config
- start with full rebuild semantics if needed

### Phase 5: Migration Cleanup

- move docs and templates to the new API
- deprecate direct `SearchIndex` authoring
- deprecate `search_db_path` from the public contract

## Test Strategy

Every implementation slice should include:

- unit tests
- at least one automated end-to-end test
- docs updates
- demo or template follow-on note

Recommended coverage:

### Unit

- canonical surface row generation
- surface-state fingerprinting
- local realizer contract
- `SearchIndex` compatibility lowering
- backend adapter config fingerprinting

### Integration

- build a surface from artifacts and query it through the local Synix tool path
- `TopicalRollup` or equivalent uses search without reading `search_db_path`
- default `SynixSearch` release produces the expected compatibility outputs

### End-to-End

- rewrite one retrieval-heavy template to use the new model
- prove the same user-facing workflow still works

## Prior Art

This design follows established patterns from other systems:

- Nix closure and realization split:
  - logical store objects are distinct from realized outputs
  - https://releases.nixos.org/nix/nix-2.18.4/manual/command-ref/nix-store/realise.html
- Materialize views and indexes:
  - queryable logical state and serving structures are distinct concepts
  - https://materialize.com/docs/concepts/views/
  - https://materialize.com/docs/sql/create-index/
- Dagster asset graphs:
  - build dependencies are modeled separately from publication or serving concerns
  - https://docs.dagster.io/

Synix should take the same lesson:

- searchable capability is part of the build closure
- publication targets are separate

## Resolved Design Choices

This RFC resolves the design questions that matter for implementation:

1. `SearchSurface` is a first-class pipeline declaration registered through `pipeline.add()`.

   Publicly, it is a searchable capability declaration rather than a transform or release target.
   Internally, Synix may reuse `Layer`-style registration mechanics for the first slice if that keeps the planner and runner changes small.

2. Build-time surface access is declared with `uses=[...]`.

   `depends_on` remains artifact flow.
   `uses` is explicit capability access.
   That keeps the graph readable and avoids pretending that a local realization is itself an upstream artifact input.

3. The first implementation should use search-specific canonical object types.

   Do not start with a generic `surface_state` abstraction.
   Start with search-specific records such as:

   - normalized search rows
   - optional embedding records
   - one `search_surface_state` manifest object

   If later surfaces emerge with the same needs, Synix can generalize from working code.

4. `SynixSearch` preserves the current local SQLite-backed experience as a compatibility target, but not as a public contract.

   The first release should keep the current user-facing behavior where practical:

   - `uvx synix search ...` still works
   - `./build/search.db` can still exist as the default local compatibility output
   - current demos and templates should keep working during migration

   But Synix should not freeze the SQLite schema, table layout, or file-level access pattern as part of the public API.

## Remaining Implementation Questions

These are implementation choices, not design blockers:

1. whether the first `TransformContext` adapter should support both `split(inputs, config)` and `split(inputs, ctx)` through one compatibility shim, or split the migration by transform class
2. how much of the current SQLite physical schema should be preserved in the first `SynixSearch` compatibility target to minimize demo churn
3. whether the first local realizer should live under `.synix/work/<run_id>/surfaces/...` immediately or start with a simpler build-dir-backed compatibility path and move after the contract is stable

## Recommendation

Synix should adopt this model:

- `SearchSurface` is the logical searchable capability
- `SynixSearch` is the default Synix-provided search realization and tooling bundle
- explicit release targets are advanced and optional
- transforms use search through tools or a build context, not a path convention
- checkpoint or bank semantics stay separate for now

This gives Synix a cleaner foundation for both ordinary pipelines and the LENS benchmark path without forcing release semantics into every internal semantic dependency.
