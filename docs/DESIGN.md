# Synix Design Document

**A build system for agent memory**

---

## Part I: Vision

### 1.0 Origin Story

In May 2025, I was deep in a rabbit hole about knowledge provenance. I'd been building voice AI agents and kept running into the same problem: the agent would say something, and I had no idea where it came from. Was it grounded in something real? Was it hallucinated? Was it something I told it three weeks ago that got compressed and mutated through layers of summarization?

So I started sketching what I called an "epistemic framework" — a system where every piece of knowledge carries traceable provenance back to its source. Derivation chains. DERIVED_FROM relationships between memory blocks. A hierarchical structure where raw chunks roll up into conversations, conversations into parent nodes, parent nodes into a root. I was calling it "an operating system for perspective itself," which tells you how deep in the weeds I was.

The ideas were real. The architecture made sense — dual-layer memory with a vector store for fuzzy recall and a knowledge graph for precise derivation tracking, specialized extractors feeding structured data into both. I demoed pieces of it at Noisebridge. I explored building it on top of Letta. I had names for things — Kairix for the cognitive engine, Apiana for the memory infrastructure.

But every implementation was bespoke. The extraction pipeline was hand-wired. The hierarchy was hardcoded into the graph schema. When I wanted to change how summarization worked, I was rewriting plumbing. When I wanted to try a different memory architecture — and I kept wanting to try different memory architectures — it was a teardown.

I moved on to other things. Built more agent experiments. Kept bumping into the same friction from different angles. Eventually started working on KP3, which was an attempt at programmable knowledge processing — closer to the right abstraction, but still hard to reason about. The pipeline concepts were there, branching was there, incremental processing was there in embryonic form. But it hadn't clicked yet.

Then in early 2026, I was doing something completely unrelated. I had my ChatGPT and Claude exports and I wanted to search through them — honestly, I was trying to prove to myself that I'd come up with certain ideas before someone else had. Petty, but real. And I realized I kept doing this. Kept wanting to look back across months of conversations and find patterns, trace how ideas evolved, see what I'd actually said versus what I remembered saying.

I thought: I already have KP3. Maybe I can use it for this. Upload the exports incrementally, skip what's already processed, build up a searchable history. But as I started mapping it out, I realized KP3's model was wrong for this. Too imperative. Too much plumbing exposed. What I actually wanted was to just *declare* the pipeline — here are my sources, here's how I want them transformed, here's how I want them grouped and summarized, here's what I want to be searchable. Like writing a CloudFormation template or a Step Functions definition, but for knowledge processing.

And then I wanted to be able to change my mind. Try a different summarization prompt without reprocessing everything. Try a different grouping strategy on a branch without blowing up what I already had. And trace any piece of the output — any monthly summary, any evolving world model — back through exactly which conversations and which transforms produced it.

That's when I realized I'd been circling the same problem for almost a year. The epistemic framework from May 2025 was the same system. The DERIVED_FROM chains were the same provenance model. The hierarchical chunks-to-conversations-to-parent-nodes structure was the same altitude concept. I just hadn't found the right interface for it.

The interface is declarative pipelines. You describe the memory architecture you want — the transforms, the hierarchy, the search surfaces — and the system handles the execution, the incrementality, the provenance tracking, and the ability to change your mind without starting over.

Because after a year of trying to design the "right" memory architecture, the thing I actually learned is that there might not be one. It depends on what you're building, what your data looks like, and what questions you need to answer. The tool that matters isn't the architecture. **It's the workbench that lets you iterate on architectures.**

That's when the build system analogy clicked. This is the same problem software had before Make, before Bazel, before dbt. Without a build system, changing your compiler flags means recompiling everything. Without a build system, you can't trace which source files produced which binary. Without a build system, experimenting with a different optimization strategy is terrifying because you might break what's working.

Synix is a build system for memory. Your chat logs and conversation exports are source files. Your prompts for summarization, aggregation, and synthesis are build rules. Your searchable indexes and world models are build artifacts. Change a rule, get an incremental rebuild. Trace any artifact back to its sources. Experiment on a branch without breaking main.

That's Synix.

### 1.1 The Three Hypotheses

Synix is built on three core beliefs about agent memory:

#### Hypothesis 1: No One-Size-Fits-All Memory Architecture

A customer service bot, a therapy companion, and a coding assistant all need fundamentally different memory structures. The existence of Mem0, Letta, Zep, and LangMem as co-existing solutions is itself evidence — if there were a "correct" architecture, the market would have converged.

#### Hypothesis 2: Memory Needs Evolve Over an Application's Lifecycle

This has a weak and strong version:

**Weak (obviously true):** As data accumulates, you need summarization and hierarchy or the context window drowns. This is a scaling problem.

**Strong (the real bet):** The *type* of cognition changes qualitatively as the agent-user relationship matures:

- **Early:** Episodic memory dominates — "what did we discuss Tuesday?"
- **Later:** Semantic memory emerges — "this person values directness"
- **Eventually:** Procedural memory crystallizes — "when this user does X, respond with Y"

These aren't different amounts of the same thing — they're structurally different architectures. The memory system that works at 10 conversations may fail at 1,000.

#### Hypothesis 3: Memory Architecture Is a Runtime Concern

You don't pick an architecture on day one and leave it. You need to evolve the pipeline as you learn what the agent actually needs. The tool that lets you discover the right architecture fastest wins.

### 1.2 The Vision

**Synix is a build system for agent memory.**

The pitch isn't "our pipeline is better than RAG." It's: "Nobody knows what the right memory architecture is for your domain yet — including you. Synix lets you find out."

| Build System | Synix |
|--------------|-------|
| Source files | Chat logs, notes, transcripts |
| Build rules | Transform, aggregate, fold, merge |
| Build artifacts | Summaries, reflections, world models |
| Dependency graph | Provenance DAG |
| Cache keys | Materialization keys |
| Incremental rebuild | Skip-if-unchanged |
| Clean build | `full=True` |
| Branch builds | Branch experiments + promote |

What if you could:

- **Declare your sources and build rules in Python** — composable, testable, versionable
- **Branch and experiment** — try different build rules in isolation
- **Incremental rebuild** — change a rule, only rebuild what's affected
- **Evolve without data loss** — change the architecture, keep the sources
- **Full lineage** — trace any artifact back through the dependency graph to its sources

### 1.3 The Killer Feature: Architecture Migration

The under-articulated superpower: **architecture migration without data loss.**

If you're on Letta and it's not working, your options today are:
1. Rip it out and start over
2. Write custom glue code
3. Cope

Synix gives you: edit the pipeline definition, run it again, and your same raw data produces a completely different memory structure. The data doesn't move — the *lens* changes.

**The pitch to engineering teams:** "You're not locked into a memory architecture decision on day one. Start simple, evolve as you learn, never lose data in the transition."

### 1.4 Tagline

**"A build system for agent memory"**

The name "Synix" comes from **synthesis** — the core action of transforming raw information into processed understanding. In chip design, synthesis turns high-level descriptions into gate-level implementations. In software, build systems turn source files into artifacts. Synix does the same for agent memory: sources become artifacts through declared build rules, with full lineage tracking and incremental rebuilds.

---

## Part II: Conceptual Grounding

### 2.1 The Build System Lineage

Synix is a build system. It inherits ideas from the same family tree as Make, Bazel, and dbt.

#### dbt (data build tool)

dbt is a build system for SQL transforms. Synix is a build system for LLM-based knowledge transforms.

**What we share:**
- DAG-based processing with explicit dependencies
- Incremental materialization (skip-if-unchanged)
- Layered architecture (raw → staging → marts / sources → summaries → world model)

**Key differences:**
- LLM transforms are **non-deterministic** — same inputs can produce different outputs. Caching matters more.
- LLM transforms are **expensive** — $0.01-$1.00 per call vs. milliseconds of SQL. Incremental rebuild matters more.
- LLM transforms produce **unstructured output** — provenance matters more because you can't just re-run a query to verify.
- Memory architecture is **unsettled** — experimentation matters more. You need branches to try different build rules safely.

#### CDK vs. CloudFormation

CloudFormation (YAML/JSON) came first, but CDK (imperative code that generates infrastructure) won with developers. Real infrastructure is conditional, looped, parameterized, and composed — all things painful in config and natural in code.

**Key decision:** Python-first, not config-first. Build rules are code, not config.

#### lakeFS / DVC / Bazel

**What we take:**
- Branching for experiments (branch builds)
- Copy-on-write semantics (efficient forking)
- Reproducible builds (or in our case: audit determinism — explain what produced what, even if not perfectly reproducible)

### 2.2 Competitive Landscape

| Product | Approach | Strength | Weakness |
|---------|----------|----------|----------|
| **Mem0** | Simple API: add/search | Dead simple, AWS-selected | No processing, no provenance |
| **Letta/MemGPT** | Agent self-manages memory | Dynamic, autonomous | Opinionated hierarchy |
| **Zep (Graphiti)** | Temporal knowledge graph | Excellent temporal reasoning | Complex, graph-specific |
| **LangMem** | Semantic/Procedural/Episodic | Clear taxonomy | Fixed categories |

**Synix positioning:**

We don't compete on simplicity (Mem0 wins) or temporal reasoning (Zep wins). We compete on **flexibility**, **provenance**, and **evolvability**:

- Define any architecture you can imagine
- Trace any memory back to its source
- Change the architecture without losing data
- Measure which architecture works best

### 2.3 Red Team Critiques (Honest Assessment)

#### "Over-engineered" (Mem0's perspective)

**Validity:** Completely fair. More concepts to learn.

**Counter:** Different use cases. Mem0 is for simple session memory. Synix is for bulk import, custom pipelines, provenance requirements, and architecture experimentation.

#### "Branching is academic"

**Validity:** Most individual developers won't branch.

**Counter:** Enterprise and research users will. And branching enables the migration story — the sharpest differentiator.

#### "Just RAG with extra steps"

**Validity:** The core is embed-and-retrieve.

**Counter:** The "extra steps" are the value. Summarization reduces noise, aggregation creates hierarchy, folding builds world models, provenance enables debugging. Raw RAG on 1,000 conversations is worse than hierarchical processed memory.

### 2.4 Market Timing

The "no one has converged" window is real right now — the agent memory space is in active churn. But if the ecosystem converges on 2-3 standard patterns (like web dev converged on REST + Postgres), the workbench value shrinks.

Capture winning patterns as templates before that happens.

---

## Part III: Domain Model

The domain model maps directly to build system concepts:

| Build System | Synix | Description |
|--------------|-------|-------------|
| Build definition | Pipeline | Container for sources, rules, and artifacts |
| Source files | Sources | Chat logs, notes, transcripts |
| Build rules | Steps | Transform, aggregate, fold, merge |
| Build artifacts | Records | The actual content produced at each step |
| Targets/outputs | Artifacts | Queryable surfaces (search index, projection) |
| Dependency graph | Provenance DAG | Which records depend on which sources |

### 3.1 Core Entities

```
┌─────────────────────────────────────────────────────────────────┐
│  PIPELINE                                                        │
│  A named container for an agent's memory processing.            │
│                                                                  │
│  Fields:                                                         │
│  - name: string (unique identifier)                              │
│  - agent: string (who this memory belongs to)                    │
│  - description: string (optional)                                │
│  - created_at: timestamp                                         │
└─────────────────────────────────────────────────────────────────┘
         │
         ├─────────────────┬─────────────────┬─────────────────┐
         ▼                 ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  SOURCE         │ │  STEP           │ │  ARTIFACT       │ │  BRANCH         │
│                 │ │                 │ │                 │ │                 │
│  Input that     │ │  Processing     │ │  Queryable      │ │  Variant for    │
│  brings data    │ │  operation      │ │  target         │ │  experiments    │
│  into pipeline  │ │                 │ │                 │ │                 │
│                 │ │  Types:         │ │  Types:         │ │                 │
│  Types:         │ │  - transform    │ │  - projection   │ │                 │
│  - file         │ │  - aggregate    │ │  - search       │ │                 │
│  - api          │ │  - fold         │ │                 │ │                 │
│  - stream       │ │  - merge        │ │                 │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
```

### 3.2 Runtime Entities

```
┌─────────────────────────────────────────────────────────────────┐
│  RECORD                                                          │
│  A single piece of content produced by the system.              │
│                                                                  │
│  Fields:                                                         │
│  - id: uuid (globally unique)                                    │
│  - content: string (the actual text/data)                        │
│  - step: string (which step produced this)                       │
│  - branch: string (which branch this belongs to)                 │
│  - sources: list[uuid] (input record IDs — provenance)           │
│  - materialization_key: string (see below)                       │
│  - run_id: uuid (which run produced this)                        │
│  - created_at: timestamp                                         │
│  - metadata: { ... } (see reserved keys)                         │
│  - audit: {                                                      │
│      prompt_template_hash: string,                               │
│      rendered_prompt_hash: string,                               │
│      model: string,                                              │
│      temperature: float,                                         │
│      raw_response: string,                                       │
│    }                                                             │
│  - embedding: vector (optional)                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  RUN                                                             │
│  An execution of one or more steps.                             │
│                                                                  │
│  Fields:                                                         │
│  - id: uuid                                                      │
│  - pipeline: string                                              │
│  - branch: string                                                │
│  - steps: list[string]                                           │
│  - status: pending | running | completed | failed                │
│  - started_at / completed_at: timestamp                          │
│  - stats: { input, output, skipped, errors, tokens, duration }   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Cache Keys (Materialization Keys)

Every build system needs cache keys to determine whether a rebuild is needed. Bazel uses content hashes. Make uses timestamps. Synix uses **materialization keys**.

The original Cursor design ("last_processed_id") was broken — UUIDs have no natural ordering, and time-ordered IDs break on late-arriving data and backfills.

**Solution:** Materialization keys = cache keys that capture everything affecting what gets produced.

Every record has a **materialization key**:
```
(branch, step_name, step_version_hash, input_fingerprints, group_key)
```

Where:
- `step_version_hash` = hash(step_type + config + prompt_template + model)
- `input_fingerprints` = hash of source record content(s)
- `group_key` = period for aggregates, sequence for folds

**Incremental semantics:** If the materialization key exists, skip. If it doesn't, process.

**Why this works:**
- Changing a prompt → step_version_hash changes → automatic reprocess
- Late-arriving data → new input_fingerprint → gets processed
- No `--full` flag needed for prompt changes

#### 3.3.1 Canonical Fingerprinting Rules

Every Record gets a `content_fingerprint` = SHA-256 hash over normalized content bytes (UTF-8, strip trailing whitespace).

Per-step materialization key recipes:

| Step Type | Materialization Key |
|-----------|-------------------|
| **transform** | `(branch, step_name, input_record_id, step_version_hash)` |
| **aggregate** | `(branch, step_name, group_key, combined_input_fingerprint, step_version_hash)` |
| **fold** | `(branch, step_name, sequence_fingerprint, step_version_hash)` |
| **merge** | `(branch, step_name, dedupe_key, step_version_hash)` |

Where:
- `combined_input_fingerprint` = SHA-256 of concatenated input `content_fingerprint` values in **stable sort order** (sorted by content_fingerprint lexicographically)
- `sequence_fingerprint` = SHA-256 of ordered list of input content_fingerprints (order determined by `meta.time.period` or explicit `order_key`)

**Why different keys per step type:** Transform is 1:1 — the input record ID is stable and sufficient. Aggregates and folds take N inputs where the set/sequence can change — fingerprinting the combined inputs detects when group composition changes.

### 3.4 Reserved Metadata Keys

```python
# Time
meta.time.created_at      # source timestamp
meta.time.period          # for aggregates: "2024-03"

# Chat context
meta.chat.conversation_id
meta.chat.message_id
meta.chat.author          # "user" | "assistant"

# Source tracking
meta.source.type          # "chatgpt-export" | "claude-export"
meta.source.path          # original file path

# Step tracking
meta.step.version_hash    # step version that produced this
meta.step.sequence        # for folds: position in sequence
```

User-defined metadata goes in `meta.custom.*` to avoid collisions.

### 3.5 Build Rule Types

Synix has four build rule types. Each defines how source artifacts become output artifacts.

| Rule Type | Input | Output | Build Analogy |
|-----------|-------|--------|---------------|
| **transform** | 1 artifact | 1 artifact | Compiling `.c` → `.o` |
| **aggregate** | N artifacts (grouped) | 1 artifact | Linking `.o` files into a library |
| **fold** | N artifacts (ordered) | 1 artifact | Rolling build with accumulated state |
| **merge** | N artifacts (from multiple sources) | N artifacts (deduplicated) | Deduplicating objects from different build paths |

#### Transform (1:1)

The workhorse build rule. One source artifact in, one output artifact out.

**V1 decision:** No `batch_size`. Batching transforms multiple inputs into a parsing problem — if the LLM doesn't return aligned output, you get silent corruption. Keep it simple.

#### Aggregate (N:1 by key)

Group records by a key (typically time period), reduce to one record.

**Incremental semantics:** When a new record arrives for an already-aggregated period, the materialization key changes (new combined_input_fingerprint). The old aggregate for that period is **replaced** — only one aggregate per (branch, step, group_key) exists at a time. This is "last write wins": the most recent aggregation with the complete set of inputs is canonical.

**Downstream impact:** If a monthly aggregate is replaced, downstream steps (e.g., fold) see a changed input fingerprint and re-process. For folds, this triggers the backfill behavior defined in section 3.8.

#### Fold (N:1 sequential)

Sequential processing with accumulated state. The key difference from aggregate: **order matters**, and each step sees previous state.

See section 3.8 for the full fold contract including invalidation rules.

#### Merge (N:N)

Combine records from multiple sources with deduplication. Merge sits at the top of most pipelines (unifying ChatGPT + Claude exports), so its semantics matter.

**Deduplication strategy:**

```python
pipeline.merge("unified",
    from_=["chatgpt", "claude"],
    dedupe="artifact_id",       # default: exact content match
    # dedupe="metadata_match",  # match on specific fields
    # dedupe="fuzzy",           # similarity threshold (future)
    conflict="prefer_latest"    # or: prefer_first, keep_all
)
```

Dedup options:
- `artifact_id` — SHA-256 of content. Exact duplicates only. Fast and safe.
- `metadata_match` — Match on specified fields (e.g., `conversation_id`). For when same conversation appears in multiple exports with different formatting.
- `fuzzy` (future) — Semantic similarity above threshold. Dangerous — requires careful tuning.

Conflict resolution (when dedup matches):
- `prefer_latest` — Keep record with most recent `meta.time.created_at`
- `prefer_first` — Keep first record encountered (stable ordering)
- `keep_all` — No dedup, emit all records (useful for debugging)

**Open question:** Should merge require an LLM call (to reconcile conflicting versions) or stay purely mechanical? V1 keeps it mechanical — LLM-based reconciliation is a transform step the user can add downstream.

### 3.6 The `from_` Parameter

The `from_` parameter accepts either a string or list, with validity depending on step type:

| Step Type | `from_` accepts | Rationale |
|-----------|-----------------|-----------|
| **transform** | string only | 1:1 mapping requires single source |
| **aggregate** | string only | Groups records from one upstream step |
| **fold** | string only | Sequential ordering requires single source |
| **merge** | list only | Combining sources is the point |
| **artifact** | string or list | Single step or union of multiple |

```python
# Transform — single source
pipeline.transform("summaries", from_="transcripts", ...)

# Aggregate — single source (groups by period within that source)
pipeline.aggregate("monthly", from_="summaries", ...)

# Fold — single source (ordering must be unambiguous)
pipeline.fold("world-model", from_="monthly", ...)

# Merge — list required
pipeline.merge("unified", from_=["chatgpt", "claude"])

# Artifact — either works
pipeline.artifact("context", from_="world-model", ...)
pipeline.artifact("memory-index", from_=["summaries", "monthly"], ...)
```

### 3.8 Fold Contract and Invalidation Rules

Fold is the most novel primitive. This section defines its exact behavior.

**Ordering contract:**
- Fold input is a **totally ordered list** by `meta.time.period` (or explicit `order_key` parameter)
- The fold output is a function of the **entire sequence**, not incremental by default
- `sequence_fingerprint = SHA-256(ordered list of input content_fingerprints)`

**Invalidation behavior:**
- Any change to any item in the sequence → sequence_fingerprint changes → cache miss → fold reruns
- Insert in middle → sequence changes → full rerun from affected point

**Checkpoint optimization (v1.0):**
- Fold checkpoints every N steps (configurable, default: 6)
- Each checkpoint stores: `prefix_fingerprint = SHA-256(first M items)`, accumulated state
- On rerun, resume from nearest prior checkpoint whose prefix_fingerprint still matches
- Example: Record inserted in month 8 of 28, checkpoint at month 6 still valid → rerun from month 7 only

**State growth management:**
- `max_state_tokens` parameter on fold (default: 8000)
- When state exceeds limit, fold automatically summarizes state before passing to next iteration
- Built-in compression, not user-configured

**Error recovery:**
- Checkpoint every N steps
- On failure: resume from last valid checkpoint
- If checkpoint corrupted (fingerprint mismatch): fall back to full rerun
- Failure logged in run stats

```python
pipeline.fold("world-model",
    from_="monthly",
    prompt=world_model_prompt,
    order_key="meta.time.period",      # explicit ordering
    checkpoint_every=6,                 # checkpoint frequency
    max_state_tokens=8000               # state size limit
)
```

### 3.9 Determinism and Audit

LLM calls are non-deterministic. We pick a lane: **audit determinism**.

Every record stores:
- `prompt_template_hash` — which template version
- `rendered_prompt_hash` — the actual prompt sent
- `model`, `temperature` — model parameters
- `raw_response` — what the LLM returned

You can always explain *why* a record exists (inputs + prompt + response), even if you can't reproduce it exactly.

**Future:** `pipeline.run(replay=True)` reuses cached responses for free rebuilds.

---

## Part IV: Python API

### 4.1 Design Decision: Python-First

TOML configuration contradicts the core thesis. TOML says "I know what I want." The whole point of Synix is "you don't know the right architecture yet."

The CDK vs. CloudFormation parallel is exact: declarative config is painful for things that are conditional, looped, parameterized, and composed — all natural in code.

The target user is already writing Python. A Python API is:
- Just as readable for simple cases
- Far more powerful for experimentation
- One less layer to implement
- Composable with the rest of their agent code

### 4.2 Pipeline Definition

```python
from synix import Pipeline

# Create pipeline
pipeline = Pipeline("personal-memory", agent="mark")

# Sources — where data comes from
pipeline.source("chatgpt",
    file="~/exports/chatgpt.json",
    format="chatgpt-export"
)
pipeline.source("claude",
    file="~/exports/claude.json",
    format="claude-export"
)

# Steps — processing operations
pipeline.merge("unified", from_=["chatgpt", "claude"])

pipeline.transform("transcripts",
    from_="unified",
    prompt=extract_transcript,
    validate=None  # Optional: callable(Record) -> bool. Reserved for v1.0.
)

pipeline.transform("summaries",
    from_="transcripts",
    prompt=summarize_conversation
)

pipeline.aggregate("monthly",
    from_="summaries",
    period="month",
    prompt=monthly_reflection
)

pipeline.fold("world-model",
    from_="monthly",
    prompt=world_model_prompt
)

# Artifacts — queryable targets
pipeline.artifact("context",
    from_="world-model",
    surface="projection"
)

pipeline.artifact("memory-index",
    from_=["summaries", "monthly"],
    surface="search"
)
```

### 4.3 Prompt Functions

Prompts are Python functions, not template files.

#### 4.3.1 Prompt Identity and Versioning

Prompt functions need stable identity for materialization keys. Two approaches:

**Default: Source hash**
```python
def summarize_conversation(record: Record) -> str:
    ...
# step_version_hash includes hash of inspect.getsource(summarize_conversation)
```

**Override: Explicit version** (for closures, generated functions, external state)
```python
@synix.prompt(version="2026-02-05")
def summarize_conversation(record: Record) -> str:
    ...
# step_version_hash uses explicit version string, ignores source
```

Source hash covers 90% of cases. Explicit version is the escape hatch when source hash is fragile.

#### 4.3.2 Prompt Signatures

```python
def summarize_conversation(record: Record) -> str:
    return f"""
Summarize this conversation in 2-3 sentences.

Include:
- Main topic or question
- Key conclusions or decisions
- Any action items mentioned

## Conversation
{record.content}

## Summary
"""

def monthly_reflection(records: list[Record], period: str) -> str:
    summaries = "\n".join(f"- {r.content}" for r in records)
    return f"""
Reflect on this month's conversations.

Month: {period}
Conversations: {len(records)}

## Summaries
{summaries}

## Reflection
"""

def world_model_prompt(record: Record, state: str) -> str:
    return f"""
Update your understanding based on new information.

Current understanding:
{state}

New reflection ({record.metadata.get('period', 'unknown')}):
{record.content}

## Updated Understanding
"""
```

### 4.4 Execution

#### 4.4.1 Pipeline Sync

Before running, sync the pipeline definition to detect changes:

```python
pipeline.sync()
# Pipeline 'personal-memory' — changes detected:
#   CHANGED  summaries (prompt hash changed) → will reprocess 487 records
#   CHANGED  monthly (upstream changed) → will reprocess 28 records
#   CHANGED  world-model (upstream changed) → will reprocess 1 record
#
# Estimated cost: $1.68
# Run with: pipeline.run()
```

`sync()` serializes the current pipeline definition (step graph + step_version_hashes), compares against stored definition, and shows what changed. This separates "define" from "execute" — same as `terraform plan` vs `terraform apply`.

#### 4.4.2 Run Commands

```python
# Run full pipeline
pipeline.run()

# Run specific step (dependencies run automatically)
pipeline.run("summaries")

# Run from a step onwards
pipeline.run(from_="transcripts")

# Force full reprocess (ignores materialization keys)
pipeline.run(full=True)

# Dry run — show what would execute
plan = pipeline.plan()
print(plan.steps)      # ['chatgpt', 'claude', 'unified', ...]
print(plan.estimates)  # {'tokens': 1_200_000, 'cost': 1.50}
```

**Parallelism (v1.0+):** v0.1 processes transforms sequentially. Parallel execution with rate limiting is a natural optimization — transforms are embarrassingly parallel. Deferred to post-MVP to keep execution engine simple.

### 4.5 Branching

```python
# Create branch for experimentation
branch = pipeline.branch("better-summaries")

# Modify configuration on branch
branch.transform("summaries",
    from_="transcripts",
    prompt=improved_summarize,  # different prompt
    model="gpt-4o"              # different model
)

# Run on branch
branch.run("summaries", full=True)

# Compare with main
diff = branch.diff()
print(diff.changed_steps)  # ['summaries', 'monthly', 'world-model']

# Evaluate both
main_score = pipeline.eval("locomo")
branch_score = branch.eval("locomo")

# If better, promote
if branch_score > main_score:
    branch.promote()  # merge to main
```

**Promote semantics:**
- `branch.promote()` performs **upsert by materialization_key** into main
- Records and audit fields are copied/upserted
- Run history remains branch-scoped (not merged)
- If materialization_key exists on both branches with different content, promoted version wins
- Downstream steps on main whose inputs changed are marked stale → re-process on next `pipeline.run()`

### 4.6 Querying

```python
# Surface search — pick your altitude
results = pipeline.search("Rust", step="monthly")

# Unscoped search — highest-altitude dedup
results = pipeline.search("Rust")
# Searches all indexed steps. If monthly summary and source summary both match,
# returns monthly (higher altitude). Lower-altitude hits available via drill-down.

# Drill down — follow provenance
for hit in results:
    print(hit.content)
    print(hit.sources())        # direct parents only (one hop up)
    print(hit.leaves())         # deduplicated leaf records

# Leaf search — straight to bottom
raw = pipeline.search("FastAPI", step="transcripts", exact=True)

# Lineage — full provenance chain
record = pipeline.get("record-uuid")
lineage = record.lineage()      # tree of sources
```

**Drill-down API contracts:**

| Method | Returns | Parameters |
|--------|---------|------------|
| `sources()` | Direct parents only (one hop up the DAG) | None |
| `leaves()` | Deduplicated leaf records (records with empty `sources[]`) | `max_depth=10`, `max_count=100` |

"Leaf" = record with no sources = original source record. Breadth-first traversal with sensible limits to prevent explosion on ambiguous graphs.

### 4.7 The Hello World Experience

Five minutes from export to searchable memory:

```bash
# One command to initialize and import
synix init personal-memory --from ~/exports/chatgpt.json

# Process everything
synix run

# Search
synix search "that rust conversation"
```

Behind the scenes, `synix init --from` creates a sensible default pipeline.

### 4.8 Cost Estimation

For a pipeline processing 487 conversations through 5 steps, you're looking at real money. Users need to understand cost before they hit run.

```python
# Always plan before running
plan = pipeline.plan()

print(plan)
# Pipeline: personal-memory
# Branch: main
#
# Steps to execute:
#   chatgpt      →    487 records (source)
#   transcripts  →    487 records (transform, ~450 tok/rec)
#   summaries    →    487 records (transform, ~800 tok/rec)
#   monthly      →     28 records (aggregate, ~12k tok/rec)
#   world-model  →      1 record  (fold, ~8k tok/rec)
#
# Estimated tokens: 1,247,000 input / 94,000 output
# Estimated cost: $1.68 (gpt-4o-mini @ $0.15/1M in, $0.60/1M out)
#
# Run with: pipeline.run()

# Access programmatically
plan.total_input_tokens   # 1_247_000
plan.total_output_tokens  # 94_000
plan.estimated_cost       # 1.68
plan.cost_by_step         # {'transcripts': 0.42, 'summaries': 0.89, ...}
```

**Branch cost warning:**

```python
branch = pipeline.branch("experiment")
branch.transform("summaries", ..., model="gpt-4o")  # expensive model

plan = branch.plan(full=True)
# ⚠️  Warning: full reprocess on branch
# Estimated cost: $24.50 (gpt-4o @ $2.50/1M in, $10/1M out)
# This is 14x more expensive than main branch.
```

Cost estimation uses:
- Record counts from previous runs (or source file analysis for new pipelines)
- Average tokens per record by step type (calibrated from sample runs)
- Current model pricing (updated in config)

### 4.9 TOML as Export Format

TOML isn't the primary interface, but pipelines can serialize for sharing:

```python
# Export for documentation/version control
pipeline.to_toml("pipeline.toml")

# Import (creates Python equivalent)
pipeline = Pipeline.from_toml("pipeline.toml")
```

---

## Part V: CLI Design

### 5.1 Quick Start

```bash
# Fastest path: import + run + search
synix init my-memory --from ~/exports/chatgpt.json
synix run
synix search "python projects"
```

### 5.2 Full Command Reference

```bash
# ============================================
# PIPELINE LIFECYCLE
# ============================================

synix init <name>                    # Create new pipeline
synix init <name> --from <file>      # Create with source
synix status                         # Show pipeline status
synix plan                           # Dry-run, show estimates

# ============================================
# EXECUTION
# ============================================

synix run                            # Run full pipeline
synix run <step>                     # Run specific step
synix run --from <step>              # Run from step onwards
synix run --full                     # Ignore materialization, reprocess all

# ============================================
# BRANCHING
# ============================================

synix branch list
synix branch create <name>
synix branch switch <name>
synix branch diff <name>
synix branch promote <name>
synix branch delete <name>

# ============================================
# QUERY
# ============================================

synix search <query>                 # Search across artifacts
synix search <query> --step <step>   # Search specific step
synix lineage <record-id>            # Show provenance tree
synix get <record-id>                # Get single record

# ============================================
# EVALUATION
# ============================================

synix eval <benchmark>               # Run benchmark
synix eval locomo --compare mem0     # Compare against baseline
synix eval report                    # Generate report

# ============================================
# STATUS
# ============================================

synix stats                          # Record counts by step
synix runs                           # List recent runs
synix runs <run-id>                  # Run details
```

---

## Part VI: Artifact Semantics

### 6.1 Two Artifact Types

#### Projection Artifact

A single record (e.g., the latest world-model) formatted and pushed somewhere.

```python
pipeline.artifact("context",
    from_="world-model",
    surface="projection",
    on_update=webhook_url  # optional: trigger on change
)
```

Use cases:
- Agent system prompt injection
- Export to file
- Webhook notification

#### Search Artifact

A queryable index over one or more steps' records.

```python
pipeline.artifact("memory-index",
    from_=["summaries", "monthly"],
    surface="search"
)
```

**What gets indexed:**

| Field | Indexed | Queryable | Notes |
|-------|---------|-----------|-------|
| `content` | FTS + embedding | text search, semantic search | Primary search target |
| `step` | exact | filter | `step="monthly"` scopes to that level |
| `meta.time.created_at` | range | filter, sort | Temporal queries |
| `meta.time.period` | exact | filter | For aggregates: "2024-03" |
| `meta.chat.conversation_id` | exact | filter | Group by conversation |

The `step` field becomes a first-class facet — this is what enables searching build artifacts at different levels of the dependency graph. When you call `pipeline.search("Rust", step="monthly")`, the step filter restricts results to artifacts produced by the `monthly` build rule.

### 6.2 Searching the Dependency Graph

The search system is DAG-aware. This is like searching compiled artifacts vs. source files — same underlying information, different level of processing.

**Altitude = distance from sources in the dependency graph.** Monthly summaries are higher altitude than individual conversation summaries, which are higher altitude than raw transcripts.

Three query modes:

#### Search at a Specific Level

Search build artifacts at a chosen level of the dependency graph.

```python
# Search high-altitude artifacts (monthly summaries)
results = pipeline.search("Rust", step="monthly")

# Search low-altitude artifacts (individual summaries)
results = pipeline.search("Rust", step="summaries")
```

Like searching compiled documentation vs. source comments — different granularity, different context.

#### Walk the Dependency Graph

From any artifact, walk the dependency graph back to its sources. Every build system has this — "what source files went into this binary?"

```python
for hit in results:
    # This monthly summary mentions Rust — what produced it?
    sources = hit.sources()     # → conversation summaries (direct dependencies)
    leaves = hit.leaves()       # → raw transcripts (original sources)
```

The drill-down path is determined by the pipeline structure, not hardcoded.

#### Search Sources Directly

Skip the derived artifacts, search original sources.

```python
# Find every transcript containing literal "FastAPI"
raw = pipeline.search("FastAPI", step="transcripts", exact=True)
```

Full-text search at the bottom of the dependency graph — the original source files.

**This is novel.** Mem0 gives flat search. Zep gives temporal search. Nobody gives "search build artifacts at different levels of the dependency graph, then walk the graph back to sources." And it falls out naturally from treating memory as a build system.

---

## Part VII: Implementation Phases

### Two Milestones: v0.1 vs v1.0

The doc previously conflated two different milestones under "v1." These have different audiences and scope.

| Milestone | Audience | Success Criteria |
|-----------|----------|------------------|
| **v0.1** | You (internal validation) | Does the DAG work on your own data? Is altitude search useful? |
| **v1.0** | Other developers (public release) | Can someone else define a pipeline and get useful results? |

---

### v0.1: Internal Validation (Weeks 1-3)

**Goal:** Prove the architecture on your own data.

#### What ships in v0.1:

| Area | In | Out |
|------|-----|------|
| **Primitives** | transform, aggregate | fold, merge |
| **Incremental** | Materialization keys, content fingerprinting | Fold checkpoints |
| **Pipeline** | `run()`, `plan()` | `sync()`, prompt versioning decorator |
| **Search** | FTS with `step=` parameter | Unscoped search, drill-down API |
| **Branching** | None (main only) | All branching |
| **Security** | None | Everything |
| **CLI** | `run`, `plan`, `search` | `serve`, `export` |

#### v0.1 Phases:

**Phase 1a: Foundation (Week 1-2)**
- [ ] Project structure
  ```
  src/synix/
  ├── __init__.py
  ├── cli.py           # Click CLI
  ├── pipeline.py      # Pipeline class
  ├── steps.py         # Step implementations
  ├── models.py        # SQLAlchemy models
  ├── db.py            # Database engine
  └── prompts.py       # Prompt utilities
  ```
- [ ] Pipeline class with Python API
- [ ] Step types: **transform, aggregate**
- [ ] Source importer: Claude export
- [ ] SQLite storage with materialization keys
- [ ] Record creation with provenance
- [ ] Run tracking and stats
- [ ] Basic CLI: init, run, status

**Deliverable:** Can run `source → transform → aggregate` pipeline

**Phase 1b: Search + Validation (Week 3)**
- [ ] FTS5 search with `step=` parameter
- [ ] `pipeline.plan()` with cost estimates
- [ ] Manual inspection of outputs

**Deliverable:** v0.1 complete — architecture validated on your own data

#### What v0.1 proves:
1. The DAG works — source → transform → aggregate produces correct outputs
2. Incremental is real — re-run only processes new records
3. Altitude search is useful — "monthly" results differ from "summary" results
4. Cost model holds — predictions roughly match actuals

---

### v1.0: Public Release (Weeks 4-10)

**Goal:** Ship to other developers.

#### Additional v1.0 scope:

**Phase 2: Advanced Primitives (Week 4-5)**
- [ ] **Fold** step with checkpointing and state management
- [ ] **Merge** step with deduplication
- [ ] ChatGPT export importer
- [ ] Multiple source support

**Deliverable:** Full four-primitive pipeline works

**Phase 3: Eval Harness (Week 6-7)**
- [ ] Benchmark runner framework
- [ ] LoCoMo integration
- [ ] LongMemEval integration
- [ ] Memory freshness metric
- [ ] Comparison mode (vs Mem0, vs raw RAG)
- [ ] `synix eval` command
- [ ] Report generation

**Rationale:** If the pitch is "find the right architecture through trial and error," then eval is load-bearing.

**Deliverable:** `synix eval locomo --compare mem0` produces comparison table

**Goal:** Prove you can cheaply compare architectures.

- [ ] Branch model and storage
- [ ] Copy-on-write semantics
- [ ] Branch create/switch/delete
- [ ] Branch diff (compare records)
- [ ] Branch promote (merge to main via materialization keys)
- [ ] Eval on branches

**Rationale:** Branching enables the migration story — the sharpest differentiator.

**Deliverable:** Can branch, modify prompts, run, compare eval scores, promote

**Phase 5: Query, Explorer + Polish (Week 10)**

**Goal:** Complete the v1.0 experience.

- [ ] Unscoped search (highest-altitude dedup)
- [ ] Drill-down API (`hit.sources()`, `hit.leaves()`)
- [ ] Lineage visualization
- [ ] Pipeline Explorer (see Part XIII)
- [ ] `synix serve` for explorer UI
- [ ] Hello World experience (`init --from`)
- [ ] Error handling and progress bars
- [ ] Documentation

**Deliverable:** v1.0 complete — ready for other developers

---

## Part VIII: MVP Reference Implementation

### 8.1 Use Case: Personal Memory from ChatGPT History

**Scenario:** 2 years of ChatGPT conversations → searchable memory with provenance.

### 8.2 Pipeline Definition

```python
from synix import Pipeline

pipeline = Pipeline("personal-memory", agent="mark")

# Source
pipeline.source("chatgpt",
    file="~/exports/chatgpt.json",
    format="chatgpt-export"
)

# Processing
pipeline.transform("transcripts",
    from_="chatgpt",
    prompt=extract_transcript
)

pipeline.transform("summaries",
    from_="transcripts",
    prompt=summarize_conversation
)

pipeline.aggregate("monthly",
    from_="summaries",
    period="month",
    prompt=monthly_reflection
)

pipeline.fold("world-model",
    from_="monthly",
    prompt=world_model_prompt
)

# Artifacts
pipeline.artifact("context", from_="world-model", surface="projection")
pipeline.artifact("memory-index", from_=["summaries", "monthly"], surface="search")
```

### 8.3 Execution

```bash
$ synix run
Running pipeline 'personal-memory' on branch 'main'...

[1/5] chatgpt ████████████████████ 487/487 (parsed)
[2/5] transcripts ████████████████████ 487/487 (2m 15s)
[3/5] summaries ████████████████████ 487/487 (3m 42s)
[4/5] monthly ████████████████████ 28/28 (45s)
[5/5] world-model ████████████████████ 1/1 (12s)

✓ Completed in 7m 14s
  Records: 1,490 created
  Tokens: 1,247,832 input / 94,521 output
```

### 8.4 Experimentation

```python
# Try a different summarization approach
branch = pipeline.branch("gpt4-summaries")

branch.transform("summaries",
    from_="transcripts",
    prompt=detailed_summary,  # different prompt
    model="gpt-4o"            # different model
)

# Always plan before running — especially on branches
plan = branch.plan(full=True)
# ⚠️  Warning: full reprocess on branch
# Estimated cost: $24.50 (gpt-4o @ $2.50/1M in, $10/1M out)
# This is 14x more expensive than main branch (gpt-4o-mini).

branch.run(from_="summaries", full=True)

# Compare
main_eval = pipeline.eval("locomo")
branch_eval = branch.eval("locomo")

print(f"Main: {main_eval.precision_at_5:.2f}")
print(f"Branch: {branch_eval.precision_at_5:.2f}")

# If better, promote
if branch_eval.precision_at_5 > main_eval.precision_at_5:
    branch.promote()
```

---

## Part IX: Evaluation Strategy

### 9.1 Standard Benchmarks

#### LoCoMo (Long Context Memory)

- Multi-session conversation memory
- Retrieval accuracy, temporal reasoning, entity tracking
- Metrics: Precision@k, temporal accuracy

#### LongMemEval

- Long-horizon memory evaluation
- Information extraction, summarization quality
- Metrics: ROUGE, factual accuracy

### 9.2 Memory Freshness

Critical for the "evolving architecture" thesis: can the system distinguish yesterday vs. two years ago?

- Surface recent information appropriately
- Don't let old memories drown current context
- Metric: recency-weighted precision

### 9.3 Custom Benchmark: SynixBench

Existing benchmarks test flat retrieval. They have no concept of hierarchical search or provenance. **Our most differentiating capability has no benchmark.**

#### Why Not Extend Existing Benchmarks?

The obvious question: "Can't you just add provenance questions to LoCoMo?"

No — the test structure is fundamentally different.

LoCoMo asks: "Retrieve fact X from the memory store." The evaluation is: did you find it?

SynixBench asks: "Retrieve fact X **at altitude Y**, then verify it traces to source Z." The evaluation is three-part:
1. Did you find it at the requested abstraction level?
2. Is the answer appropriate for that level (summary vs. detail)?
3. Does the provenance chain correctly link to source records?

This is a different evaluation shape. LoCoMo treats memory as a flat key-value store. SynixBench treats memory as a DAG with meaningful structure. You can't bolt DAG-awareness onto a flat benchmark — you need questions designed around the hierarchy.

#### What SynixBench Tests

**Multi-altitude retrieval:** Can the system find the right answer at different abstraction levels? Are answers appropriately different at summary vs. raw transcript level?

**Provenance navigation:** Can the system trace hits to specific source records? Is lineage complete and correct?

**Leaf bypass:** Can the system find verbatim phrases in raw transcripts even when summaries abstracted them away?

**Emergent vs. source knowledge:** Some information only exists at higher levels — a monthly reflection might identify a trend ("shifting from web dev to systems programming") that doesn't appear in any single conversation. The benchmark tests whether the system distinguishes:
- Information at a specific altitude (emergent synthesis)
- Information that traces to a source record
- Information that appears differently at different altitudes

This is the killer differentiator. No existing benchmark touches this.

**Status:** Discovery phase. Define once pipeline and query model are working.

### 9.4 Comparison Targets

| System | Configuration |
|--------|---------------|
| **Mem0** | Same data, add() then search() |
| **Raw RAG** | Embed sources directly, no processing |
| **Synix** | Full pipeline with hierarchy |

### 9.5 Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Precision@5 | Top 5 retrieval accuracy | >0.80 |
| Temporal Accuracy | Correct time-based queries | >0.75 |
| Memory Freshness | Recency-appropriate surfacing | >0.70 |
| Provenance Completeness | Full lineage available | 100% |
| Incremental Speedup | Re-run / full run time | <0.1x |

---

## Part X: Open Design Questions

### 10.1 Search Strategy

**Decision: FTS-only for v0.1.**

- SQLite FTS5 for text search
- No similarity scores in v0.1 results
- API includes `mode="fts"` parameter so interface is stable when semantic is added
- Semantic search as pluggable addon in v1.0+ with clean boundary: optional sqlite-vss or external vector store, same query API, `mode="semantic"` or `mode="hybrid"`
- Eval numbers in v0.1 use FTS retrieval, clearly labeled

```python
# v0.1 — FTS only
results = pipeline.search("Rust", mode="fts")  # default

# v1.0+ — semantic available
results = pipeline.search("Rust", mode="semantic")
results = pipeline.search("Rust", mode="hybrid")
```

### 10.2 Error Model and Output Validation

What happens when an LLM call returns garbage?

**Failure modes:**
- Transform produces unparseable output (e.g., summary that's just "I don't know")
- Aggregate hallucinates facts not present in inputs
- Fold state update contradicts previous state
- Model returns refusal instead of content

The audit trail captures what happened, but there's no validation or recovery.

**Open questions:**

#### Validation
- Should transforms have optional validators? `pipeline.transform(..., validate=is_valid_summary)`
- What's the failure behavior — skip record, retry, halt pipeline?
- How do you validate "summarization quality" without another LLM call?

#### Retry Policy
- Exponential backoff for rate limits (obvious)
- Retry on malformed output? How many times?
- Different prompts on retry (e.g., add "You must provide a summary")?

#### Quality Gates
- Minimum content length?
- Required fields in structured output?
- Semantic similarity to input (anti-hallucination)?

**V1 stance:** Minimal validation. Retry on API errors (rate limit, timeout). Log malformed outputs but don't halt. Quality gates are a v2 feature.

The audit trail means you can always find bad outputs after the fact and reprocess. Perfect is the enemy of shipped.

---

## Part XI: Security and Privacy

### 11.1 The PII Problem

Bulk importing years of ChatGPT/Claude history means ingesting names, emails, medical conversations, financial details — everything a user ever discussed with an LLM.

Provenance makes this *worse*: sensitive data isn't just stored, it's linked and traceable through lineage.

### 11.2 Capability Scope by Version

| Capability | v0.1 | v1.0 | v1.5+ |
|-----------|------|------|-------|
| **Encryption at rest** | ❌ | Optional via SQLCipher (documented how-to) | ✓ |
| **Cascade purge** | ❌ | ✓ (provenance makes it straightforward) | ✓ |
| **PII detection** | ❌ | `pii="flag"` only — regex + optional Presidio | ✓ |
| **PII redaction** | ❌ | ❌ | Full redaction with LLM-based detection |
| **Field-level encryption** | ❌ | ❌ | ✓ |

**v0.1 stance:** No security features. You're running on your own data locally. This is fine and honest.

### 11.3 v1.0 Capabilities

#### PII Detection (Flag Mode)

First-class primitive, flag-only in v1.0.

```python
pipeline.transform("transcripts",
    from_="chatgpt",
    prompt=extract_transcript,
    pii="flag"  # v1.0: flag only. v1.5+: "redact" available
)
```

#### Encryption at Rest

SQLCipher for SQLite encryption — documented how-to, not built-in.

#### Selective Purge (GDPR)

Delete a source record and cascade through all derived records.

```python
# Delete and cascade
pipeline.purge("record-uuid", cascade=True)

# Preview what would be deleted
cascade = pipeline.purge("record-uuid", dry_run=True)
print(cascade.affected_records)  # all downstream records
```

### 11.4 Design Implications

Not v1-blocking, but data model must support:
- Cascade deletion via provenance links
- Encryption hooks at storage layer
- PII processor as built-in step type

---

## Part XII: Relationship to KP3

Synix is a more compelling formulation of what KP3 was building toward.

KP3 built: programmable knowledge processing, passages with provenance, semantic search.

Synix reframes it as: **declarative memory architecture with evolution as a first-class concern**.

The reframing elevates the concept from "a pipeline tool" to "infrastructure for a problem that doesn't have a settled answer yet."

Key insights carried forward:
- Passages → Records
- Refs/Branches → Branch model
- Processors → Step types
- Provenance tracking → Materialization keys + source links

---

## Part XIII: Pipeline Explorer

The strongest novelty is altitude search + provenance drill-down. This should be the first demo — it's visceral. Search monthly reflections, click into source summaries, click into raw transcripts. The hierarchy becomes real.

### 13.1 Explorer UI (v1.0)

Lightweight web explorer served by `synix serve`:

**What it shows:**
- Pipeline DAG visualization (steps as nodes, dependencies as edges)
- Record browser by step (pick altitude, see records)
- Search with step scoping (surface search)
- **Drill-down view:** Click a record → see sources → click source → see its sources → down to leaves

### 13.2 Three-Column Drill-Down

```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│  SEARCH RESULTS     │  SOURCES            │  DETAIL             │
│  (monthly)          │  (summaries)        │  (transcript)       │
├─────────────────────┼─────────────────────┼─────────────────────┤
│                     │                     │                     │
│ > March 2024        │ > Conv about Rust   │ User: I'm thinking  │
│   Rust exploration  │   ownership model   │ about learning      │
│   and systems...    │                     │ Rust...             │
│                     │ > Conv about Python │                     │
│   April 2024        │   vs Rust perf      │ Assistant: Great    │
│   Backend arch...   │                     │ choice! The...      │
│                     │ > Conv about...     │                     │
│   May 2024          │                     │                     │
│   ...               │                     │                     │
│                     │                     │                     │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

Left column = search results at current altitude. Middle = selected record's sources. Right = selected source's content. Click through to navigate.

### 13.3 Minimum Record Metadata for Explorer

Every record needs enough metadata to render usefully:

| Field | Required For |
|-------|--------------|
| `id` | Linking |
| `content` (or truncated preview) | Display |
| `step` | Altitude label |
| `created_at` | Sorting |
| `source_count` | "3 sources" indicator |
| `meta.time.period` | For aggregates |
| `meta.chat.conversation_id` | Grouping |

### 13.4 Implementation Note

Could be as simple as local FastAPI + htmx. The drill-down is the hard part visually — need to show hierarchy without overwhelming.

**Phasing:** v1.0 feature, but the query API it depends on (`sources()`, `leaves()`, record metadata) must be designed during v0.1.

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Pipeline** | Named container for memory processing |
| **Source** | Input that brings data into the system |
| **Step** | Processing operation (transform, aggregate, fold, merge) |
| **Artifact** | Queryable target (search index or projection) |
| **Record** | Single piece of content with provenance |
| **Run** | One execution of steps |
| **Branch** | Isolated variant for experimentation |
| **Materialization Key** | Unique identifier for incremental processing |
| **Provenance** | Chain from record back to sources |
| **DAG** | Directed Acyclic Graph of step dependencies |

## Appendix B: Why Not Just Use...

### "Why not just use Mem0?"

Use Mem0 if you need simple add/search. Use Synix if you need bulk import, custom processing, provenance, or architecture experimentation.

### "Why not just use a vector database?"

Vector DBs are storage, not processing. They don't summarize, aggregate, track provenance, or run pipelines. Synix uses vector DBs as a backend.

### "Why not just use dbt?"

dbt is for SQL transformations on structured data. Synix is for LLM transformations on unstructured text with provenance tracking.

## Appendix C: Design Decisions Log

### Python over TOML

**Decision:** Primary interface is Python code, not TOML config.

**Rationale:** Experimentation is a code activity. Config contradicts "you don't know the right architecture yet."

### Materialization Keys over Cursors

**Decision:** Replace cursor-as-last-id with materialization keys.

**Rationale:** Cursors break on late-arriving data and prompt changes. Materialization keys handle both automatically.

### Audit Determinism over Reproducibility

**Decision:** Store audit trail rather than guaranteeing reproducible builds.

**Rationale:** LLM outputs are non-deterministic. We can explain why a record exists, even if we can't reproduce it.

### No Batch Transforms in V1

**Decision:** Transform is strictly 1:1, no batch_size.

**Rationale:** Batching creates parsing problems. Keep it simple until structured output is robust.
