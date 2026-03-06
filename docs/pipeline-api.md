# Pipeline API

Pipelines are defined in Python. Layers are real objects — `Source` for inputs, transform classes for LLM steps, `SearchSurface` for build-time retrieval, and `SynixSearch` / `FlatFile` for outputs. Dependencies are expressed as object references, not strings.

```python
# pipeline.py
from synix import FlatFile, Pipeline, SearchSurface, Source, SynixSearch
from synix.ext import EpisodeSummary, MonthlyRollup, CoreSynthesis

pipeline = Pipeline("personal-memory")
pipeline.source_dir = "./sources"
pipeline.build_dir = "./build"
pipeline.llm_config = {
    "model": "claude-sonnet-4-20250514",
    "temperature": 0.3,
    "max_tokens": 1024,
}

transcripts = Source("transcripts")
episodes = EpisodeSummary("episodes", depends_on=[transcripts])
monthly = MonthlyRollup("monthly", depends_on=[episodes])
core = CoreSynthesis("core", depends_on=[monthly], context_budget=10000)

memory_search = SearchSurface(
    "memory-search",
    sources=[episodes, monthly, core],
    modes=["fulltext", "semantic"],
    embedding_config={"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5"},
)

pipeline.add(transcripts, episodes, monthly, core, memory_search)

pipeline.add(
    SynixSearch("search", surface=memory_search)
)
pipeline.add(
    FlatFile("context-doc", sources=[core], output_path="./build/context.md")
)
```

## Generic Transforms (`synix.transforms`)

Most LLM pipeline steps follow one of four generic patterns. The `synix.transforms` module provides those platform transforms directly — just pass a prompt string and parameters instead of writing a custom class.

```python
from synix.transforms import MapSynthesis, GroupSynthesis, ReduceSynthesis, FoldSynthesis
```

All generic transforms share these behaviors:
- **Prompt templates** with placeholders — changing the prompt automatically invalidates the cache
- **Deterministic ordering** — multi-input transforms sort by `artifact_id` before building prompts
- **LLM calls** via the standard pipeline `llm_config`
- **Callable fingerprinting** — if you pass a callable for `label_fn`, `group_by`, or `sort_by`, changes to the function source code invalidate the cache

### MapSynthesis (1:1)

Apply a prompt to each input artifact independently. Each input produces one output.

```python
work_styles = MapSynthesis(
    "work_styles",
    depends_on=[bios],
    prompt="Infer this person's work style in 2-3 sentences:\n\n{artifact}",
    artifact_type="work_style",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | *required* | Template with `{artifact}`, `{label}`, `{artifact_type}` placeholders |
| `label_fn` | `Callable[[Artifact], str] \| None` | `None` | Custom label derivation. Default: `{name}-{input_label}` |
| `metadata_fn` | `Callable[[Artifact], dict] \| None` | `None` | Custom metadata derivation from the input artifact. Merged on top of auto-propagated input metadata |
| `artifact_type` | `str` | `"summary"` | Output artifact type |

**Metadata propagation:** MapSynthesis automatically copies input artifact metadata to the output (1:1 natural inheritance). The `metadata_fn` result and `source_label` are merged on top.

**Custom labels:**

```python
# t-text-alice → ws-alice
work_styles = MapSynthesis(
    "work_styles",
    depends_on=[bios],
    prompt="...",
    label_fn=lambda a: f"ws-{a.label.replace('t-text-', '')}",
    artifact_type="work_style",
)
```

### GroupSynthesis (N:M)

Group inputs by a metadata key (or callable), produce one output per group.

```python
by_customer = GroupSynthesis(
    "customer-summaries",
    depends_on=[episodes],
    group_by="customer_id",
    prompt="Summarize interactions for customer '{group_key}':\n\n{artifacts}",
    artifact_type="customer_summary",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `group_by` | `str \| Callable[[Artifact], str]` | *required* | Metadata key or callable that returns the group key |
| `prompt` | `str` | *required* | Template with `{group_key}`, `{artifacts}`, `{count}`, `{artifact_type}` placeholders |
| `label_prefix` | `str \| None` | `None` | Prefix for output labels. Default: derived from `group_by` key |
| `metadata_fn` | `Callable[[str, list[Artifact]], dict] \| None` | `None` | Custom metadata from `(group_key, inputs)`. Merged on top of default `group_key` and `input_count` |
| `artifact_type` | `str` | `"summary"` | Output artifact type |
| `on_missing` | `str` | `"group"` | Behavior when metadata key is absent (see below) |
| `missing_key` | `str` | `"_ungrouped"` | Group name for missing-key artifacts when `on_missing="group"` |

**`on_missing` behavior:**

| Value | What happens |
|-------|-------------|
| `"group"` | Collect under `missing_key`, print warning |
| `"skip"` | Drop artifacts without the key, print warning |
| `"error"` | Raise `ValueError` immediately |

**Grouping by callable:**

```python
by_quarter = GroupSynthesis(
    "quarterly",
    depends_on=[episodes],
    group_by=lambda a: f"Q{(int(a.metadata.get('month', 1)) - 1) // 3 + 1}",
    prompt="Summarize Q{group_key} activity:\n\n{artifacts}",
)
```

### ReduceSynthesis (N:1)

Combine all inputs into a single output. One LLM call.

```python
team_dynamics = ReduceSynthesis(
    "team_dynamics",
    depends_on=[work_styles],
    prompt="Analyze team dynamics from these profiles:\n\n{artifacts}",
    label="team-dynamics",
    artifact_type="team_dynamics",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | *required* | Template with `{artifacts}`, `{count}` placeholders |
| `label` | `str` | *required* | Fixed output label |
| `metadata_fn` | `Callable[[list[Artifact]], dict] \| None` | `None` | Custom metadata from the input list. Merged on top of default `input_count` |
| `artifact_type` | `str` | `"summary"` | Output artifact type |

### FoldSynthesis (N:1 sequential)

Process inputs one at a time, building up an accumulated result through sequential LLM calls. Use this when the input set is too large for a single context window, or when order matters.

```python
progressive = FoldSynthesis(
    "progressive-summary",
    depends_on=[episodes],
    prompt=(
        "Update this running summary with the new information.\n\n"
        "Current summary:\n{accumulated}\n\n"
        "New input:\n{artifact}"
    ),
    initial="No information yet.",
    sort_by="date",
    label="progressive-summary",
    artifact_type="progressive_summary",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | *required* | Template with `{accumulated}`, `{artifact}`, `{label}`, `{step}`, `{total}` placeholders |
| `initial` | `str` | `""` | Starting value for the accumulator |
| `sort_by` | `str \| Callable \| None` | `None` | Metadata key, callable, or `None` (sort by `artifact_id`) |
| `label` | `str` | *required* | Fixed output label |
| `metadata_fn` | `Callable[[list[Artifact]], dict] \| None` | `None` | Custom metadata from the input list. Merged on top of default `input_count` |
| `artifact_type` | `str` | `"summary"` | Output artifact type |

FoldSynthesis always runs synchronously (never batched) because each step depends on the previous.

### Choosing a Transform

| If your step... | Use |
|-----------------|-----|
| Processes each input independently | `MapSynthesis` |
| Groups inputs by a key, one output per group | `GroupSynthesis` |
| Combines all inputs into one output | `ReduceSynthesis` |
| Needs to accumulate through inputs in order | `FoldSynthesis` |

## Bundled Ext Transforms (`synix.ext`)

Synix also ships a small set of opinionated memory-oriented transforms in `synix.ext`:

| Class | Pattern | Description |
|-------|---------|-------------|
| `EpisodeSummary` | 1:1 | 1 transcript → 1 episode summary |
| `MonthlyRollup` | N:M | Group episodes by calendar month, synthesize each |
| `TopicalRollup` | N:M | Group episodes by user-declared topics. Requires `config={"topics": [...]}` |
| `CoreSynthesis` | N:1 | All rollups → single core memory document. Respects `context_budget` |

These are bundled convenience transforms rather than the generic platform primitives.

## Other Platform Transforms

Import from `synix.transforms`:

| Class | Pattern | Description |
|-------|---------|-------------|
| `Merge` | N:M | Group artifacts by content similarity (Jaccard), merge above threshold |

## Sources

Drop files into `source_dir` — the parser auto-detects format by file structure.

| Format | Extensions | Notes |
|--------|-----------|-------|
| **ChatGPT** | `.json` | `conversations.json` exports. Handles regeneration branches via `current_node` |
| **Claude** | `.json` | Claude conversation exports with `chat_messages` arrays |
| **Claude Code** | `.jsonl` | Claude Code session transcripts. Extracts user/assistant turns, skips tool blocks |
| **Text / Markdown** | `.txt`, `.md` | YAML frontmatter support. Auto-detects conversation turns (`User:` / `Assistant:` prefixes) |

## Projections

`SearchSurface` is the build-time searchable capability declaration. `SynixSearch` is the canonical local search output. `SearchIndex` remains compatibility sugar for older direct-layer pipelines.

Import from `synix`:

| Class | Output | Description |
|-------|--------|-------------|
| `SearchSurface` | local compatibility realization under `build/surfaces/` today | Named build-time search surface over selected layers. Use with `uses=[surface]` on transforms that need retrieval during the build. Treat the on-disk path as internal; the supported interface is `ctx.search(...)` |
| `SynixSearch` | `build/search.db` by default | Default Synix search output over a declared `SearchSurface`. Keeps the current local SQLite-backed experience as a compatibility target without making the file/schema the public contract |
| `SearchIndex` | `build/search.db` | Legacy compatibility projection over direct source layers. This does not satisfy `uses=[...]`; prefer `SearchSurface + SynixSearch` for new pipelines |
| `FlatFile` | `build/context.md` | Renders artifacts as markdown. Ready to paste into an LLM system prompt |

## Config Change Demo

Swap the rollup strategy — transcripts and episodes stay cached:

```python
from synix import SearchSurface
from synix.ext import TopicalRollup

episode_search = SearchSurface(
    "episode-search",
    sources=[episodes],
    modes=["fulltext"],
)
topics = TopicalRollup(
    "topics",
    depends_on=[episodes],
    uses=[episode_search],
    config={"topics": ["career", "technical-projects", "san-francisco"]},
)
core = CoreSynthesis("core", depends_on=[topics], context_budget=10000)
pipeline.add(transcripts, episodes, episode_search, topics, core)
```

## Dynamic Layer Generation

Because it's Python, you can generate layers programmatically:

```python
for topic in ["work", "health", "projects", "relationships"]:
    pipeline.add(TopicalRollup(
        f"topic-{topic}", depends_on=[episodes],
        config={"topic": topic},
    ))
```

## Custom Transforms

When you need logic beyond prompt templating — filtering inputs, conditional branching, multi-step LLM chains — extend `Transform` directly:

```python
from synix import SearchSurface, Transform, TransformContext
from synix.build.llm_transforms import _get_llm_client, _logged_complete
from synix.core.models import Artifact

episode_search = SearchSurface("episode-search", sources=[episodes], modes=["fulltext"])

class CompetitiveIntel(Transform):
    def execute(self, inputs: list[Artifact], ctx: TransformContext) -> list[Artifact]:
        ctx = self.get_context(ctx)
        client = _get_llm_client(ctx)
        search = ctx.search("episode-search")
        related = search.query("pricing", layers=["episodes"], limit=5) if search is not None else []
        # Your custom logic here — filter, branch, chain multiple LLM calls
        response = _logged_complete(
            client, ctx,
            messages=[{
                "role": "user",
                "content": (
                    f"Analyze:\n{inputs[0].content}\n\n"
                    f"Related episode hits: {[hit.label for hit in related]}"
                ),
            }],
            artifact_desc="competitive-intel",
        )
        return [Artifact(
            label="intel-report",
            artifact_type="competitive_intel",
            content=response.content,
            input_ids=[a.artifact_id for a in inputs],
            prompt_id="competitive_intel_v1",
            model_config=ctx.llm_config,
        )]

    def split(self, inputs: list[Artifact], ctx: TransformContext):
        # Optional: decompose into parallel work units (default is 1:1)
        return [(inputs, {})]
```

Use it like any transform:

```python
intel = CompetitiveIntel(
    "competitive_intel",
    depends_on=[competitor_docs, product_specs],
    uses=[episode_search],
)
pipeline.add(intel)
```

Use `ctx.search(...)` or `self.get_search_surface(ctx, required=True)` for build-time retrieval. `TransformContext` stays mapping-compatible during migration so legacy custom transforms keep running, but the public interface is the typed context and search handle rather than `search_db_path` or a hard-coded SQLite path.

## Validators and Fixers (Experimental)

> **Note:** The validate/fix workflow is experimental. APIs and output formats may change.

```python
from synix.validators import PII, SemanticConflict, Citation, MutualExclusion, RequiredField
from synix.fixers import SemanticEnrichment, CitationEnrichment

pipeline.add_validator(PII(severity="warning"))
pipeline.add_validator(SemanticConflict())
pipeline.add_validator(Citation(layers=["strategy", "call_prep"]))
pipeline.add_validator(MutualExclusion(fields=["customer_id"]))
pipeline.add_validator(RequiredField(layers=[report], field="input_count"))

pipeline.add_fixer(SemanticEnrichment())
pipeline.add_fixer(CitationEnrichment())
```

| Validator | What it checks |
|-----------|---------------|
| `MutualExclusion` | Merged artifacts don't mix values of a metadata field |
| `RequiredField` | Artifacts in specified layers have a required metadata field |
| `PII` | Detects credit cards, SSNs, emails, phone numbers |
| `SemanticConflict` | LLM-based detection of contradictions across artifacts |
| `Citation` | Verifies artifacts cite their source artifacts with valid URIs |

| Fixer | What it fixes |
|-------|--------------|
| `SemanticEnrichment` | Resolves semantic conflicts by rewriting with source context |
| `CitationEnrichment` | Adds missing citation references |
