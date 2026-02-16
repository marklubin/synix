# Pipeline Definition (Python API)

Pipelines are defined in Python. Layers are real objects — `Source` for inputs, transform classes for LLM steps, `SearchIndex` and `FlatFile` for outputs. Dependencies are expressed as object references, not strings.

```python
# pipeline.py — Personal Memory Pipeline
from synix import Pipeline, Source, SearchIndex, FlatFile
from synix.transforms import EpisodeSummary, MonthlyRollup, CoreSynthesis

pipeline = Pipeline("personal-memory")

# Sources
pipeline.source_dir = "./sources"
pipeline.build_dir = "./build"

# LLM defaults
pipeline.llm_config = {
    "model": "claude-sonnet-4-20250514",
    "temperature": 0.3,
    "max_tokens": 1024,
}

# Layers — the memory hierarchy
transcripts = Source("transcripts")

episodes = EpisodeSummary("episodes", depends_on=[transcripts])

monthly = MonthlyRollup("monthly", depends_on=[episodes])

core = CoreSynthesis("core", depends_on=[monthly], context_budget=10000)

pipeline.add(transcripts, episodes, monthly, core)

# Projections — how artifacts become usable
pipeline.add(
    SearchIndex(
        "memory-index",
        sources=[episodes, monthly, core],
        search=["fulltext", "semantic"],
        embedding_config={
            "provider": "fastembed",
            "model": "BAAI/bge-small-en-v1.5",
        },
    )
)

pipeline.add(
    FlatFile(
        "context-doc",
        sources=[core],
        output_path="./build/context.md",
    )
)
```

## Demo Config Change: Monthly → Topical

Swap the rollup strategy — transcripts and episodes stay cached:

```python
# pipeline_topical.py — just change level 2
from synix.transforms import TopicalRollup

topics = TopicalRollup(
    "topics",
    depends_on=[episodes],
    config={"topics": ["career", "technical-projects", "san-francisco", "personal-growth", "ai-and-agents"]},
)

core = CoreSynthesis("core", depends_on=[topics], context_budget=10000)

pipeline.add(transcripts, episodes, topics, core)
```

## Dynamic Layer Generation

Because it's Python, you can do things like:

```python
from synix.transforms import TopicalRollup

for topic in ["work", "health", "projects", "relationships"]:
    pipeline.add(TopicalRollup(
        f"topic-{topic}",
        depends_on=[episodes],
        config={"topic": topic},
    ))
```

This is why code > config. You can't do that in YAML.

## Validators and Fixers

Validators and fixers are typed Python objects with explicit constructors:

```python
from synix.validators import PII, SemanticConflict, Citation, MutualExclusion
from synix.fixers import SemanticEnrichment, CitationEnrichment

pipeline.add_validator(PII(severity="warning"))
pipeline.add_validator(SemanticConflict())
pipeline.add_validator(Citation(layers=["strategy", "call_prep"]))
pipeline.add_validator(MutualExclusion(fields=["customer_id"]))

pipeline.add_fixer(SemanticEnrichment())
pipeline.add_fixer(CitationEnrichment())
```

## Custom Transforms

Extend `Transform` from `synix.core.models` to define custom transforms:

```python
from synix.core.models import Artifact, Transform

class CompetitiveIntel(Transform):
    prompt_name = "competitive_intel"  # loads prompts/competitive_intel.txt

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        # Your transform logic here
        ...

    def split(self, inputs: list[Artifact], config: dict) -> list[tuple[list[Artifact], dict]]:
        # Optional: decompose into parallel work units
        return [(inputs, config)]
```

Then use it in your pipeline like any built-in transform:

```python
intel = CompetitiveIntel("competitive_intel", depends_on=[competitor_docs, product_specs])
pipeline.add(intel)
```
