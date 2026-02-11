# Pipeline Definition (Python API)

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

## Demo Config Change: Monthly → Topical

```python
# pipeline_topical.py — just change level 2
pipeline.add_layer(Layer(
    name="topics",
    level=2,
    depends_on=["episodes"],
    transform="topical_rollup",
    grouping="by_topic",
    config={"topics": ["career", "technical-projects", "san-francisco", "personal-growth", "ai-and-agents"]},
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

## Dynamic Layer Generation

Because it's Python, you can do things like:
```python
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
