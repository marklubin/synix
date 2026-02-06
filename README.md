# Synix

**Memory architectures that evolve with your agent**

Synix is a memory workbench for building custom cognitive architectures. Define processing pipelines in Python, experiment with branches, measure with benchmarks, and evolve without losing data.

## The Problem

Current agent memory solutions are either too simple (Mem0: add/search) or too opinionated (Letta: fixed hierarchy). Nobody knows what the right memory architecture is for your domain — including you.

Synix lets you find out.

## Quick Start

```bash
# Import and process in one command
synix init personal-memory --from ~/exports/chatgpt.json
synix run
synix search "that rust conversation"
```

## Python API

```python
from synix import Pipeline

pipeline = Pipeline("personal-memory", agent="mark")

# Sources
pipeline.source("chatgpt", file="~/exports/chatgpt.json", format="chatgpt-export")

# Processing
pipeline.transform("summaries", from_="chatgpt", prompt=summarize)
pipeline.aggregate("monthly", from_="summaries", period="month", prompt=reflect)
pipeline.fold("world-model", from_="monthly", prompt=world_model)

# Outputs
pipeline.output("context", from_="world-model", surface="projection")
pipeline.output("search", from_=["summaries", "monthly"], surface="search")

# Run
pipeline.run()

# Experiment
branch = pipeline.branch("better-summaries")
branch.transform("summaries", from_="chatgpt", prompt=improved_summarize)
branch.run(full=True)

# Compare
if branch.eval("locomo") > pipeline.eval("locomo"):
    branch.promote()
```

## Key Features

- **Python-first** — Define pipelines in code, not config
- **Full provenance** — Trace any memory back to its source
- **Branching** — Experiment with different architectures in isolation
- **Incremental** — Only process new data on re-runs
- **Evolvable** — Change architecture without losing source data
- **Measurable** — Built-in benchmarks (LoCoMo, LongMemEval)

## Why Synix?

The killer feature: **architecture migration without data loss**.

If your current memory system isn't working, you normally have to start over. Synix lets you change the processing pipeline while keeping all source data. The data doesn't move — the lens changes.

## Documentation

See [DESIGN.md](./DESIGN.md) for the full design document.

## Status

🚧 **Early Development** — Design phase complete. Implementation in progress.

## License

MIT
