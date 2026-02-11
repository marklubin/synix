# 01 — Chatbot Export Synthesis

Synthesize ChatGPT and Claude conversation exports into hierarchical memory with full provenance.

## Setup

Drop your exports into `./fixtures/`:

```
fixtures/
├── chatgpt_export.json    # ChatGPT "Export data" JSON
└── claude_export.json     # Claude export JSON
```

## Run

```bash
cd examples/01-chatbot-export-synthesis

# Full build — transcripts → episodes → monthly rollups → core memory
synix build pipeline_monthly.py

# Search your memory
synix search "what do I think about anthropic?"

# View the generated context document
cat build/context.md

# Swap to topic-based rollups (episodes cached, only rollups + core rebuild)
synix build pipeline_topical.py

# Search again — different results, same data
synix search "what do I think about anthropic?"
```

## Pipelines

- **`pipeline_monthly.py`** — Groups episodes by calendar month for rollups
- **`pipeline_topical.py`** — Groups episodes by topic (career, technical projects, etc.) for rollups

Run monthly first, then topical to see incremental rebuild in action.
