# 01 — Chatbot Export Synthesis

Synthesize ChatGPT and Claude conversation exports into hierarchical memory with full provenance.

## What This Demonstrates

- Hierarchical memory synthesis: transcripts -> episodes -> rollups -> core memory
- Two rollup strategies: monthly (`pipeline_monthly.py`) vs topical (`pipeline_topical.py`)
- Incremental rebuild: swap pipelines and only upper layers rebuild (transcripts + episodes stay cached)
- Full provenance tracking through every layer

## Sample Data

Place your ChatGPT and Claude exports into `sources/`:

```
sources/
├── chatgpt_export.json    # ChatGPT "Export data" JSON
└── claude_export.json     # Claude export JSON
```

## Run

```bash
cd examples/01-chatbot-export-synthesis
cp .env.example .env       # add your API key

# Full build — transcripts -> episodes -> monthly rollups -> core memory
synix build pipeline_monthly.py

# Search your memory
synix search "what do I think about anthropic?"

# View the generated context document
cat build/context.md

# Swap to topic-based rollups (episodes cached, only rollups + core rebuild)
synix build pipeline_topical.py
```

## Pipelines

- **`pipeline_monthly.py`** — Groups episodes by calendar month for rollups
- **`pipeline_topical.py`** — Groups episodes by topic (career, technical projects, etc.) for rollups

Run monthly first, then topical to see incremental rebuild in action.

## Use Your Own Data

**ChatGPT:** Settings -> Data Controls -> Export data. Download the zip, extract `conversations.json`, and copy it into `sources/`.

**Claude:** Settings -> Export data. Download and copy the JSON file into `sources/`.

The parse transform auto-detects ChatGPT vs Claude export format.
