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
uvx synix build pipeline_monthly.py

# Search your memory
uvx synix search "what do I think about anthropic?"

# View the generated context document
cat build/context.md

# Swap to topic-based rollups (episodes cached, only rollups + core rebuild)
uvx synix build pipeline_topical.py
```

## Pipelines

- **`pipeline_monthly.py`** — Groups episodes by calendar month for rollups
- **`pipeline_topical.py`** — Groups episodes by topic (career, technical projects, etc.) for rollups

Run monthly first, then topical to see incremental rebuild in action.

## Use Your Own Data

### ChatGPT

1. Go to [chatgpt.com/settings](https://chatgpt.com/#settings/DataControls) → **Data Controls** → **Export data**
2. Click **Export** — you'll receive an email within a few minutes
3. Download the zip from the email link, then:

```bash
unzip chatgpt-export-*.zip -d /tmp/chatgpt-export
cp /tmp/chatgpt-export/conversations.json sources/
```

### Claude

1. Go to [claude.ai/settings](https://claude.ai/settings) → **Export data**
2. Click **Export** — you'll receive an email within a few minutes
3. Download the zip from the email link, then:

```bash
unzip Claude-export-*.zip -d /tmp/claude-export
cp /tmp/claude-export/*.json sources/
```

### Build

```bash
uvx synix build pipeline_monthly.py
uvx synix search "your query here"
```

The parse transform auto-detects ChatGPT vs Claude export format.
