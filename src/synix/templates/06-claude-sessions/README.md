# Claude Sessions — Synix Mesh Template

Build agent memory from Claude Code sessions across multiple machines.

## What it does

1. **Watches** `~/.claude/projects` for `.jsonl` session files
2. **Submits** new sessions to the mesh server
3. **Builds** a 4-layer memory pipeline:
   - Sessions → Episode Summaries → Monthly Rollups → Core Synthesis
4. **Deploys** context documents back to all machines

## Quick start

```bash
# 1. Create the mesh
uvx synix mesh create --name claude-sessions --pipeline ./pipeline.py

# 2. Provision this machine as server
uvx synix mesh provision --name claude-sessions --role server

# 3. On other machines, provision as client
uvx synix mesh provision --name claude-sessions --role client --server YOUR_SERVER:7433

# 4. Check status
uvx synix mesh status --name claude-sessions
```

## Pipeline layers

| Layer | Type | Description |
|-------|------|-------------|
| sessions | Source | Raw Claude Code JSONL files |
| summaries | EpisodeSummary | Per-conversation summaries |
| rollups | MonthlyRollup | Monthly activity overviews |
| core | CoreSynthesis | Distilled identity + knowledge |
| context | FlatFile | Markdown context doc for Claude |

## Configuration

Edit `synix-mesh.toml` to customize:
- `[source].watch_dir` — where to look for session files
- `[server].build_*` — build scheduling thresholds
- `[cluster].leader_candidates` — machines eligible for leader role
- `[deploy.client].commands` — post-deploy hooks (e.g., symlink to `~/.claude/`)
