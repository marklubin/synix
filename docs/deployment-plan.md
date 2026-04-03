<!-- updated: 2026-04-03 -->
# Deployment & Verification Plan

End-to-end deployment of the Synix knowledge server infrastructure.

## Pre-requisites

| Item | Status |
|------|--------|
| synix 0.22.1 on PyPI | Done — released via GitHub Actions |
| Knowledge server module (`synix serve`) | Done — in synix core |
| Claude Code plugin (`plugin/`) | Done — in synix repo |
| MCP router memory backend | Done — deployed to oxnard |
| REST endpoints (search, buckets) | Done — in synix 0.22.1 |
| Pipeline config (`workspace.py`) | TODO — domain-specific |
| Server TOML config | TODO |

## Step 1: Pipeline Config (workspace.py)

Write the domain-specific pipeline — this runs on salinas. Needs:
- Three sources: sessions, documents, reports (matching bucket config)
- EpisodeSummary (1:1)
- FoldSynthesis layers: weekly, research, hypotheses, open-threads, core, work-status
- SearchIndex over all layers
- FlatFile projections: context-doc, weekly-brief

Prompts extracted from `synix-agent-mesh/src/synix_agent_mesh/pipeline.py`.

## Step 2: Deploy Synix Server on Salinas

```bash
ssh salinas

# --- Project setup ---
sudo mkdir -p /srv/synix/project/sources/{sessions,documents,reports}
sudo chown -R mark:mark /srv/synix

cd /srv/synix
uv venv .venv
uv pip install 'synix[server]==0.22.1'

# --- Init synix project ---
.venv/bin/synix init /srv/synix/project

# --- Copy configs ---
# workspace.py → /srv/synix/project/pipeline.py
# synix-server.toml → /srv/synix/project/synix-server.toml

# --- Data backfill ---
# A. Existing artifacts (3,554 objects)
cp -r ~/synix-agent-mesh/.synix/ /srv/synix/project/.synix/

# B. Session transcripts (591 files)
find ~/unified-memory/sessions/ \( -name "*.jsonl.gz" -o -name "*.jsonl" \) \
  -exec cp {} /srv/synix/project/sources/sessions/ \;

# C. agent-memory-disco corpus (359 .md files)
cd ~/agent-memory-disco
find desks/ intel/ log/ receipts/ synix/ docs/ pipeline/ \
  -name "*.md" -not -name "ARCHIVED.md" \
  -exec sh -c 'cp "$1" /srv/synix/project/sources/documents/$(echo "$1" | tr "/" "-")' _ {} \;

# --- First build ---
cd /srv/synix/project
ANTHROPIC_API_KEY=<key> /srv/synix/.venv/bin/synix build pipeline.py
/srv/synix/.venv/bin/synix release HEAD --to local

# --- systemd service ---
sudo tee /etc/systemd/system/synix-server.service << 'UNIT'
[Unit]
Description=Synix Knowledge Server
After=network.target

[Service]
Type=simple
User=mark
WorkingDirectory=/srv/synix/project
ExecStart=/srv/synix/.venv/bin/synix serve --config synix-server.toml
Environment=ANTHROPIC_API_KEY=<key>
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable synix-server
sudo systemctl start synix-server
```

## Step 3: Update Caddy on Salinas

```caddyfile
http://view.salinas:80 {
    bind 100.120.96.128
    reverse_proxy localhost:9471
}

http://mcp.salinas:80 {
    bind 100.120.96.128
    reverse_proxy localhost:8200
}
```

Remove `mesh.salinas` and `dash.salinas` routes.

## Step 4: Install Plugin on Obispo + Salinas

```bash
# On obispo (laptop)
ln -sf ~/synix/plugin ~/.claude/plugins/synix

# On salinas
ssh salinas 'cd ~/synix && git pull && ln -sf ~/synix/plugin ~/.claude/plugins/synix'

# Set server URL (both machines)
# fish:
set -Ux SYNIX_SERVER_URL http://salinas:8200
# bash:
echo 'export SYNIX_SERVER_URL="http://salinas:8200"' >> ~/.bashrc
```

## Step 5: Verification

Run in order — each step depends on the previous.

### 5.1 Server Health (from obispo)

```bash
curl -sf http://salinas:8200/api/v1/health
# ✓ {"status": "ok"}
```

### 5.2 Buckets

```bash
curl -sf http://salinas:8200/api/v1/buckets | python3 -m json.tool
# ✓ Shows documents, sessions, reports
```

### 5.3 Ingest via REST

```bash
curl -X POST http://salinas:8200/api/v1/ingest/documents \
  -H 'Content-Type: application/json' \
  -d '{"content":"Deployment verification test","filename":"verify-deploy.md"}'
# ✓ {"status": "ok", "bucket": "documents", ...}
```

### 5.4 Context Document

```bash
curl -sf http://salinas:8200/api/v1/flat-file/context-doc | head -20
# ✓ Returns synthesized core memory markdown
```

### 5.5 Search

```bash
curl -sf 'http://salinas:8200/api/v1/search?q=memory+architecture&limit=3' | python3 -m json.tool
# ✓ Returns results with labels, scores, content
```

### 5.6 Viewer

```bash
curl -sf http://salinas:9471/api/status | python3 -m json.tool
# ✓ Shows artifact count
# Or: open http://view.salinas in browser
```

### 5.7 MCP Router Bridge (from obispo, via oxnard)

```bash
# Verify oxnard can reach salinas
ssh oxnard 'curl -sf http://100.120.96.128:8200/api/v1/health'
# ✓ {"status": "ok"}
```

### 5.8 Claude Code Plugin — Session Start Hook

Start a new Claude Code session on obispo. The SessionStart hook should:
1. Fetch context-doc from salinas
2. Inject it as `additionalContext`

Verify by asking Claude "what do you know about the current project context?"
— it should have awareness from the injected context doc.

### 5.9 Claude Code Plugin — MCP Tools

In a Claude Code session:
```
> search memory for "distributed systems"
> what buckets are available?
> ingest this as a document: "Test from Claude Code verification"
```

Each should invoke the synix MCP tools and return results from salinas.

### 5.10 Claude Code Plugin — Session End Hook

End the Claude Code session. Then verify:

```bash
ssh salinas 'ls -lt /srv/synix/project/sources/sessions/ | head -5'
# ✓ New .jsonl file with recent timestamp
```

### 5.11 Auto-Builder

After the session push, wait for auto-builder to detect the new file
(scan_interval + cooldown). Monitor:

```bash
ssh salinas 'journalctl -u synix-server -f --no-pager' | grep Auto-build
# ✓ "bucket content changed, starting build"
# ✓ "complete (N built, M cached)"
```

### 5.12 Claude Desktop via Oxnard Bridge

Open Claude Desktop (uses CF tunnel → oxnard → mcp-router). Test:
```
> use memory_search to find "synix architecture"
> use memory_get_context
> use memory_list_buckets
```

Each should proxy through oxnard to salinas and return results.

### 5.13 End-to-End Cycle

After auto-build completes, search should find content from the verification
session:

```bash
curl -sf 'http://salinas:8200/api/v1/search?q=deployment+verification' | python3 -m json.tool
# ✓ Returns results from the just-built session episode
```

## Rollback

If anything fails:

```bash
# Stop server
ssh salinas 'sudo systemctl stop synix-server'

# Revert to synix-agent-mesh
ssh salinas 'cd ~/synix-agent-mesh && sam serve'

# Revert oxnard router (remove memory backend)
ssh oxnard 'cd ~/mcp-infrastructure && git stash && sudo systemctl restart mcp-router@$USER'
```

## Post-Deploy

- [ ] Redirect cron outputs to `sources/reports/` on salinas
- [ ] Retire synix-agent-mesh service
- [ ] Update `CONTEXT.md` in agent-memory-disco to note the new architecture
- [ ] Remove old `unified-memory` Synix MCP from any Claude Code configs
