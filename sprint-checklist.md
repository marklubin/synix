# Synix Sprint Reference Card
## Fri-Sun Feb 7-9, 2026

### Friday Night (NOW — 2 hrs)
- [ ] Fill in YC app text fields
- [ ] Record 1-min founder video (iPhone, one take)
- [ ] Submit YC app with GitHub repo link
- [ ] Set up Synix repo with CLAUDE.md + pyproject.toml
- [ ] `uv init` + `uv sync`
- [ ] Prepare test data: full ChatGPT + Claude exports in ./exports/

### Saturday (main build day — 8 hrs)
**Morning (Phase 1-2: Foundation + Parsers)**
- [ ] Artifact store + tests
- [ ] Provenance + tests
- [ ] ChatGPT parser + tests
- [ ] Claude parser + tests
- [ ] Transform base class + prompt template loading
- [ ] `uv run pytest` — all green before continuing

**Afternoon (Phase 3: Pipeline Engine)**
- [ ] Config parser + tests
- [ ] DAG resolver + tests
- [ ] Pipeline runner + tests (mock LLM)
- [ ] LLM transforms (episode, monthly, topical, core) + tests
- [ ] `uv run pytest` — all green before continuing

**Evening (Phase 4: Projections + CLI)**
- [ ] Search index projection + tests
- [ ] Flat file projection + tests
- [ ] CLI commands (run, search, lineage, status) + tests
- [ ] Rich formatting (progress bars, colored output, tree views)
- [ ] `uv run pytest` — all green before continuing

### Sunday (integration + demo — 6 hrs)
**Morning**
- [ ] Integration tests (pipeline run, incremental rebuild, config change)
- [ ] Run on full dataset (1000+ conversations)
- [ ] Fix bugs, re-run tests
- [ ] `uv run pytest` — ALL tests green

**Afternoon**
- [ ] Run test_demo_flow.py — the automated demo sequence
- [ ] Demo rehearsal — manual run-through for recording
- [ ] Screen record demo video
- [ ] Upload to YC app

### Monday (buffer — 3 hrs)
- [ ] Bug fixes if needed
- [ ] Re-record if needed
- [ ] Final submission before 8pm PT

---

### Budget
- LLM API: ~$15-30 (Sonnet for all transforms, full dataset)
- 1000+ convos × episode summary = ~1000 LLM calls (layer 1)
- ~12 monthly rollups or ~5 topical rollups (layer 2)
- 1 core synthesis (layer 3)
- Second run (config change): only layer 2+3 rebuild = ~6-10 calls

### Fallback Demo
If CLI polish isn't clean enough, record a Python REPL session:
```python
from synix.pipeline.runner import run
from synix.projections.search_index import SearchIndex

result = run("pipeline.py", "./exports")
# show: 50 built, 0 cached, projections materialized

index = SearchIndex("./build/search.db")
results = index.query("anthropic")
# show: results with provenance chains

# show the context doc projection
print(open("./build/context.md").read()[:500])
# show: agent system prompt output

# change config...
result2 = run("pipeline_topical.py", "./exports")
# show: 0 transcripts rebuilt (cached), 0 episodes rebuilt (cached),
#        8 topic rollups built, 1 core rebuilt, projections re-materialized
```

### Scope Fence — DO NOT BUILD
❌ Stateful artifacts / scratchpad
❌ Pipeline branching / A/B testing
❌ Postgres / Neo4j / any external DB
❌ Web UI / API server
❌ Eval harness / cost estimation
❌ Docker / deployment
❌ Anything not needed for the 90-second demo
