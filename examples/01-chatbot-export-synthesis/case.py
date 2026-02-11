"""chatbot_export_synthesis demo case — hierarchical memory from chat exports.

Flow:
  1. Plan    — show the DAG and what will be built
  2. Build   — parse exports, summarize episodes, roll up monthly, synthesize core
  3. Search  — query across all layers
  4. Rebuild — second run, everything cached

This demonstrates:
  - ChatGPT/Claude export parsing into transcripts (6 conversations)
  - Episode summarization (1:1 per conversation)
  - Monthly rollup aggregation (groups by calendar month)
  - Core memory synthesis
  - Hierarchical search with provenance
  - Incremental rebuild (second run is instant)
"""

case = {
    "name": "chatbot_export_synthesis",
    "pipeline": "pipeline_monthly.py",
    "steps": [
        # Step 1: Plan
        {"name": "note_plan", "command": ["synix", "demo", "note",
            "1/4 Planning build..."]},
        {"name": "plan", "command": ["synix", "plan", "PIPELINE"]},

        # Step 2: Build
        {"name": "note_build", "command": ["synix", "demo", "note",
            "2/4 Building: transcripts \u2192 episodes \u2192 monthly rollups \u2192 core memory..."]},
        {"name": "build", "command": ["synix", "build", "PIPELINE"]},

        # Step 3: Search
        {"name": "note_search", "command": ["synix", "demo", "note",
            "3/4 Searching across all layers..."]},
        {"name": "search", "command": [
            "synix", "search", "docker containers", "--mode", "keyword", "--limit", "3"]},

        # Step 4: Rebuild — everything cached
        {"name": "note_rebuild", "command": ["synix", "demo", "note",
            "4/4 Rebuilding (nothing changed \u2192 all cached)..."]},
        {"name": "rebuild", "command": ["synix", "build", "PIPELINE"]},
    ],
    "goldens": {},
}
