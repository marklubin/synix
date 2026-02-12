"""team_report demo case — init-style pipeline with multi-level DAG.

Flow:
  1. Plan     — show the DAG and what will be built
  2. Build    — parse bios + brief, infer work styles, roll up team dynamics,
                synthesize final report
  3. Search   — query across all layers
  4. Validate — check final report length
  5. Rebuild  — second run, everything cached

This demonstrates:
  - Two independent level-0 source layers (bios + project_brief)
  - 1:1 LLM transform (bio → work style profile)
  - Many:1 rollup (work styles → team dynamics)
  - Multi-input synthesis (team dynamics + brief → final report)
  - Full-text search across all 5 layers with provenance
  - Incremental rebuild (second run is instant)
"""

case = {
    "name": "team_report",
    "pipeline": "pipeline.py",
    "steps": [
        # Step 1: Plan
        {"name": "note_plan", "command": ["synix", "demo", "note", "1/5 Planning build..."]},
        {"name": "plan", "command": ["synix", "plan", "PIPELINE"]},
        # Step 2: Build
        {
            "name": "note_build",
            "command": ["synix", "demo", "note", "2/5 Building: bios → work styles → team dynamics → final report..."],
        },
        {"name": "build", "command": ["synix", "build", "PIPELINE"]},
        # Step 3: Search
        {"name": "note_search", "command": ["synix", "demo", "note", "3/5 Searching across all layers..."]},
        {"name": "search", "command": ["synix", "search", "climate dashboard", "--mode", "keyword", "--limit", "3"]},
        # Step 4: Validate
        {"name": "note_validate", "command": ["synix", "demo", "note", "4/5 Validating final report length..."]},
        {"name": "validate", "command": ["synix", "validate", "PIPELINE", "--json"], "capture_json": True},
        # Step 5: Rebuild — everything cached
        {
            "name": "note_rebuild",
            "command": ["synix", "demo", "note", "5/5 Rebuilding (nothing changed → all cached)..."],
        },
        {"name": "rebuild", "command": ["synix", "build", "PIPELINE"]},
    ],
    "goldens": {
        "validate": "validate.json",
    },
}
