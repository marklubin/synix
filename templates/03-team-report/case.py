"""team_report demo case — ext transforms pipeline with multi-level DAG.

Flow:
  1. Plan     — show the DAG and what will be built
  2. Build    — parse bios + brief, map work styles, reduce team dynamics,
                fold final report
  3. Release  — materialize search index to a release target
  4. Search   — query across all layers via release
  5. Validate — check final report has input_count
  6. Rebuild  — second run, everything cached
  7. Explain  — show cache decisions inline

This demonstrates:
  - Two independent level-0 source layers (bios + project_brief)
  - MapSynthesis: 1:1 LLM transform (bio → work style profile)
  - ReduceSynthesis: N:1 rollup (work styles → team dynamics)
  - FoldSynthesis: sequential accumulation (team dynamics + brief → final report)
  - Release workflow: build → release → query
  - Full-text search across all 5 layers with provenance
  - Incremental rebuild (second run is instant)
  - Explain-cache with inline fingerprint breakdown
"""

case = {
    "name": "team_report",
    "pipeline": "pipeline.py",
    "steps": [
        # Step 1: Plan
        {"name": "note_plan", "command": ["synix", "demo", "note", "1/7 Planning build..."]},
        {"name": "plan", "command": ["synix", "plan", "PIPELINE"]},
        # Step 2: Build
        {
            "name": "note_build",
            "command": ["synix", "demo", "note", "2/7 Building: bios → work styles → team dynamics → final report..."],
        },
        {"name": "build", "command": ["synix", "build", "PIPELINE"]},
        # Step 3: Release — materialize projections to a named release target
        {
            "name": "note_release",
            "command": ["synix", "demo", "note", "3/7 Releasing: materializing search index..."],
        },
        {"name": "release", "command": ["synix", "release", "HEAD", "--to", "local"]},
        # Step 4: Search
        {"name": "note_search", "command": ["synix", "demo", "note", "4/7 Searching across all layers..."]},
        {"name": "search", "command": ["synix", "search", "climate dashboard", "--mode", "keyword", "--limit", "3"]},
        # Step 5: Validate
        {"name": "note_validate", "command": ["synix", "demo", "note", "5/7 Validating final report..."]},
        {"name": "validate", "command": ["synix", "validate", "PIPELINE", "--json"], "capture_json": True},
        # Step 6: Rebuild — everything cached
        {
            "name": "note_rebuild",
            "command": ["synix", "demo", "note", "6/7 Rebuilding (nothing changed → all cached)..."],
        },
        {"name": "rebuild", "command": ["synix", "build", "PIPELINE"]},
        # Step 7: Explain cache decisions
        {"name": "note_explain", "command": ["synix", "demo", "note", "7/7 Explaining cache decisions..."]},
        {"name": "explain", "command": ["synix", "plan", "PIPELINE", "--explain-cache"]},
    ],
    "goldens": {
        "validate": "validate.json",
    },
}
