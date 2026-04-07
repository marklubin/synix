"""agent_driven demo case — agent-backed transforms pipeline.

Flow:
  1. Plan    — show DAG with agent-backed transforms
  2. Build   — bios + brief → work styles (analyst) → team dynamics (analyst)
               → final report (reporter)
  3. Release — materialize search index
  4. Search  — query across all layers
  5. Rebuild — second run, everything cached
  6. Explain — show cache decisions (includes agent fingerprint)

This demonstrates:
  - Same agent (analyst) reused across MapSynthesis and ReduceSynthesis
  - Different agent (reporter) for FoldSynthesis
  - Transform prompt (task structure) composes with agent instructions (persona)
  - Agent fingerprint in cache decisions
  - Provenance: agent_id + agent_fingerprint on artifacts
"""

case = {
    "name": "agent_driven",
    "pipeline": "pipeline.py",
    "steps": [
        # Step 1: Plan
        {"name": "note_plan", "command": ["synix", "demo", "note", "1/6 Planning agent-driven build..."]},
        {"name": "plan", "command": ["synix", "plan", "PIPELINE"]},
        # Step 2: Build
        {
            "name": "note_build",
            "command": ["synix", "demo", "note", "2/6 Building with analyst + reporter agents..."],
        },
        {"name": "build", "command": ["synix", "build", "PIPELINE"]},
        # Step 3: Release
        {
            "name": "note_release",
            "command": ["synix", "demo", "note", "3/6 Releasing: materializing search index..."],
        },
        {"name": "release", "command": ["synix", "release", "HEAD", "--to", "local"]},
        # Step 4: Search
        {"name": "note_search", "command": ["synix", "demo", "note", "4/6 Searching built artifacts..."]},
        {"name": "search", "command": ["synix", "search", "collaboration", "--mode", "keyword", "--limit", "3"]},
        # Step 5: Rebuild — everything cached
        {
            "name": "note_rebuild",
            "command": ["synix", "demo", "note", "5/6 Rebuilding (nothing changed → all cached)..."],
        },
        {"name": "rebuild", "command": ["synix", "build", "PIPELINE"]},
        # Step 6: Explain cache decisions
        {"name": "note_explain", "command": ["synix", "demo", "note", "6/6 Explaining cache decisions..."]},
        {"name": "explain", "command": ["synix", "plan", "PIPELINE", "--explain-cache"]},
    ],
    "goldens": {},
    # Agent-backed transforms hit the real LLM (no cassette support yet).
    # Content varies between runs, so mask LLM-dependent output lines.
    "output_masks": {
        "build": [r"Pipeline failed"],  # mask error details if API varies
        "rebuild": [r"Pipeline failed"],
        "release": [r"Released|artifacts"],  # snapshot OIDs vary
        "search": [r"^\s{2,}"],  # mask content lines in search results
    },
}
