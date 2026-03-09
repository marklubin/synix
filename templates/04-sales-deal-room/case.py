"""sales_deal_room demo case — citation-backed competitive intelligence pipeline.

Flow:
  Phase A: Build + Release + Search
    1. Plan     — show the 4-level DAG
    2. Build    — 10 sources → 3 intel → 1 strategy → 1 call prep
    3. Release  — materialize search index to a release target
    4. Search   — query competitive intel for pricing

  Phase B: Invalidation cascade (new intel arrives)
    5. Copy staged intel update into sources
    6. Plan     — shows cascade: competitor_docs → intel → strategy → call_prep
    7. Build    — incremental cascade rebuild
    8. Release  — re-release with updated artifacts
    9. Search   — query new competitive data
   10. Explain  — cache fingerprint breakdown

This demonstrates:
  - Multi-source, multi-stage pipeline with citation-backed synthesis
  - Release workflow: build → release → query
  - Full invalidation cascade when new competitive intel arrives
  - Incremental rebuild (only affected artifacts rebuild)
"""

case = {
    "name": "sales_deal_room",
    "pipeline": "pipeline.py",
    "steps": [
        # Phase A: Build + Release + Search
        {"name": "note_plan", "command": ["synix", "demo", "note", "1/10 Planning build..."]},
        {"name": "plan", "command": ["synix", "plan", "PIPELINE"]},
        {
            "name": "note_build",
            "command": ["synix", "demo", "note", "2/10 Building competitive intelligence pipeline..."],
        },
        {"name": "build", "command": ["synix", "build", "PIPELINE"]},
        # Release — materialize projections to a named release target
        {
            "name": "note_release",
            "command": ["synix", "demo", "note", "3/10 Releasing: materializing search index..."],
        },
        {"name": "release", "command": ["synix", "release", "HEAD", "--to", "local"]},
        {"name": "note_search", "command": ["synix", "demo", "note", "4/10 Searching competitive intel..."]},
        {"name": "search", "command": ["synix", "search", "pricing", "--mode", "keyword", "--limit", "3"]},
        # Phase B: Invalidation cascade
        {"name": "note_new_intel", "command": ["synix", "demo", "note", "5/10 New intel arrives!"]},
        {"name": "copy_staged", "command": ["python3", "copy_staged.py"]},
        {
            "name": "note_plan_cascade",
            "command": ["synix", "demo", "note", "6/10 Planning rebuild..."],
        },
        {"name": "plan_cascade", "command": ["synix", "plan", "PIPELINE"]},
        {
            "name": "note_build_cascade",
            "command": ["synix", "demo", "note", "7/10 Rebuilding with new intel..."],
        },
        {"name": "build_cascade", "command": ["synix", "build", "PIPELINE"]},
        # Release after cascade rebuild
        {
            "name": "note_release_cascade",
            "command": ["synix", "demo", "note", "8/10 Re-releasing with updated artifacts..."],
        },
        {"name": "release_cascade", "command": ["synix", "release", "HEAD", "--to", "local"]},
        {
            "name": "note_search_cascade",
            "command": ["synix", "demo", "note", "9/10 Searching for new intel..."],
        },
        {
            "name": "search_cascade",
            "command": ["synix", "search", "price cut", "--mode", "keyword", "--limit", "3"],
        },
        {
            "name": "note_explain",
            "command": ["synix", "demo", "note", "10/10 Explaining cache decisions..."],
        },
        {"name": "explain", "command": ["synix", "plan", "PIPELINE", "--explain-cache"]},
    ],
    "goldens": {},
    # Files created by steps that should be cleaned up between runs
    "cleanup": [
        "sources/competitors/acme_q1_update.md",
    ],
}
