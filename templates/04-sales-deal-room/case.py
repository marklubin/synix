"""sales_deal_room demo case — citation-backed competitive intelligence pipeline.

Flow:
  Phase A: Build + Validate + Fix (full citation lifecycle)
    1. Plan     — show the 4-level DAG
    2. Build    — 10 sources → 3 intel → 1 strategy → 1 call prep
    3. Search   — query competitive intel for pricing
    4. Validate — citation validator finds ungrounded claims
    5. Fix      — citation_enrichment fixer adds citations / removes claims
    6. Validate — should be clean (0 violations)

  Phase B: Invalidation cascade (new intel arrives)
    7. Copy staged intel update into sources
    8. Plan     — shows cascade: competitor_docs → intel → strategy → call_prep
    9. Build    — incremental cascade rebuild
   10. Search   — query new competitive data
   11. Validate — should pass on fresh LLM output
   12. Explain  — cache fingerprint breakdown

This demonstrates:
  - Multi-source, multi-stage pipeline with citation-backed synthesis
  - Citation validator catches ungrounded claims in synthesized artifacts
  - Citation fixer adds proper citations or removes unsupported content
  - Full invalidation cascade when new competitive intel arrives
  - Incremental rebuild (only affected artifacts rebuild)
"""

case = {
    "name": "sales_deal_room",
    "pipeline": "pipeline.py",
    "steps": [
        # Phase A: Build + Validate + Fix
        {"name": "note_plan", "command": ["synix", "demo", "note", "1/12 Planning build..."]},
        {"name": "plan", "command": ["synix", "plan", "PIPELINE"]},
        {
            "name": "note_build",
            "command": ["synix", "demo", "note", "2/12 Building competitive intelligence pipeline..."],
        },
        {"name": "build", "command": ["synix", "build", "PIPELINE"]},
        {"name": "note_search", "command": ["synix", "demo", "note", "3/12 Searching competitive intel..."]},
        {"name": "search", "command": ["synix", "search", "pricing", "--mode", "keyword", "--limit", "3"]},
        {"name": "note_validate", "command": ["synix", "demo", "note", "4/12 Validating citations..."]},
        {
            "name": "validate_initial",
            "command": ["synix", "validate", "PIPELINE", "--json"],
            "capture_json": True,
        },
        {"name": "note_fix", "command": ["synix", "demo", "note", "5/12 Fixing ungrounded claims..."]},
        {"name": "fix", "command": ["synix", "fix", "PIPELINE"], "stdin": "a\na\na\na\na\na\na\na\na\na\na\na\na\n"},
        {"name": "note_revalidate", "command": ["synix", "demo", "note", "6/12 Re-validating..."]},
        {
            "name": "validate_clean",
            "command": ["synix", "validate", "PIPELINE", "--json"],
            "capture_json": True,
        },
        # Phase B: Invalidation cascade
        {"name": "note_new_intel", "command": ["synix", "demo", "note", "7/12 New intel arrives!"]},
        {"name": "copy_staged", "command": ["python3", "copy_staged.py"]},
        {
            "name": "note_plan_cascade",
            "command": ["synix", "demo", "note", "8/12 Planning rebuild..."],
        },
        {"name": "plan_cascade", "command": ["synix", "plan", "PIPELINE"]},
        {
            "name": "note_build_cascade",
            "command": ["synix", "demo", "note", "9/12 Rebuilding with new intel..."],
        },
        {"name": "build_cascade", "command": ["synix", "build", "PIPELINE"]},
        {
            "name": "note_search_cascade",
            "command": ["synix", "demo", "note", "10/12 Searching for new intel..."],
        },
        {
            "name": "search_cascade",
            "command": ["synix", "search", "price cut", "--mode", "keyword", "--limit", "3"],
        },
        {
            "name": "note_validate_cascade",
            "command": ["synix", "demo", "note", "11/12 Validating rebuilt artifacts..."],
        },
        {
            "name": "validate_cascade",
            "command": ["synix", "validate", "PIPELINE", "--json"],
            "capture_json": True,
        },
        {
            "name": "note_explain",
            "command": ["synix", "demo", "note", "12/12 Explaining cache decisions..."],
        },
        {"name": "explain", "command": ["synix", "plan", "PIPELINE", "--explain-cache"]},
    ],
    "goldens": {
        "validate_initial": "validate_initial.json",
        "validate_clean": "validate_clean.json",
        "validate_cascade": "validate_cascade.json",
    },
}
