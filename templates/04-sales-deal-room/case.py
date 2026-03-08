"""sales_deal_room demo case — citation-backed competitive intelligence pipeline.

Flow:
  Phase A: Build + Release + Validate + Fix (full citation lifecycle)
    1. Plan     — show the 4-level DAG
    2. Build    — 10 sources → 3 intel → 1 strategy → 1 call prep
    3. Release  — materialize search index to a release target
    4. Search   — query competitive intel for pricing
    5. Validate — citation validator finds ungrounded claims
    6. Fix      — citation_enrichment fixer adds citations / removes claims
    7. Validate — should be clean (0 violations)

  Phase B: Invalidation cascade (new intel arrives)
    8. Copy staged intel update into sources
    9. Plan     — shows cascade: competitor_docs → intel → strategy → call_prep
   10. Build    — incremental cascade rebuild
   11. Search   — query new competitive data
   12. Validate — surfaces the newly introduced citation gaps after rebuild
   13. Explain  — cache fingerprint breakdown

This demonstrates:
  - Multi-source, multi-stage pipeline with citation-backed synthesis
  - Release workflow: build → release → query
  - Citation validator catches ungrounded claims in synthesized artifacts
  - Citation fixer adds proper citations or removes unsupported content
  - Full invalidation cascade when new competitive intel arrives
  - Revalidation can surface fresh citation gaps after new inputs land
  - Incremental rebuild (only affected artifacts rebuild)
"""

case = {
    "name": "sales_deal_room",
    "pipeline": "pipeline.py",
    "steps": [
        # Phase A: Build + Release + Validate + Fix
        {"name": "note_plan", "command": ["synix", "demo", "note", "1/13 Planning build..."]},
        {"name": "plan", "command": ["synix", "plan", "PIPELINE"]},
        {
            "name": "note_build",
            "command": ["synix", "demo", "note", "2/13 Building competitive intelligence pipeline..."],
        },
        {"name": "build", "command": ["synix", "build", "PIPELINE"]},
        # Release — materialize projections to a named release target
        {
            "name": "note_release",
            "command": ["synix", "demo", "note", "3/13 Releasing: materializing search index..."],
        },
        {"name": "release", "command": ["synix", "release", "HEAD", "--to", "local"]},
        {"name": "note_search", "command": ["synix", "demo", "note", "4/13 Searching competitive intel..."]},
        {"name": "search", "command": ["synix", "search", "pricing", "--mode", "keyword", "--limit", "3"]},
        {"name": "note_validate", "command": ["synix", "demo", "note", "5/13 Validating citations..."]},
        {
            "name": "validate_initial",
            "command": ["synix", "validate", "PIPELINE", "--json"],
            "capture_json": True,
        },
        {"name": "note_fix", "command": ["synix", "demo", "note", "6/13 Fixing ungrounded claims..."]},
        {"name": "fix", "command": ["synix", "fix", "PIPELINE"], "stdin": "a\na\n"},
        {"name": "note_revalidate", "command": ["synix", "demo", "note", "7/13 Re-validating..."]},
        {
            "name": "validate_clean",
            "command": ["synix", "validate", "PIPELINE", "--json"],
            "capture_json": True,
        },
        # Phase B: Invalidation cascade
        {"name": "note_new_intel", "command": ["synix", "demo", "note", "8/13 New intel arrives!"]},
        {"name": "copy_staged", "command": ["python3", "copy_staged.py"]},
        {
            "name": "note_plan_cascade",
            "command": ["synix", "demo", "note", "9/13 Planning rebuild..."],
        },
        {"name": "plan_cascade", "command": ["synix", "plan", "PIPELINE"]},
        {
            "name": "note_build_cascade",
            "command": ["synix", "demo", "note", "10/13 Rebuilding with new intel..."],
        },
        {"name": "build_cascade", "command": ["synix", "build", "PIPELINE"]},
        {
            "name": "note_search_cascade",
            "command": ["synix", "demo", "note", "11/13 Searching for new intel..."],
        },
        {
            "name": "search_cascade",
            "command": ["synix", "search", "price cut", "--mode", "keyword", "--limit", "3"],
        },
        {
            "name": "note_validate_cascade",
            "command": ["synix", "demo", "note", "12/13 Validating rebuilt artifacts..."],
        },
        {
            "name": "validate_cascade",
            "command": ["synix", "validate", "PIPELINE", "--json"],
            "capture_json": True,
        },
        {
            "name": "note_explain",
            "command": ["synix", "demo", "note", "13/13 Explaining cache decisions..."],
        },
        {"name": "explain", "command": ["synix", "plan", "PIPELINE", "--explain-cache"]},
    ],
    "goldens": {
        "validate_initial": "validate_initial.json",
        "validate_clean": "validate_clean.json",
        "validate_cascade": "validate_cascade.json",
    },
    # Regex masks — lines matching any pattern are excluded from golden comparison.
    # Used for non-deterministic output that can't be normalized to stable placeholders.
    "output_masks": {
        # Fix output is highly LLM-dependent (violation count, proposed rewrites, etc.)
        "fix": [r"^.*$"],  # skip entire output — content depends on LLM non-determinism
        # Validate stdout varies in violation details
        "validate_initial": [r"active violation"],
        "validate_clean": [r"active violation"],
    },
    # Files created by steps that should be cleaned up between runs
    "cleanup": [
        "sources/competitors/acme_q1_update.md",
    ],
}
