"""tv_returns demo case — full build-validate-fix lifecycle.

Flow:
  1. Plan    — show what the pipeline will build (DAG + cost estimate)
  2. Build   — build CS product briefs from catalog + policies (source has PII)
  3. Search  — query the search index (keyword mode)
  4. Verify  — structural integrity check on the build
  5. Validate — find domain violations (2 semantic conflicts + 1 PII)
  6. Fix PII at source — edit vendor_offers.json, not the artifact
  7. Rebuild — partial rebuild (source changed, PII gone from fresh artifacts)
  8. Re-validate — PII gone, semantic conflicts remain
  9. Fix semantic conflicts — LLM auto-rewrite (artifact-level fix)
 10. Validate — should be clean (0 violations)

This demonstrates:
  - Pipeline planning and cost estimation
  - Search across built artifacts with provenance
  - Build integrity verification
  - Two different fix strategies (source correction vs LLM auto-fix)
  - Incremental rebuild (only affected artifacts rebuild when source changes)
  - Full violation lifecycle (detect → fix → verify → clean)
"""

case = {
    "name": "tv_returns",
    "pipeline": "pipeline.py",
    "steps": [
        # Step 1: Plan
        {"name": "note_plan", "command": ["synix", "demo", "note", "1/10 Planning build..."]},
        {"name": "plan", "command": ["synix", "plan", "PIPELINE"]},
        # Step 2: Build
        {
            "name": "note_build",
            "command": ["synix", "demo", "note", "2/10 Building product briefs from catalog + policies..."],
        },
        {"name": "build", "command": ["synix", "build", "PIPELINE"]},
        # Step 3: Search
        {"name": "note_search", "command": ["synix", "demo", "note", "3/10 Searching built artifacts..."]},
        {"name": "search", "command": ["synix", "search", "return policy", "--mode", "keyword", "--limit", "3"]},
        # Step 4: Verify
        {"name": "note_verify", "command": ["synix", "demo", "note", "4/10 Verifying build integrity..."]},
        {"name": "verify", "command": ["synix", "verify", "--build-dir", "build"]},
        # Step 5: Validate — finds semantic conflicts + PII
        {
            "name": "note_validate",
            "command": ["synix", "demo", "note", "5/10 Validating for contradictions and PII..."],
        },
        {"name": "validate_initial", "command": ["synix", "validate", "PIPELINE", "--json"], "capture_json": True},
        # Step 6: Fix PII at source — edit the data, not the artifact
        {"name": "note_pii", "command": ["synix", "demo", "note", "6/10 Fixing PII at the source..."]},
        {"name": "fix_source", "command": ["python3", "fix_source.py"]},
        # Step 7: Rebuild (source changed → partial rebuild, PII gone)
        {"name": "note_rebuild", "command": ["synix", "demo", "note", "7/10 Rebuilding (source data changed)..."]},
        {"name": "build_mid", "command": ["synix", "build", "PIPELINE"]},
        # Step 8: Re-validate — PII gone, semantic conflicts remain
        {"name": "note_revalidate", "command": ["synix", "demo", "note", "8/10 Re-validating (PII should be gone)..."]},
        {"name": "validate_mid", "command": ["synix", "validate", "PIPELINE", "--json"], "capture_json": True},
        # Step 9: Fix semantic conflicts (LLM-powered, artifact-level)
        {"name": "note_fix", "command": ["synix", "demo", "note", "9/10 Fixing semantic conflicts with LLM..."]},
        {"name": "fix", "command": ["synix", "fix", "PIPELINE"], "stdin": "a\na\na\n"},
        # Step 10: Final validate — should be clean
        {"name": "validate_final", "command": ["synix", "validate", "PIPELINE", "--json"], "capture_json": True},
    ],
    "goldens": {
        "validate_initial": "validate_initial.json",
        "validate_mid": "validate_mid.json",
        "validate_final": "validate_final.json",
    },
}
