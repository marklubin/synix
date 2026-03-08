"""tv_returns demo case — full build-validate-fix lifecycle.

Flow:
  1. Plan    — show what the pipeline will build (DAG + cost estimate)
  2. Build   — build CS product briefs from catalog + policies (source has PII)
  3. Release — materialize search index to a release target
  4. Search  — query the search index (keyword mode)
  5. Verify  — structural integrity check on the build
  6. Validate — find domain violations (2 semantic conflicts + 1 PII)
  7. Fix PII at source — edit vendor_offers.json, not the artifact
  8. Rebuild — partial rebuild (source changed, PII gone from fresh artifacts)
  9. Re-validate — PII gone, semantic conflicts remain
 10. Fix semantic conflicts — LLM auto-rewrite (artifact-level fix)
 11. Validate — should be clean (0 violations)

This demonstrates:
  - Pipeline planning and cost estimation
  - Release workflow: build → release → query
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
        {"name": "note_plan", "command": ["synix", "demo", "note", "1/11 Planning build..."]},
        {"name": "plan", "command": ["synix", "plan", "PIPELINE"]},
        # Step 2: Build
        {
            "name": "note_build",
            "command": ["synix", "demo", "note", "2/11 Building product briefs from catalog + policies..."],
        },
        {"name": "build", "command": ["synix", "build", "PIPELINE"]},
        # Step 3: Release — materialize projections to a named release target
        {
            "name": "note_release",
            "command": ["synix", "demo", "note", "3/11 Releasing: materializing search index..."],
        },
        {"name": "release", "command": ["synix", "release", "HEAD", "--to", "local"]},
        # Step 4: Search
        {"name": "note_search", "command": ["synix", "demo", "note", "4/11 Searching built artifacts..."]},
        {"name": "search", "command": ["synix", "search", "return policy", "--mode", "keyword", "--limit", "3"]},
        # Step 5: Verify
        {"name": "note_verify", "command": ["synix", "demo", "note", "5/11 Verifying build integrity..."]},
        {"name": "verify", "command": ["synix", "verify", "--build-dir", "build"]},
        # Step 6: Validate — finds semantic conflicts + PII
        {
            "name": "note_validate",
            "command": ["synix", "demo", "note", "6/11 Validating for contradictions and PII..."],
        },
        {"name": "validate_initial", "command": ["synix", "validate", "PIPELINE", "--json"], "capture_json": True},
        # Step 7: Fix PII at source — edit the data, not the artifact
        {"name": "note_pii", "command": ["synix", "demo", "note", "7/11 Fixing PII at the source..."]},
        {"name": "fix_source", "command": ["python3", "fix_source.py"]},
        # Step 8: Rebuild (source changed → partial rebuild, PII gone)
        {"name": "note_rebuild", "command": ["synix", "demo", "note", "8/11 Rebuilding (source data changed)..."]},
        {"name": "build_mid", "command": ["synix", "build", "PIPELINE"]},
        # Step 9: Re-validate — PII gone, semantic conflicts remain
        {"name": "note_revalidate", "command": ["synix", "demo", "note", "9/11 Re-validating (PII should be gone)..."]},
        {"name": "validate_mid", "command": ["synix", "validate", "PIPELINE", "--json"], "capture_json": True},
        # Step 10: Fix semantic conflicts (LLM-powered, artifact-level)
        {"name": "note_fix", "command": ["synix", "demo", "note", "10/11 Fixing semantic conflicts with LLM..."]},
        {"name": "fix", "command": ["synix", "fix", "PIPELINE"], "stdin": "a\na\na\n"},
        # Step 11: Final validate — should be clean
        {"name": "validate_final", "command": ["synix", "validate", "PIPELINE", "--json"], "capture_json": True},
    ],
    "goldens": {
        "validate_initial": "validate_initial.json",
        "validate_mid": "validate_mid.json",
        "validate_final": "validate_final.json",
    },
    # Regex masks — lines matching any pattern are excluded from golden comparison.
    "output_masks": {
        # Trace artifact labels contain content hashes that change when prompts change
        "verify": [r"trace-"],
    },
}
