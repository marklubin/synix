"""tv_returns demo case — ecommerce CS knowledge pipeline.

Flow:
  1. Plan    — show what the pipeline will build (DAG + cost estimate)
  2. Build   — build CS product briefs from catalog + policies
  3. Release — materialize search index to a release target
  4. Search  — query the search index (keyword mode)
  5. Rebuild — second run, everything cached
  6. Explain — show cache decisions inline

This demonstrates:
  - Pipeline planning and cost estimation
  - Custom transforms (structured join + LLM extraction + LLM enrichment)
  - Release workflow: build → release → query
  - Search across built artifacts with provenance
  - Incremental rebuild (second run instant — everything cached)
  - Explain-cache with inline fingerprint breakdown
"""

case = {
    "name": "tv_returns",
    "pipeline": "pipeline.py",
    "steps": [
        # Step 1: Plan
        {"name": "note_plan", "command": ["synix", "demo", "note", "1/6 Planning build..."]},
        {"name": "plan", "command": ["synix", "plan", "PIPELINE"]},
        # Step 2: Build
        {
            "name": "note_build",
            "command": ["synix", "demo", "note", "2/6 Building product briefs from catalog + policies..."],
        },
        {"name": "build", "command": ["synix", "build", "PIPELINE"]},
        # Step 3: Release — materialize projections to a named release target
        {
            "name": "note_release",
            "command": ["synix", "demo", "note", "3/6 Releasing: materializing search index..."],
        },
        {"name": "release", "command": ["synix", "release", "HEAD", "--to", "local"]},
        # Step 4: Search
        {"name": "note_search", "command": ["synix", "demo", "note", "4/6 Searching built artifacts..."]},
        {"name": "search", "command": ["synix", "search", "return policy", "--mode", "keyword", "--limit", "3"]},
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
}
