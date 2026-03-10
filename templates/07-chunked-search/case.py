"""chunked_search demo case — Chunk transform with paragraph splitting.

Flow:
  1. Plan     — show the DAG and what will be built
  2. Build    — parse articles, split into paragraph chunks (no LLM)
  3. Release  — materialize search index to a release target
  4. Search   — query across chunks via release
  5. Rebuild  — second run, everything cached
  6. Explain  — show cache decisions inline

This demonstrates:
  - Chunk transform: 1:N pure text splitting (no LLM calls)
  - Paragraph-level chunking via separator="\\n\\n"
  - Chunk metadata: source_label, chunk_index, chunk_total
  - Full-text search across chunk artifacts with provenance
  - Incremental rebuild (second run is instant)
"""

case = {
    "name": "chunked_search",
    "pipeline": "pipeline.py",
    "steps": [
        # Step 1: Plan
        {"name": "note_plan", "command": ["synix", "demo", "note", "1/6 Planning build..."]},
        {"name": "plan", "command": ["synix", "plan", "PIPELINE"]},
        # Step 2: Build
        {
            "name": "note_build",
            "command": ["synix", "demo", "note", "2/6 Building: articles → chunks (no LLM)..."],
        },
        {"name": "build", "command": ["synix", "build", "PIPELINE"]},
        # Step 3: Release — materialize projections to a named release target
        {
            "name": "note_release",
            "command": ["synix", "demo", "note", "3/6 Releasing: materializing search index..."],
        },
        {"name": "release", "command": ["synix", "release", "HEAD", "--to", "local"]},
        # Step 4: Search
        {"name": "note_search", "command": ["synix", "demo", "note", "4/6 Searching across chunks..."]},
        {"name": "search", "command": ["synix", "search", "encryption", "--mode", "keyword", "--limit", "3"]},
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
