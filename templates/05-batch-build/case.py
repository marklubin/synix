"""batch_build demo case — batch API at level 1, sync rollup at level 2.

Flow (batch path):
  0. Clean    — remove any previous build artifacts
  1. Plan     — show which layers batch vs sync
  2. Run      — parse bios, batch work styles (OpenAI), sync team summary (Anthropic)
  3. List     — show batch build instances
  4. Status   — detailed status for latest build (--latest)
  5. Artifacts — list all artifacts across layers
  6. Show     — render the final team summary artifact

Flow (sync path — same pipeline, different execution):
  7. Clean    — reset for sync path
  8. Build    — synix build (all layers synchronous)
  9. Artifacts — verify same artifact structure produced
 10. Show     — verify team summary content produced

This demonstrates:
  - OpenAI Batch API for 1:1 transforms (work styles)
  - Anthropic sync inference for N:1 rollup (team summary)
  - Mixed providers in a single pipeline
  - Batch build plan showing mode per layer
  - Every batch-build subcommand (plan, run, list, status)
  - Same pipeline works via both batch and sync paths
"""

case = {
    "name": "batch_build",
    "pipeline": "pipeline.py",
    "steps": [
        # --- Batch path ---
        # Step 0: Clean previous build
        {"name": "clean", "command": ["synix", "clean", "-y"]},
        # Step 1: Batch build plan
        {"name": "note_plan", "command": ["synix", "demo", "note", "1/7 Batch build plan..."]},
        {"name": "plan", "command": ["synix", "batch-build", "plan", "PIPELINE"]},
        # Step 2: Run batch build
        {
            "name": "note_run",
            "command": [
                "synix",
                "demo",
                "note",
                "2/7 Running batch build (bios \u2192 work styles \u2192 team summary)...",
            ],
        },
        {"name": "run", "command": ["synix", "batch-build", "run", "PIPELINE", "--poll"]},
        # Step 3: List batch builds
        {"name": "note_list", "command": ["synix", "demo", "note", "3/7 Listing batch builds..."]},
        {"name": "list", "command": ["synix", "batch-build", "list"]},
        # Step 4: Status for latest build
        {
            "name": "note_status",
            "command": ["synix", "demo", "note", "4/7 Batch build status (--latest)..."],
        },
        {"name": "status", "command": ["synix", "batch-build", "status", "--latest"]},
        # Step 5: List artifacts
        {"name": "note_artifacts", "command": ["synix", "demo", "note", "5/7 Listing artifacts..."]},
        {"name": "artifacts", "command": ["synix", "list"]},
        # Step 6: Show the final summary
        {"name": "note_show", "command": ["synix", "demo", "note", "6/7 Showing team summary..."]},
        {"name": "show", "command": ["synix", "show", "team-summary"]},
        # --- Sync path (same pipeline, standard build) ---
        # Step 7: Clean for sync path
        {"name": "clean_sync", "command": ["synix", "clean", "-y"]},
        # Step 8: Sync build
        {
            "name": "note_sync",
            "command": [
                "synix",
                "demo",
                "note",
                "7/7 Sync build (same pipeline, no Batch API)...",
            ],
        },
        {"name": "sync_build", "command": ["synix", "build", "PIPELINE"]},
        # Step 9: List artifacts (sync)
        {"name": "artifacts_sync", "command": ["synix", "list"]},
        # Step 10: Show team summary (sync)
        {"name": "show_sync", "command": ["synix", "show", "team-summary"]},
    ],
    "goldens": {},
    "output_masks": {
        "clean": [r"(?:Cleaned|Nothing to clean)"],
        "run": [r"Build ID:", r"batch-[0-9a-f]+", r"\bTime:"],
        "list": [r"batch-[0-9a-f]+", r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}"],
        "status": [r"batch-[0-9a-f]+", r"\d{4}-\d{2}-\d{2}"],
        "clean_sync": [r"(?:Cleaned|Nothing to clean)"],
    },
}
