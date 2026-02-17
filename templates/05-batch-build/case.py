"""batch_build demo case — batch API at level 1, sync rollup at level 2.

Flow:
  1. Clean    — remove any previous build artifacts
  2. Plan     — show which layers batch vs sync
  3. Build    — parse bios, batch work styles (OpenAI), sync team summary (Anthropic)
  4. List     — show batch build instances
  5. Artifacts — list all artifacts across layers
  6. Show     — render the final team summary artifact

This demonstrates:
  - OpenAI Batch API for 1:1 transforms (work styles)
  - Anthropic sync inference for N:1 rollup (team summary)
  - Mixed providers in a single pipeline
  - Batch build plan showing mode per layer
"""

case = {
    "name": "batch_build",
    "pipeline": "pipeline.py",
    "steps": [
        # Step 0: Clean previous build
        {"name": "clean", "command": ["synix", "clean", "-y"]},
        # Step 1: Batch build plan
        {"name": "note_plan", "command": ["synix", "demo", "note", "1/5 Batch build plan..."]},
        {"name": "plan", "command": ["synix", "batch-build", "plan", "PIPELINE"]},
        # Step 2: Run batch build
        {
            "name": "note_run",
            "command": [
                "synix",
                "demo",
                "note",
                "2/5 Running batch build (bios \u2192 work styles \u2192 team summary)...",
            ],
        },
        {"name": "run", "command": ["synix", "batch-build", "run", "PIPELINE", "--poll"]},
        # Step 3: List batch builds
        {"name": "note_list", "command": ["synix", "demo", "note", "3/5 Listing batch builds..."]},
        {"name": "list", "command": ["synix", "batch-build", "list"]},
        # Step 4: List artifacts
        {"name": "note_artifacts", "command": ["synix", "demo", "note", "4/5 Listing artifacts..."]},
        {"name": "artifacts", "command": ["synix", "list"]},
        # Step 5: Show the final summary
        {"name": "note_show", "command": ["synix", "demo", "note", "5/5 Showing team summary..."]},
        {"name": "show", "command": ["synix", "show", "team-summary"]},
    ],
    "goldens": {},
    "output_masks": {
        "run": [r"Build ID:", r"batch-[0-9a-f]+", r"\bTime:"],
        "list": [r"batch-[0-9a-f]+", r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}"],
    },
}
