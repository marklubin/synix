# 03 — Team Report

Multi-layer team analysis pipeline: parse bios and a project brief, infer work styles, analyze team dynamics, and synthesize a staffing report.

## What This Demonstrates

- Two independent level-0 source layers (bios + project brief)
- 1:1 LLM transform (bio -> work style profile)
- Many:1 rollup (work styles -> team dynamics analysis)
- Multi-input synthesis (team dynamics + project brief -> final staffing report)
- Custom validator (max_length on final report)
- Full-text search across all 5 layers with provenance
- Incremental rebuild (second run is instant — everything cached)

## Sample Data

```
sources/
├── bios/
│   ├── alice.md         # Backend engineer, distributed systems
│   ├── bob.md           # Product designer, accessibility focus
│   └── carol.md         # Data scientist, climate modeling
└── brief/
    └── project_brief.md # Climate sensor dashboard project
```

## Run

```bash
cd examples/03-team-report
cp .env.example .env     # add your API key

uvx synix build pipeline.py
uvx synix validate pipeline.py
uvx synix search 'hiking'
```

## Try It Yourself

### Add yourself to the team

Create `sources/bios/yourname.md`:

```markdown
# Your Name

Senior frontend engineer, 6 years experience. Built design systems at two
startups. Passionate about animation and micro-interactions. Prefers
pair programming and async standups over daily meetings.
```

Then rebuild — only the new bio and downstream layers recompute:

```bash
uvx synix build pipeline.py    # bios: 1 new, 3 cached
uvx synix search 'frontend'
```

### Change the project

Edit `sources/brief/project_brief.md` with your own project description. The pipeline will regenerate work style matches, team dynamics, and the staffing report based on the new brief.

### Customize the pipeline

Open `pipeline.py` to see how layers, transforms, and validators are wired. Key things to try:

- Add a new layer (e.g., `skills_matrix` that cross-references bios against the project brief)
- Change the LLM model or temperature in `pipeline.llm_config`
- Add a validator (e.g., `pii` to check for email addresses in bios)
