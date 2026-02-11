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

synix build pipeline.py
synix validate pipeline.py
synix search 'hiking'
```

## Use Your Own Data

- Replace the markdown files in `sources/bios/` with your team members' backgrounds
- Edit `sources/brief/project_brief.md` with your project description
- The pipeline will infer work styles, analyze team dynamics, and generate a staffing report tailored to your inputs
