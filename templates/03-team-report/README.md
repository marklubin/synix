# 03 — Team Report

Multi-layer team analysis pipeline: parse bios and a project brief, infer work styles, analyze team dynamics, and synthesize a staffing report.

## What This Demonstrates

- **`synix.transforms` generic transforms** — no custom Transform subclasses needed
- `MapSynthesis` (1:1): bio → work style profile
- `ReduceSynthesis` (N:1): work styles → team dynamics analysis
- `FoldSynthesis` (sequential N:1): team dynamics + project brief → staffing report
- Two independent level-0 source layers (bios + project brief)
- Domain validator (`RequiredField` on final report)
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
cd templates/03-team-report
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

Edit `sources/brief/project_brief.md` with your own project description. The pipeline will regenerate the team dynamics and staffing report based on the new brief.

### Customize the pipeline

Open `pipeline.py` to see how layers and generic platform transforms are wired. Key things to try:

- Change the prompts in `MapSynthesis`, `ReduceSynthesis`, or `FoldSynthesis`
- Add a new layer (e.g., `skills_matrix` that cross-references bios against the project brief)
- Change the LLM model or temperature in `pipeline.llm_config`
- Add a validator (e.g., `pii` to check for email addresses in bios)

## Custom Transforms

The generic transforms cover common patterns, but you can always write a custom `Transform` subclass for full control. Here's the equivalent of `MapSynthesis` as a custom class:

```python
from synix import Transform
from synix.build.llm_transforms import _get_llm_client, _logged_complete
from synix.core.models import Artifact

class WorkStyleTransform(Transform):
    """One bio -> one work style profile. Default split gives 1:1."""

    def execute(self, inputs, config):
        bio = inputs[0]
        client = _get_llm_client(config)
        response = _logged_complete(
            client, config,
            messages=[{"role": "user", "content": f"Infer work style:\n\n{bio.content}"}],
            artifact_desc=f"work-style {bio.label}",
        )
        return [Artifact(
            label=f"ws-{bio.label}",
            artifact_type="work_style",
            content=response.content,
            input_ids=[bio.artifact_id],
            prompt_id="work_style_v1",
            model_config=config.get("llm_config"),
        )]
```

Use custom transforms when you need logic beyond simple prompt templating — e.g., filtering inputs, conditional branching, or multi-step LLM chains within a single transform.
