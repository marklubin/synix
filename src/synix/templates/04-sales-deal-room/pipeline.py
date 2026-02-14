# pipeline.py — Sales Deal Room Pipeline
#
# DAG:
#   Level 0: competitor_docs  [parse] ← sources/competitors/ (3 md files)
#   Level 0: product_specs    [parse] ← sources/product/ (2 md files)
#   Level 0: deal_context     [parse] ← sources/deal/ (3 md files)
#   Level 0: win_reports      [parse] ← sources/wins/ (2 md files)
#   Level 1: competitive_intel [deal_competitive_intel] ← competitor_docs + product_specs
#   Level 2: strategy          [deal_strategy] ← competitive_intel + deal_context + win_reports
#   Level 3: call_prep         [deal_call_prep] ← strategy + deal_context
#
# Projection: search_index on competitive_intel, strategy, call_prep
# Validator: citation on strategy, call_prep
# Fixer: citation_enrichment
#
# Usage:
#   synix build pipeline.py
#   synix validate pipeline.py
#   synix search "pricing"

import transforms  # noqa: F401 — registers custom transforms

from synix import Layer, Pipeline, Projection
from synix.core.models import FixerDecl, ValidatorDecl

# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════

pipeline = Pipeline("sales-deal-room")
pipeline.source_dir = "./sources"
pipeline.build_dir = "./build"
pipeline.llm_config = {
    "provider": "anthropic",
    "model": "claude-haiku-4-5-20251001",
    "temperature": 0.3,
    "max_tokens": 2048,
}

# ── Level 0: four independent parse layers ────────────────────────────────

pipeline.add_layer(
    Layer(
        name="competitor_docs",
        level=0,
        transform="parse",
        config={"source_dir": "./sources/competitors"},
    )
)
pipeline.add_layer(
    Layer(
        name="product_specs",
        level=0,
        transform="parse",
        config={"source_dir": "./sources/product"},
    )
)
pipeline.add_layer(
    Layer(
        name="deal_context",
        level=0,
        transform="parse",
        config={"source_dir": "./sources/deal"},
    )
)
pipeline.add_layer(
    Layer(
        name="win_reports",
        level=0,
        transform="parse",
        config={"source_dir": "./sources/wins"},
    )
)

# ── Level 1: per-competitor analysis (1:competitor + all product specs) ───

pipeline.add_layer(
    Layer(
        name="competitive_intel",
        level=1,
        depends_on=["competitor_docs", "product_specs"],
        transform="deal_competitive_intel",
    )
)

# ── Level 2: strategic positioning (N:1) ──────────────────────────────────

pipeline.add_layer(
    Layer(
        name="strategy",
        level=2,
        depends_on=["competitive_intel", "deal_context", "win_reports"],
        transform="deal_strategy",
    )
)

# ── Level 3: call prep brief (N:1) ───────────────────────────────────────

pipeline.add_layer(
    Layer(
        name="call_prep",
        level=3,
        depends_on=["strategy", "deal_context"],
        transform="deal_call_prep",
    )
)

# ── Projection: full-text search on intel + strategy + call prep ──────────

pipeline.add_projection(
    Projection(
        name="deal-search",
        projection_type="search_index",
        sources=[
            {"layer": "competitive_intel", "search": ["fulltext"]},
            {"layer": "strategy", "search": ["fulltext"]},
            {"layer": "call_prep", "search": ["fulltext"]},
        ],
    )
)

# ── Validator: citation grounding check on strategy + call_prep ──────────

pipeline.add_validator(
    ValidatorDecl(
        name="citation",
        config={
            "layers": ["strategy", "call_prep"],
            "llm_config": {
                "provider": "anthropic",
                "model": "claude-haiku-4-5-20251001",
                "temperature": 0.0,
            },
        },
    )
)

# ── Fixer: LLM adds citations or removes ungrounded claims ──────────────

pipeline.add_fixer(
    FixerDecl(
        name="citation_enrichment",
        config={"max_context_episodes": 5, "temperature": 0.3},
    )
)
