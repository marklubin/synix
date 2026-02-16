# pipeline.py — Sales Deal Room Pipeline
#
# DAG:
#   Level 0: competitor_docs  [parse] <- sources/competitors/ (3 md files)
#   Level 0: product_specs    [parse] <- sources/product/ (2 md files)
#   Level 0: deal_context     [parse] <- sources/deal/ (3 md files)
#   Level 0: win_reports      [parse] <- sources/wins/ (2 md files)
#   Level 1: competitive_intel [deal_competitive_intel] <- competitor_docs + product_specs
#   Level 2: strategy          [deal_strategy] <- competitive_intel + deal_context + win_reports
#   Level 3: call_prep         [deal_call_prep] <- strategy + deal_context
#
# Projection: search_index on competitive_intel, strategy, call_prep
# Validator: citation on strategy, call_prep
# Fixer: citation_enrichment
#
# Usage:
#   uvx synix build pipeline.py
#   uvx synix validate pipeline.py
#   uvx synix search "pricing"

from transforms import (
    DealCallPrepTransform,
    DealCompetitiveIntelTransform,
    DealStrategyTransform,
)

from synix import Pipeline, SearchIndex, Source
from synix.fixers import CitationEnrichment
from synix.validators import Citation

# ======================================================================
# PIPELINE DEFINITION
# ======================================================================

pipeline = Pipeline("sales-deal-room")
pipeline.source_dir = "./sources"
pipeline.build_dir = "./build"
pipeline.llm_config = {
    "provider": "anthropic",
    "model": "claude-haiku-4-5-20251001",
    "temperature": 0.3,
    "max_tokens": 2048,
}

# -- Level 0: four independent parse layers ----------------------------

competitor_docs = Source("competitor_docs", dir="./sources/competitors")
product_specs = Source("product_specs", dir="./sources/product")
deal_context = Source("deal_context", dir="./sources/deal")
win_reports = Source("win_reports", dir="./sources/wins")

# -- Level 1: per-competitor analysis (1:competitor + all product specs) ---

competitive_intel = DealCompetitiveIntelTransform(
    "competitive_intel",
    depends_on=[competitor_docs, product_specs],
)

# -- Level 2: strategic positioning (N:1) ------------------------------

strategy = DealStrategyTransform(
    "strategy",
    depends_on=[competitive_intel, deal_context, win_reports],
)

# -- Level 3: call prep brief (N:1) -----------------------------------

call_prep = DealCallPrepTransform(
    "call_prep",
    depends_on=[strategy, deal_context],
)

pipeline.add(
    competitor_docs,
    product_specs,
    deal_context,
    win_reports,
    competitive_intel,
    strategy,
    call_prep,
)

# -- Projection: full-text search on intel + strategy + call prep ------

pipeline.add(
    SearchIndex(
        "deal-search",
        sources=[competitive_intel, strategy, call_prep],
        search=["fulltext"],
    )
)

# -- Validator: citation grounding check on strategy + call_prep -------

pipeline.add_validator(
    Citation(
        layers=[strategy, call_prep],
        llm_config={
            "provider": "anthropic",
            "model": "claude-haiku-4-5-20251001",
            "temperature": 0.0,
        },
    )
)

# -- Fixer: LLM adds citations or removes ungrounded claims -----------

pipeline.add_fixer(CitationEnrichment(max_context_episodes=5, temperature=0.3))
