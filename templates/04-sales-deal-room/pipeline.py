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
# Search: SynixSearch over a declared deal search surface
#
# Usage:
#   uvx synix build pipeline.py
#   uvx synix search "pricing"

from transforms import (
    DealCallPrepTransform,
    DealCompetitiveIntelTransform,
    DealStrategyTransform,
)

from synix import Pipeline, SearchSurface, Source, SynixSearch

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

deal_search = SearchSurface(
    "deal-search",
    sources=[
        competitor_docs,
        product_specs,
        deal_context,
        win_reports,
        competitive_intel,
        strategy,
        call_prep,
    ],
    modes=["fulltext"],
)

pipeline.add(
    competitor_docs,
    product_specs,
    deal_context,
    win_reports,
    competitive_intel,
    strategy,
    call_prep,
    deal_search,
)

# -- Projection: full-text search on all layers (sources + transforms) --

pipeline.add(SynixSearch("search", surface=deal_search))
