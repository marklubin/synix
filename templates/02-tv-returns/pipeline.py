# pipeline.py — TV Returns Demo Pipeline
#
# Synix pipelines are defined in Python. Each pipeline declares:
#   - Layers:      the DAG of transforms (source -> derived -> enriched)
#   - Projections: how artifacts become searchable (FTS5, embeddings)
#   - Validators:  domain rules to check (semantic conflicts, PII)
#   - Fixers:      LLM-powered auto-repair for violations
#
# This pipeline builds customer service product briefs for a TV retailer.
# Source data: product catalog + vendor offers + store policies.
# Output: searchable CS reference briefs with validated, conflict-free content.
#
# Custom transforms live in transforms.py — imported and used as layers.

from transforms import (
    DemoEnrichCSBriefTransform,
    DemoExtractPoliciesTransform,
    DemoLoadPoliciesTransform,
    DemoLoadProductOffersTransform,
)

from synix import Pipeline, SearchIndex
from synix.fixers import SemanticEnrichment
from synix.validators import PII, SemanticConflict

# ======================================================================
# PIPELINE DEFINITION — edit this to change what Synix builds
# ======================================================================

pipeline = Pipeline("tv-returns-demo")
pipeline.source_dir = "./sources"  # source data
pipeline.build_dir = "./build"  # output artifacts
pipeline.llm_config = {  # default LLM for all layers
    "provider": "anthropic",
    "model": "claude-haiku-4-5-20251001",
    "temperature": 0.3,
    "max_tokens": 2048,
}

# -- Layers: the DAG of transforms ------------------------------------
# Level 0 = source (reads files), Level 1+ = derived (LLM transforms).
# Synix hashes inputs and skips layers when nothing changed.

product_offers = DemoLoadProductOffersTransform("product_offers")  # reads product_catalog + vendor_offers
policies = DemoLoadPoliciesTransform("policies")  # reads policy markdown files
policy_index = DemoExtractPoliciesTransform(  # LLM extracts actionable rules
    "policy_index",
    depends_on=[policies],  # from raw policy documents
)
cs_product_brief = DemoEnrichCSBriefTransform(  # LLM merges product data + policy
    "cs_product_brief",
    depends_on=[product_offers, policy_index],  # rules into CS reference briefs
)

pipeline.add(product_offers, policies, policy_index, cs_product_brief)

# -- Projection: makes artifacts searchable ----------------------------
pipeline.add(
    SearchIndex(
        "cs-search",
        sources=[cs_product_brief],  # SQLite FTS5 + semantic embeddings
        search=["fulltext", "semantic"],
        embedding_config={
            "provider": "fastembed",
            "model": "BAAI/bge-small-en-v1.5",
            "dimensions": 384,
        },
    )
)

# -- Validators: domain rules checked after build ---------------------
pipeline.add_validator(
    SemanticConflict(  # LLM detects contradictions in briefs
        artifact_ids=["cs-brief-SAM-OLED-65"],
        llm_config={"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "temperature": 0.0},
    )
)
pipeline.add_validator(
    PII(  # regex catches emails, phones, etc.
        layers=[cs_product_brief],
        patterns=["email"],
        severity="error",
    )
)

# -- Fixer: LLM auto-repairs violations (you accept/reject each) ------
pipeline.add_fixer(
    SemanticEnrichment(  # rewrites conflicting artifact sections
        max_context_episodes=5,
        temperature=0.3,
    )
)
