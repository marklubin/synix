# pipeline.py — TV Returns Demo Pipeline
#
# Synix pipelines are defined in Python. Each pipeline declares:
#   - Layers:      the DAG of transforms (source → derived → enriched)
#   - Projections: how artifacts become searchable (FTS5, embeddings)
#   - Validators:  domain rules to check (semantic conflicts, PII)
#   - Fixers:      LLM-powered auto-repair for violations
#
# This pipeline builds customer service product briefs for a TV retailer.
# Source data: product catalog + vendor offers + store policies.
# Output: searchable CS reference briefs with validated, conflict-free content.
#
# Custom transforms live in transforms.py — imported here to register them.

import transforms  # noqa: F401 — registers custom transforms

from synix import Layer, Pipeline, Projection
from synix.core.models import FixerDecl, ValidatorDecl

# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE DEFINITION — edit this to change what Synix builds
# ═══════════════════════════════════════════════════════════════════════════

pipeline = Pipeline("tv-returns-demo")
pipeline.source_dir = "./sources"                        # source data
pipeline.build_dir = "./build"                          # output artifacts
pipeline.llm_config = {                                 # default LLM for all layers
    "provider": "anthropic",
    "model": "claude-haiku-4-5-20251001",
    "temperature": 0.3,
    "max_tokens": 2048,
}

# ── Layers: the DAG of transforms ────────────────────────────────────────
# Level 0 = source (reads files), Level 1+ = derived (LLM transforms).
# Synix hashes inputs and skips layers when nothing changed.

pipeline.add_layer(Layer(
    name="product_offers", level=0,         # reads product_catalog + vendor_offers
    transform="demo_load_product_offers",
))
pipeline.add_layer(Layer(
    name="policies", level=0,               # reads policy markdown files
    transform="demo_load_policies",
))
pipeline.add_layer(Layer(
    name="policy_index", level=1,           # LLM extracts actionable rules
    depends_on=["policies"],                #   from raw policy documents
    transform="demo_extract_policies",
))
pipeline.add_layer(Layer(
    name="cs_product_brief", level=2,       # LLM merges product data + policy
    depends_on=["product_offers", "policy_index"],  # rules into CS reference briefs
    transform="demo_enrich_cs_brief",
))

# ── Projection: makes artifacts searchable ───────────────────────────────
pipeline.add_projection(Projection(
    name="cs-search",
    projection_type="search_index",         # SQLite FTS5 + semantic embeddings
    sources=[{"layer": "cs_product_brief", "search": ["fulltext", "semantic"]}],
    config={"embedding_config": {
        "provider": "fastembed", "model": "BAAI/bge-small-en-v1.5", "dimensions": 384,
    }},
))

# ── Validators: domain rules checked after build ────────────────────────
pipeline.add_validator(ValidatorDecl(
    name="semantic_conflict",               # LLM detects contradictions in briefs
    config={
        "artifact_ids": ["cs-brief-SAM-OLED-65"],
        "llm_config": {"provider": "anthropic",
                        "model": "claude-haiku-4-5-20251001", "temperature": 0.0},
    },
))
pipeline.add_validator(ValidatorDecl(
    name="pii",                             # regex catches emails, phones, etc.
    config={"layers": ["cs_product_brief"], "patterns": ["email"], "severity": "error"},
))

# ── Fixer: LLM auto-repairs violations (you accept/reject each) ─────────
pipeline.add_fixer(FixerDecl(
    name="semantic_enrichment",             # rewrites conflicting artifact sections
    config={"max_context_episodes": 5, "temperature": 0.3},
))
