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

import hashlib
import json
from pathlib import Path

from synix import Layer, Pipeline, Projection
from synix.build.transforms import BaseTransform, register_transform
from synix.core.models import Artifact, FixerDecl, ValidatorDecl

# ---------------------------------------------------------------------------
# Custom transforms — these define HOW each layer processes its inputs.
# Synix ships built-in transforms, but pipelines can define their own.
# ---------------------------------------------------------------------------

@register_transform("demo_load_product_offers")
class DemoLoadProductOffersTransform(BaseTransform):
    """Read product_catalog.json + vendor_offers.json, join by SKU."""

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        source_dir = Path(config["source_dir"])

        catalog_path = source_dir / "product_catalog.json"
        offers_path = source_dir / "vendor_offers.json"
        catalog = json.loads(catalog_path.read_text())
        offers = json.loads(offers_path.read_text())
        offer_by_sku = {o["sku"]: o for o in offers}

        cat_hash = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
        off_hash = hashlib.sha256(offers_path.read_bytes()).hexdigest()
        combined_hash = f"sha256:{hashlib.sha256(f'{cat_hash}:{off_hash}'.encode()).hexdigest()}"

        artifacts = []
        for product in catalog:
            sku = product["sku"]
            offer = offer_by_sku.get(sku, {})
            merged = {**product, "offer": offer}
            content = json.dumps(merged, indent=2)

            artifacts.append(Artifact(
                artifact_id=f"product-offer-{sku}",
                artifact_type="product_offer_view",
                content=content,
                input_hashes=[combined_hash],
                metadata={
                    "sku": sku,
                    "product_name": product["name"],
                    "category": product["category"],
                    "brand": product.get("brand", ""),
                    "price": offer.get("our_price", 0),
                },
            ))
        return artifacts


@register_transform("demo_load_policies")
class DemoLoadPoliciesTransform(BaseTransform):
    """Read policy markdown files from sources/policies/."""

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        source_dir = Path(config["source_dir"])
        policies_dir = source_dir / "policies"

        artifacts = []
        for md_file in sorted(policies_dir.glob("*.md")):
            content = md_file.read_text()
            file_hash = f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"
            name = md_file.stem

            artifacts.append(Artifact(
                artifact_id=f"policy-{name}",
                artifact_type="policy",
                content=content,
                input_hashes=[file_hash],
                metadata={"policy_name": name},
            ))
        return artifacts


@register_transform("demo_extract_policies")
class DemoExtractPoliciesTransform(BaseTransform):
    """Extract key actionable rules from policy documents using LLM."""

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        from synix.build.llm_transforms import _get_llm_client, _logged_complete

        client = _get_llm_client(config)
        prompt_id = "demo_extract_policies"
        model_config = config.get("llm_config", {})

        results: list[Artifact] = []
        for policy in inputs:
            prompt = (
                "Extract the key actionable rules from this store policy document.\n\n"
                "<policy>\n"
                f"{policy.content}\n"
                "</policy>\n\n"
                "Return a structured summary with:\n"
                "1. Policy name and effective date\n"
                "2. Each key rule as a specific, standalone statement\n"
                "   (e.g. \"Electronics over $500: 15-day return window from delivery\")\n"
                "3. Important exceptions and conditions\n\n"
                "Be precise — preserve exact numbers, time windows, conditions, and fees.\n"
                "Format as a bulleted list grouped by product category or topic."
            )
            response = _logged_complete(
                client, config,
                messages=[{"role": "user", "content": prompt}],
                artifact_desc=f"policy extraction {policy.artifact_id}",
            )
            results.append(Artifact(
                artifact_id=f"idx-{policy.metadata['policy_name']}",
                artifact_type="policy_index",
                content=response.content,
                input_hashes=[policy.content_hash],
                prompt_id=prompt_id,
                model_config=model_config,
                metadata={"policy_name": policy.metadata["policy_name"]},
            ))
        return results


@register_transform("demo_enrich_cs_brief")
class DemoEnrichCSBriefTransform(BaseTransform):
    """Combine product/offer data with policy rules into a CS agent brief."""

    def split(
        self, inputs: list[Artifact], config: dict
    ) -> list[tuple[list[Artifact], dict]]:
        products = sorted(
            [a for a in inputs if a.artifact_type == "product_offer_view"],
            key=lambda a: a.artifact_id,
        )
        policies = sorted(
            [a for a in inputs if a.artifact_type == "policy_index"],
            key=lambda a: a.artifact_id,
        )
        return [([product] + policies, {}) for product in products]

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        from synix.build.llm_transforms import _get_llm_client, _logged_complete

        client = _get_llm_client(config)
        prompt_id = "demo_enrich_cs_brief"
        model_config = config.get("llm_config", {})

        products = sorted(
            [a for a in inputs if a.artifact_type == "product_offer_view"],
            key=lambda a: a.artifact_id,
        )
        policies = sorted(
            [a for a in inputs if a.artifact_type == "policy_index"],
            key=lambda a: a.artifact_id,
        )

        policies_text = "\n\n---\n\n".join(
            f"### {p.metadata.get('policy_name', p.artifact_id)}\n{p.content}"
            for p in policies
        )

        artifacts = []
        for product in products:
            prompt = (
                "You are writing a customer service reference brief for this product.\n"
                "The CS agent will use this as their ONLY reference — it must be complete.\n\n"
                "<product_and_offer>\n"
                f"{product.content}\n"
                "</product_and_offer>\n\n"
                "<store_policies>\n"
                f"{policies_text}\n"
                "</store_policies>\n\n"
                "Write a comprehensive customer service brief covering:\n"
                "1. Product overview (name, key features, specs)\n"
                "2. Pricing and any active promotions\n"
                "3. Availability and shipping (method, timeline, costs)\n"
                "4. Returns & defects — write ONE unified section covering what a customer "
                "should do if they want to return the product OR if it arrives defective. "
                "Include all relevant timeframes and conditions from both store policy "
                "and manufacturer terms.\n"
                "5. Known issues or vendor notes (include any notes from the vendor data verbatim "
                "— the CS agent needs full context)\n"
                "6. Any other details a CS agent should know\n\n"
                "Be specific — include exact timeframes, dollar amounts, and conditions.\n"
                "Write in a clear, scannable format with headers.\n"
                "The CS agent should not need to look up anything else."
            )

            response = _logged_complete(
                client, config,
                messages=[{"role": "user", "content": prompt}],
                artifact_desc=f"cs brief {product.metadata['sku']}",
            )

            sku = product.metadata["sku"]
            artifacts.append(Artifact(
                artifact_id=f"cs-brief-{sku}",
                artifact_type="cs_product_brief",
                content=response.content,
                input_hashes=(
                    [product.content_hash] + [p.content_hash for p in policies]
                ),
                prompt_id=prompt_id,
                model_config=model_config,
                metadata={
                    "sku": sku,
                    "product_name": product.metadata.get("product_name", ""),
                    "category": product.metadata.get("category", ""),
                },
            ))
        return artifacts


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
