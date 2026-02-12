# Ecommerce Demo Pipeline Spec
# tv_demo.py — pipeline config for the CS product knowledge demo

## Overview

Two independent ingestion paths converge via LLM enrichment into a final
per-product CS brief. The brief is what a customer service agent reads
when answering customer questions.

## Sources

sources/product_catalog.json    — 3 products (Samsung TV, Dyson vacuum, Shun knife)
sources/vendor_offers.json      — 3 vendor offers with pricing, manufacturer warranty terms
sources/policies/return_policy.md    — store return policy (updated Nov 2025)
sources/policies/shipping_policy.md  — store shipping policy
sources/policies/warranty_policy.md  — warranty claims handling (store vs manufacturer)

## Pipeline Structure

### Path 1: Product + Offer Join (no LLM)

Step: join_product_offers
  Input: product_catalog.json, vendor_offers.json
  Transform type: structured join on SKU
  Output: one artifact per SKU in layer "product_offer_view"
  
  Each product_offer_view artifact contains:
    - product name, brand, category, specs
    - price, stock status, fulfillment method
    - manufacturer warranty terms (verbatim from vendor offer)
    - active promos

  This is a pure data join — no LLM, no summarization. Just merge the
  two JSON records by SKU into one combined view.

### Path 2: Policy Extraction (lightweight LLM or structured)

Step: extract_policies
  Input: policies/*.md (all policy docs)
  Transform type: chunk + extract per policy doc
  Output: one artifact per policy doc in layer "policy_index"
  
  Each policy_index artifact contains:
    - policy name
    - effective date
    - key rules extracted as structured claims
      e.g. "Electronics over $500: 15-day return window"
      e.g. "Manufacturer warranty != store return policy"
      e.g. "Large items: free threshold delivery over $500"

  Light extraction — pull out the actionable rules, drop the boilerplate.

### Path 3: LLM Enrichment (convergence point)

Step: enrich_with_policies
  Input: product_offer_view/* (from Path 1), policy_index/* (from Path 2)
  Transform type: LLM enrichment — per product_offer_view artifact
  Output: one artifact per SKU in layer "cs_product_brief"

  For each product_offer_view:
    1. Determine which policies are relevant based on product category and price
       (e.g. Samsung TV triggers: electronics return policy, large item shipping,
       warranty claims policy)
    2. Synthesize a customer service brief that a CS agent can read to answer
       any question about this product
    3. Brief should cover: product summary, price, availability, shipping,
       return policy, warranty info, active promos

  This is where the contradiction gets introduced. The LLM sees
  "90-day defect replacement" from the vendor offer and "15-day return
  window" from the store policy and is likely to conflate them.

## Layer Summary

Layer 0 (sources):       product_catalog.json, vendor_offers.json, policies/*.md
Layer 1 (intermediate):  product_offer_view/SAM-OLED-65, DYS-V15-DETECT, KAI-PRO-KNIFE8
Layer 1 (intermediate):  policy_index/return_policy, shipping_policy, warranty_policy
Layer 2 (final):         cs_product_brief/SAM-OLED-65, DYS-V15-DETECT, KAI-PRO-KNIFE8

Total: 5 sources → 6 intermediate → 3 final = 14 artifacts

## Validators

validators:
  - semantic_conflict:
      layers: ["cs_product_brief"]
      llm_config: { model: "claude-sonnet-4-20250514", temperature: 0.0 }
      max_artifacts: 10

  - pii:
      layers: ["cs_product_brief", "product_offer_view"]
      patterns: ["credit_card", "ssn", "email", "phone"]

## Fixers

fixers:
  - semantic_enrichment:
      max_context_episodes: 5
      temperature: 0.3

## Expected Contradictions

The validate step should catch these in the cs_product_brief layer:

1. SAM-OLED-65 (HIGH confidence):
   - Claim A: "90-day return/replacement" (from vendor manufacturer warranty)
   - Claim B: "15-day return window for electronics over $500" (from store return policy)
   - Root cause: LLM conflated manufacturer defect warranty with store return policy
   - Fix: Distinguish store return (15 days) from Samsung defect warranty (90 days, direct to Samsung)

2. DYS-V15-DETECT (MEDIUM confidence):
   - Claim A: "2-year warranty with free repair/replacement" (from vendor warranty terms)
   - Claim B: "30-day return window for small appliances" (from store return policy)
   - Root cause: Brief may imply store handles warranty when Dyson handles it directly
   - Fix: Clarify warranty is through Dyson, store return is separate 30-day window

3. KAI-PRO-KNIFE8 (MEDIUM confidence):
   - Claim A: "Lifetime warranty, free sharpening" (from vendor warranty)
   - Claim B: "Knives that have been sharpened are non-returnable" (from store return policy)
   - Root cause: Getting knife sharpened by KAI could be misread as voiding store return
   - Fix: Clarify KAI sharpening is warranty service, store return is separate 30-day window for unused knives

## Demo Flow

$ synix build tv_demo.py
Built 14 artifacts (5 sources → 6 intermediate → 3 cs_product_brief)

$ synix search "return policy samsung tv"
[product_offer_view]  SAM-OLED-65: "90-day defect replacement direct to Samsung"
[policy_index]        return_policy: "Electronics over $500: 15-day return window"
[policy_index]        warranty_policy: "We do NOT process warranty claims for manufacturers"
[cs_product_brief]    SAM-OLED-65: "Returns accepted within 90 days..."

$ synix validate tv_demo.py
⚠ semantic_conflict: cs_product_brief/SAM-OLED-65 (high)
  Claim A: "90-day return for full replacement"
    [source: product_offer_view ← vendor_offers.json]
  Claim B: "15-day return window, 15% restocking fee"
    [source: policy_index ← return_policy.md]
  Reasoning: 90-day figure is manufacturer defect warranty (direct to
    Samsung). Store return window is 15 days for electronics over $500.
    LLM enrichment conflated the two.

$ synix validate tv_demo.py --fix
── Fix proposal for cs_product_brief/SAM-OLED-65 ──
  Evidence: vendor_offers.json, return_policy.md, warranty_policy.md

  ── Diff ──
  - Return policy: Customers may return within 90 days for full replacement.
  + Returns: 15-day store return window from delivery (15% restocking fee
  + if opened). Manufacturer warranty: Samsung offers 90-day defect
  + replacement handled directly through Samsung at 1-800-SAMSUNG — this
  + is not a store return.

  [a]ccept / [d]eny / [i]gnore > a
  ✅ Fixed (cs_product_brief/SAM-OLED-65 rewritten)

$ synix build tv_demo.py
Rebuilding cs_product_brief/SAM-OLED-65 (source changed)...
Done. 1 artifact rebuilt.

$ synix validate tv_demo.py
✅ No violations found
