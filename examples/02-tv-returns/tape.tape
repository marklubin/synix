# tv_returns demo — deterministic VHS recording
#
# Usage:
#   cd demos/tv_returns && ./record.sh

Output tv_returns.mp4
Set Shell "bash"
Set FontSize 14
Set Width 1200
Set Height 1200
Set Padding 20
Set Theme "Solarized Dark Higher Contrast"
Set TypingSpeed 60ms

# Set up environment for deterministic replay
Env SYNIX_CASSETTE_MODE "replay"
Env SYNIX_CASSETTE_DIR "./cassettes"
Env SYNIX_DEMO "1"
Env TOKENIZERS_PARALLELISM "false"
Env HF_HUB_DISABLE_PROGRESS_BARS "1"
Env TRANSFORMERS_NO_ADVISORY_WARNINGS "1"
Env PATH "/home/mark/synix/.venv/bin:/usr/bin:/bin"

# ── Intro ──────────────────────────────────────────────────────
Type "synix info"
Enter
Sleep 6s

Ctrl+L
Type "# This demo builds CS product briefs for a TV retailer from"
Enter
Type "# product catalog + vendor offers + store return policies."
Enter
Type "# Here's the pipeline:"
Enter
Sleep 2s

Type "python3 show_pipeline.py"
Enter
Sleep 8s

# ── Plan ───────────────────────────────────────────────────────
Ctrl+L
Type "# PLAN: analyze the DAG and estimate LLM cost."
Enter
Type "# Fresh build — all 4 layers are new, 6 LLM calls needed."
Enter
Sleep 1.5s
Type "synix plan pipeline.py"
Enter
Sleep 7s

# ── Build ──────────────────────────────────────────────────────
Ctrl+L
Type "# BUILD: walk the DAG, run LLM transforms, materialize search index."
Enter
Type "# Loading 3 products + 3 policies, then 6 LLM calls to extract"
Enter
Type "# policy rules and enrich each product into a CS brief."
Enter
Sleep 1.5s
Type "synix build pipeline.py"
Enter
Sleep 12s

# ── Search ─────────────────────────────────────────────────────
Ctrl+L
Type "# SEARCH: query the search index projection built from the artifacts."
Enter
Type "# Each result shows content, layer, score, and provenance chain"
Enter
Type "# tracing back through the DAG to source files."
Enter
Sleep 1.5s
Type "synix search 'return policy' --mode keyword --limit 3"
Enter
Sleep 5s

# ── Validate ───────────────────────────────────────────────────
Ctrl+L
Type "# VALIDATE: check domain rules declared in the pipeline."
Enter
Type "# semantic_conflict — LLM checks for contradictory return timeframes."
Enter
Type "# pii — regex catches a leaked email from vendor data."
Enter
Type "# Expecting violations from both."
Enter
Sleep 2s
Type "synix validate pipeline.py"
Enter
Sleep 14s

# ── Fix PII at source ─────────────────────────────────────────
Ctrl+L
Type "# FIX STRATEGY 1: PII is a source data problem."
Enter
Type "# Find the leaked email in the vendor data, edit it out, then rebuild."
Enter
Type "# Fix the input, not the output."
Enter
Sleep 1.5s
Type "grep -n 'john.smith' sources/vendor_offers.json"
Enter
Sleep 5s
Type "sed -i 's/john.smith@gmail.com/A customer/g' sources/vendor_offers.json"
Enter
Sleep 2s

# ── Plan (show what needs rebuilding) ─────────────────────────
Ctrl+L
Type "# PLAN: source data changed — let's see what needs rebuilding."
Enter
Type "# The plan shows which layers are cached vs need new LLM calls."
Enter
Sleep 1.5s
Type "synix plan pipeline.py"
Enter
Sleep 7s

# ── Rebuild ────────────────────────────────────────────────────
Ctrl+L
Type "# REBUILD: source changed, so Synix detects the hash mismatch."
Enter
Type "# Only affected layers rebuild — unchanged layers served from cache."
Enter
Type "# Watch the built vs cached counts."
Enter
Sleep 1.5s
Type "synix build pipeline.py"
Enter
Sleep 12s

# ── Re-validate ──────────────────────────────────────────────
Ctrl+L
Type "# RE-VALIDATE: PII is gone (fixed at source). But semantic"
Enter
Type "# conflicts persist — the LLM still generated contradictory policy"
Enter
Type "# language. Those need an LLM-powered fix."
Enter
Sleep 2s
Type "synix validate pipeline.py"
Enter
Sleep 14s

# ── Fix semantic conflicts ────────────────────────────────────
Ctrl+L
Type "# FIX STRATEGY 2: the fixer walks the provenance chain to find"
Enter
Type "# all source documents, then asks the LLM to resolve the conflict."
Enter
Type "# If enough info exists, it rewrites the section. If not, it flags"
Enter
Type "# for human review. Accept, reject, or skip each proposal."
Enter
Sleep 2s
Type "synix fix pipeline.py"
Enter
Sleep 14s
Type "a"
Enter
Sleep 8s
Type "a"
Enter
Sleep 8s
Type "a"
Enter
Sleep 6s

# ── Final validate ─────────────────────────────────────────────
Ctrl+L
Type "# FINAL VALIDATE: PII fixed at source, semantic conflicts"
Enter
Type "# fixed by LLM rewrite. Should be clean now."
Enter
Sleep 1.5s
Type "synix validate pipeline.py"
Enter
Sleep 8s

Type "# Clean. Two fix strategies, one pipeline, full provenance."
Enter
Sleep 4s
