# sales_deal_room demo — deterministic VHS recording
#
# Usage:
#   cd templates/04-sales-deal-room && vhs tape.tape

Output deal_room.mp4
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
Type "# This demo builds citation-backed competitive intelligence for a"
Enter
Type "# sales deal room: competitor analysis, strategy, and call prep."
Enter
Type "# Every claim traces to a source document via synix:// citations."
Enter
Sleep 2s

# ── Plan ───────────────────────────────────────────────────────
Ctrl+L
Type "# PLAN: 4-level DAG — sources → intel → strategy → call prep."
Enter
Type "# Fresh build — 10 source files, 5 LLM calls needed."
Enter
Sleep 1.5s
Type "synix plan pipeline.py"
Enter
Sleep 7s

# ── Build ──────────────────────────────────────────────────────
Ctrl+L
Type "# BUILD: parse 10 source files, then 3 competitive analyses,"
Enter
Type "# 1 strategy document, and 1 call prep brief — all with citations."
Enter
Sleep 1.5s
Type "synix build pipeline.py"
Enter
Sleep 12s

# ── Search ─────────────────────────────────────────────────────
Ctrl+L
Type "# SEARCH: query competitive intel for pricing data."
Enter
Type "# Results include provenance chains back to source documents."
Enter
Sleep 1.5s
Type "synix search 'pricing' --mode keyword --limit 3"
Enter
Sleep 5s

# ── Validate (expect ungrounded claims) ────────────────────────
Ctrl+L
Type "# VALIDATE: citation validator checks that claims in strategy"
Enter
Type "# and call prep are grounded by synix:// citation links."
Enter
Type "# Expecting some ungrounded claims in the initial build."
Enter
Sleep 2s
Type "synix validate pipeline.py"
Enter
Sleep 14s

# ── Fix ungrounded claims ─────────────────────────────────────
Ctrl+L
Type "# FIX: the citation_enrichment fixer searches source documents,"
Enter
Type "# then adds [text](synix://label) citations or removes claims."
Enter
Type "# Accept, reject, or skip each proposed fix."
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

# ── Re-validate (should be clean) ─────────────────────────────
Ctrl+L
Type "# RE-VALIDATE: all claims should now be properly cited."
Enter
Sleep 1.5s
Type "synix validate pipeline.py"
Enter
Sleep 8s

# ── New intel arrives ──────────────────────────────────────────
Ctrl+L
Type "# NEW INTEL: Acme Analytics just announced a 30% price cut."
Enter
Type "# Copy the update into sources — this triggers a cascade rebuild."
Enter
Sleep 1.5s
Type "python3 copy_staged.py"
Enter
Sleep 3s

# ── Plan cascade ──────────────────────────────────────────────
Ctrl+L
Type "# PLAN: new source file → competitor_docs rebuilds → cascades"
Enter
Type "# through intel → strategy → call prep. Other sources cached."
Enter
Sleep 1.5s
Type "synix plan pipeline.py"
Enter
Sleep 7s

# ── Rebuild cascade ───────────────────────────────────────────
Ctrl+L
Type "# REBUILD: only affected artifacts rebuild. Watch built vs cached."
Enter
Sleep 1.5s
Type "synix build pipeline.py"
Enter
Sleep 12s

# ── Search new intel ──────────────────────────────────────────
Ctrl+L
Type "# SEARCH: query for the new competitive data."
Enter
Sleep 1.5s
Type "synix search 'price cut' --mode keyword --limit 3"
Enter
Sleep 5s

# ── Validate cascade ──────────────────────────────────────────
Ctrl+L
Type "# VALIDATE: fresh LLM output should be well-cited."
Enter
Sleep 1.5s
Type "synix validate pipeline.py"
Enter
Sleep 14s

# ── Explain cache ──────────────────────────────────────────────
Ctrl+L
Type "# EXPLAIN: cache fingerprint breakdown — why each artifact was"
Enter
Type "# rebuilt or served from cache."
Enter
Sleep 1.5s
Type "synix plan pipeline.py --explain-cache"
Enter
Sleep 7s

Type "# Citation-backed intelligence with full provenance. Done."
Enter
Sleep 4s
