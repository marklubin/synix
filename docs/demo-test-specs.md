# Synix Demo Scenario Test Specifications

These tests verify that the system can execute the exact flows shown in the three demo scenarios. They are end-to-end tests that exercise the full pipeline and should be run against both mock LLM (deterministic, CI) and real LLM (pre-demo validation, manual).

All three test specs use a shared test corpus and mock LLM fixture. The mock LLM returns deterministic but realistic responses — not echo/passthrough, but canned responses that simulate real summarization, extraction, and aggregation behavior. This is critical because the demo narratives depend on the *content* of artifacts being plausible, not just the *mechanics* working.

---

## Shared Test Infrastructure

### Test Corpus: `fixtures/demo_corpus/`

**Demo 1 corpus:** 30 synthetic personal conversations spanning 3 months, covering 4 distinct topics (database migration, career planning, Rust learning, side project). Conversations are designed so that:
- The "database migration" topic spans 8 conversations across 3 months (tests cross-month topical clustering)
- Some conversations cover multiple topics (tests episode splitting)
- Some conversations are very short, 2–3 turns (edge case)
- Some conversations are long, 40+ turns (tests chunking resilience)

**Demo 2 corpus:** 50 synthetic financial advisor conversations across 10 simulated customers. Each customer has 3–7 conversations. Conversations are designed so that:
- Customer "sarah_287" has clear behavioral evolution over 4 conversations (the diff showcase)
- At least 2 customers have minimal data (1 conversation each — edge case)
- Conversations contain both structured financial data (numbers, rates) and unstructured behavioral signals (emotional language, contradictions between stated goals and actions)

**Demo 3 corpus:** 100 synthetic support conversations across 20 simulated customers. Conversations are designed so that:
- Customers "alice" (conv_8834) and "bob" (conv_8891) both mention product "BillingEngine" and error code "ERR-4401" in the same week — this is the merge trap
- Two additional customer pairs have similar overlap (the "3 incidents" verification)
- The remaining conversations have no cross-customer similarity above any reasonable threshold

### Mock LLM Fixture: `fixtures/mock_responses/`

The mock LLM server has two modes:

**Deterministic mode (CI):** Returns fixture-based responses keyed on input content hash. For each step in each demo, pre-generated response fixtures exist. These are hand-reviewed to be realistic — they read like actual LLM output, not test stubs.

**Model-switching mode (Demo 2 Beat 2):** Returns different fixture sets based on the model parameter. When `model=gpt-4o`, returns the GPT-4o fixture set. When `model=claude-sonnet-4-5`, returns the Sonnet fixture set. The Sonnet fixtures are written to be qualitatively different — more behavioral/linguistic nuance, matching the demo narrative.

### Assertion Helpers

```python
def assert_artifact_exists(store, step, record_id)
def assert_artifact_cached(run_log, step, record_id)       # was not rebuilt
def assert_artifact_rebuilt(run_log, step, record_id)       # was rebuilt
def assert_provenance_chain(store, artifact_id, expected_chain)
def assert_search_returns(search, query, index_name, expected_ids, top_k=5)
def assert_diff_nonempty(diff_result, artifact_id)
def assert_diff_empty(diff_result, artifact_id)
def assert_verify_passes(verify_result)
def assert_verify_fails(verify_result, expected_check, expected_count)
def assert_steps_share_sources(store, step_a, step_b)      # same source artifacts
def assert_steps_have_distinct_artifacts(store, step_a, step_b)
def assert_search_indexes_are_independent(search, index_a, index_b, query)
def count_llm_calls(run_log)
def count_llm_calls_for_step(run_log, step_name)
def count_cache_hits(run_log)
def count_cache_hits_for_step(run_log, step_name)
```

---

## DT-1: "A Thousand Conversations" Test Spec

**Test file:** `tests/e2e/test_demo_1_personal.py`

### DT-1.1: Fresh Build — Full Pipeline

```
GIVEN  Demo 1 corpus (30 conversations, clean state, no prior build)
       Pipeline config:
         sources → episodes → summaries → monthly (group_by=month) → core
         Single search output from [summaries, monthly]

WHEN   synix plan
THEN   Plan shows:
         - 30 conversations marked as "new"
         - Episode count > 30 (multi-topic conversations produce multiple episodes)
         - 30 summaries
         - 3 monthly aggregations (3 months of data)
         - 1 core synthesis
         - Estimated cost > $0
         - Estimated time > 0
         - 0 cached artifacts

WHEN   synix build
THEN   - Exit code 0
       - Run log shows LLM calls for every step (0 cache hits on fresh build)
       - Artifacts exist at every step: episodes, summaries, monthly, core
       - Every artifact has valid provenance (synix verify passes)
       - Run summary printed with total LLM calls, tokens, cost, time
```

### DT-1.2: Search After Build

```
GIVEN  Completed build from DT-1.1
WHEN   synix search "database migration tradeoffs"
THEN   - At least 1 result from step "monthly" in top 3
       - At least 1 result from step "summaries" in top 5
       - Results include relevance scores
       - No results from unrelated topics (career planning) in top 3

WHEN   synix search "database migration" --step monthly
THEN   - All results are from step "monthly"
       - Result count <= 3 (only 3 monthly artifacts exist)

WHEN   synix search "database migration" --trace
THEN   - Each result includes a provenance chain
       - Chain depth >= 2 (monthly → summaries → episodes or similar)
       - Leaf nodes in chain reference source conversation IDs
```

### DT-1.3: Reconfigure Aggregation — Monthly to Topical

```
GIVEN  Completed build from DT-1.1
WHEN   Pipeline config is changed: aggregation step switches from
       group_by=month to group_by=topic_cluster
AND    synix plan
THEN   Plan shows:
         - Episodes: ALL cached (source data unchanged, episode prompt unchanged)
         - Summaries: ALL cached (summary prompt unchanged)
         - Aggregation: N topical clusters marked as "new" (new step config)
           where N is determined by clustering (expect 3-6 for this corpus)
         - Core: marked as "rebuild" (inputs changed)
         - Estimated cost < cost of DT-1.1 fresh build
         - Estimated LLM calls < DT-1.1 total LLM calls

WHEN   synix build
THEN   - Exit code 0
       - count_llm_calls_for_step(log, "episodes") == 0
       - count_llm_calls_for_step(log, "summaries") == 0
       - count_llm_calls_for_step(log, "aggregation") > 0
       - count_llm_calls_for_step(log, "core") > 0
       - cache_hit_rate > 0.7

WHEN   synix diff core
THEN   - Diff is non-empty
       - Old version contains month-oriented language ("In March...", "During April...")
       - New version contains topic-oriented language (project names, theme names)
       - Diff metadata shows which build rule changed
```

### DT-1.4: No-Change Rebuild — Full Cache Hit

```
GIVEN  Completed build from DT-1.3
WHEN   synix plan (no config or source changes)
THEN   - ALL artifacts marked as "cached"
       - Estimated cost = $0.00
       - Estimated LLM calls = 0

WHEN   synix build
THEN   - Exit code 0
       - count_llm_calls(log) == 0
       - cache_hit_rate == 1.0
       - Build completes in < 5 seconds (no LLM latency)
```

---

## DT-2: "The Pivot" Test Spec

**Test file:** `tests/e2e/test_demo_2_startup.py`

### DT-2.1: Initial Build — Financial Profile Pipeline

```
GIVEN  Demo 2 corpus (50 conversations, 10 customers, clean state)
       Pipeline config:
         sources → extract (profile prompt, model=gpt-4o) → aggregate
         (group_by=customer_id, profile summary prompt) → global_model
         Single search output from [extract, aggregate]

WHEN   synix build
THEN   - Exit code 0
       - 50 extraction artifacts exist
       - 10 customer aggregation artifacts exist (one per customer)
       - 1 global model exists
       - All artifacts have valid provenance
       - synix verify passes

       Store for later comparison:
         - total_llm_calls_fresh = count_llm_calls(log)
         - extract_llm_calls_fresh = count_llm_calls_for_step(log, "extract")
         - sarah_287_aggregate_content = content of aggregate:customer_287
```

### DT-2.2: Beat 1 — Change Aggregation, Preserve Extractions

```
GIVEN  Completed build from DT-2.1
WHEN   ONLY the aggregation prompt is changed to:
       "Build a temporal narrative... Track how behaviors evolve..."
       (extraction step is UNCHANGED — same prompt, same model)
AND    synix plan
THEN   Plan shows:
         - Sources: 50 conversations (cached ✓)
         - Extract: 50 extractions (cached ✓) — THIS IS THE KEY ASSERTION
         - Aggregate: 10 customer aggregations (rebuild — prompt changed)
         - Global: 1 model (rebuild — inputs changed)
         - Estimated LLM calls == 11 (10 aggregations + 1 global)
         - Estimated LLM calls << total_llm_calls_fresh

WHEN   synix build
THEN   - Exit code 0
       - count_cache_hits_for_step(log, "extract") == 50
       - count_llm_calls_for_step(log, "extract") == 0
       - count_llm_calls_for_step(log, "aggregate") == 10
       - count_llm_calls_for_step(log, "global_model") == 1
       - total LLM calls == 11
       - Extraction artifact IDs unchanged from DT-2.1 (same IDs, same content)

WHEN   synix diff aggregate:customer_287
THEN   - Diff is non-empty
       - Old content (sarah_287_aggregate_content) resembles flat profile
         (contains patterns like "Income:", "Risk tolerance:", snapshot-style)
       - New content resembles temporal narrative
         (contains patterns like "Month 1:", "evolution", "trend", timeline-style)
       - Diff metadata shows: build rule changed, sources unchanged
```

### DT-2.3: Beat 2 — Parallel Pipeline Paths for Model Comparison

This test verifies the A/B experimentation flow using parallel paths in a
single pipeline definition — not a branching system. Both extraction steps
read from the same source layer. Each path produces its own artifacts and
feeds its own search index. The client queries whichever index it wants.

The DAG naturally deduplicates: sources are ingested once, and each
extraction step references the same source artifacts by ID. This is not a
special feature — it's what falls out of the DAG having two steps that
declare `from: sources`.

```
GIVEN  Demo 2 corpus (50 conversations, clean state — fresh build)
       Pipeline config with parallel paths:

         sources:
           type: source
           adapter: chatgpt_json
           path: ./conversations/

         extract_gpt4o:
           type: transform
           from: sources
           model: gpt-4o
           prompt: "Extract behavioral patterns..."

         extract_sonnet:
           type: transform
           from: sources
           model: claude-sonnet-4-5
           prompt: "Extract behavioral patterns..."

         aggregate_gpt4o:
           type: aggregate
           from: extract_gpt4o
           group_by: customer_id
           prompt: "Build temporal narrative..."

         aggregate_sonnet:
           type: aggregate
           from: extract_sonnet
           group_by: customer_id
           prompt: "Build temporal narrative..."

         global_gpt4o:
           type: aggregate
           from: aggregate_gpt4o

         global_sonnet:
           type: aggregate
           from: aggregate_sonnet

         outputs:
           search_main:
             from: [extract_gpt4o, aggregate_gpt4o]
           search_experiment:
             from: [extract_sonnet, aggregate_sonnet]

WHEN   synix plan
THEN   Plan shows:
         - Sources: 50 conversations (ingested once)
         - extract_gpt4o: 50 extractions
         - extract_sonnet: 50 extractions
         - aggregate_gpt4o: 10 aggregations
         - aggregate_sonnet: 10 aggregations
         - global_gpt4o: 1
         - global_sonnet: 1
         - Total LLM calls: 122 (50+50 extracts, 10+10 aggs, 1+1 globals)

WHEN   synix build
THEN   - Exit code 0
       - Source artifacts exist once (not duplicated)
       - Both extraction steps have 50 artifacts each
       - Both aggregation steps have 10 artifacts each
       - Both global models exist
       - extract_gpt4o artifacts differ in content from extract_sonnet
         (mock LLM returns different responses per model parameter)
       - synix verify passes (all provenance valid across both paths)

       Verify source sharing:
       - assert_steps_share_sources(store, "extract_gpt4o", "extract_sonnet")
       - Source artifact IDs referenced by both extraction steps are identical
       - Sources were NOT parsed/ingested twice (verify from run log)
```

### DT-2.4: Querying Parallel Search Indexes

```
GIVEN  Completed build from DT-2.3
WHEN   synix search "emotional triggers" --index search_main
THEN   - Returns results from gpt4o path artifacts only
       - No sonnet-path artifacts appear in results

WHEN   synix search "emotional triggers" --index search_experiment
THEN   - Returns results from sonnet path artifacts only
       - No gpt4o-path artifacts appear in results
       - Result count > search_main result count for same query
         (mock fixtures designed so sonnet captures more behavioral nuance)

       Verify independence:
       - assert_search_indexes_are_independent(
           search, "search_main", "search_experiment", "emotional triggers")
       - No cross-index contamination
```

### DT-2.5: Diffing Across Parallel Paths

```
GIVEN  Completed build from DT-2.3
WHEN   synix diff extract_gpt4o:customer_287 extract_sonnet:customer_287
       (diff two artifacts from parallel paths for the same source customer)
THEN   - Diff shows two distinct versions, labeled by step name
       - Both versions are non-empty and plausible
       - Content differs meaningfully (not just whitespace):
         - GPT-4o version: more structured/factual extraction
         - Sonnet version: more behavioral/linguistic nuance
       - Diff metadata includes: model used, prompt (same), source (same)
```

### DT-2.6: Incremental Update — New Conversations Rebuild in Both Paths

```
GIVEN  Completed build from DT-2.3
WHEN   3 new conversations are added for customer sarah_287
AND    synix plan
THEN   Plan shows:
         - Sources: 3 new, 50 cached
         - extract_gpt4o: 3 new extractions, 50 cached
         - extract_sonnet: 3 new extractions, 50 cached
         - aggregate_gpt4o: 1 rebuild (sarah_287 inputs changed), 9 cached
         - aggregate_sonnet: 1 rebuild (sarah_287 inputs changed), 9 cached
         - global_gpt4o: 1 rebuild (inputs changed)
         - global_sonnet: 1 rebuild (inputs changed)
         - Total LLM calls: 10 (3+3 extracts, 1+1 aggs, 1+1 globals)
         - NOT 122 (full rebuild)

WHEN   synix build
THEN   - count_llm_calls(log) == 10
       - count_cache_hits_for_step(log, "extract_gpt4o") == 50
       - count_cache_hits_for_step(log, "extract_sonnet") == 50
       - Both search indexes updated with new artifacts
       - DT-2.4 assertions still hold (indexes still independent)
```

---

## DT-3: "The Incident" Test Spec

**Test file:** `tests/e2e/test_demo_3_incident.py`

### DT-3.1: Initial Build with Merge Trap

```
GIVEN  Demo 3 corpus (100 conversations, 20 customers, clean state)
       Pipeline config:
         sources → extract → merge (similarity_threshold=0.85,
         NO customer_id constraint) → aggregate (group_by=customer_id)
         → global_model
         Single search output from [extract, aggregate]

WHEN   synix build
THEN   - Exit code 0
       - Merge step produces artifacts where some merge groups contain
         records from multiple customers (this is the bug, by design)
       - Specifically: merge artifact for "billing_issues_week_47" contains
         episodes from both conv_8834 (alice) and conv_8891 (bob)
       - Customer summary for alice_account contains information
         derived from bob's conversations (the contamination)
       - synix verify with standard checks PASSES
         (the merge is structurally valid, just semantically wrong —
          standard checks verify provenance integrity, not business logic)
```

### DT-3.2: Provenance Trace — Finding the Bad Merge

```
GIVEN  Completed build from DT-3.1
WHEN   synix search "billing dispute" --customer alice_account --trace
THEN   - Results include alice's customer summary
       - Provenance trace for this result contains:
         - aggregate:alice_account
           └─ merge:billing_issues_week_47 (merged from 2+ sources)
              ├─ extract:conv_8834 (alice)     ✓ correct source
              └─ extract:conv_8891 (bob)        ✗ WRONG CUSTOMER
       - The cross-customer merge is VISIBLE in the provenance chain
       - Bob's conversation ID appears in alice's provenance tree
```

### DT-3.3: Verify — Scoping the Blast Radius

```
GIVEN  Completed build from DT-3.1
WHEN   synix verify --check merge_integrity
       (custom check: flag merge artifacts containing records from
        multiple distinct customer_id values in their source provenance)
THEN   - Verify returns FAIL (exit code 1)
       - Exactly 3 merge artifacts flagged:
         - merge:billing_issues_week_47   (alice + bob)
         - merge:shipping_issues_week_51  (carol + dave)
         - merge:refund_requests_week_52  (eve + frank)
       - Exactly 6 customer IDs listed as affected
       - All other merge artifacts PASS (single-customer sources)
       - Output includes artifact IDs of all contaminated merge artifacts
       - Output includes customer IDs of all affected customers
       - Output is structured (parseable with --json flag)
```

### DT-3.4: Fix and Incremental Rebuild

```
GIVEN  Completed build from DT-3.1, verified failure from DT-3.3
WHEN   Merge config is changed:
         constraints: ["NEVER merge records with different customer_id"]
         similarity_threshold: 0.92 (was 0.85)
AND    synix plan
THEN   Plan shows:
         - Sources: 100 conversations (cached ✓)
         - Extract: all extractions (cached ✓)
         - Merge: rebuild (config changed — threshold + constraints
           are part of the materialization key, so all merge artifacts
           invalidated)
         - Aggregate: affected customer summaries rebuild (inputs changed)
         - Estimated cost << full rebuild cost
         - count_cache_hits_for_step in plan == 100 for extraction

WHEN   synix build
THEN   - Exit code 0
       - count_llm_calls_for_step(log, "extract") == 0 (all cached)
       - count_llm_calls_for_step(log, "merge") > 0 (merges rebuilt)
       - count_llm_calls_for_step(log, "aggregate") >= 6 (affected customers)
       - Total LLM calls < DT-3.1 total (extraction layer fully saved)

WHEN   synix verify --check merge_integrity
THEN   - Verify returns PASS (exit code 0)
       - All merge artifacts contain single-customer records only
       - Zero cross-customer contamination
```

### DT-3.5: Post-Fix Verification — No Collateral Damage

```
GIVEN  Completed rebuild from DT-3.4
WHEN   synix search "billing dispute" --customer alice_account --trace
THEN   - Alice's summary NO LONGER contains bob's information
       - Provenance trace shows only alice's episodes as sources
       - Bob's conversation ID does NOT appear in alice's provenance chain

WHEN   synix search "billing dispute" --customer bob_account --trace
THEN   - Bob's information is intact in his own customer summary
       - Bob's data was separated, not deleted
       - Bob has his own clean provenance chain

WHEN   synix verify (full verification, all standard checks)
THEN   - All checks pass
       - No orphaned artifacts
       - No broken provenance chains
       - Search index consistent with build output
       - No artifacts reference deleted or missing sources
```

### DT-3.6: Incident Timeline Reconstruction

```
GIVEN  Run logs from DT-3.1 build and DT-3.4 build
       Verify output from DT-3.3
       Provenance data from DT-3.2
THEN   The following can be reconstructed programmatically:

       - WHEN created: timestamp from DT-3.1 build log for
         merge:billing_issues_week_47
       - ROOT CAUSE: merge step, threshold=0.85, no customer_id constraint
         (from pipeline config at build time)
       - WHICH conversations: conv_8834 (alice), conv_8891 (bob)
         (from provenance trace)
       - WHY merged: both reference "BillingEngine" + "ERR-4401" same week
         (from merge artifact metadata / similarity score)
       - BLAST RADIUS: 3 merge artifacts, 6 customers
         (from verify output)
       - WHEN fixed: timestamp from DT-3.4 build log
       - WHAT changed: threshold 0.85→0.92, added customer_id constraint
         (from config diff between builds)
       - VERIFIED CLEAN: verify passes post-fix
         (from DT-3.4 verify output)

       Every field is extractable from structured data
       (logs, verify output, provenance, config).
       This is the machine-readable post-mortem.
```

---

## Test Execution Matrix

| Test ID | Mock LLM (CI) | Real LLM (Pre-Demo) | Depends On |
|---|---|---|---|
| DT-1.1 | ✓ every commit | ✓ before demo | — |
| DT-1.2 | ✓ | ✓ | DT-1.1 |
| DT-1.3 | ✓ | ✓ | DT-1.1 |
| DT-1.4 | ✓ | ✓ | DT-1.3 |
| DT-2.1 | ✓ | ✓ | — |
| DT-2.2 | ✓ | ✓ | DT-2.1 |
| DT-2.3 | ✓ | ✓ | — (fresh build) |
| DT-2.4 | ✓ | ✓ | DT-2.3 |
| DT-2.5 | ✓ | ✓ | DT-2.3 |
| DT-2.6 | ✓ | ✓ | DT-2.3 |
| DT-3.1 | ✓ | ✓ | — |
| DT-3.2 | ✓ | ✓ | DT-3.1 |
| DT-3.3 | ✓ | ✓ | DT-3.1 |
| DT-3.4 | ✓ | ✓ | DT-3.1 |
| DT-3.5 | ✓ | ✓ | DT-3.4 |
| DT-3.6 | ✓ | ✓ | DT-3.4 |

### CI Integration

```yaml
demo-scenario-tests:
  stage: e2e
  needs: [integration-tests]
  script:
    - export SYNIX_LLM_BASE_URL=http://localhost:9999/v1
    - python -m synix.tests.mock_server &
    - sleep 2
    - pytest tests/e2e/test_demo_1_personal.py -v
    - pytest tests/e2e/test_demo_2_startup.py -v
    - pytest tests/e2e/test_demo_3_incident.py -v

demo-validation-real-llm:
  stage: manual
  when: manual
  script:
    - export SYNIX_LLM_API_KEY=$OPENAI_API_KEY
    - pytest tests/e2e/test_demo_1_personal.py -v --real-llm
    - pytest tests/e2e/test_demo_2_startup.py -v --real-llm
    - pytest tests/e2e/test_demo_3_incident.py -v --real-llm
```

### Failure Modes Covered

| Failure Mode | Covered By | What We Assert |
|---|---|---|
| Cache skips record that should rebuild | DT-1.3, DT-2.2, DT-3.4 | Changed config → downstream rebuilds |
| Cache rebuilds record that shouldn't | DT-1.4, DT-2.2, DT-2.6 | Unchanged records stay cached |
| Parallel paths contaminate each other | DT-2.3, DT-2.4 | Indexes independent, no cross-path artifacts |
| Parallel paths duplicate shared sources | DT-2.3 | Sources exist once, both paths reference same IDs |
| Incremental build misses a parallel path | DT-2.6 | New conversations rebuild in BOTH paths |
| Provenance chain broken after rebuild | DT-3.5 | Verify passes, trace shows valid chain |
| Search returns stale results after rebuild | DT-1.3, DT-3.5 | Search reflects new artifacts, not old |
| Merge combines cross-customer data | DT-3.1 | By design — verify catches it |
| Fix introduces new problems | DT-3.5 | Full verify after fix, bob's data intact |
| Fix loses unrelated data | DT-3.5 | Bob's own summary still correct |
| Diff empty when artifacts changed | DT-1.3, DT-2.2, DT-2.5 | Diff non-empty after rebuild |
| Plan estimates wildly wrong | DT-1.1, DT-2.2 | Compare plan call count to actual |
| Model parameter doesn't affect output | DT-2.3, DT-2.5 | gpt4o and sonnet artifacts differ |
| Verify misses contamination | DT-3.3 | Exactly 3 flagged, exactly 6 customers |
| Verify false-positives on clean merges | DT-3.4 | Post-fix verify passes, no false flags |

### Test Corpus Design Requirements

The test corpora are the foundation of these tests. They must be hand-authored
(not generated) and version-controlled. Key design constraints:

**Demo 1 corpus must have:**
- At least 4 clearly distinct topics (for topical clustering)
- At least 1 topic spanning 3+ months (for cross-month topical aggregation)
- At least 2 multi-topic conversations (for episode splitting)
- At least 1 conversation with 2–3 turns and 1 with 40+ turns (edge cases)

**Demo 2 corpus must have:**
- Customer sarah_287 with 4 conversations showing clear behavioral evolution
- At least 2 customers with only 1 conversation (minimal data edge case)
- Content that produces meaningfully different extractions per model
  (mock LLM uses model parameter to select fixture set)
- Financial data mixed with emotional/behavioral language

**Demo 3 corpus must have:**
- Exactly 3 customer pairs with cross-customer semantic overlap:
  - alice + bob: same product ("BillingEngine"), same error ("ERR-4401"), same week
  - carol + dave: same product, same issue type, same week
  - eve + frank: same product, same refund reason, same week
- Overlap must trigger merge at threshold 0.85 but NOT at 0.92
- All other customer pairs: similarity well below 0.85 (no accidental merges)
- Each trapped customer also has clean conversations unaffected by the fix
  (to verify no collateral damage in DT-3.5)
