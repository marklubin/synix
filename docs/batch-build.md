# Batch Build — OpenAI Batch API Support

> **[Experimental]** This feature may change in future releases.

## Overview

LLM calls dominate Synix build cost. OpenAI's [Batch API](https://platform.openai.com/docs/guides/batch) offers **50% cheaper** inference with async processing (typically completes in minutes, 24h SLA). `synix batch-build` uses Batch API for eligible layers while keeping the existing `synix build` completely unchanged.

The only new core primitive is a `BatchLLMClient` — a drop-in replacement for `LLMClient`. Transforms don't change at all. When a transform calls `client.complete()`, the batch client intercepts: it queues the request, raises a control-flow exception, and the runner handles the batch lifecycle. On resume, the same transform re-executes, the client returns the cached batch result, and artifacts are created normally.

**Transform contract for batch mode:** Transforms are already required to be stateless during execution (see `Transform` docstring in `core/models.py`). Batch mode adds one additional requirement: `execute()` must be **idempotent with respect to the LLM call**. Specifically, given the same inputs and config, `execute()` must produce the same `client.complete()` arguments. This is already true for all built-in transforms (they derive prompts deterministically from inputs). Custom transforms that have non-deterministic prompt construction (e.g., timestamp injection, random sampling) will produce different request keys on resume, causing cache misses and duplicate batch submissions. If your transform has side effects beyond returning artifacts (e.g., writing to external systems, sending notifications), guard them with a check against the batch state or make them idempotent.

## When to Use

Use `batch-build` when:

- Your pipeline has OpenAI layers with many artifacts (episodes, rollups)
- You want to cut LLM cost by 50%
- You can tolerate async wait times (minutes to hours, 24h SLA)

Don't use `batch-build` when:

- All your layers use Anthropic (not supported by OpenAI Batch API)
- You need results immediately
- You're iterating on prompts and need fast feedback loops

## Architecture

### Build Instances

Each `batch-build` creates a **named build instance** with its own ID and lifecycle. This is the on-ramp to snapshot builds — each instance tracks which layers are complete, which batches are in flight, and which results have been collected.

```
<build_dir>/builds/<build_id>/
    manifest.json      # BuildInstance metadata (status, layers completed)
    batch_state.json   # Requests, batches, results, errors
```

Artifacts live in the shared cache (existing `ArtifactStore`). The build instance only tracks batch lifecycle state.

### Request Flow

```
Transform.execute()
    └── client.complete()
         ├── Result cached? → return LLMResponse (normal path)
         ├── Batch complete? → download, cache, return LLMResponse
         ├── Batch in-progress? → raise BatchInProgress
         └── New request? → queue, raise BatchCollecting
```

1. **Collect phase**: Runner walks the DAG. For each batchable layer, it calls `split()` then `execute()` per unit. The `BatchLLMClient` queues each request and raises `BatchCollecting`. After all units are attempted, queued requests are submitted as an OpenAI batch.

2. **Wait phase**: Either poll (`--poll`) or exit with resume instructions.

3. **Resume phase**: Runner re-walks the DAG. Completed layers are skipped. For in-progress layers, the batch client checks batch status, downloads results, and returns them through the normal `complete()` → `LLMResponse` path. Transforms execute identically — they don't know they're in batch mode.

### Batch Mode Resolution

Each transform has an optional `batch` parameter:

| `batch` value | Provider is OpenAI | Behavior |
|---|---|---|
| `None` (default) | Yes, and `split_count > 1` | Auto-batch |
| `None` (default) | No, or single unit | Sync |
| `True` | Yes | Force batch |
| `True` | No | **Hard error** (`ValueError`) |
| `False` | Any | Force sync |

Layers that run sync in a batch-build execute identically to `synix build` — same code path, same runner logic.

### Mixed Providers

A pipeline can mix Anthropic and OpenAI layers. Only OpenAI layers participate in batching; Anthropic layers run synchronously as normal:

```python
episodes = EpisodeSummary("episodes", depends_on=[transcripts])  # Uses pipeline default (Anthropic) → sync
monthly = MonthlyRollup("monthly", depends_on=[episodes],
    config={"llm_config": {"provider": "openai", "model": "gpt-4o-mini"}})  # → batched
```

## CLI Commands

All commands are under the `batch-build` group. Every invocation prints an experimental warning.

### `synix batch-build run`

Create a build instance and submit the first batch.

```bash
uvx synix batch-build run pipeline.py              # Submit and exit
uvx synix batch-build run pipeline.py --poll        # Submit and wait
uvx synix batch-build run pipeline.py --poll --poll-interval 30
```

Options:
- `--poll`: Stay alive and poll until the build completes
- `--poll-interval N`: Seconds between status checks (default: 60)

### `synix batch-build resume`

Resume an existing build instance. Checks batch status, downloads results, submits next layer's batch if needed.

```bash
uvx synix batch-build resume <build-id>                          # Check and advance
uvx synix batch-build resume <build-id> --poll                   # Check and wait
uvx synix batch-build resume <build-id> --allow-pipeline-mismatch  # Resume despite pipeline changes
uvx synix batch-build resume <build-id> --reset-state            # Restart layer after state corruption
```

Options:
- `--poll`: Stay alive and poll until the build completes
- `--allow-pipeline-mismatch`: Resume even if the pipeline fingerprint has changed since submission (dangerous — can mix artifacts from incompatible pipeline definitions)
- `--reset-state`: Acknowledge corrupted state and restart the current layer from scratch (discards pending/in-flight requests for the corrupted layer)

### `synix batch-build list`

Show all build instances and their status.

```bash
uvx synix batch-build list
```

Output (Rich table):

```
Build ID     Status      Pipeline Hash  Created          Layers Done  Current
batch-a1b2   submitted   fp:3e8a...     2026-02-16 10:30  1/4         episodes
batch-c3d4   completed   fp:3e8a...     2026-02-15 14:00  4/4         —
```

### `synix batch-build status`

Detailed status for a specific build instance.

```bash
uvx synix batch-build status <build-id>
```

Output: per-layer breakdown, per-batch status, request counts, errors.

### `synix batch-build plan`

Dry-run showing which layers would batch vs sync, estimated batch count.

```bash
uvx synix batch-build plan pipeline.py
```

## Pipeline API

### `batch` Parameter on Transform

```python
from synix.transforms import EpisodeSummary, MonthlyRollup, CoreSynthesis

# Auto (default) — batches if OpenAI and multiple work units
episodes = EpisodeSummary("episodes", depends_on=[transcripts])

# Force batch — error if not OpenAI
monthly = MonthlyRollup("monthly", depends_on=[episodes], batch=True)

# Force sync — never batch, even if OpenAI
core = CoreSynthesis("core", depends_on=[monthly], batch=False)
```

The `batch` parameter is `None` by default (auto-detect). It has no effect on `synix build` — only `synix batch-build` reads it.

## Error Handling

### Hard Errors (fail fast)

| Condition | Error |
|---|---|
| `batch-build` with zero batchable OpenAI layers | `UsageError` |
| `batch=True` on non-OpenAI layer | `ValueError` at validation |
| Resume with unknown `build_id` | `UsageError` |
| Resume with changed pipeline (fingerprint mismatch) | `UsageError` (override with `--allow-pipeline-mismatch`) |
| `OPENAI_API_KEY` not set for batchable layers | `ValueError` |
| `SYNIX_CASSETTE_MODE` set to `replay` | `UsageError` (incompatible — batch client manages its own replay via `batch_state.json`) |

### Recoverable

| Condition | Behavior |
|---|---|
| Batch expired (24h SLA) | Re-queue failed requests, submit new batch, log warning |
| Individual request failed in batch | Mark request as failed, report per-artifact error, continue with others (see Build Completion Semantics below) |
| Network error checking batch | If polling, retry next interval; if not, save state and exit |
| Corrupted `batch_state.json` | Quarantine corrupted file, set build to `failed`, require `--reset-state` to restart (see State Persistence) |

### Edge Cases

| Condition | Behavior |
|---|---|
| Pipeline changed between submit/resume | **Hard error** by default (fingerprint mismatch). Override with `--allow-pipeline-mismatch` to resume anyway. |
| All layers cached | "Nothing to build", no batch submission |
| Empty layer (no inputs) | Skip |

Pipeline fingerprint mismatch is a hard error because resuming a build with a different pipeline definition can silently mix artifacts from incompatible prompts, models, or transform logic. The `--allow-pipeline-mismatch` flag on `resume` opts into this explicitly.

## Build Completion Semantics

A build instance has the following terminal states:

| Status | Meaning |
|---|---|
| `completed` | All layers finished, all requests succeeded, all artifacts created |
| `completed_with_errors` | All layers attempted, but some requests failed (artifacts missing) |
| `failed` | Unrecoverable error (corrupted state, validation failure, etc.) |

**Per-request failure policy:** When individual requests fail within a completed batch (e.g., content filter, context length exceeded):

1. The failed request is recorded in `errors` with its error code and message.
2. The corresponding artifact is **not created** — there is no silent fallback or placeholder.
3. The layer continues processing all other requests normally.
4. The build status becomes `completed_with_errors` (not `completed`).

**Downstream layer behavior:** When a layer has failed requests, downstream layers that depend on it receive only the successfully-produced artifacts as inputs. This means:

- A 1:1 transform (e.g., `EpisodeSummary`) with 2 of 50 failures produces 48 episodes. Downstream `MonthlyRollup` processes those 48.
- An N:1 transform (e.g., `CoreSynthesis`) receives whatever is available.

The `status` command shows per-layer error counts. The `list` command shows the terminal status. This makes partial failures visible without blocking the entire pipeline.

## State Persistence

### `manifest.json`

```json
{
  "build_id": "batch-a1b2c3d4",
  "pipeline_hash": "fp:3e8a...",
  "status": "submitted",
  "created_at": 1708099200.0,
  "layers_completed": ["transcripts"],
  "current_layer": "episodes",
  "failed_requests": 0,
  "error": null
}
```

Valid `status` values: `pending`, `collecting`, `submitted`, `completed`, `completed_with_errors`, `failed`.

### `batch_state.json`

```json
{
  "pending": {
    "<request_key>": {"layer": "episodes", "body": {...}, "desc": "episode ep-conv-123"}
  },
  "batch_map": {
    "<request_key>": "<batch_id>"
  },
  "batches": {
    "<batch_id>": {"layer": "episodes", "keys": ["<key1>", "<key2>"], "status": "completed"}
  },
  "results": {
    "<request_key>": {"content": "...", "model": "gpt-4o-mini", "tokens": {"input": 500, "output": 200}}
  },
  "errors": {
    "<request_key>": {"code": "content_filter", "message": "Content filtered"}
  }
}
```

All writes use `atomic_write()` for crash safety.

**Corruption recovery:** If `batch_state.json` is corrupted (invalid JSON, truncated write), the system does **not** silently start from empty. Instead:

1. The corrupted file is quarantined to `batch_state.json.corrupt` with a timestamp suffix.
2. The build status is set to `failed` with an error message describing the corruption.
3. The `resume` command refuses to continue until the user either:
   - Fixes or restores the state file manually, or
   - Runs `resume --reset-state` to acknowledge data loss and restart the current layer from scratch.

This prevents orphaned in-flight batches and duplicate submissions. The `manifest.json` (which tracks high-level build status and is much smaller) uses the same quarantine behavior.

## Request Keying

Batch request keys reuse `compute_cassette_key()` from the cassette system. The key is a SHA256 over the **complete set of output-affecting parameters**:

- `provider` — which API backend
- `model` — model identifier
- `messages` — canonicalized for hashing (JSON dict keys sorted, trailing whitespace on string values stripped, `\r\n` → `\n`). Message content bytes are preserved — this is serialization normalization, not content mutation.
- `max_tokens` — max output length
- `temperature` — sampling temperature

**Scope limitation:** The current key covers all parameters that Synix's `LLMClient.complete()` accepts. Parameters like `top_p`, `seed`, `response_format`, `tools`, and `frequency_penalty` are not part of the `complete()` interface and therefore not in the key. If `LLMClient.complete()` is extended with new output-affecting parameters in the future, `compute_cassette_key()` must be updated to include them — both for cassette correctness and batch correctness. This is a single point of change (one function).

Properties:
- Identical requests produce identical keys (deduplication)
- A cassette-recorded response and a batch response for the same request have the same key
- The key is deterministic across sessions

## Design Decisions

### Sync-only layers in batch-build

`batch-build run` processes all layers in DAG order. Source layers always run synchronously. Transform layers that resolve to sync (Anthropic provider, `batch=False`, single work unit) also run synchronously inline. The command only exits to await a batch when it encounters the first batchable layer with pending requests. This means a pipeline with sync-only layers at the top runs those immediately — there's no wasted round-trip.

If the pipeline has **zero** batchable layers, `batch-build run` fails immediately with a `UsageError` rather than doing a full sync build (use `synix build` for that).

### Permanent request failures and downstream layers

When requests fail permanently (content filter, validation error), the failed artifacts are simply absent from that layer's output. Downstream layers receive only successfully-produced artifacts as inputs. This follows Synix's existing pattern — transforms already handle variable-length inputs (e.g., `MonthlyRollup` groups whatever episodes exist for a month).

No special "partial input" or "structured error" objects are injected into downstream transforms. The errors are recorded in `batch_state.json` and visible via `status`, but they don't propagate as data through the DAG. This keeps the transform interface clean — `execute()` never receives error sentinels.

## Demo Template: `05-batch-build`

A self-contained demo that exercises the full batch-build lifecycle using the existing team-report pipeline sources (bios + project brief), but with an OpenAI layer for the 1:1 work-style transform. This is both a user-facing template and the source of truth for e2e cassette replay tests.

### Why a Separate Template

The existing demos (01-04) all use Anthropic. This demo needs at least one OpenAI layer to exercise batch mode. Rather than retrofitting an existing template, a dedicated `05-batch-build` template keeps the batch-specific concerns isolated and gives users a clear example of how to set up batch builds.

### Pipeline Design

```
Level 0: bios [parse]              → 3 artifacts (reuses 03-team-report sources)
Level 1: work_styles [WorkStyle]   → 3 artifacts, OpenAI gpt-4o-mini, batch=True
Level 2: summary [TeamSummary]     → 1 artifact, Anthropic (sync, batch=False)
```

Key properties:
- **Mixed providers**: Level 1 is OpenAI (batched), Level 2 is Anthropic (sync)
- **batch=True explicit**: Forces batch mode on the OpenAI layer
- **batch=False explicit**: Forces sync on the Anthropic layer
- **Small enough to record**: 3 OpenAI batch requests + 1 Anthropic sync call

### Directory Layout

```
templates/05-batch-build/
├── README.md              # User-facing instructions
├── pipeline.py            # Pipeline definition (mixed providers)
├── sources/
│   └── bios/              # Symlink or copy from 03-team-report
│       ├── alice.md
│       ├── bob.md
│       └── carol.md
├── cassettes/
│   ├── llm.yaml           # Anthropic cassette (sync layer)
│   └── batch_responses.json  # Recorded OpenAI Batch API responses
├── golden/
│   ├── plan.stdout.txt
│   ├── batch_run.stdout.txt
│   ├── batch_status.stdout.txt
│   ├── batch_resume.stdout.txt
│   └── list.stdout.txt
└── case.py                # Demo case definition
```

### Cassette Recording Strategy

The batch-build flow involves two distinct kinds of LLM calls that need recording:

1. **Sync Anthropic calls** (Level 2 summary): Use the existing cassette system (`SYNIX_CASSETTE_MODE=record`). These are identical to how other demos work.

2. **OpenAI Batch API calls** (Level 1 work styles): The Batch API is async — you upload a JSONL file, poll for completion, then download results. We can't use the standard cassette system for this because the batch client talks to OpenAI's file/batch endpoints, not the chat completions endpoint.

   **Recording approach**: Run `batch-build` once against the real OpenAI Batch API. The `BatchLLMClient` records:
   - The JSONL request file it would upload (deterministic from inputs)
   - The batch ID returned by OpenAI
   - The JSONL response file downloaded on completion

   These are saved to `cassettes/batch_responses.json` as a map from request key to response. On replay, the `BatchLLMClient` checks this file before hitting OpenAI — same pattern as the cassette store, just for batch results.

   **In practice**: Since `BatchLLMClient` stores results in `batch_state.json` during a real run, and on resume it checks `batch_state.get_result(key)` before anything else, we can seed the cassette by copying the results section from a completed `batch_state.json`. The replay path never touches OpenAI at all.

### Case Definition (`case.py`)

```python
case = {
    "name": "batch_build",
    "pipeline": "pipeline.py",
    "steps": [
        # Step 1: Plan — show batch sequencing
        {"name": "note_plan", "command": ["synix", "demo", "note", "1/5 Planning batch build..."]},
        {"name": "plan", "command": ["synix", "batch-build", "plan", "PIPELINE"]},

        # Step 2: Run — submit batch (mocked: results pre-seeded)
        {"name": "note_run", "command": ["synix", "demo", "note", "2/5 Submitting batch..."]},
        {"name": "batch_run", "command": ["synix", "batch-build", "run", "PIPELINE", "--poll"]},

        # Step 3: List — show build instances
        {"name": "note_list", "command": ["synix", "demo", "note", "3/5 Listing builds..."]},
        {"name": "list", "command": ["synix", "batch-build", "list"]},

        # Step 4: Status — detailed per-batch info
        {"name": "note_status", "command": ["synix", "demo", "note", "4/5 Build status..."]},
        {"name": "batch_status", "command": ["synix", "batch-build", "status", "BATCH_ID"]},

        # Step 5: Verify artifacts
        {"name": "note_verify", "command": ["synix", "demo", "note", "5/5 Verifying artifacts..."]},
        {"name": "verify", "command": ["synix", "list"]},
    ],
    "goldens": {},
}
```

> Note: `BATCH_ID` placeholder will need special handling in the demo runner (resolve from the `builds/` directory), or the case.py can shell out to discover it.

### E2E Test (`tests/e2e/test_batch_build.py`)

The e2e test does **not** use the demo runner. It directly exercises the batch-build CLI commands with mocked OpenAI:

```python
# Mock strategy: monkeypatch openai.OpenAI to return controlled responses
# - files.create() → returns mock file ID
# - batches.create() → returns mock batch with status "completed"
# - batches.retrieve() → returns completed batch
# - files.content() → returns JSONL with pre-recorded responses
```

Test cases:
- Full flow: `run --poll` → completes, artifacts created
- Resume flow: `run` (exits with pending) → `resume --poll` (completes)
- `list` and `status` show correct info
- `plan` shows batch sequencing
- No OpenAI layers → hard error
- Mixed providers: OpenAI batches, Anthropic runs sync
- `batch=False` layer runs sync even with OpenAI
- All cached → "nothing to build"
- Experimental warning displayed on every command
- Fingerprint mismatch on resume → hard error without `--allow-pipeline-mismatch`
- Fingerprint mismatch with `--allow-pipeline-mismatch` → proceeds with warning
- Corrupted `batch_state.json` → quarantine file, set status to `failed`
- Corrupted state with `--reset-state` → restarts current layer
- Corrupted state without `--reset-state` → refuses to continue
- Partial request failures → `completed_with_errors` status, downstream receives partial input
- Resume is idempotent (repeated resume doesn't duplicate artifacts or batch submissions)
- `SYNIX_CASSETTE_MODE=replay` with `batch-build` → hard error
- `SYNIX_CASSETTE_MODE=record` with `synix build` (not `batch-build`) → allowed (recording workflow)
- Request key correctness: distinct keys when any output-affecting param changes (model, temperature, messages, max_tokens)

### Recording the Initial Cassettes

One-time manual step (requires both `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`):

```bash
cd templates/05-batch-build

# Step 1: Record Anthropic cassette for the sync layer (standard synix build)
export ANTHROPIC_API_KEY=sk-ant-...
export SYNIX_CASSETTE_MODE=record
export SYNIX_CASSETTE_DIR=./cassettes
uvx synix build pipeline.py   # records llm.yaml for the Anthropic call
unset SYNIX_CASSETTE_MODE SYNIX_CASSETTE_DIR

# Step 2: Run batch-build against real OpenAI Batch API
# (cassette mode must NOT be set — batch-build rejects SYNIX_CASSETTE_MODE=replay,
#  and batch results are captured via batch_state.json, not the cassette system)
export OPENAI_API_KEY=sk-...
uvx synix batch-build run pipeline.py --poll

# Step 3: Seed batch cassette from completed build state
# Copy the "results" section from build/builds/<id>/batch_state.json
# into cassettes/batch_responses.json
python -c "
import json
from pathlib import Path
builds = sorted(Path('build/builds').iterdir())[-1]
state = json.loads((builds / 'batch_state.json').read_text())
Path('cassettes/batch_responses.json').write_text(
    json.dumps(state['results'], indent=2))
"

# Step 4: Clean build dir and generate golden outputs
rm -rf build/
uvx synix demo run . --update-goldens

# Step 5: Verify replay works (no API keys needed)
rm -rf build/
uvx synix demo run .
```

After this, CI runs the demo in replay mode with no API keys needed. The cassette system handles the Anthropic call; the batch client finds pre-seeded results in `cassettes/batch_responses.json` and never contacts OpenAI.

## What Doesn't Change

- `synix build` — completely untouched, same code path
- All existing transforms — zero interface changes
- `ArtifactStore` — shared cache, content-addressed, no changes
- `synix clean` — already deletes entire `build_dir` which includes `builds/`
- Cassette system — `compute_cassette_key()` reused as-is, not modified
