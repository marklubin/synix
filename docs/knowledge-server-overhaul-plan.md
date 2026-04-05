# Synix Knowledge Server Overhaul

## Context

The deployed server at `/srv/synix/project/` uses Anthropic Haiku (expensive, cloud-dependent), a polling auto-builder with no per-document tracking, a viewer with rendering bugs (`[Object object]` metadata, flat chunk display), hardcoded inline prompts, and no build progress visibility.

This plan: replaces inference with local vLLM (Qwen3.5-9B-AWQ, RTX 3060), replaces polling with an event-driven SQLite queue with durable status tracking, adds a standalone prompt management system (colocated but independent of synix core), fixes viewer bugs systematically, adds build status + pipeline DAG + prompt editing tabs.

---

## Architecture

```
  ingest (MCP/REST)                    synix serve
  ─ doc + hash + client_id ──►  ┌────────────────────────────────────┐
       ◄── doc_id ────────────  │                                    │
                                │  DocumentQueue (.synix/queue.db)   │
                                │    │                               │
                                │    ▼  window trigger (N sec)       │
                                │  BuildWorker                       │
                                │    acquire lock → claim batch →    │
                                │    project.build() →               │
                                │    release_to("local") →           │
                                │    mark status (durable) →         │
                                │    release lock                    │
                                │         │                          │
                                │         ▼                          │
                                │    vLLM (localhost:8100)           │
                                │    Qwen3.5-9B-AWQ, thinking=off   │
                                │                                    │
  viewer (:9471)                │                                    │
  ◄─ /api/build-status ────────│  queue.db (read-only)              │
  ◄─ /api/dag ─────────────────│  pipeline introspection            │
  ◄─ /api/prompts/* ───────────│  PromptStore (standalone .db)      │
                                └────────────────────────────────────┘
```

**Key decisions:**
- **Queue is durable coordination layer** — all status transitions persisted to SQLite, builds are idempotent (re-runnable on crash), status visible to viewer at all times
- **Fixed-window batching** — first new doc starts an N-second timer; all docs arriving in that window form one batch; any arriving after are blocked until the current build releases its lock
- **Prompts are a standalone subsystem** — `PromptStore` is independent of synix core; colocated in `.synix/prompts.db` but does NOT modify `Transform.load_prompt()` or any core models
- **Client tracking** — document queue records `client_id` (e.g. `Claude@Salinas`) passed via MCP param or REST header
- **vLLM thinking disabled at inference level** — `--default-chat-template-kwargs '{"enable_thinking": false}'`, verified with tok/s measurement
- **Backup before wipe** — `tar.gz` of `.synix/` before clearing for rebuild

---

## Stream A: SQLite Document Queue

### A1: `src/synix/server/queue.py` (new)

Two tables, WAL mode:

```sql
document_queue(
  doc_id TEXT PRIMARY KEY,       -- uuid4
  bucket TEXT NOT NULL,
  filename TEXT NOT NULL,
  content_hash TEXT NOT NULL,    -- sha256 dedup
  file_path TEXT NOT NULL,
  client_id TEXT,                -- e.g. "Claude@Salinas", from MCP param or REST header
  status TEXT DEFAULT 'pending', -- pending|processing|built|released|failed
  error_message TEXT,
  created_at TEXT NOT NULL,      -- ISO 8601
  processing_started_at TEXT,
  built_at TEXT,
  released_at TEXT,
  build_run_id TEXT,
  UNIQUE(content_hash)           -- dedup constraint
)

build_runs(
  run_id TEXT PRIMARY KEY,
  status TEXT DEFAULT 'pending', -- pending|running|completed|failed
  started_at TEXT, completed_at TEXT,
  documents_count INTEGER DEFAULT 0,
  built_count INTEGER DEFAULT 0,
  cached_count INTEGER DEFAULT 0,
  error_message TEXT
)
```

Class `DocumentQueue(db_path)`:
- `enqueue(bucket, filename, content_hash, file_path, client_id=None) -> doc_id` — dedup: returns existing doc_id if content_hash already built/released; records client_id
- `pending_count() -> int`
- `last_enqueue_time() -> float | None` — for window trigger
- `claim_pending_batch(run_id) -> list[doc_id]` — atomic `UPDATE ... SET status='processing', build_run_id=? WHERE status='pending'`
- `mark_built(run_id, built_count, cached_count)` — updates build_runs and doc statuses
- `mark_released(run_id)` — final status transition
- `mark_failed(run_id, error)` — sets error_message, transitions docs back to `pending` for retry
- `document_status(doc_id) -> dict` — includes queue position, client_id, timestamps
- `recent_history(limit=50) -> list[dict]`
- `queue_stats() -> dict` — total processed, avg build time, etc.

**Durability/idempotency guarantees:**
- All state transitions in single SQLite transactions
- `mark_failed` resets docs to `pending` so they're retried on next run
- `claim_pending_batch` is atomic (no doc claimed by two runs)
- WAL mode allows concurrent viewer reads during worker writes

**Tests:** `tests/unit/test_queue.py` — all methods, dedup, state transitions, crash recovery (mark_failed re-queues), concurrent access

### A2: Config update — `src/synix/server/config.py`

Replace `AutoBuildConfig` (lines 20-26) with `BuildQueueConfig`:
```python
@dataclass
class BuildQueueConfig:
    enabled: bool = True
    window: int = 30   # seconds — first doc starts timer, batch all within window
```

Update `ServerConfig.auto_build` field type. Parse from `[auto_build]` TOML section (backward-compatible: `enabled` carries over; `scan_interval`/`cooldown` ignored, `window` is new).

**Tests:** Update `tests/unit/test_server_config.py`

### A3: Queue-driven build worker — `src/synix/server/serve.py`

Replace `run_auto_builder()` (lines 53-153) with `run_build_worker(config, queue)`:

```
loop:
  sleep 2s
  if queue.pending_count() == 0: continue
  if window not started:
    window_start = now
    continue
  if now - window_start < config.auto_build.window: continue
  # Window expired — build
  run_id = uuid4()
  claim_pending_batch(run_id)
  acquire build lock (threading.Lock)
  try:
    project.build(accept_existing=True)
    mark_built(run_id, result.built, result.cached)
    project.release_to("local")
    mark_released(run_id)
  except:
    mark_failed(run_id, error)  # docs go back to pending
  finally:
    release lock
    window_start = None
```

The lock prevents concurrent builds. Docs arriving during a build accumulate in `pending` and trigger the next window after lock release.

In `serve()` (line 187+): init `DocumentQueue(.synix/queue.db)`, store in `_state["queue"]`, create build lock, pass to worker.

### A4: Ingestion integration

**`src/synix/server/mcp_tools.py`:**
- `ingest(bucket, content, filename, client_id=None)` — add optional `client_id` param, compute `content_hash = hashlib.sha256(content.encode()).hexdigest()`, call `queue.enqueue(...)`, return `doc_id` in response string
- New tool: `document_status(doc_id) -> str` — returns JSON-like status including queue position

**`src/synix/server/api.py`:**
- `ingest_to_bucket()` — read `client_id` from `X-Client-Id` header or JSON body, include `doc_id` in response
- New route: `GET /api/v1/document/{doc_id}` — returns full document status

**Tests:** `tests/e2e/test_queue_integration.py` — ingest → queue → verify pending → (mock build) → verify released

---

## Stream B: vLLM Integration

### B1: `src/synix/server/vllm_manager.py` (new)

```python
@dataclass
class VLLMConfig:
    enabled: bool = False
    model: str = "QuantTrio/Qwen3.5-9B-AWQ"
    gpu_device: int = 0
    port: int = 8100
    max_model_len: int = 2048      # conservative for 12GB VRAM
    gpu_memory_utilization: float = 0.85
    extra_args: list[str] = field(default_factory=list)
    startup_timeout: int = 120

class VLLMManager:
    async def start(self) -> None
    async def health_check(self) -> bool
    async def stop(self) -> None       # SIGTERM → wait 10s → SIGKILL
    async def measure_throughput(self) -> dict  # tok/s verification
    @property
    def base_url(self) -> str          # http://localhost:{port}/v1
```

**Launch command constructed by `start()`:**
```bash
CUDA_VISIBLE_DEVICES={gpu_device} CUDA_DEVICE_ORDER=PCI_BUS_ID \
vllm serve {model} \
  --port {port} \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization {gpu_memory_utilization} \
  --max-model-len {max_model_len} \
  --quantization awq \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --enable-prefix-caching \
  {extra_args}
```

**Critical: Thinking mode is disabled at inference level** via `--default-chat-template-kwargs '{"enable_thinking": false}'` — this tells the Qwen3.5 chat template to not enter thinking mode, not just strip output.

**`measure_throughput()`**: After startup, send a short test prompt and measure output tok/s. Log result. Expected: 60-150 tok/s on RTX 3060 with AWQ quantization.

**Tests:** `tests/unit/test_vllm_manager.py` — lifecycle with mocked subprocess, health check

### B2: Config — `src/synix/server/config.py`

Add `VLLMConfig` dataclass, parse `[vllm]` TOML section, add `vllm: VLLMConfig` field to `ServerConfig` (default: `enabled=False`).

### B3: Server orchestration — `src/synix/server/serve.py`

In `serve()`:
1. If `config.vllm.enabled`: create `VLLMManager`, `await manager.start()`, `await manager.measure_throughput()` (log tok/s)
2. Override pipeline llm_config: `{"provider": "openai-compatible", "model": config.vllm.model, "base_url": manager.base_url, "api_key": "not-needed"}`
3. Store in `_state["llm_config_override"]` — build worker applies before each build
4. Shutdown handler: `await manager.stop()`

### B4: Deploy config — `/srv/synix/project/synix-server.toml`

```toml
[vllm]
enabled = true
model = "QuantTrio/Qwen3.5-9B-AWQ"
gpu_device = 0
port = 8100
max_model_len = 2048
gpu_memory_utilization = 0.85
```

---

## Stream C: Prompt Management (Standalone Subsystem)

The prompt store is an **independent subsystem** colocated in `.synix/`. It does NOT modify `Transform.load_prompt()` or any synix core code. The server manages it; the viewer exposes a UI for it; pipelines can optionally use it.

### C1: `src/synix/server/prompt_store.py` (new)

SQLite DB at `{project_dir}/.synix/prompts.db`:
```sql
prompts(
  key TEXT NOT NULL,
  version INTEGER NOT NULL,
  content TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (key, version)
)
```

Class `PromptStore(db_path)`:
- `get(key, version=None) -> str | None` — latest if version omitted
- `get_with_meta(key, version=None) -> dict | None`
- `put(key, content) -> dict` — no-op if content_hash matches latest
- `list_keys() -> list[str]`
- `history(key) -> list[dict]`
- `seed_from_files(prompts_dir) -> int` — import `.txt` files as v1, skip existing

**Tests:** `tests/unit/test_prompt_store.py` — CRUD, versioning, dedup, seed

### C2: Server integration

**Server startup in `serve()`:** Create `PromptStore(.synix/prompts.db)`, seed from `PROMPTS_DIR`, store in `_state["prompt_store"]`.

**No core synix modifications.** The prompt store is accessed only through server/viewer APIs.

### C3: Prompt CRUD API

**`src/synix/server/mcp_tools.py`:** New tools: `list_prompts`, `get_prompt`, `update_prompt`, `prompt_history`

**`src/synix/server/api.py`:** Routes:
- `GET /api/v1/prompts` — list keys
- `GET /api/v1/prompts/{key}` — get latest content
- `PUT /api/v1/prompts/{key}` — update (creates new version)
- `GET /api/v1/prompts/{key}/history` — version history

### C4: Viewer prompt tab

**`src/synix/viewer/server.py`:** Flask routes proxying to `.synix/prompts.db`

**`src/synix/viewer/static/app.js`:** Prompts tab — list of keys with version count, textarea editor with save, version history sidebar

---

## Stream D: Build Status + Pipeline DAG Visualization

### D1: Build status API — `src/synix/viewer/server.py`

`GET /api/build-status` — queue depth, active build, recent history, stats

### D2: Pipeline DAG API — `src/synix/viewer/server.py`

`GET /api/dag` — nodes (layers with type/level/count), edges (depends_on), projections

### D3: Viewer Pipeline tab — `src/synix/viewer/static/app.js`

DAG visualization (Canvas/SVG), build queue panel, recent completions/failures, auto-refresh 3s

---

## Stream E: Viewer Bug Fixes

### E1: Systematic metadata value formatting — `src/synix/viewer/static/app.js`

Replace `String(v)` with `formatMetaValue(v)` that handles objects/arrays. Skip `_`-prefixed keys.

### E2: Group chunks by source document — `src/synix/viewer/static/app.js`

Detect chunk layers, group by `source_label`, collapsible section headers.

---

## Stream F: Tab Infrastructure

### F1: Tab navigation — `index.html` + `app.js` + `style.css`

Tab bar: Browse / Pipeline / Prompts. Tab switching logic, show/hide panels.

---

## Stream G: Backup + Rebuild

1. `tar czf .synix-backup-$(date +%Y%m%d-%H%M%S).tar.gz .synix/`
2. Clear build artifacts (keep queue.db, prompts.db)
3. Seed prompt store, restart server
4. Monitor rebuild via Pipeline tab

---

## Execution Order

```
Phase 1 (parallel): A1 | B1 | E1+E2
Phase 2 (parallel): A2+A3 | B2 | C1 | F1
Phase 3 (parallel): A4 | B3 | C2
Phase 4 (parallel): C3 | C4 | D1+D2+D3
Phase 5: B4 | G1
```

## Agent Assignments

| Agent | Streams | Key Files |
|-------|---------|-----------|
| Agent 1: Queue + Build Worker | A1-A4 | `server/queue.py`, `server/serve.py`, `server/mcp_tools.py`, `server/api.py`, `server/config.py` |
| Agent 2: vLLM Integration | B1-B3 | `server/vllm_manager.py`, `server/config.py`, `server/serve.py` |
| Agent 3: Prompt Management | C1-C3 | `server/prompt_store.py` |
| Agent 4: Viewer Overhaul | E1, E2, F1, C4, D1-D3 | `viewer/static/app.js`, `viewer/static/index.html`, `viewer/static/style.css`, `viewer/server.py` |

## Verification

1. `uv run pytest tests/unit/test_queue.py tests/unit/test_vllm_manager.py tests/unit/test_prompt_store.py -v`
2. `uv run pytest tests/e2e/test_queue_integration.py -v`
3. `uv run release`
4. Manual: viewer tabs, vLLM tok/s, queue status transitions, prompt editing, DAG rendering
5. Full rebuild with Qwen3.5-9B-AWQ after backup + wipe
