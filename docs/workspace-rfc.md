# Workspace Abstraction for Synix

## Context

Synix has no single named entity for "a project with its config, pipeline, buckets, build state, and runtime services." Configuration is scattered across 5 disconnected pieces:

- **Project** (sdk.py) — knows `.synix/` + optional pipeline. No bucket/server/queue awareness.
- **ServerConfig** (server/config.py) — knows project_dir, buckets, ports, vllm. Doesn't hold a Project.
- **Pipeline** (core/models.py) — knows layers, llm_config, paths. Decoupled from project.
- **`_state` dict** (mcp_tools.py) — runtime glue bag: project, config, queue, prompt_store, llm_override, build_lock.
- **CLI options** — per-command discovery of pipeline, build dir, source dir.

This causes: 60 lines of manual wiring in serve.py, viewer can't start without a release, CLI bypasses Project entirely, nothing enforces coherence.

**Goal:** Define `Workspace` as the first-class unit — one namespace = pipeline + buckets + .synix state + releases + runtime services. Everything binds to a Workspace.

## Scope and constraints

- **Single-process model.** Like the current `_state` dict, one Workspace is active per process. Cross-process coordination (e.g. CLI while server is running) uses SQLite WAL and filesystem refs — same as today. This RFC does not introduce multi-workspace or multi-process serving.
- **`Workspace` is not a god object.** It is a thin composition layer — identity + config + delegation to `Project`. It holds no build logic, no transform execution, no search indexing. Those stay in `Project`, `runner.py`, and `search/`.
- **Buckets are a workspace concept, not a pipeline concept.** Buckets define how content enters the workspace (ingestion). `Pipeline.source_dir` defines where the pipeline reads from during build. These may overlap (buckets write to source_dir) but are distinct concerns.

---

## Design

### Workspace wraps Project via composition

```python
class Workspace:
    _project: Project           # delegates build/release/refs
    _config: WorkspaceConfig    # parsed from synix.toml (buckets, auto_build, vllm, pipeline_path)
    _runtime: WorkspaceRuntime | None  # set when serve() activates (queue, prompts, vllm, lock)
```

Not a subclass — `Project` stays unchanged, `open_project()` keeps working. New `open_workspace()` returns `Workspace`.

### When to use which

- **`open_project(path)`** — SDK/library consumers doing programmatic builds. Returns `Project`. No config file needed. This is the low-level API for scripts, tests, and pipelines that manage their own configuration.
- **`open_workspace(path)`** — Server, viewer, CLI `serve` command, and anything that needs the full configuration context (buckets, vllm, auto_build). Returns `Workspace`. Discovers and parses `synix.toml`.
- **Rule of thumb:** If you're writing a pipeline.py or a test, use `Project`. If you're deploying or operating, use `Workspace`.

### Lifecycle states (computed, never persisted)

```python
class WorkspaceState(Enum):
    FRESH = "fresh"           # .synix/ exists, no pipeline, no builds
    CONFIGURED = "configured" # pipeline loaded or buckets defined
    BUILT = "built"           # HEAD ref points to a valid snapshot
    RELEASED = "released"     # at least one release receipt exists
    SERVING = "serving"       # runtime services active
```

Derived from disk state on access. No manifest file. States describe *capability* — what operations are possible — not recency. A workspace with an old release is still RELEASED because search/viewer can serve it. Staleness is a separate concern (detectable by comparing HEAD ref timestamp to source mtimes).

### Config split: workspace vs server

**Workspace-scoped** (describes the project):
```toml
[workspace]
name = "my-agent-memory"
pipeline_path = "pipeline.py"

[buckets.sessions]
dir = "sources/sessions"
patterns = ["**/*.jsonl.gz"]

[auto_build]
enabled = true
window = 30

[vllm]
enabled = true
model = "Qwen/Qwen3.5-2B"
```

**Server-scoped** (describes the process):
```toml
[server]
mcp_port = 8200
viewer_port = 9471
```

`project_dir` disappears from config — workspace knows its own root from discovery.

### Config file

One well-known location: **`synix.toml`** at the workspace root. No fallback chain, no precedence rules, no discovery logic.

```
my-project/
├── synix.toml        ← workspace config (required for serve/viewer)
├── pipeline.py       ← pipeline definition
├── sources/          ← bucket directories
└── .synix/           ← build state
```

`open_workspace()` looks for `synix.toml` in the project root. If it doesn't exist, the workspace has no config (bare mode — just a Project with .synix/). The old `synix-server.toml` format is handled by a one-time migration: `synix migrate-config` renames and restructures the file.

---

## User workflow

### Create

```bash
$ synix init agent-memory

Created workspace: agent-memory/
  synix.toml       (edit buckets, vllm, etc.)
  pipeline.py      (edit your pipeline DAG)
  sources/         (put content here)
  .synix/          (build state)
```

`synix init` scaffolds everything — config, pipeline template, source directories, `.synix/` state. One command, complete workspace.

### Disk layout

```
agent-memory/
├── synix.toml              ← workspace config
├── pipeline.py             ← pipeline DAG (Python)
├── sources/                ← bucket directories
│   ├── sessions/
│   ├── documents/
│   ├── reference/
│   ├── exports/
│   └── reports/
└── .synix/                 ← build state (managed by synix)
    ├── objects/            ← content-addressed artifacts
    ├── refs/               ← build + release refs
    ├── releases/           ← materialized projections
    ├── queue.db            ← document ingestion queue
    ├── prompts.db          ← versioned prompt templates
    └── HEAD
```

### Serve

```bash
$ cd agent-memory/
$ synix serve               # starts vLLM + MCP + viewer + build queue
```

Or containerized:
```bash
$ podman run --network=host --device nvidia.com/gpu=0 \
    -v /srv/synix/agent-memory:/workspace:z \
    -v /srv/synix/models:/models:z \
    synix-server
```

### Ingest → Build → Query

Content enters via buckets (drop files or use API):
```bash
# Ingest via API
$ curl -X POST localhost:8200/api/v1/ingest/documents \
    -d '{"content":"...", "filename":"note.md"}'
# → {"doc_id": "abc123"}

# Track processing
$ curl localhost:8200/api/v1/document/abc123
# → {"status": "released"}

# Search
$ curl localhost:8200/api/v1/search?q=synix
# → {"count": 5, "results": [...]}

# Or use the viewer
$ open http://localhost:9471
```

### Inspect

```bash
$ synix status              # workspace state + release info + queue depth
$ synix plan pipeline.py    # dry-run build plan
$ synix list                # all artifacts
$ synix lineage <id>        # provenance tree
```

---

## API

```python
# workspace.py

class Workspace:
    # Identity
    name: str                    # from config or directory name
    root: Path                   # project root directory
    synix_dir: Path              # .synix/ path
    state: WorkspaceState        # computed lifecycle state

    # Configuration
    config: WorkspaceConfig      # parsed TOML (buckets, auto_build, vllm)
    buckets: list[BucketConfig]  # shortcut to config.buckets
    bucket_dir(name) -> Path     # resolve bucket path

    # Pipeline (delegates to Project)
    pipeline                     # current pipeline or None
    load_pipeline(path=None)     # loads from config.pipeline_path if no arg

    # Build/Release (delegates to Project)
    build(**kwargs) -> BuildResult
    release_to(name, ref="HEAD")
    release(name) -> Release
    releases() -> list[str]

    # Runtime services
    runtime: WorkspaceRuntime | None
    activate_runtime(queue, prompt_store, build_lock, ...) -> WorkspaceRuntime

    # Direct access
    project: Project             # underlying Project for SDK consumers


def open_workspace(path=".", config_path=None) -> Workspace
def init_workspace(path, pipeline=None, config_path=None) -> Workspace
```

### WorkspaceRuntime replaces `_state` dict

```python
@dataclass
class WorkspaceRuntime:
    queue: DocumentQueue
    prompt_store: PromptStore
    build_lock: asyncio.Lock
    vllm_manager: VLLMManager | None = None
    llm_config_override: dict | None = None
```

---

## Implementation

### Phase 1: Add workspace.py (purely additive, nothing breaks)

**New: `src/synix/workspace.py`** (~250 lines)
- `WorkspaceState` enum
- `BucketConfig` dataclass (moved from server/config.py)
- `BuildQueueConfig` dataclass (moved from server/config.py)
- `VLLMConfig` dataclass (moved from server/config.py)
- `WorkspaceConfig` dataclass (name, pipeline_path, buckets, auto_build, vllm)
- `ServerBindings` dataclass (mcp_port, viewer_port, viewer_host, allowed_hosts)
- `WorkspaceRuntime` dataclass
- `Workspace` class
- `open_workspace()`, `init_workspace()` factories
- TOML parser handling both old and new formats
- `load_server_bindings()` — parses just the `[server]` section

**Modify: `src/synix/server/config.py`** — config types move to workspace.py. This file becomes a thin backward-compat layer:
- Re-exports `BucketConfig`, `BuildQueueConfig`, `VLLMConfig` from `synix.workspace`
- `ServerConfig` stays here (composes `WorkspaceConfig` + `ServerBindings`)
- `load_config()` stays here, returns `ServerConfig` (calls workspace TOML parser internally)

**Dependency direction:** `workspace.py` owns the config types. `server/config.py` imports from workspace. This is correct — workspace is the foundational abstraction, server is a consumer.

**New: `tests/unit/test_workspace.py`** (~150 lines)
- State computation for each lifecycle
- TOML parsing (new format, old format, both files present, minimal, missing)
- `open_workspace()` discovery
- Delegation to Project
- `bucket_dir()` resolution
- `activate_runtime()` state transition

**New: `tests/e2e/test_workspace_e2e.py`** (~100 lines)
- Full path: `init_workspace() → load_pipeline → build → release_to → releases() → search`
- Verifies Workspace delegates correctly through the entire lifecycle

**Modify: `src/synix/__init__.py`** — add exports: `Workspace`, `open_workspace`

**Gate:** `uv run release` passes, nothing existing breaks.

### Phase 2: Migrate serve.py

**Modify: `src/synix/server/serve.py`**
- `serve()` takes `(workspace: Workspace, bindings: ServerBindings)` instead of `ServerConfig`
- Delete 40+ lines of manual wiring (open project, load pipeline, init queue, init prompts, start vllm)
- Replace with: `workspace.load_pipeline()`, `workspace.activate_runtime(...)`
- Build worker reads from workspace instead of `_state`
- `run_viewer()` passes workspace to viewer
- Add adapter: `serve_from_config(config: ServerConfig)` for backward compat from CLI

**Modify: `src/synix/cli/serve_commands.py`**
- Use `open_workspace()` + `load_server_bindings()` instead of `load_config()`
- Fallback to old path if `synix-server.toml` detected

### Phase 3: Replace `_state` dict

**Modify: `src/synix/server/mcp_tools.py`**
- Replace `_state: dict = {"project": None, "config": None}` with `_workspace: Workspace | None = None`
- `_require_project()` → `_workspace.project`
- `_require_config()` → `_workspace.config`
- `_resolve_bucket_dir(name)` → `_workspace.bucket_dir(name)`
- Queue access: `_workspace.runtime.queue`
- Prompt store: `_workspace.runtime.prompt_store`

**Modify: `src/synix/server/api.py`**
- Import `_workspace` instead of `_state`
- Update queue/prompt_store references (~10 lines)

### Phase 4: Viewer binds to Workspace

**Modify: `src/synix/viewer/__init__.py`**
- `serve()` and `create_app()` accept optional `workspace` parameter
- When workspace provided, extract project + release from it

**Modify: `src/synix/viewer/server.py`**
- `ViewerState.__init__` accepts optional `workspace`
- Add `ViewerState.from_workspace(workspace, title)` classmethod
- `try_discover_release()` uses `workspace.releases()` instead of re-opening project
- No route handler changes needed — they already use `state.release` and `state.project`

### Phase 5: Viewer shows workspace identity

The viewer becomes workspace-aware in the UI:

**Header/sidebar:** Shows workspace name and state badge (FRESH / CONFIGURED / BUILT / RELEASED / SERVING). Replaces the generic "Viewer" logo text.

**Pipeline tab:** Shows workspace config summary — name, pipeline path, buckets list, vllm model, auto_build window. This is the "what am I looking at" context that's currently invisible.

**Status API:** `GET /api/status` returns workspace identity:
```json
{
  "workspace": "my-agent-memory",
  "state": "released",
  "loaded": true,
  "artifact_count": 42,
  "release": "local"
}
```

### Phase 6: Deploy config update

**Rename: `deploy/synix-server.toml` → `deploy/synix.toml`**
**Modify: `deploy/pipeline.py`** — no changes needed (pipeline is workspace-independent)

---

## Files changed

| File | Change | Size |
|------|--------|------|
| `src/synix/workspace.py` | NEW — config types, Workspace, factories | ~250 lines |
| `src/synix/__init__.py` | Add exports: `Workspace`, `open_workspace` | ~2 lines |
| `src/synix/server/config.py` | Re-export config types from workspace, slim down | ~40 lines changed |
| `src/synix/server/serve.py` | Refactor serve() to use Workspace | ~60 lines changed |
| `src/synix/server/mcp_tools.py` | Replace _state with _workspace | ~30 lines changed |
| `src/synix/server/api.py` | Update state references | ~10 lines changed |
| `src/synix/viewer/__init__.py` | Accept workspace param | ~10 lines |
| `src/synix/viewer/server.py` | ViewerState.from_workspace | ~20 lines |
| `src/synix/cli/serve_commands.py` | Use open_workspace | ~10 lines |
| `deploy/synix-server.toml` | Add [workspace] section | ~3 lines |
| `tests/unit/test_workspace.py` | NEW — state, config, discovery, delegation | ~150 lines |
| `tests/unit/test_workspace_config.py` | NEW — TOML parsing, precedence, backward compat | ~100 lines |
| `tests/e2e/test_workspace_e2e.py` | NEW — full lifecycle: init → build → release → search | ~100 lines |

## Verification

1. `uv run pytest tests/unit/test_workspace.py tests/unit/test_workspace_config.py -v` — unit tests
2. `uv run pytest tests/e2e/test_workspace_e2e.py -v` — full lifecycle e2e
3. `uv run pytest tests/ --ignore=tests/e2e/mesh` — full regression
4. `uv run release` — full gate (lint + test + demos)
5. `podman build -t synix-server -f deploy/Containerfile . && bash deploy/smoke-test/run.sh` — container e2e
6. Verify old `synix-server.toml` format still works: `synix serve --config synix-server.toml`

## Decisions log

Resolutions from review feedback:

| Concern | Resolution |
|---------|------------|
| Config types in server/ (backwards coupling) | Types move to workspace.py, server/config.py re-exports for compat |
| Both `open_project()` and `open_workspace()` exported | Documented guidance: Project for SDK/scripts, Workspace for operations |
| Config precedence complexity | Single well-known location: `synix.toml`. No fallback chain. Migration CLI for old format. |
| God object risk | Workspace is composition only — no build/search/transform logic |
| Multi-process safety | Single-process model (explicit constraint), same as current _state |
| WorkspaceState misleading for stale releases | States describe capability not recency; staleness is a separate concern |
| Missing e2e test | Added `tests/e2e/test_workspace_e2e.py` to plan |
| `serve_from_config()` permanent or transitional? | Transitional — kept for one release cycle, then removed |
