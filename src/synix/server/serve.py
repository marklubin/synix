"""Server orchestrator — runs MCP HTTP + REST API + auto-builder + viewer."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import time
import uuid

from synix.server.config import ServerConfig
from synix.server.queue import DocumentQueue
from synix.workspace import ServerBindings, Workspace

logger = logging.getLogger(__name__)


async def run_mcp_http(bindings: ServerBindings) -> None:
    """Start the MCP HTTP server with REST API routes mounted alongside."""
    import socket

    import uvicorn
    from mcp.server.transport_security import TransportSecuritySettings

    from synix.server.api import api_routes
    from synix.server.mcp_tools import server_mcp

    hostname = socket.gethostname()
    allowed = [
        f"localhost:{bindings.mcp_port}",
        f"127.0.0.1:{bindings.mcp_port}",
        f"{hostname}:{bindings.mcp_port}",
        f"{hostname}:*",
    ]
    allowed.extend(bindings.allowed_hosts)

    server_mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed,
    )

    # Get the Starlette app from FastMCP
    app = server_mcp.streamable_http_app()

    # Mount REST API routes on the same app
    app.routes.extend(api_routes)

    uv_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=bindings.mcp_port,
        log_level="info",
    )
    server = uvicorn.Server(uv_config)
    await server.serve()


def _run_build(workspace: Workspace):
    """Run build synchronously (called in executor)."""
    project = workspace.project
    if project._pipeline is None:
        workspace.load_pipeline()

    logger.info("Build queue: starting pipeline execution")
    result = project.build(accept_existing=True)
    logger.info(
        "Build queue: build done — %d built, %d cached in %.1fs",
        result.built,
        result.cached,
        result.total_time,
    )
    return result


def _run_release(workspace: Workspace):
    """Run release synchronously (called in executor)."""
    from synix.server.mcp_tools import _RELEASE_NAME

    logger.info("Build queue: materializing release %r", _RELEASE_NAME)
    workspace.release_to(_RELEASE_NAME)
    logger.info("Build queue: release complete")


async def run_build_worker(workspace: Workspace, queue: DocumentQueue, build_lock: asyncio.Lock) -> None:
    """Event-driven build worker. Watches queue, triggers builds on window expiry.

    First pending document starts a timer. All documents arriving within
    the configured window seconds form one batch. Documents arriving
    during a build accumulate and trigger the next window after completion.
    """
    config = workspace.config
    if not config.auto_build.enabled:
        logger.info("Build queue: disabled in config")
        return

    window = config.auto_build.window

    # Let the rest of the server start first
    await asyncio.sleep(5)
    logger.info("Build queue: watching (window %ds)", window)

    window_start: float | None = None

    while True:
        await asyncio.sleep(2)

        pending = queue.pending_count()
        if pending == 0:
            window_start = None
            continue

        # Start window timer on first pending doc
        if window_start is None:
            window_start = time.monotonic()
            logger.info("Build queue: %d pending, window started (%ds)", pending, window)
            continue

        # Wait for window to expire
        elapsed = time.monotonic() - window_start
        if elapsed < window:
            continue

        # Window expired — trigger build
        run_id = uuid.uuid4().hex
        claimed = queue.claim_pending_batch(run_id)
        if not claimed:
            window_start = None
            continue

        logger.info("Build queue: batch of %d docs (run %s)", len(claimed), run_id[:8])

        async with build_lock:
            try:
                loop = asyncio.get_running_loop()

                # Apply LLM config override if vLLM is managing inference
                llm_override = workspace.runtime.llm_config_override if workspace.runtime else None
                project = workspace.project
                if llm_override and project._pipeline:
                    project._pipeline.llm_config = llm_override

                result = await loop.run_in_executor(None, _run_build, workspace)
                queue.mark_built(run_id, result.built, result.cached)

                try:
                    await loop.run_in_executor(None, _run_release, workspace)
                except Exception as release_exc:
                    # Release failed after successful build — requeue docs
                    queue.mark_failed(run_id, f"release failed: {release_exc}")
                    logger.error("Build queue: release failed for run %s: %s", run_id[:8], release_exc)
                else:
                    queue.mark_released(run_id)
                    logger.info(
                        "Build queue: run %s complete — %d built, %d cached",
                        run_id[:8],
                        result.built,
                        result.cached,
                    )
            except Exception as exc:
                queue.mark_failed(run_id, str(exc))
                logger.error("Build queue: run %s failed: %s", run_id[:8], exc)

        window_start = None


def run_viewer(workspace: Workspace, bindings: ServerBindings) -> None:
    """Start the synix viewer Flask app (blocking, meant for thread).

    Starts immediately with just the project — the viewer discovers
    releases lazily via before_request hook.
    """
    try:
        from synix.viewer import serve as viewer_serve
    except ImportError:
        logger.warning("Viewer: synix[viewer] extra not installed — viewer disabled")
        return

    try:
        from synix.server.mcp_tools import _RELEASE_NAME

        project = workspace.project

        # Try to bind to an existing release, but start either way
        release = None
        names = workspace.releases()
        if names:
            release_name = _RELEASE_NAME if _RELEASE_NAME in names else names[0]
            release = workspace.release(release_name)
            logger.info("Viewer: serving release %r", release_name)
        else:
            logger.info("Viewer: no release yet — starting in discovery mode")

        viewer_serve(
            release,
            host=bindings.viewer_host,
            port=bindings.viewer_port,
            project=project,
        )
    except Exception as exc:
        logger.error("Viewer failed to start: %s", exc)


async def serve(workspace: Workspace, bindings: ServerBindings, *, viewer: bool = True) -> None:
    """Start the synix knowledge server.

    Components:
    - MCP HTTP server with REST API on mcp_port
    - Auto-builder watching bucket directories
    - Viewer (optional, threaded Flask) on viewer_port
    """

    from synix.server import mcp_tools

    config = workspace.config

    # Startup banner
    logger.info("=" * 60)
    logger.info("Synix Knowledge Server starting")
    logger.info("  workspace:  %s", workspace.name)
    logger.info("  project:    %s", workspace.root)
    logger.info("  pipeline:   %s", config.pipeline_path)
    logger.info("  MCP HTTP:   :%d", bindings.mcp_port)
    if viewer:
        logger.info("  Viewer:     :%d", bindings.viewer_port)
    logger.info("  Buckets:    %s", ", ".join(b.name for b in config.buckets) or "(none)")
    logger.info(
        "  Build queue: %s (window %ds)", "on" if config.auto_build.enabled else "off", config.auto_build.window
    )
    logger.info("=" * 60)

    # Load pipeline if it exists
    pipeline_path = workspace.root / config.pipeline_path
    if pipeline_path.exists() and workspace.pipeline is None:
        try:
            workspace.load_pipeline()
            logger.info("Loaded pipeline from %s", pipeline_path)
        except Exception as exc:
            logger.warning("Could not load pipeline: %s", exc)

    # Initialize document queue
    queue_db = workspace.synix_dir / "queue.db"
    queue = DocumentQueue(queue_db)
    build_lock = asyncio.Lock()
    logger.info("Document queue initialized at %s", queue_db)

    # Initialize prompt store
    from synix.server.prompt_store import PromptStore

    prompts_db = workspace.synix_dir / "prompts.db"
    prompt_store = PromptStore(prompts_db)
    # Seed from bundled prompt templates
    try:
        from synix.build.transforms import PROMPTS_DIR

        seeded = prompt_store.seed_from_files(PROMPTS_DIR)
        if seeded:
            logger.info("Seeded %d prompts from %s", seeded, PROMPTS_DIR)
    except Exception as exc:
        logger.warning("Could not seed prompts: %s", exc)

    # Start vLLM if configured (non-blocking — runs in background)
    vllm_manager = None
    llm_config_override = None
    if config.vllm.enabled:
        from synix.server.vllm_manager import VLLMManager

        vllm_manager = VLLMManager(config.vllm)

        # Set LLM config override immediately so build worker knows to use vLLM
        # (builds will wait for vLLM health before running via the build worker)
        llm_config_override = {
            "provider": "openai-compatible",
            "model": config.vllm.model,
            "base_url": vllm_manager.base_url,
            "api_key": "not-needed",
            "temperature": 0.3,
            "max_tokens": 4096,
        }

        async def _start_vllm():
            logger.info("Starting vLLM: %s on GPU %d", config.vllm.model, config.vllm.gpu_device)
            try:
                await vllm_manager.start()
                await vllm_manager.measure_throughput()
                logger.info("vLLM ready: %s via %s", config.vllm.model, vllm_manager.base_url)
            except Exception as exc:
                logger.error("vLLM failed to start: %s", exc)

        asyncio.create_task(_start_vllm())

    # Activate runtime services on workspace
    workspace.activate_runtime(
        queue=queue,
        prompt_store=prompt_store,
        build_lock=build_lock,
        vllm_manager=vllm_manager,
        llm_config_override=llm_config_override,
    )

    # Set module-level workspace BEFORE starting workers/viewer
    mcp_tools._workspace = workspace

    # Report existing state
    try:
        releases = workspace.releases()
        if releases:
            logger.info("Available releases: %s", ", ".join(releases))
        else:
            logger.info("No releases yet — first build will create one")
    except Exception:
        logger.info("No releases yet — first build will create one")

    loop = asyncio.get_running_loop()
    tasks = []

    # MCP HTTP + REST API (async)
    tasks.append(asyncio.create_task(run_mcp_http(bindings)))

    # Build queue worker (async)
    tasks.append(asyncio.create_task(run_build_worker(workspace, queue, build_lock)))

    # Viewer (threaded Flask)
    if viewer:
        loop.run_in_executor(None, run_viewer, workspace, bindings)

    # Handle shutdown
    stop = asyncio.Event()

    def _signal_handler():
        logger.info("Shutting down...")
        stop.set()
        for t in tasks:
            t.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*tasks)
    finally:
        if vllm_manager is not None:
            logger.info("Stopping vLLM...")
            await vllm_manager.stop()


async def serve_from_config(config: ServerConfig, *, viewer: bool = True) -> None:
    """Backward-compat adapter -- creates Workspace from ServerConfig."""
    from synix.workspace import open_workspace

    workspace = open_workspace(config.project_dir)

    # Overlay ServerConfig fields onto the workspace config so that CLI
    # overrides (pipeline_path, buckets, auto_build, vllm) are respected.
    ws_config = workspace.config
    if ws_config is None:
        from synix.workspace import WorkspaceConfig

        ws_config = WorkspaceConfig()
        workspace._config = ws_config
    ws_config.pipeline_path = config.pipeline_path
    ws_config.buckets = config.buckets
    ws_config.auto_build = config.auto_build
    ws_config.vllm = config.vllm

    bindings = ServerBindings(
        mcp_port=config.mcp_port,
        viewer_port=config.viewer_port,
        viewer_host=config.viewer_host,
        allowed_hosts=config.allowed_hosts,
    )
    await serve(workspace, bindings, viewer=viewer)
