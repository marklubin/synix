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

logger = logging.getLogger(__name__)


async def run_mcp_http(config: ServerConfig) -> None:
    """Start the MCP HTTP server with REST API routes mounted alongside."""
    import socket

    import uvicorn
    from mcp.server.transport_security import TransportSecuritySettings

    from synix.server.api import api_routes
    from synix.server.mcp_tools import server_mcp

    hostname = socket.gethostname()
    allowed = [
        f"localhost:{config.mcp_port}",
        f"127.0.0.1:{config.mcp_port}",
        f"{hostname}:{config.mcp_port}",
        f"{hostname}:*",
    ]
    allowed.extend(config.allowed_hosts)

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
        port=config.mcp_port,
        log_level="info",
    )
    server = uvicorn.Server(uv_config)
    await server.serve()


def _run_build(config: ServerConfig):
    """Run build synchronously (called in executor)."""
    from pathlib import Path

    from synix.server.mcp_tools import _state

    project = _state["project"]
    pipeline_path = Path(config.project_dir) / config.pipeline_path
    if pipeline_path.exists() and project._pipeline is None:
        project.load_pipeline(str(pipeline_path))

    logger.info("Build queue: starting pipeline execution")
    result = project.build(accept_existing=True)
    logger.info(
        "Build queue: build done — %d built, %d cached in %.1fs",
        result.built,
        result.cached,
        result.total_time,
    )
    return result


def _run_release(config: ServerConfig):
    """Run release synchronously (called in executor)."""
    from synix.server.mcp_tools import _RELEASE_NAME, _state

    project = _state["project"]
    logger.info("Build queue: materializing release %r", _RELEASE_NAME)
    project.release_to(_RELEASE_NAME)
    logger.info("Build queue: release complete")


async def run_build_worker(config: ServerConfig, queue: DocumentQueue, build_lock: asyncio.Lock) -> None:
    """Event-driven build worker. Watches queue, triggers builds on window expiry.

    First pending document starts a timer. All documents arriving within
    ``config.auto_build.window`` seconds form one batch. Documents arriving
    during a build accumulate and trigger the next window after completion.
    """
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
                from synix.server.mcp_tools import _state

                llm_override = _state.get("llm_config_override")
                project = _state["project"]
                if llm_override and project._pipeline:
                    project._pipeline.llm_config = llm_override

                result = await loop.run_in_executor(None, _run_build, config)
                queue.mark_built(run_id, result.built, result.cached)

                try:
                    await loop.run_in_executor(None, _run_release, config)
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


def run_viewer(config: ServerConfig) -> None:
    """Start the synix viewer Flask app (blocking, meant for thread)."""
    try:
        import synix
        from synix.viewer import serve as viewer_serve
    except ImportError:
        logger.warning("Viewer: synix[viewer] extra not installed — viewer disabled")
        return

    try:
        from synix.server.mcp_tools import _RELEASE_NAME

        project = synix.open_project(config.project_dir)
        names = project.releases()
        if not names:
            logger.warning("Viewer: no releases found — build and release first")
            return

        release_name = _RELEASE_NAME if _RELEASE_NAME in names else names[0]
        release = project.release(release_name)
        logger.info("Viewer: serving release %r", release_name)
        viewer_serve(
            release,
            host=config.viewer_host,
            port=config.viewer_port,
            project=project,
        )
    except Exception as exc:
        logger.error("Viewer failed to start: %s", exc)


async def serve(config: ServerConfig, *, viewer: bool = True) -> None:
    """Start the synix knowledge server.

    Components:
    - MCP HTTP server with REST API on mcp_port
    - Auto-builder watching bucket directories
    - Viewer (optional, threaded Flask) on viewer_port
    """
    from pathlib import Path

    import synix
    from synix.server.mcp_tools import _state

    # Startup banner
    logger.info("=" * 60)
    logger.info("Synix Knowledge Server starting")
    logger.info("  project:    %s", config.project_dir)
    logger.info("  pipeline:   %s", config.pipeline_path)
    logger.info("  MCP HTTP:   :%d", config.mcp_port)
    if viewer:
        logger.info("  Viewer:     :%d", config.viewer_port)
    logger.info("  Buckets:    %s", ", ".join(b.name for b in config.buckets) or "(none)")
    logger.info(
        "  Build queue: %s (window %ds)", "on" if config.auto_build.enabled else "off", config.auto_build.window
    )
    logger.info("=" * 60)

    # Open project and configure MCP state
    project = synix.open_project(config.project_dir)
    _state["project"] = project
    _state["config"] = config
    logger.info("Opened project at %s", config.project_dir)

    # Load pipeline if it exists
    pipeline_path = Path(config.project_dir) / config.pipeline_path
    if pipeline_path.exists():
        try:
            project.load_pipeline(str(pipeline_path))
            logger.info("Loaded pipeline from %s", pipeline_path)
        except Exception as exc:
            logger.warning("Could not load pipeline: %s", exc)

    # Initialize document queue
    queue_db = Path(config.project_dir) / ".synix" / "queue.db"
    queue = DocumentQueue(queue_db)
    _state["queue"] = queue
    build_lock = asyncio.Lock()
    _state["build_lock"] = build_lock
    logger.info("Document queue initialized at %s", queue_db)

    # Initialize prompt store
    from synix.server.prompt_store import PromptStore

    prompts_db = Path(config.project_dir) / ".synix" / "prompts.db"
    prompt_store = PromptStore(prompts_db)
    _state["prompt_store"] = prompt_store
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
    if config.vllm.enabled:
        from synix.server.vllm_manager import VLLMManager

        vllm_manager = VLLMManager(config.vllm)

        # Set LLM config override immediately so build worker knows to use vLLM
        # (builds will wait for vLLM health before running via the build worker)
        _state["llm_config_override"] = {
            "provider": "openai-compatible",
            "model": config.vllm.model,
            "base_url": vllm_manager.base_url,
            "api_key": "not-needed",
            "temperature": 0.3,
            "max_tokens": 4096,
        }
        _state["vllm_manager"] = vllm_manager

        async def _start_vllm():
            logger.info("Starting vLLM: %s on GPU %d", config.vllm.model, config.vllm.gpu_device)
            try:
                await vllm_manager.start()
                await vllm_manager.measure_throughput()
                logger.info("vLLM ready: %s via %s", config.vllm.model, vllm_manager.base_url)
            except Exception as exc:
                logger.error("vLLM failed to start: %s", exc)

        asyncio.create_task(_start_vllm())

    # Report existing state
    try:
        releases = project.releases()
        if releases:
            logger.info("Available releases: %s", ", ".join(releases))
        else:
            logger.info("No releases yet — first build will create one")
    except Exception:
        logger.info("No releases yet — first build will create one")

    loop = asyncio.get_running_loop()
    tasks = []

    # MCP HTTP + REST API (async)
    tasks.append(asyncio.create_task(run_mcp_http(config)))

    # Build queue worker (async)
    tasks.append(asyncio.create_task(run_build_worker(config, queue, build_lock)))

    # Viewer (threaded Flask)
    if viewer:
        loop.run_in_executor(None, run_viewer, config)

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
