"""Server orchestrator — runs MCP HTTP + REST API + auto-builder + viewer."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import time

from synix.server.config import ServerConfig

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
        app, host="0.0.0.0", port=config.mcp_port, log_level="info",
    )
    server = uvicorn.Server(uv_config)
    await server.serve()


async def run_auto_builder(config: ServerConfig) -> None:
    """Watch bucket directories and trigger builds when new files appear.

    Scans buckets every ``scan_interval`` seconds. When the file count
    changes, waits ``cooldown`` seconds, then runs build + release.
    """
    if not config.auto_build.enabled:
        logger.info("Auto-build: disabled in config")
        return

    from pathlib import Path

    scan_interval = config.auto_build.scan_interval
    cooldown = config.auto_build.cooldown

    # Let the rest of the server start first
    await asyncio.sleep(5)
    logger.info(
        "Auto-build: watching buckets (scan every %ds, cooldown %ds)",
        scan_interval,
        cooldown,
    )

    def _count_bucket_files() -> int:
        """Count files matching bucket patterns."""
        total = 0
        for bucket in config.buckets:
            bucket_dir = Path(bucket.dir)
            if not bucket_dir.is_absolute():
                bucket_dir = Path(config.project_dir) / bucket_dir
            if not bucket_dir.exists():
                continue
            for pattern in bucket.patterns:
                total += sum(1 for _ in bucket_dir.glob(pattern))
        return total

    def _run_build() -> str:
        """Run build + release synchronously (called in executor)."""
        import synix

        project = synix.open_project(config.project_dir)
        pipeline_path = Path(config.project_dir) / config.pipeline_path
        if pipeline_path.exists():
            project.load_pipeline(str(pipeline_path))
        result = project.build()
        project.release_to("local")
        return f"{result.built} built, {result.cached} cached"

    last_count = _count_bucket_files()
    last_build_time = 0.0

    while True:
        await asyncio.sleep(scan_interval)

        current_count = _count_bucket_files()
        if current_count == last_count:
            continue

        # New files detected — wait for cooldown
        now = time.monotonic()
        since_last = now - last_build_time if last_build_time > 0 else cooldown
        if since_last < cooldown:
            remaining = cooldown - since_last
            logger.info(
                "Auto-build: %d new files, waiting %.0fs cooldown",
                current_count - last_count,
                remaining,
            )
            await asyncio.sleep(remaining)

        logger.info(
            "Auto-build: file count changed %d → %d, starting build",
            last_count,
            current_count,
        )

        try:
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(None, _run_build)
            last_build_time = time.monotonic()
            last_count = _count_bucket_files()  # re-count after build
            logger.info("Auto-build: complete (%s)", summary)
        except Exception as exc:
            logger.error("Auto-build: failed: %s", exc)
            last_count = current_count  # update count to avoid retry loop


def run_viewer(config: ServerConfig) -> None:
    """Start the synix viewer Flask app (blocking, meant for thread)."""
    try:
        import synix
        from synix.viewer import serve as viewer_serve
    except ImportError:
        logger.warning("Viewer: synix[viewer] extra not installed — viewer disabled")
        return

    try:
        project = synix.open_project(config.project_dir)
        names = project.releases()
        if not names:
            logger.warning("Viewer: no releases found — build and release first")
            return

        release_name = "local" if "local" in names else names[0]
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

    loop = asyncio.get_event_loop()
    tasks = []

    # MCP HTTP + REST API (async)
    tasks.append(asyncio.create_task(run_mcp_http(config)))

    # Auto-builder (async)
    tasks.append(asyncio.create_task(run_auto_builder(config)))

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

    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.gather(*tasks)
