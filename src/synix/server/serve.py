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
    """Watch bucket directories and trigger builds when content changes.

    Scans buckets every ``scan_interval`` seconds. Detects changes via a
    fingerprint of file paths and modification times — catches new files,
    edits, renames, and deletions. Waits ``cooldown`` seconds after the
    last change before building.
    """
    if not config.auto_build.enabled:
        logger.info("Auto-build: disabled in config")
        return

    import hashlib
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

    def _bucket_fingerprint() -> str:
        """Hash of (path, size, mtime) for all files in all buckets.

        Detects: new files, deletions, renames, and content edits.
        """
        entries: list[str] = []
        for bucket in config.buckets:
            bucket_dir = Path(bucket.dir)
            if not bucket_dir.is_absolute():
                bucket_dir = Path(config.project_dir) / bucket_dir
            if not bucket_dir.exists():
                continue
            for pattern in bucket.patterns:
                for f in sorted(bucket_dir.glob(pattern)):
                    if f.is_file():
                        stat = f.stat()
                        entries.append(f"{f}:{stat.st_size}:{stat.st_mtime_ns}")
        return hashlib.sha256("\n".join(entries).encode()).hexdigest()[:16]

    def _run_build() -> str:
        """Run build + release synchronously (called in executor)."""
        import synix

        project = synix.open_project(config.project_dir)
        pipeline_path = Path(config.project_dir) / config.pipeline_path
        if pipeline_path.exists():
            project.load_pipeline(str(pipeline_path))
        from synix.server.mcp_tools import _RELEASE_NAME

        result = project.build()
        project.release_to(_RELEASE_NAME)
        return f"{result.built} built, {result.cached} cached"

    last_fingerprint = _bucket_fingerprint()
    last_build_time = 0.0

    while True:
        await asyncio.sleep(scan_interval)

        current_fingerprint = _bucket_fingerprint()
        if current_fingerprint == last_fingerprint:
            continue

        # Changes detected — wait for cooldown
        now = time.monotonic()
        since_last = now - last_build_time if last_build_time > 0 else cooldown
        if since_last < cooldown:
            remaining = cooldown - since_last
            logger.info(
                "Auto-build: changes detected, waiting %.0fs cooldown",
                remaining,
            )
            await asyncio.sleep(remaining)

        logger.info("Auto-build: bucket content changed, starting build")

        try:
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(None, _run_build)
            last_build_time = time.monotonic()
            last_fingerprint = _bucket_fingerprint()  # re-fingerprint after build
            logger.info("Auto-build: complete (%s)", summary)
        except Exception as exc:
            logger.error("Auto-build: failed: %s", exc)
            # Re-fingerprint so we retry on next actual change, not immediately
            last_fingerprint = _bucket_fingerprint()


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
    logger.info("  Auto-build: %s (scan %ds, cooldown %ds)",
                "on" if config.auto_build.enabled else "off",
                config.auto_build.scan_interval,
                config.auto_build.cooldown)
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

    # Report existing state
    try:
        releases = project.releases()
        if releases:
            logger.info("Available releases: %s", ", ".join(releases))
        else:
            logger.info("No releases yet — first build will create one")
    except Exception:
        logger.info("No releases yet — first build will create one")

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
