"""CLI command for the synix knowledge server."""

from __future__ import annotations

import asyncio
import logging

import click


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default="synix-server.toml",
    help="Path to server config TOML file.",
)
@click.option("--mcp-port", default=None, type=int, help="Override MCP HTTP port.")
@click.option("--viewer/--no-viewer", default=True, help="Enable/disable viewer.")
@click.option("-v", "--verbose", is_flag=True, help="Debug-level logging.")
def serve(config_path: str, mcp_port: int | None, viewer: bool, verbose: bool) -> None:
    """Start the synix knowledge server."""
    from synix.server.config import load_config
    from synix.server.serve import serve as run_serve

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Ensure synix build logs are visible (layer starts, artifact progress, etc.)
    logging.getLogger("synix.build").setLevel(logging.INFO)
    logging.getLogger("synix.server").setLevel(logging.INFO)
    # Suppress noisy httpx request logging unless verbose
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)

    config = load_config(config_path)

    if mcp_port is not None:
        config.mcp_port = mcp_port

    asyncio.run(run_serve(config, viewer=viewer))
