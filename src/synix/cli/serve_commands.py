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
def serve(config_path: str, mcp_port: int | None, viewer: bool) -> None:
    """Start the synix knowledge server."""
    from synix.server.config import load_config
    from synix.server.serve import serve as run_serve

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    config = load_config(config_path)

    if mcp_port is not None:
        config.mcp_port = mcp_port

    asyncio.run(run_serve(config, viewer=viewer))
