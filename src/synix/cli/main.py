"""Synix CLI — main entry point and shared utilities."""

from __future__ import annotations

import click
from rich.console import Console

console = Console()

# Color scheme per layer level
LAYER_COLORS = {
    0: "dim",       # transcripts
    1: "blue",      # episodes
    2: "green",     # monthly/topics
    3: "yellow",    # core
}


def get_layer_style(level: int) -> str:
    """Return Rich style string for a given layer level."""
    return LAYER_COLORS.get(level, "white")


@click.group()
def main():
    """Synix — A build system for agent memory."""
    pass


# Import subcommand modules to register commands
from synix.cli.build_commands import build, plan, run_alias  # noqa: E402, F401
from synix.cli.search_commands import search  # noqa: E402, F401
from synix.cli.verify_commands import diff, lineage, status, verify  # noqa: E402, F401

# Register commands
main.add_command(build)
main.add_command(plan)
main.add_command(run_alias, name="run")
main.add_command(search)
main.add_command(lineage)
main.add_command(status)
main.add_command(verify)
main.add_command(diff)
