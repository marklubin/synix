"""Synix CLI — main entry point and shared utilities."""

from __future__ import annotations

import os
import sys
from pathlib import Path

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


def is_demo_mode() -> bool:
    """Check if running in demo mode (deterministic output, no timestamps/timings)."""
    return os.environ.get("SYNIX_DEMO", "").strip() == "1"


def _resolve_pipeline_path(
    ctx: click.Context, param: click.Parameter, value: str | None,
) -> str:
    """Click callback: default to ./pipeline.py when no argument is given."""
    if value is not None:
        return value
    default = str(Path.cwd() / "pipeline.py")
    if not Path(default).exists():
        console.print(
            "[red]Error:[/red] No pipeline file specified and "
            "[bold]pipeline.py[/bold] not found in the current directory."
        )
        sys.exit(1)
    return default


def pipeline_argument(fn):
    """Shared Click argument decorator for PIPELINE_PATH with ./pipeline.py default."""
    return click.argument(
        "pipeline_path",
        required=False,
        default=None,
        callback=_resolve_pipeline_path,
        is_eager=False,
    )(fn)


@click.group()
def main():
    """Synix — A build system for agent memory."""
    pass


def cli():
    """Entrypoint that loads .env before running the CLI."""
    from dotenv import load_dotenv

    load_dotenv()
    main()


# Import subcommand modules to register commands
from synix.cli.artifact_commands import list_artifacts, show_artifact  # noqa: E402, F401
from synix.cli.build_commands import build, plan, run_alias  # noqa: E402, F401
from synix.cli.clean_commands import clean  # noqa: E402, F401
from synix.cli.demo_commands import demo  # noqa: E402, F401
from synix.cli.fix_commands import fix  # noqa: E402, F401
from synix.cli.info_commands import info  # noqa: E402, F401
from synix.cli.init_commands import init  # noqa: E402, F401
from synix.cli.search_commands import search  # noqa: E402, F401
from synix.cli.validate_commands import validate  # noqa: E402, F401
from synix.cli.verify_commands import diff, lineage, status, verify  # noqa: E402, F401

# Register commands
main.add_command(build)
main.add_command(plan)
main.add_command(run_alias, name="run")
main.add_command(search)
main.add_command(list_artifacts, name="list")
main.add_command(show_artifact, name="show")
main.add_command(lineage)
main.add_command(status)
main.add_command(verify)
main.add_command(validate)
main.add_command(fix)
main.add_command(diff)
main.add_command(clean)
main.add_command(demo)
main.add_command(info)
main.add_command(init)
