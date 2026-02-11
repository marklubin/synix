"""Init command — scaffold a new Synix project."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import click

from synix.cli.main import console


def _get_template_dir() -> Path:
    """Return the path to the bundled init template directory."""
    return Path(__file__).resolve().parent.parent / "templates" / "init"


@click.command()
@click.argument("project_name")
def init(project_name: str):
    """Create a new Synix project with a minimal working example.

    PROJECT_NAME is the directory name to create (e.g., my-project).
    """
    target = Path(project_name)

    if target.exists():
        console.print(
            f"[red]Error:[/red] Directory [bold]{project_name}[/bold] already exists."
        )
        sys.exit(1)

    template_dir = _get_template_dir()
    if not template_dir.is_dir():
        console.print(
            "[red]Error:[/red] Init template not found. "
            "This is a bug — please report it."
        )
        sys.exit(1)

    shutil.copytree(template_dir, target)

    console.print(
        f"[green]Created project[/green] [bold]{project_name}/[/bold]\n"
        f"\n"
        f"  cd {project_name}\n"
        f"  synix build\n"
        f"  synix validate\n"
        f"  synix search 'hiking'"
    )
