"""Init command — scaffold a new Synix project."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import click

from synix.cli.main import console


def _get_templates_root() -> Path:
    """Return the path to the bundled templates directory."""
    return Path(__file__).resolve().parent.parent / "templates"


def _get_template_dir(template_name: str) -> Path:
    """Return the path to a specific bundled template directory."""
    return _get_templates_root() / template_name


def _available_templates() -> list[str]:
    """Return sorted list of available template names."""
    root = _get_templates_root()
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir())


@click.command()
@click.argument("project_name", default="synix")
@click.option(
    "--template",
    "-t",
    default="03-team-report",
    help="Template to use (default: 03-team-report). Use --list to see available templates.",
)
@click.option(
    "--list",
    "list_templates",
    is_flag=True,
    default=False,
    help="List available templates and exit.",
)
def init(project_name: str, template: str, list_templates: bool):
    """Create a new Synix project from an example template.

    PROJECT_NAME is the directory name to create (default: synix).
    """
    if list_templates:
        templates = _available_templates()
        if not templates:
            console.print("[red]Error:[/red] No templates found. Run scripts/sync-templates first.")
            sys.exit(1)
        console.print("[bold]Available templates:[/bold]\n")
        for name in templates:
            marker = " [dim](default)[/dim]" if name == "03-team-report" else ""
            console.print(f"  {name}{marker}")
        return

    target = Path(project_name)

    if target.exists():
        console.print(f"[red]Error:[/red] Directory [bold]{project_name}[/bold] already exists.")
        sys.exit(1)

    template_dir = _get_template_dir(template)
    if not template_dir.is_dir():
        available = _available_templates()
        console.print(f"[red]Error:[/red] Template [bold]{template}[/bold] not found.")
        if available:
            console.print(f"Available templates: {', '.join(available)}")
        else:
            console.print("No templates found. Run scripts/sync-templates first.")
        sys.exit(1)

    shutil.copytree(template_dir, target)

    console.print(
        f"[green]Created project[/green] [bold]{project_name}/[/bold] "
        f"[dim](template: {template})[/dim]\n"
        f"\n"
        f"  cd {project_name}\n"
        f"  cp .env.example .env   [dim]# add your API key[/dim]\n"
        f"  synix build\n"
        f"  synix validate\n"
        f"  synix search 'hiking'"
    )
