"""Clean command — remove build artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path

import click

from synix.cli.main import console


@click.command()
@click.argument("build_dir", default="./build")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt")
def clean(build_dir: str, yes: bool):
    """Remove all build artifacts.

    BUILD_DIR defaults to ./build in the current directory.
    """
    build_path = Path(build_dir)

    if not build_path.exists():
        console.print("[dim]Nothing to clean — build directory does not exist.[/dim]")
        return

    if not yes:
        console.print(f"This will delete [bold]{build_path}[/bold] and all its contents.")
        if not click.confirm("Continue?"):
            console.print("[dim]Aborted.[/dim]")
            return

    shutil.rmtree(build_path)
    console.print(f"[green]Cleaned:[/green] {build_path}")
