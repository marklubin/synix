"""Clean command — remove build artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path

import click

from synix.cli.main import console, pipeline_argument


@click.command()
@pipeline_argument
@click.option("--build-dir", default=None, help="Override build directory")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt")
def clean(pipeline_path: str, build_dir: str | None, yes: bool):
    """Remove all build artifacts for a pipeline.

    PIPELINE_PATH defaults to pipeline.py in the current directory.

    Deletes the entire build directory. Use --yes to skip the confirmation prompt.
    """
    from synix.build.pipeline import load_pipeline

    try:
        pipeline = load_pipeline(pipeline_path)
    except Exception as e:
        console.print(f"[red]Error loading pipeline:[/red] {e}")
        raise SystemExit(1) from e

    if build_dir:
        pipeline.build_dir = build_dir

    build_path = Path(pipeline.build_dir)

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
