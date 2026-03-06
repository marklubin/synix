"""Run and snapshot inspection commands."""

from __future__ import annotations

from datetime import datetime

import click
from rich import box
from rich.table import Table

from synix.build.refs import synix_dir_for_build_dir
from synix.build.snapshots import list_runs
from synix.cli.main import console


@click.group("runs")
def runs_group():
    """Inspect immutable Synix run snapshots."""


@runs_group.command("list")
@click.option("--build-dir", default="./build", help="Build directory")
@click.option("--synix-dir", default=None, help="Explicit .synix directory")
def list_runs_command(build_dir: str, synix_dir: str | None):
    """List recorded run refs and snapshot ids."""
    resolved_synix_dir = synix_dir_for_build_dir(build_dir, configured_synix_dir=synix_dir)
    if not resolved_synix_dir.exists():
        console.print("[red]No snapshot store found.[/red] Run [bold]synix build[/bold] first.")
        raise SystemExit(1)

    runs = list_runs(build_dir, synix_dir=resolved_synix_dir)
    if not runs:
        console.print("[dim]No run snapshots found.[/dim]")
        return

    table = Table(title="Run Snapshots", box=box.ROUNDED)
    table.add_column("Run Ref", style="bold")
    table.add_column("Snapshot", no_wrap=True)
    table.add_column("Created", no_wrap=True)
    table.add_column("Pipeline")

    for run in runs:
        created_at = run["created_at"]
        if created_at:
            try:
                created_at = datetime.fromisoformat(created_at).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        table.add_row(run["ref"], run["snapshot_oid"][:12], created_at, run["pipeline_name"])

    console.print()
    console.print(table)
