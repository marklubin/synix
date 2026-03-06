"""Run and snapshot inspection commands."""

from __future__ import annotations

import json
from datetime import datetime

import click
from rich import box
from rich.table import Table

from synix.build.refs import synix_dir_for_build_dir
from synix.build.snapshots import list_runs
from synix.cli.main import console

RUNS_LIST_JSON_SCHEMA_VERSION = 1


@click.group("runs")
def runs_group():
    """Inspect immutable Synix artifact snapshots."""


@runs_group.command("list")
@click.option("--build-dir", default="./build", help="Build directory")
@click.option("--synix-dir", default=None, help="Explicit .synix directory")
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON instead of a table")
def list_runs_command(build_dir: str, synix_dir: str | None, json_output: bool):
    """List recorded run refs and artifact snapshot ids."""
    resolved_synix_dir = synix_dir_for_build_dir(build_dir, configured_synix_dir=synix_dir)
    if not resolved_synix_dir.exists():
        console.print("[red]No snapshot store found.[/red] Run [bold]synix build[/bold] first.")
        raise SystemExit(1)

    runs = list_runs(build_dir, synix_dir=resolved_synix_dir)
    if json_output:
        console.print_json(
            json.dumps(
                {
                    "schema_version": RUNS_LIST_JSON_SCHEMA_VERSION,
                    "runs": runs,
                }
            )
        )
        return
    if not runs:
        console.print("[dim]No run snapshots found.[/dim]")
        return

    table = Table(title="Run Artifact Snapshots", box=box.ROUNDED)
    table.add_column("Run ID", style="bold")
    table.add_column("Snapshot", no_wrap=True)
    table.add_column("Created", no_wrap=True)
    table.add_column("Pipeline")
    table.add_column("Ref")

    for run in runs:
        created_at = run["created_at"]
        if created_at:
            try:
                created_at = datetime.fromisoformat(created_at).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        table.add_row(run["run_id"], run["snapshot_oid"][:12], created_at, run["pipeline_name"], run["ref"])

    console.print()
    console.print(table)
