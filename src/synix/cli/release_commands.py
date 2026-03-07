"""Release and revert CLI commands."""

from __future__ import annotations

import json

import click
from rich import box
from rich.table import Table

from synix.build.refs import synix_dir_for_build_dir
from synix.build.release_engine import execute_release, get_release, list_releases
from synix.cli.main import console


@click.command()
@click.argument("ref", default="HEAD")
@click.option("--to", "release_name", required=True, help="Release target name (e.g. 'local', 'prod')")
@click.option("--target", "target_path", default=None, help="Override target directory")
@click.option("--build-dir", default="./build", help="Build directory (used to locate .synix)")
@click.option("--synix-dir", default=None, help="Explicit .synix directory")
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON")
def release(
    ref: str,
    release_name: str,
    target_path: str | None,
    build_dir: str,
    synix_dir: str | None,
    json_output: bool,
):
    """Release a snapshot to a named target.

    Materializes projections (search index, context docs) from the snapshot
    at REF into the release target. Every release writes a receipt.

    Examples:
        synix release HEAD --to local
        synix release refs/runs/20260307T120000Z --to prod
    """
    resolved_synix_dir = synix_dir_for_build_dir(build_dir, configured_synix_dir=synix_dir)
    if not resolved_synix_dir.exists():
        console.print("[red]Error:[/red] No snapshot store found. Run [bold]synix build[/bold] first.")
        raise SystemExit(1)

    try:
        receipt = execute_release(
            resolved_synix_dir,
            ref=ref,
            release_name=release_name,
            target=target_path,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc

    if json_output:
        console.print_json(json.dumps(receipt.to_dict()))
        return

    console.print()
    short_oid = receipt.snapshot_oid[:12]
    console.print(f"[bold green]Released[/bold green] [bold]{short_oid}[/bold] → [cyan]{release_name}[/cyan]")
    console.print(f"  Pipeline: {receipt.pipeline_name}")
    console.print(f"  Source:   {ref}")
    console.print()

    if receipt.adapters:
        for name, adapter_data in receipt.adapters.items():
            status_style = "green" if adapter_data.get("status") == "success" else "red"
            console.print(
                f"  [{status_style}]●[/{status_style}] {name}  "
                f"{adapter_data.get('adapter', '')}  →  {adapter_data.get('target', '')}  "
                f"({adapter_data.get('artifacts_applied', 0)} artifacts)"
            )
    console.print()


@click.command()
@click.argument("ref")
@click.option("--to", "release_name", required=True, help="Release target name")
@click.option("--target", "target_path", default=None, help="Override target directory")
@click.option("--build-dir", default="./build", help="Build directory (used to locate .synix)")
@click.option("--synix-dir", default=None, help="Explicit .synix directory")
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON")
def revert(
    ref: str,
    release_name: str,
    target_path: str | None,
    build_dir: str,
    synix_dir: str | None,
    json_output: bool,
):
    """Revert a release target to an older snapshot.

    This is equivalent to releasing an older snapshot — there is no
    special revert machinery. The adapter sees the desired state and
    reconciles.

    Example:
        synix revert refs/runs/20260306T120000Z --to prod
    """
    # Revert is just release of an older snapshot
    resolved_synix_dir = synix_dir_for_build_dir(build_dir, configured_synix_dir=synix_dir)
    if not resolved_synix_dir.exists():
        console.print("[red]Error:[/red] No snapshot store found. Run [bold]synix build[/bold] first.")
        raise SystemExit(1)

    try:
        receipt = execute_release(
            resolved_synix_dir,
            ref=ref,
            release_name=release_name,
            target=target_path,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc

    if json_output:
        console.print_json(json.dumps(receipt.to_dict()))
        return

    console.print()
    short_oid = receipt.snapshot_oid[:12]
    console.print(f"[bold yellow]Reverted[/bold yellow] [cyan]{release_name}[/cyan] → [bold]{short_oid}[/bold]")
    console.print(f"  Pipeline: {receipt.pipeline_name}")
    console.print(f"  Source:   {ref}")
    console.print()


@click.group("releases")
def releases_group():
    """Inspect release state and receipts."""


@releases_group.command("list")
@click.option("--build-dir", default="./build", help="Build directory")
@click.option("--synix-dir", default=None, help="Explicit .synix directory")
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON")
def releases_list(build_dir: str, synix_dir: str | None, json_output: bool):
    """List all releases with receipt info."""
    resolved_synix_dir = synix_dir_for_build_dir(build_dir, configured_synix_dir=synix_dir)
    if not resolved_synix_dir.exists():
        console.print("[dim]No snapshot store found.[/dim]")
        return

    releases = list_releases(resolved_synix_dir)

    if json_output:
        console.print_json(json.dumps({"releases": releases}))
        return

    if not releases:
        console.print("[dim]No releases found.[/dim]")
        return

    table = Table(title="Releases", box=box.ROUNDED)
    table.add_column("Release", style="bold cyan")
    table.add_column("Snapshot", no_wrap=True)
    table.add_column("Released", no_wrap=True)
    table.add_column("Pipeline")
    table.add_column("Adapters")

    for r in releases:
        snapshot = r.get("snapshot_oid", "")[:12]
        released = r.get("released_at", "")
        if released:
            try:
                from datetime import datetime

                released = datetime.fromisoformat(released).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass
        adapters = ", ".join(r.get("adapters", {}).keys())
        table.add_row(r.get("release_name", ""), snapshot, released, r.get("pipeline_name", ""), adapters)

    console.print()
    console.print(table)


@releases_group.command("show")
@click.argument("name")
@click.option("--build-dir", default="./build", help="Build directory")
@click.option("--synix-dir", default=None, help="Explicit .synix directory")
@click.option("--json", "json_output", is_flag=True, help="Emit machine-readable JSON")
def releases_show(name: str, build_dir: str, synix_dir: str | None, json_output: bool):
    """Show receipt details for a release."""
    resolved_synix_dir = synix_dir_for_build_dir(build_dir, configured_synix_dir=synix_dir)
    if not resolved_synix_dir.exists():
        console.print("[red]Error:[/red] No snapshot store found.")
        raise SystemExit(1)

    receipt = get_release(resolved_synix_dir, name)
    if receipt is None:
        console.print(f"[red]Error:[/red] No release named [bold]{name}[/bold] found.")
        raise SystemExit(1)

    if json_output:
        console.print_json(json.dumps(receipt.to_dict()))
        return

    console.print()
    console.print(f"  [bold]Release:[/bold]   [cyan]{receipt.release_name}[/cyan]")
    console.print(f"  [bold]Snapshot:[/bold]  {receipt.snapshot_oid[:12]}")
    console.print(f"  [bold]Released:[/bold]  {receipt.released_at}")
    console.print(f"  [bold]Pipeline:[/bold]  {receipt.pipeline_name}")
    console.print(f"  [bold]Source:[/bold]    {receipt.source_ref}")
    console.print()

    if receipt.adapters:
        console.print("  [bold]Adapters:[/bold]")
        for name, data in receipt.adapters.items():
            status_style = "green" if data.get("status") == "success" else "red"
            console.print(
                f"    [{status_style}]●[/{status_style}] {name}  "
                f"{data.get('adapter', '')}  →  {data.get('target', '')}  "
                f"({data.get('artifacts_applied', 0)} artifacts)"
            )
    console.print()
