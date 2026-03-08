"""Ref inspection CLI commands."""

from __future__ import annotations

import json

import click
from rich import box
from rich.table import Table

from synix.build.object_store import ObjectStore
from synix.build.refs import RefStore, synix_dir_for_build_dir
from synix.cli.main import console


@click.group("refs")
def refs_group():
    """Inspect Synix refs (build + release pointers)."""


@refs_group.command("list")
@click.option("--build-dir", default="./build", help="Build directory")
@click.option("--synix-dir", default=None, help="Explicit .synix directory")
@click.option("--json", "json_output", is_flag=True, help="Emit JSON")
def refs_list(build_dir, synix_dir, json_output):
    """List all refs (build heads, runs, releases)."""
    resolved = synix_dir_for_build_dir(build_dir, configured_synix_dir=synix_dir)
    if not resolved.exists():
        console.print("[dim]No snapshot store found.[/dim]")
        return

    ref_store = RefStore(resolved)

    # Gather all refs from known prefixes
    all_refs = []
    for prefix in ("refs/heads", "refs/runs", "refs/releases"):
        for ref_name, oid in ref_store.iter_refs(prefix):
            all_refs.append({"ref": ref_name, "oid": oid})

    if json_output:
        console.print_json(json.dumps({"refs": all_refs}))
        return

    if not all_refs:
        console.print("[dim]No refs found.[/dim]")
        return

    table = Table(title="Refs", box=box.ROUNDED)
    table.add_column("Ref", style="bold")
    table.add_column("Snapshot OID", no_wrap=True)

    for r in all_refs:
        table.add_row(r["ref"], r["oid"][:12])

    console.print()
    console.print(table)


@refs_group.command("show")
@click.argument("ref")
@click.option("--build-dir", default="./build", help="Build directory")
@click.option("--synix-dir", default=None, help="Explicit .synix directory")
@click.option("--json", "json_output", is_flag=True, help="Emit JSON")
def refs_show(ref, build_dir, synix_dir, json_output):
    """Resolve a ref and show snapshot details."""
    resolved = synix_dir_for_build_dir(build_dir, configured_synix_dir=synix_dir)
    if not resolved.exists():
        console.print("[red]Error:[/red] No snapshot store found.")
        raise SystemExit(1)

    ref_store = RefStore(resolved)
    oid = ref_store.read_ref(ref)
    if oid is None:
        console.print(f"[red]Error:[/red] ref {ref!r} does not resolve.")
        raise SystemExit(1)

    object_store = ObjectStore(resolved)
    snapshot = object_store.get_json(oid)

    if json_output:
        console.print_json(json.dumps({"ref": ref, "oid": oid, "snapshot": snapshot}))
        return

    console.print()
    console.print(f"  [bold]Ref:[/bold]       {ref}")
    console.print(f"  [bold]OID:[/bold]       {oid[:12]}")
    console.print(f"  [bold]Type:[/bold]      {snapshot.get('type', 'unknown')}")
    console.print(f"  [bold]Pipeline:[/bold]  {snapshot.get('pipeline_name', '')}")
    console.print(f"  [bold]Created:[/bold]   {snapshot.get('created_at', '')}")
    console.print(f"  [bold]Run ID:[/bold]    {snapshot.get('run_id', '')}")

    manifest_oid = snapshot.get("manifest_oid")
    if manifest_oid:
        console.print(f"  [bold]Manifest:[/bold]  {manifest_oid[:12]}")

    parents = snapshot.get("parent_snapshot_oids", [])
    if parents:
        console.print(f"  [bold]Parents:[/bold]   {', '.join(p[:12] for p in parents)}")
    console.print()
