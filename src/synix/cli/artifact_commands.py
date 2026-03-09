"""Artifact browsing commands — synix list, synix show."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich import box
from rich.markdown import Markdown
from rich.table import Table

from synix.cli.main import console, get_layer_style


def _resolve_synix_dir(build_dir: str, synix_dir: str | None) -> Path | None:
    """Resolve the .synix directory from explicit option or build-dir fallback.

    Returns the resolved Path, or None if no snapshot store can be found.
    """
    from synix.build.refs import synix_dir_for_build_dir

    if synix_dir:
        resolved = Path(synix_dir)
        if resolved.exists():
            return resolved
        return None

    build_path = Path(build_dir)
    try:
        resolved = synix_dir_for_build_dir(build_path)
    except ValueError:
        # Ambiguous store resolution — caller will handle error display
        return None

    if resolved.exists():
        return resolved
    return None


@click.command("list")
@click.argument("layer", required=False)
@click.option("--build-dir", default="./build", help="Build directory")
@click.option("--synix-dir", default=None, help="Path to .synix directory")
@click.option("--ref", default="HEAD", help="Snapshot ref to read (default: HEAD)")
def list_artifacts(layer: str | None, build_dir: str, synix_dir: str | None, ref: str):
    """List artifacts in the build, optionally filtered by layer.

    LAYER is an optional layer name to filter (e.g., episodes, monthly, core).
    Without it, lists all artifacts grouped by layer.
    """
    from synix.build.snapshot_view import SnapshotView

    resolved_synix_dir = _resolve_synix_dir(build_dir, synix_dir)
    if resolved_synix_dir is None:
        console.print("[dim]No artifacts found.[/dim]")
        return

    try:
        view = SnapshotView.open(resolved_synix_dir, ref=ref)
    except ValueError:
        console.print("[dim]No artifacts found.[/dim]")
        return

    artifacts = view.list_artifacts()

    if not artifacts:
        console.print("[dim]No artifacts found.[/dim]")
        return

    # Group by layer, sorted by level
    by_layer: dict[str, list[dict]] = {}
    for art in artifacts:
        metadata = art.get("metadata", {})
        layer_name = metadata.get("layer_name", "unknown")
        if layer and layer_name != layer:
            continue
        by_layer.setdefault(layer_name, []).append(art)

    if not by_layer:
        if layer:
            console.print(f"[dim]No artifacts found in layer:[/dim] {layer}")
        else:
            console.print("[dim]No artifacts found.[/dim]")
        return

    # Sort layers by level
    sorted_layers = sorted(
        by_layer.items(),
        key=lambda x: x[1][0].get("metadata", {}).get("layer_level", 0) if x[1] else 0,
    )

    for layer_name, entries in sorted_layers:
        level = entries[0].get("metadata", {}).get("layer_level", 0) if entries else 0
        style = get_layer_style(level)

        table = Table(
            title=f"[{style} bold]{layer_name}[/{style} bold] (L{level}) — {len(entries)} artifacts",
            box=box.ROUNDED,
            show_header=True,
        )
        table.add_column("Label", style="bold", no_wrap=True)
        table.add_column("Artifact ID", style="dim", no_wrap=True)
        table.add_column("Date", no_wrap=True)
        table.add_column("Title / Summary", max_width=60)

        for art in sorted(entries, key=lambda x: x.get("label", "")):
            art_label = art.get("label", "")

            # Short artifact ID (7 chars like git)
            raw_id = (art.get("artifact_id", "") or "").removeprefix("sha256:")
            short_id = raw_id[:7] if raw_id else "-"

            # Extract date and title from metadata
            metadata = art.get("metadata", {})
            date = metadata.get("date", "")
            if not date and metadata.get("month"):
                date = metadata["month"]
            title = metadata.get("title", "")

            # If no title, load content and use first line as summary
            if not title:
                try:
                    content = view.get_content(art_label)
                    if content:
                        first_line = content.strip().split("\n")[0]
                        # Strip markdown heading markers
                        title = first_line.lstrip("# ").strip()
                        if len(title) > 60:
                            title = title[:57] + "..."
                except (KeyError, OSError):
                    title = ""

            table.add_row(art_label, short_id, str(date), title)

        console.print(table)
        console.print()


@click.command("show")
@click.argument("artifact_id")
@click.option("--build-dir", default="./build", help="Build directory")
@click.option("--synix-dir", default=None, help="Path to .synix directory")
@click.option("--ref", default="HEAD", help="Snapshot ref to read (default: HEAD)")
@click.option("--raw", is_flag=True, default=False, help="Show raw JSON instead of rendered content")
def show_artifact(artifact_id: str, build_dir: str, synix_dir: str | None, ref: str, raw: bool):
    """Display the full content of an artifact.

    ARTIFACT_ID is the artifact to display.
    """
    import json

    from synix.build.snapshot_view import SnapshotView

    resolved_synix_dir = _resolve_synix_dir(build_dir, synix_dir)
    if resolved_synix_dir is None:
        console.print("[red]No snapshot store found.[/red] Run [bold]synix build[/bold] first.")
        sys.exit(1)

    try:
        view = SnapshotView.open(resolved_synix_dir, ref=ref)
    except ValueError as e:
        console.print(f"[red]Cannot open snapshot:[/red] {e}")
        sys.exit(1)

    # Resolve prefix (git-like: artifact ID prefix or content hash prefix)
    try:
        resolved_label = view.resolve_prefix(artifact_id)
    except ValueError as e:
        console.print(f"[red]Ambiguous:[/red] {e}")
        sys.exit(1)

    if resolved_label is None:
        console.print(f"[red]Artifact not found:[/red] {artifact_id}")
        sys.exit(1)

    try:
        artifact = view.get_artifact(resolved_label)
    except KeyError:
        console.print(f"[red]Artifact not found:[/red] {resolved_label}")
        sys.exit(1)

    if raw:
        # Show full JSON representation
        data = {
            "label": artifact.get("label"),
            "artifact_type": artifact.get("artifact_type"),
            "artifact_id": artifact.get("artifact_id"),
            "input_ids": artifact.get("input_ids", []),
            "prompt_id": artifact.get("prompt_id"),
            "model_config": artifact.get("model_config"),
            "metadata": artifact.get("metadata", {}),
            "content": artifact.get("content", ""),
        }
        console.print(json.dumps(data, indent=2))
        return

    # Render content as markdown in a panel
    metadata = artifact.get("metadata", {})
    level = metadata.get("layer_level", 0)
    layer_name = metadata.get("layer_name", "unknown")
    style = get_layer_style(level)

    # Metadata header
    meta_parts = [f"[dim]Layer:[/dim] [{style}]{layer_name}[/{style}] (L{level})"]
    meta_parts.append(f"[dim]Type:[/dim] {artifact.get('artifact_type', 'unknown')}")
    if metadata.get("date"):
        meta_parts.append(f"[dim]Date:[/dim] {metadata['date']}")
    if metadata.get("month"):
        meta_parts.append(f"[dim]Month:[/dim] {metadata['month']}")
    if metadata.get("title"):
        meta_parts.append(f"[dim]Title:[/dim] {metadata['title']}")
    if metadata.get("episode_count"):
        meta_parts.append(f"[dim]Episodes:[/dim] {metadata['episode_count']}")
    meta_header = "  ".join(meta_parts)

    console.print(f"\n{meta_header}\n")

    # Render content as markdown
    content = artifact.get("content", "")
    console.print(Markdown(content))
    console.print()
