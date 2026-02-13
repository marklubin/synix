"""Artifact browsing commands — synix list, synix show."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich import box
from rich.markdown import Markdown
from rich.table import Table

from synix.cli.main import console, get_layer_style


@click.command("list")
@click.argument("layer", required=False)
@click.option("--build-dir", default="./build", help="Build directory")
def list_artifacts(layer: str | None, build_dir: str):
    """List artifacts in the build, optionally filtered by layer.

    LAYER is an optional layer name to filter (e.g., episodes, monthly, core).
    Without it, lists all artifacts grouped by layer.
    """
    from synix.build.artifacts import ArtifactStore

    build_path = Path(build_dir)
    if not build_path.exists():
        console.print("[red]No build directory found.[/red] Run [bold]synix build[/bold] first.")
        sys.exit(1)

    store = ArtifactStore(build_dir)
    manifest = store._manifest

    if not manifest:
        console.print("[dim]No artifacts found.[/dim]")
        return

    # Group by layer, sorted by level
    by_layer: dict[str, list[tuple[str, dict]]] = {}
    for art_label, info in manifest.items():
        layer_name = info.get("layer", "unknown")
        if layer and layer_name != layer:
            continue
        by_layer.setdefault(layer_name, []).append((art_label, info))

    if not by_layer:
        if layer:
            console.print(f"[dim]No artifacts found in layer:[/dim] {layer}")
        else:
            console.print("[dim]No artifacts found.[/dim]")
        return

    # Sort layers by level
    sorted_layers = sorted(
        by_layer.items(),
        key=lambda x: x[1][0][1].get("level", 0) if x[1] else 0,
    )

    for layer_name, entries in sorted_layers:
        level = entries[0][1].get("level", 0) if entries else 0
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

        # Load each artifact to get metadata for display
        for art_label, _info in sorted(entries, key=lambda x: x[0]):
            artifact = store.load_artifact(art_label)
            if artifact is None:
                table.add_row(art_label, "-", "-", "[dim]<missing>[/dim]")
                continue

            # Short artifact ID (7 chars like git)
            raw_id = (artifact.artifact_id or "").removeprefix("sha256:")
            short_id = raw_id[:7] if raw_id else "-"

            # Extract date and title from metadata
            date = artifact.metadata.get("date", "")
            if not date and artifact.metadata.get("month"):
                date = artifact.metadata["month"]
            title = artifact.metadata.get("title", "")

            # If no title, use first line of content as summary
            if not title and artifact.content:
                first_line = artifact.content.strip().split("\n")[0]
                # Strip markdown heading markers
                title = first_line.lstrip("# ").strip()
                if len(title) > 60:
                    title = title[:57] + "..."

            table.add_row(art_label, short_id, str(date), title)

        console.print(table)
        console.print()


@click.command("show")
@click.argument("artifact_id")
@click.option("--build-dir", default="./build", help="Build directory")
@click.option("--raw", is_flag=True, default=False, help="Show raw JSON instead of rendered content")
def show_artifact(artifact_id: str, build_dir: str, raw: bool):
    """Display the full content of an artifact.

    ARTIFACT_ID is the artifact to display.
    """
    import json

    from synix.build.artifacts import ArtifactStore

    build_path = Path(build_dir)
    if not build_path.exists():
        console.print("[red]No build directory found.[/red] Run [bold]synix build[/bold] first.")
        sys.exit(1)

    store = ArtifactStore(build_dir)

    # Resolve prefix (git-like: artifact ID prefix or content hash prefix)
    try:
        resolved_id = store.resolve_prefix(artifact_id)
    except ValueError as e:
        console.print(f"[red]Ambiguous:[/red] {e}")
        sys.exit(1)

    if resolved_id is None:
        console.print(f"[red]Artifact not found:[/red] {artifact_id}")
        sys.exit(1)

    artifact = store.load_artifact(resolved_id)
    if artifact is None:
        console.print(f"[red]Artifact not found:[/red] {resolved_id}")
        sys.exit(1)

    if raw:
        # Show full JSON representation
        data = {
            "label": artifact.label,
            "artifact_type": artifact.artifact_type,
            "artifact_id": artifact.artifact_id,
            "input_ids": artifact.input_ids,
            "prompt_id": artifact.prompt_id,
            "model_config": artifact.model_config,
            "created_at": artifact.created_at.isoformat(),
            "metadata": artifact.metadata,
            "content": artifact.content,
        }
        console.print(json.dumps(data, indent=2))
        return

    # Render content as markdown in a panel
    info = store._manifest.get(artifact_id, {})
    level = info.get("level", 0)
    layer_name = info.get("layer", "unknown")
    style = get_layer_style(level)

    # Metadata header
    meta_parts = [f"[dim]Layer:[/dim] [{style}]{layer_name}[/{style}] (L{level})"]
    meta_parts.append(f"[dim]Type:[/dim] {artifact.artifact_type}")
    if artifact.metadata.get("date"):
        meta_parts.append(f"[dim]Date:[/dim] {artifact.metadata['date']}")
    if artifact.metadata.get("month"):
        meta_parts.append(f"[dim]Month:[/dim] {artifact.metadata['month']}")
    if artifact.metadata.get("title"):
        meta_parts.append(f"[dim]Title:[/dim] {artifact.metadata['title']}")
    if artifact.metadata.get("episode_count"):
        meta_parts.append(f"[dim]Episodes:[/dim] {artifact.metadata['episode_count']}")
    meta_header = "  ".join(meta_parts)

    console.print(f"\n{meta_header}\n")

    # Render content as markdown
    console.print(Markdown(artifact.content))
    console.print()
