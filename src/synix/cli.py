"""Synix CLI — Click commands with Rich formatting."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich import box

console = Console()

# Color scheme per layer level
LAYER_COLORS = {
    0: "dim",       # transcripts
    1: "blue",      # episodes
    2: "green",     # monthly/topics
    3: "yellow",    # core
}


def get_layer_style(level: int) -> str:
    """Return Rich style string for a given layer level."""
    return LAYER_COLORS.get(level, "white")


@click.group()
def main():
    """Synix — A build system for agent memory."""
    pass


@main.command()
@click.argument("pipeline_path", type=click.Path(exists=True))
@click.option("--source-dir", default=None, help="Override source directory")
@click.option("--build-dir", default=None, help="Override build directory")
def run(pipeline_path: str, source_dir: str | None, build_dir: str | None):
    """Process exports through a memory pipeline.

    PIPELINE_PATH is the Python file defining the pipeline (e.g., pipeline.py).
    """
    from synix.pipeline.config import load_pipeline
    from synix.pipeline.runner import run as run_pipeline

    try:
        pipeline = load_pipeline(pipeline_path)
    except Exception as e:
        console.print(f"[red]Error loading pipeline:[/red] {e}")
        sys.exit(1)

    if source_dir:
        pipeline.source_dir = source_dir
    if build_dir:
        pipeline.build_dir = build_dir

    console.print(Panel(
        f"[bold]Pipeline:[/bold] {pipeline.name}\n"
        f"[bold]Source:[/bold] {pipeline.source_dir}\n"
        f"[bold]Build:[/bold] {pipeline.build_dir}\n"
        f"[bold]Layers:[/bold] {len(pipeline.layers)}",
        title="[bold cyan]Synix Build[/bold cyan]",
        border_style="cyan",
    ))

    start_time = time.time()

    try:
        result = run_pipeline(pipeline, source_dir=source_dir)
    except Exception as e:
        console.print(f"\n[red]Pipeline failed:[/red] {e}")
        sys.exit(1)

    elapsed = time.time() - start_time

    # Summary table
    table = Table(title="Build Summary", box=box.ROUNDED)
    table.add_column("Layer", style="bold")
    table.add_column("Level", justify="center")
    table.add_column("Built", justify="right", style="green")
    table.add_column("Cached", justify="right", style="cyan")
    table.add_column("Skipped", justify="right", style="dim")

    for stats in result.layer_stats:
        style = get_layer_style(stats.level)
        table.add_row(
            f"[{style}]{stats.name}[/{style}]",
            str(stats.level),
            str(stats.built),
            str(stats.cached),
            str(stats.skipped),
        )

    console.print()
    console.print(table)
    console.print(
        f"\n[bold]Total:[/bold] {result.built} built, "
        f"{result.cached} cached, {result.skipped} skipped"
    )
    console.print(f"[bold]Time:[/bold] {elapsed:.1f}s")


@main.command()
@click.argument("query")
@click.option("--layers", default=None, help="Comma-separated layer names to search")
@click.option("--build-dir", default="./build", help="Build directory")
@click.option("--limit", default=10, help="Max results to return")
def search(query: str, layers: str | None, build_dir: str, limit: int):
    """Search across memory layers.

    QUERY is the search text.
    """
    from synix.artifacts.provenance import ProvenanceTracker
    from synix.projections.search_index import SearchIndexProjection

    db_path = Path(build_dir) / "search.db"
    if not db_path.exists():
        console.print("[red]No search index found.[/red] Run [bold]synix run[/bold] first.")
        sys.exit(1)

    layer_filter = [l.strip() for l in layers.split(",")] if layers else None

    provenance = ProvenanceTracker(build_dir)
    projection = SearchIndexProjection(build_dir)
    results = projection.query(
        query,
        layers=layer_filter,
        provenance_tracker=provenance,
    )

    if not results:
        console.print(f"[dim]No results for:[/dim] {query}")
        return

    console.print(f"\n[bold]Search results for:[/bold] \"{query}\"\n")

    for i, result in enumerate(results[:limit], 1):
        level = result.layer_level
        style = get_layer_style(level)

        # Content snippet (truncate long content)
        snippet = result.content[:300]
        if len(result.content) > 300:
            snippet += "..."

        panel = Panel(
            f"{snippet}\n\n"
            f"[dim]Artifact:[/dim] {result.artifact_id}\n"
            f"[dim]Score:[/dim] {result.score:.2f}",
            title=f"[{style} bold]{result.layer_name}[/{style} bold] (L{level})",
            border_style=style,
            subtitle=f"Result {i}",
        )
        console.print(panel)

        # Provenance chain
        if result.provenance_chain:
            tree = Tree("[dim]Provenance[/dim]")
            current = tree
            for pid in result.provenance_chain:
                current = current.add(f"[dim]{pid}[/dim]")
            console.print(tree)

        console.print()


@main.command()
@click.argument("artifact_id")
@click.option("--build-dir", default="./build", help="Build directory")
def lineage(artifact_id: str, build_dir: str):
    """Show provenance chain for an artifact.

    ARTIFACT_ID is the artifact to trace.
    """
    from synix.artifacts.provenance import ProvenanceTracker
    from synix.artifacts.store import ArtifactStore

    provenance = ProvenanceTracker(build_dir)
    store = ArtifactStore(build_dir)

    chain = provenance.get_chain(artifact_id)

    if not chain:
        console.print(f"[red]No provenance found for:[/red] {artifact_id}")
        sys.exit(1)

    console.print(f"\n[bold]Lineage for:[/bold] {artifact_id}\n")

    tree = Tree(f"[bold]{artifact_id}[/bold]")

    # Build tree recursively
    def add_parents(node, aid):
        record = next((r for r in chain if r.artifact_id == aid), None)
        if record:
            for parent_id in record.parent_artifact_ids:
                artifact = store.load_artifact(parent_id)
                label = parent_id
                if artifact:
                    label += f" [dim]({artifact.artifact_type})[/dim]"
                child_node = node.add(label)
                add_parents(child_node, parent_id)

    add_parents(tree, artifact_id)
    console.print(tree)


@main.command()
@click.option("--build-dir", default="./build", help="Build directory")
def status(build_dir: str):
    """Show build status summary."""
    from synix.artifacts.store import ArtifactStore

    build_path = Path(build_dir)
    if not build_path.exists():
        console.print("[red]No build directory found.[/red] Run [bold]synix run[/bold] first.")
        sys.exit(1)

    store = ArtifactStore(build_dir)

    table = Table(title="Build Status", box=box.ROUNDED)
    table.add_column("Layer", style="bold")
    table.add_column("Artifacts", justify="right")
    table.add_column("Last Build", justify="center")

    # Group artifacts by layer from manifest
    manifest = store._manifest
    layers: dict[str, dict] = {}
    for _aid, info in manifest.items():
        layer = info.get("layer", "unknown")
        level = info.get("level", 0)
        if layer not in layers:
            layers[layer] = {"count": 0, "level": level}
        layers[layer]["count"] += 1

    # Sort by level
    for layer_name, info in sorted(layers.items(), key=lambda x: x[1]["level"]):
        style = get_layer_style(info["level"])
        table.add_row(
            f"[{style}]{layer_name}[/{style}]",
            str(info["count"]),
            "-",
        )

    console.print()
    console.print(table)

    # Search index status
    search_db = build_path / "search.db"
    if search_db.exists():
        console.print(f"\n[green]Search index:[/green] {search_db} exists")
    else:
        console.print(f"\n[yellow]Search index:[/yellow] not built yet")

    # Context doc status
    context_doc = build_path / "context.md"
    if context_doc.exists():
        size = context_doc.stat().st_size
        console.print(f"[green]Context doc:[/green] {context_doc} ({size} bytes)")
    else:
        console.print(f"[yellow]Context doc:[/yellow] not built yet")
