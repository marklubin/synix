"""Search commands — synix search."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.panel import Panel
from rich.tree import Tree

from synix.cli.main import console, get_layer_style


@click.command()
@click.argument("query")
@click.option("--layers", default=None, help="Comma-separated layer names to search")
@click.option("--step", default=None, help="Filter to a specific pipeline step/layer (alias for --layers)")
@click.option("--build-dir", default="./build", help="Build directory")
@click.option("--limit", default=10, help="Max results to return (alias for --top-k)")
@click.option(
    "--mode",
    type=click.Choice(["keyword", "semantic", "hybrid"], case_sensitive=False),
    default="keyword",
    help="Search mode: keyword (FTS5), semantic (embeddings), or hybrid (both + RRF fusion)",
)
@click.option("--top-k", default=None, type=int, help="Max results to return")
@click.option("--trace", is_flag=True, default=False, help="Show provenance chain below each result")
@click.option("--customer", default=None, help="Filter results by customer_id metadata")
def search(
    query: str,
    layers: str | None,
    step: str | None,
    build_dir: str,
    limit: int,
    mode: str,
    top_k: int | None,
    trace: bool,
    customer: str | None,
):
    """Search across memory layers.

    QUERY is the search text.

    Modes:
      keyword  — FTS5 full-text search (default, no embedding config needed)
      semantic — cosine-similarity search over embeddings
      hybrid   — combines keyword + semantic via Reciprocal Rank Fusion
    """
    from synix.build.artifacts import ArtifactStore
    from synix.build.provenance import ProvenanceTracker
    from synix.search.indexer import SearchIndex, SearchIndexProjection
    from synix.search.retriever import HybridRetriever

    # top_k takes precedence over limit if both given
    effective_top_k = top_k if top_k is not None else limit

    db_path = Path(build_dir) / "search.db"
    if not db_path.exists():
        console.print("[red]No search index found.[/red] Run [bold]synix build[/bold] first.")
        sys.exit(1)

    # Combine --layers and --step: both specify layer names to filter
    layer_names: list[str] = []
    if layers:
        layer_names.extend(l.strip() for l in layers.split(","))
    if step:
        layer_names.extend(s.strip() for s in step.split(","))
    layer_filter = layer_names if layer_names else None

    provenance = ProvenanceTracker(build_dir)

    if mode == "keyword":
        # Fast path: use existing FTS5 query directly
        projection = SearchIndexProjection(build_dir)
        results = projection.query(
            query,
            layers=layer_filter,
            provenance_tracker=provenance,
        )
        results = results[:effective_top_k]
    else:
        # Semantic or hybrid: need embedding provider
        embedding_provider = None
        if mode in ("semantic", "hybrid"):
            embedding_provider = _load_embedding_provider(build_dir)
            if embedding_provider is None and mode == "semantic":
                console.print(
                    "[red]Semantic search requires embeddings.[/red] "
                    "No embedding config found and no cached embeddings available."
                )
                sys.exit(1)

        search_index = SearchIndex(db_path)
        retriever = HybridRetriever(
            search_index=search_index,
            embedding_provider=embedding_provider,
            provenance_tracker=provenance,
        )
        results = retriever.query(
            query,
            mode=mode,
            layers=layer_filter,
            top_k=effective_top_k,
        )

    # Filter by customer metadata if requested
    if customer is not None:
        store = ArtifactStore(build_dir)
        filtered = []
        for result in results:
            artifact = store.load_artifact(result.artifact_id)
            if artifact is not None:
                if artifact.metadata.get("customer_id") == customer:
                    filtered.append(result)
            else:
                # Fall back to the metadata already on the search result
                if result.metadata.get("customer_id") == customer:
                    filtered.append(result)
        results = filtered

    if not results:
        if customer is not None:
            console.print(f"[dim]No results for customer:[/dim] {customer}")
        else:
            console.print(f"[dim]No results for:[/dim] {query}")
        return

    mode_label = {"keyword": "keyword", "semantic": "semantic", "hybrid": "hybrid"}
    console.print(
        f"\n[bold]Search results for:[/bold] \"{query}\" "
        f"[dim]({mode_label[mode]} mode)[/dim]\n"
    )

    for i, result in enumerate(results[:effective_top_k], 1):
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

        # Provenance chain — show when --trace is set OR when provenance data exists (legacy behavior)
        if trace and result.provenance_chain:
            prov_tree = Tree("[bold]Provenance:[/bold]")

            def _build_trace_tree(node, aid, visited=None):
                if visited is None:
                    visited = set()
                if aid in visited:
                    return
                visited.add(aid)
                rec = provenance.get_record(aid)
                if rec:
                    for parent_id in rec.parent_artifact_ids:
                        # Try to get layer info from the parent's provenance record
                        parent_rec = provenance.get_record(parent_id)
                        label = f"[dim]{parent_id}[/dim]"
                        child = node.add(label)
                        _build_trace_tree(child, parent_id, visited)

            _build_trace_tree(prov_tree, result.artifact_id)
            console.print(prov_tree)
        elif not trace and result.provenance_chain:
            # Legacy behavior: show simple provenance tree without --trace
            tree = Tree(f"[dim]{result.artifact_id}[/dim]")

            def _add_parents(node, aid, visited=None):
                if visited is None:
                    visited = set()
                if aid in visited:
                    return
                visited.add(aid)
                rec = provenance.get_record(aid)
                if rec:
                    for parent_id in rec.parent_artifact_ids:
                        child = node.add(f"[dim]{parent_id}[/dim]")
                        _add_parents(child, parent_id, visited)

            _add_parents(tree, result.artifact_id)
            console.print(tree)

        console.print()


def _load_embedding_provider(build_dir: str):
    """Try to load an EmbeddingProvider from cached embeddings in build_dir.

    Returns the provider if embeddings directory exists, otherwise None.
    """
    from synix.core.config import EmbeddingConfig
    from synix.search.embeddings import EmbeddingProvider

    embeddings_dir = Path(build_dir) / "embeddings"
    if not embeddings_dir.exists():
        return None

    # Use default config — the provider will use cached embeddings
    # and only call the API for uncached content (the query itself).
    config = EmbeddingConfig()
    return EmbeddingProvider(config, build_dir)
