"""Search commands — synix search."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
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
    type=click.Choice(["keyword", "semantic", "hybrid", "layered"], case_sensitive=False),
    default=None,
    help="Search mode: keyword (FTS5), semantic (embeddings), hybrid (both + RRF fusion), "
    "or layered (hybrid + higher layers boosted). Auto-detects hybrid when embeddings exist.",
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
      layered  — like hybrid, but boosts higher-level layers in semantic scoring
    """
    from synix.build.artifacts import ArtifactStore
    from synix.build.provenance import ProvenanceTracker
    from synix.search.indexer import SearchIndex, SearchIndexProjection
    from synix.search.retriever import HybridRetriever

    # top_k takes precedence over limit if both given
    effective_top_k = top_k if top_k is not None else limit

    # Auto-detect mode when not explicitly set
    if mode is None:
        embeddings_dir = Path(build_dir) / "embeddings"
        manifest_path = embeddings_dir / "manifest.json"
        if manifest_path.exists():
            mode = "hybrid"
        else:
            mode = "keyword"

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
        # Semantic, hybrid, or layered: need embedding provider
        embedding_provider = None
        if mode in ("semantic", "hybrid", "layered"):
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
            artifact = store.load_artifact(result.label)
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

    mode_label = {"keyword": "keyword", "semantic": "semantic", "hybrid": "hybrid", "layered": "layered"}
    console.print(f'\n[bold]Search results for:[/bold] "{query}" [dim]({mode_label[mode]} mode)[/dim]\n')

    for i, result in enumerate(results[:effective_top_k], 1):
        level = result.layer_level
        style = get_layer_style(level)

        # Build a snippet: first few lines + keyword-matched chunks
        content_snippet = _build_snippet(result.content, query)

        # Build score display with breakdown
        score_parts = [f"{result.score:.2f}"]
        search_mode_label = getattr(result, "search_mode", mode)
        keyword_sc = getattr(result, "keyword_score", None)
        semantic_sc = getattr(result, "semantic_score", None)

        if search_mode_label in ("hybrid", "layered"):
            breakdown = []
            if keyword_sc is not None:
                breakdown.append(f"keyword: {keyword_sc:.1f}")
            if semantic_sc is not None:
                breakdown.append(f"semantic: {semantic_sc:.2f}")
            if breakdown:
                score_parts.append(f"  [{', '.join(breakdown)}]")
        elif search_mode_label == "semantic" and semantic_sc is not None:
            score_parts.append("  (cosine similarity)")

        score_display = "".join(score_parts)

        footer = Text.from_markup(
            f"[dim]Label:[/dim] {result.label}  "
            f"[dim]Score:[/dim] {score_display}  [dim]Mode:[/dim] {search_mode_label}"
        )

        panel = Panel(
            Group(content_snippet, "", footer),
            title=f"[{style} bold]{result.layer_name}[/{style} bold] (L{level})",
            border_style=style,
            subtitle=f"Result {i}",
            padding=(1, 2),
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
                    for parent_label in sorted(rec.parent_labels):
                        tree_label = f"[dim]{parent_label}[/dim]"
                        child = node.add(tree_label)
                        _build_trace_tree(child, parent_label, visited)

            _build_trace_tree(prov_tree, result.label)
            console.print(prov_tree)
        elif not trace and result.provenance_chain:
            # Legacy behavior: show simple provenance tree without --trace
            tree = Tree(f"[dim]{result.label}[/dim]")

            def _add_parents(node, aid, visited=None):
                if visited is None:
                    visited = set()
                if aid in visited:
                    return
                visited.add(aid)
                rec = provenance.get_record(aid)
                if rec:
                    for parent_label in sorted(rec.parent_labels):
                        child = node.add(f"[dim]{parent_label}[/dim]")
                        _add_parents(child, parent_label, visited)

            _add_parents(tree, result.label)
            console.print(tree)

        console.print()


def _build_snippet(content: str, query: str, head_lines: int = 4, context_lines: int = 2) -> Markdown:
    """Build a content snippet with keyword highlighting.

    Shows the first few lines, then any keyword-matched regions with
    surrounding context.  Adjacent/overlapping regions are merged into
    a single block so nothing is duplicated.
    """
    keywords = [k for k in query.lower().split() if k]
    lines = content.splitlines()

    if not keywords or not lines:
        # No query terms — just show first chunk
        preview = "\n".join(lines[: head_lines * 2])
        if len(lines) > head_lines * 2:
            preview += "\n..."
        return Markdown(preview)

    # Find which line indices contain a keyword match
    match_indices: set[int] = set()
    for idx, line in enumerate(lines):
        line_lower = line.lower()
        for kw in keywords:
            if kw in line_lower:
                match_indices.add(idx)
                break

    # Build ranges: head block + each match with context
    # A range is (start, end) inclusive
    ranges: list[tuple[int, int]] = []

    # Always include head
    head_end = min(head_lines - 1, len(lines) - 1)
    ranges.append((0, head_end))

    # Add context around each match
    for idx in sorted(match_indices):
        start = max(0, idx - context_lines)
        end = min(len(lines) - 1, idx + context_lines)
        ranges.append((start, end))

    # Merge overlapping/adjacent ranges
    ranges.sort()
    merged: list[tuple[int, int]] = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 2:  # gap of 1 line → merge
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    # Build output with highlights
    parts: list[str] = []
    for i, (start, end) in enumerate(merged):
        if i > 0:
            parts.append("...")
        for idx in range(start, end + 1):
            line = lines[idx]
            # Highlight keywords by wrapping in ** (bold markdown)
            highlighted = line
            for kw in keywords:
                # Case-insensitive replace, wrap matches in bold
                pos = 0
                result_parts: list[str] = []
                lower = highlighted.lower()
                while pos < len(highlighted):
                    found = lower.find(kw, pos)
                    if found == -1:
                        result_parts.append(highlighted[pos:])
                        break
                    result_parts.append(highlighted[pos:found])
                    result_parts.append(f"**{highlighted[found : found + len(kw)]}**")
                    pos = found + len(kw)
                highlighted = "".join(result_parts)
            parts.append(highlighted)

    # Trailing ellipsis if we didn't show the end
    if merged[-1][1] < len(lines) - 1:
        parts.append("...")

    return Markdown("\n".join(parts))


def _load_embedding_provider(build_dir: str):
    """Try to load an EmbeddingProvider from cached embeddings in build_dir.

    Reads the stored ``_config`` metadata from the embedding manifest to
    reconstruct the provider with the same config that was used at build time.
    This prevents query-time config mismatches that would silently produce
    garbage cosine similarities.

    Returns the provider if embeddings directory exists, otherwise None.
    """
    import json as _json

    from synix.core.config import EmbeddingConfig
    from synix.search.embeddings import EmbeddingProvider

    embeddings_dir = Path(build_dir) / "embeddings"
    if not embeddings_dir.exists():
        return None

    manifest_path = embeddings_dir / "manifest.json"
    stored_config = None
    if manifest_path.exists():
        try:
            manifest = _json.loads(manifest_path.read_text())
            stored_config = manifest.get("_config")
        except (_json.JSONDecodeError, OSError):
            pass

    if stored_config:
        config = EmbeddingConfig.from_dict(stored_config)
    else:
        config = EmbeddingConfig()
        console.print(
            "[yellow]Warning:[/yellow] Embeddings were built without config metadata. "
            "Query results may be unreliable.\n"
            "  Rebuild with: [bold]synix build <pipeline.py>[/bold]"
        )

    from synix.build.cassette import maybe_wrap_embedding_provider

    return maybe_wrap_embedding_provider(EmbeddingProvider(config, build_dir))
