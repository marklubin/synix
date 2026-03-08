"""Search commands — synix search."""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from pathlib import Path

import click
from rich.console import Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from synix.build.search_outputs import SearchOutputResolutionError, resolve_search_output
from synix.cli.main import console, get_layer_style

logger = logging.getLogger(__name__)


class ReleaseProvenanceProvider:
    """Read provenance chains from a released search.db.

    Released search databases contain a ``provenance_chains`` table that
    bakes in the full chain for every indexed artifact.  This provider
    reads that table and exposes the same ``get_chain`` / ``get_record``
    interface that :class:`ProvenanceTracker` provides so the search
    display layer can use it interchangeably.
    """

    def __init__(self, db_path: str | Path):
        self._chains: dict[str, list[str]] = {}
        db = Path(db_path)
        if not db.exists():
            return
        conn = sqlite3.connect(str(db))
        try:
            # Check if the table exists before querying
            table_check = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='provenance_chains'"
            ).fetchone()
            if table_check is not None:
                for row in conn.execute("SELECT label, chain FROM provenance_chains"):
                    try:
                        self._chains[row[0]] = json.loads(row[1])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning(
                            "Failed to parse provenance chain for label %r",
                            row[0],
                            exc_info=True,
                        )
        finally:
            conn.close()

    def get_chain(self, label: str) -> list[_ProvenanceShim]:
        """Return the provenance chain for a label as shim records.

        Returns a list of shim objects with ``.label`` attribute, matching
        the interface expected by SearchIndex.query and HybridRetriever.
        """
        chain_labels = self._chains.get(label, [label])
        return [_ProvenanceShim(label=l, parent_labels=[]) for l in chain_labels]

    def get_record(self, label: str) -> _ProvenanceShim | None:
        """Return a shim record with parent_labels derived from the chain.

        The chain stores the full lineage. The first entry is the artifact
        itself; subsequent entries are its ancestors.  We reconstruct a
        minimal parent_labels list from the stored chain.
        """
        chain = self._chains.get(label)
        if chain is None:
            return None
        # Chain is ordered: [self, parent1, grandparent1, ...]
        # parent_labels = everything except the label itself
        parent_labels = [l for l in chain if l != label]
        return _ProvenanceShim(label=label, parent_labels=parent_labels)


class _ProvenanceShim:
    """Minimal shim matching the ProvenanceRecord interface needed by display."""

    __slots__ = ("label", "parent_labels")

    def __init__(self, label: str, parent_labels: list[str]):
        self.label = label
        self.parent_labels = parent_labels


def _resolve_synix_dir(build_dir: str, synix_dir: str | None) -> Path:
    """Resolve the .synix directory from explicit option or build_dir convention."""
    from synix.build.refs import synix_dir_for_build_dir

    if synix_dir is not None:
        return Path(synix_dir).resolve()
    return synix_dir_for_build_dir(Path(build_dir).resolve())


def _list_release_names(synix_dir: Path) -> list[str]:
    """List available release names under .synix/releases/."""
    releases_dir = synix_dir / "releases"
    if not releases_dir.is_dir():
        return []
    return sorted(d.name for d in releases_dir.iterdir() if d.is_dir() and (d / "search.db").exists())


def _resolve_release_db(synix_dir: Path, release_name: str | None) -> Path | None:
    """Resolve a release search.db path.

    If *release_name* is given, return that release's search.db.
    If *release_name* is None, auto-detect when exactly one release exists.
    Returns None when no release can be resolved.
    """
    if release_name is not None:
        db_path = synix_dir / "releases" / release_name / "search.db"
        if db_path.exists():
            return db_path
        console.print(f"[red]Error:[/red] No search.db in release {release_name!r} (looked at {db_path})")
        raise SystemExit(1)

    # Auto-detect: list releases that have a search.db
    available = _list_release_names(synix_dir)
    if len(available) == 1:
        return synix_dir / "releases" / available[0] / "search.db"
    if len(available) > 1:
        names = ", ".join(available)
        console.print(
            f"[red]Error:[/red] Multiple releases found ({names}). Re-run with [bold]--release <name>[/bold]."
        )
        raise SystemExit(1)
    return None


@click.command()
@click.argument("query")
@click.option("--layers", default=None, help="Comma-separated layer names to search")
@click.option("--step", default=None, help="Filter to a specific pipeline step/layer (alias for --layers)")
@click.option("--build-dir", default="./build", help="Build directory")
@click.option("--synix-dir", default=None, help="Path to .synix directory (auto-detected from build-dir by default)")
@click.option(
    "--release",
    "release_name",
    default=None,
    help="Release target to query (searches .synix/releases/<name>/search.db)",
)
@click.option(
    "--ref",
    "ref",
    default=None,
    help="Snapshot ref for scratch realization (build ephemeral search.db, query, discard)",
)
@click.option(
    "--projection",
    default=None,
    help="Search output name to query when a build has multiple local outputs",
)
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
    synix_dir: str | None,
    release_name: str | None,
    ref: str | None,
    projection: str | None,
    limit: int,
    mode: str,
    top_k: int | None,
    trace: bool,
    customer: str | None,
):
    """Search across memory layers.

    QUERY is the search text.

    Source resolution (in priority order):
      --ref <ref>       scratch realization from a snapshot ref (ephemeral)
      --release <name>  query a materialized release target
      (auto)            single release auto-detected, or legacy build/ path

    Modes:
      keyword  — FTS5 full-text search (default, no embedding config needed)
      semantic — cosine-similarity search over embeddings
      hybrid   — combines keyword + semantic via Reciprocal Rank Fusion
      layered  — like hybrid, but boosts higher-level layers in semantic scoring

    Output selection:
      one local search output        — use it automatically
      several outputs, one named search
                                    — use that one (SynixSearch before SearchIndex)
      several outputs, one SynixSearch
                                    — use that one
      otherwise                     — re-run with --projection <name>
    """
    from synix.search.indexer import SearchIndex, SearchIndexProjection
    from synix.search.retriever import HybridRetriever

    # top_k takes precedence over limit if both given
    effective_top_k = top_k if top_k is not None else limit

    # Combine --layers and --step: both specify layer names to filter
    layer_names: list[str] = []
    if layers:
        layer_names.extend(l.strip() for l in layers.split(","))
    if step:
        layer_names.extend(s.strip() for s in step.split(","))
    layer_filter = layer_names if layer_names else None

    # --- Source resolution ---
    db_path: Path | None = None
    provenance: object | None = None  # ProvenanceTracker or ReleaseProvenanceProvider
    embeddings_base_dir: str | None = None  # directory containing embeddings/

    if ref is not None:
        # Scratch realization from a snapshot ref
        db_path, provenance = _scratch_realize(build_dir, synix_dir, ref)
        # Scratch realization doesn't support embeddings yet
        embeddings_base_dir = None
    elif release_name is not None:
        # Explicit release target
        resolved_synix_dir = _resolve_synix_dir(build_dir, synix_dir)
        db_path = _resolve_release_db(resolved_synix_dir, release_name)
        if db_path is not None:
            provenance = ReleaseProvenanceProvider(db_path)
            # Check for embeddings in the release directory
            embeddings_base_dir = str(db_path.parent)
    else:
        # Auto-detect: try release first, fall back to build dir
        resolved_synix_dir = _resolve_synix_dir(build_dir, synix_dir)
        release_db = _resolve_release_db(resolved_synix_dir, None)
        if release_db is not None:
            db_path = release_db
            provenance = ReleaseProvenanceProvider(db_path)
            embeddings_base_dir = str(db_path.parent)
        else:
            # Fall back to legacy build/ path
            try:
                search_output = resolve_search_output(build_dir, projection_name=projection)
            except SearchOutputResolutionError as exc:
                console.print(f"[red]{exc}[/red]")
                sys.exit(1)

            if search_output is not None and search_output.db_path.exists():
                db_path = search_output.db_path
                embeddings_base_dir = build_dir
            else:
                db_path = None

            from synix.build.refs import synix_dir_for_build_dir
            from synix.build.snapshot_view import SnapshotArtifactCache

            try:
                sd = synix_dir_for_build_dir(Path(build_dir))
                provenance = SnapshotArtifactCache(sd)
            except (ValueError, OSError):
                # No snapshot store — provenance display will be empty
                provenance = None

    if db_path is None or not db_path.exists():
        console.print(
            "[red]No search index found.[/red] Run [bold]synix build[/bold] first, "
            "or specify [bold]--release <name>[/bold]."
        )
        sys.exit(1)

    # Auto-detect mode when not explicitly set
    if mode is None:
        if embeddings_base_dir is not None:
            embeddings_dir = Path(embeddings_base_dir) / "embeddings"
            manifest_path = embeddings_dir / "manifest.json"
            if manifest_path.exists():
                mode = "hybrid"
            else:
                mode = "keyword"
        else:
            mode = "keyword"

    if mode == "keyword":
        # Fast path: use existing FTS5 query directly
        search_projection = SearchIndexProjection(build_dir, db_path)
        results = search_projection.query(
            query,
            layers=layer_filter,
            provenance_tracker=provenance,
        )
        search_projection.close()
        results = results[:effective_top_k]
    else:
        # Semantic, hybrid, or layered: need embedding provider
        embedding_provider = None
        if mode in ("semantic", "hybrid", "layered") and embeddings_base_dir is not None:
            embedding_provider = _load_embedding_provider(embeddings_base_dir)
            if embedding_provider is None and mode == "semantic":
                console.print(
                    "[red]Semantic search requires embeddings.[/red] "
                    "No embedding config found and no cached embeddings available."
                )
                sys.exit(1)
        elif mode == "semantic":
            console.print(
                "[red]Semantic search requires embeddings.[/red] "
                "No embedding config found and no cached embeddings available."
            )
            sys.exit(1)

        search_index = SearchIndex(db_path)
        try:
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
        finally:
            search_index.close()

    # Filter by customer metadata if requested
    if customer is not None:
        filtered = _filter_by_customer(results, customer, build_dir)
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
            f"[dim]Label:[/dim] {result.label}  [dim]Score:[/dim] {score_display}  [dim]Mode:[/dim] {search_mode_label}"
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


def _scratch_realize(build_dir: str, synix_dir_option: str | None, ref: str) -> tuple[Path, ReleaseProvenanceProvider]:
    """Build an ephemeral search.db from a snapshot ref and return (db_path, provenance).

    This requires the release adapter infrastructure (ReleaseClosure, get_adapter).
    If those modules are not yet available, we fail with a clear error message.
    """
    import tempfile

    resolved_synix_dir = _resolve_synix_dir(build_dir, synix_dir_option)

    try:
        from synix.build.adapters import get_adapter  # noqa: F811
        from synix.build.release import ReleaseClosure  # noqa: F811
    except ImportError as exc:
        console.print(
            "[red]Error:[/red] Scratch realization (--ref) requires the release adapter "
            "infrastructure which is not yet available. Use [bold]--release <name>[/bold] "
            "to query a materialized release, or omit --ref to use the build directory."
        )
        raise SystemExit(1) from exc

    closure = ReleaseClosure.from_ref(resolved_synix_dir, ref=ref)

    # Find the search projection
    search_proj = None
    for _name, proj in closure.projections.items():
        if proj.adapter == "synix_search":
            search_proj = proj
            break

    if search_proj is None:
        console.print("[red]Error:[/red] No search projection in snapshot")
        raise SystemExit(1)

    # Build ephemeral search.db in a work directory
    work_dir_path = resolved_synix_dir / "work"
    work_dir_path.mkdir(parents=True, exist_ok=True)
    work_dir = tempfile.mkdtemp(dir=str(work_dir_path), prefix="scratch_search_")

    adapter = get_adapter("synix_search")
    plan = adapter.plan(closure, search_proj, None)
    adapter.apply(plan, work_dir)

    db_path = Path(work_dir) / "search.db"
    provenance = ReleaseProvenanceProvider(db_path)
    return db_path, provenance


def _filter_by_customer(results: list, customer: str, build_dir: str) -> list:
    """Filter search results by customer_id metadata.

    Tries the ArtifactStore first for full metadata, falls back to
    the metadata already on the search result.
    """
    try:
        from synix.build.artifacts import ArtifactStore

        store = ArtifactStore(build_dir)
    except Exception:
        store = None
        logger.debug("Could not load ArtifactStore for customer filtering", exc_info=True)

    filtered = []
    for result in results:
        if store is not None:
            artifact = store.load_artifact(result.label)
            if artifact is not None:
                if artifact.metadata.get("customer_id") == customer:
                    filtered.append(result)
                continue
        # Fall back to the metadata already on the search result
        if result.metadata.get("customer_id") == customer:
            filtered.append(result)
    return filtered


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
