"""Interactive memory viewer — browse artifacts, search, inspect config."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import httpx
from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from synix.cli.main import get_layer_style
from synix.mesh.auth import auth_headers
from synix.mesh.dashboard import _format_uptime

logger = logging.getLogger(__name__)


@dataclass
class MeshContext:
    """Shared state across all viewer views."""

    name: str
    config_path: Path
    mesh_dir: Path
    build_dir: Path
    state: dict
    server_url: str
    token: str
    role: str
    console: Console
    pipeline_path: str = ""


@dataclass
class ViewState:
    """Tracks current view and its parameters."""

    view: str = "overview"
    layer_filter: str = ""
    artifact_label: str = ""
    search_query: str = ""
    search_results: list = field(default_factory=list)
    artifact_labels: list = field(default_factory=list)


def load_mesh_context(name: str, console: Console | None = None) -> MeshContext:
    """Load mesh config, state, and resolve build_dir."""
    from synix.mesh.config import DEFAULT_MESH_ROOT, load_mesh_config

    mesh_dir = DEFAULT_MESH_ROOT / name
    config_path = mesh_dir / "synix-mesh.toml"

    if not config_path.exists():
        raise ValueError(f"Mesh '{name}' not found at {config_path}")

    config = load_mesh_config(config_path)

    state_path = mesh_dir / "state.json"
    state: dict = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read state.json: %s", exc)

    role = state.get("role", "server")
    server_url = state.get("server_url", "")
    build_dir = mesh_dir / role / "build"

    return MeshContext(
        name=name,
        config_path=config_path,
        mesh_dir=mesh_dir,
        build_dir=build_dir,
        state=state,
        server_url=server_url,
        token=config.token,
        role=role,
        console=console or Console(),
        pipeline_path=config.pipeline_path,
    )


def _fetch_json(ctx: MeshContext, path: str) -> dict | None:
    """GET a JSON endpoint from the mesh server. Returns None on failure."""
    if not ctx.server_url:
        return None
    headers = auth_headers(ctx.token) if ctx.token else {}
    try:
        resp = httpx.get(f"{ctx.server_url}{path}", headers=headers, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        logger.debug("Failed to fetch %s: %s", path, exc)
    return None


def _post_json(ctx: MeshContext, path: str, body: dict) -> dict | None:
    """POST JSON to the mesh server. Returns None on failure."""
    if not ctx.server_url:
        return None
    headers = auth_headers(ctx.token) if ctx.token else {}
    try:
        resp = httpx.post(f"{ctx.server_url}{path}", json=body, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        logger.debug("Failed to POST %s: %s", path, exc)
    return None


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


def render_overview(ctx: MeshContext) -> None:
    """Render the overview: memory tree + status side by side."""
    from synix.build.artifacts import ArtifactStore

    c = ctx.console

    # -- Left: Memory Tree --
    tree = Tree("[bold]Memory Tree[/bold]")
    layer_counts: dict[str, tuple[int, int]] = {}  # name -> (level, count)

    if ctx.build_dir.exists():
        store = ArtifactStore(ctx.build_dir)
        for _label, info in store._manifest.items():
            layer_name = info.get("layer", "unknown")
            level = info.get("level", 0)
            prev = layer_counts.get(layer_name, (level, 0))
            layer_counts[layer_name] = (level, prev[1] + 1)

    for layer_name, (level, count) in sorted(layer_counts.items(), key=lambda x: x[1][0]):
        style = get_layer_style(level)
        tree.add(f"[{style}]{layer_name}[/{style}] (L{level}) \u2014 {count}")

    if not layer_counts:
        tree.add("[dim]No artifacts yet[/dim]")

    # -- Right: Status --
    status_lines: list[str] = []
    server_status = _fetch_json(ctx, "/api/v1/status")

    if server_status:
        sessions = server_status.get("sessions", {})
        total = sessions.get("total", 0)
        processed = sessions.get("processed", 0)
        pending = sessions.get("pending", 0)
        builds = server_status.get("build_count", 0)

        status_lines.append("[bold]Sessions[/bold]")
        status_lines.append(f"  Total: {total}  Processed: {processed}")
        status_lines.append(f"  Pending: {pending}  Builds: {builds}")
        status_lines.append("")
    else:
        status_lines.append("[dim](server unreachable)[/dim]")
        status_lines.append("")

    # Cluster info
    leader = ctx.state.get("term", {}).get("leader_id", "unknown")
    hostname = ctx.state.get("my_hostname", "unknown")
    status_lines.append(f"[bold]Cluster:[/bold] {hostname}")
    if leader:
        status_lines.append(f"  Leader: {leader}")
    if server_status:
        status_lines.append(f"  Uptime: {_format_uptime(server_status.get('uptime_seconds', 0))}")

    status_text = "\n".join(status_lines)

    # Render side by side in a panel
    left_panel = Panel(tree, title="Memory", border_style="green", expand=True)
    right_panel = Panel(status_text, title="Status", border_style="blue", expand=True)

    c.print(
        Panel(
            Columns([left_panel, right_panel], equal=True, expand=True),
            title=f"[bold]Synix Mesh: {ctx.name}[/bold]",
            border_style="white",
        )
    )


def render_artifacts(ctx: MeshContext, layer_filter: str = "") -> list[str]:
    """Render artifact list, optionally filtered by layer. Returns ordered labels."""
    from synix.build.artifacts import ArtifactStore

    c = ctx.console

    if not ctx.build_dir.exists():
        c.print("[dim]No build directory found.[/dim]")
        return []

    store = ArtifactStore(ctx.build_dir)
    manifest = store._manifest

    if not manifest:
        c.print("[dim]No artifacts found.[/dim]")
        return []

    # Group by layer, sorted by level
    by_layer: dict[str, list[tuple[str, dict]]] = {}
    for art_label, info in manifest.items():
        layer_name = info.get("layer", "unknown")
        if layer_filter and layer_name != layer_filter:
            continue
        by_layer.setdefault(layer_name, []).append((art_label, info))

    if not by_layer:
        if layer_filter:
            c.print(f"[dim]No artifacts found in layer:[/dim] {layer_filter}")
        else:
            c.print("[dim]No artifacts found.[/dim]")
        return []

    sorted_layers = sorted(
        by_layer.items(),
        key=lambda x: x[1][0][1].get("level", 0) if x[1] else 0,
    )

    row_num = 0
    ordered_labels: list[str] = []

    for layer_name, entries in sorted_layers:
        level = entries[0][1].get("level", 0) if entries else 0
        style = get_layer_style(level)

        table = Table(
            title=f"[{style} bold]{layer_name}[/{style} bold] (L{level}) \u2014 {len(entries)} artifacts",
            box=box.ROUNDED,
            show_header=True,
        )
        table.add_column("#", style="dim", no_wrap=True, width=4)
        table.add_column("Label", style="bold", no_wrap=True)
        table.add_column("Artifact ID", style="dim", no_wrap=True)
        table.add_column("Date", no_wrap=True)
        table.add_column("Title / Summary", max_width=60)

        for art_label, _info in sorted(entries, key=lambda x: x[0]):
            row_num += 1
            ordered_labels.append(art_label)

            artifact = store.load_artifact(art_label)
            if artifact is None:
                table.add_row(str(row_num), art_label, "-", "-", "[dim]<missing>[/dim]")
                continue

            raw_id = (artifact.artifact_id or "").removeprefix("sha256:")
            short_id = raw_id[:7] if raw_id else "-"

            date = artifact.metadata.get("date", "")
            if not date and artifact.metadata.get("month"):
                date = artifact.metadata["month"]
            title = artifact.metadata.get("title", "")

            if not title and artifact.content:
                first_line = artifact.content.strip().split("\n")[0]
                title = first_line.lstrip("# ").strip()
                if len(title) > 60:
                    title = title[:57] + "..."

            table.add_row(str(row_num), art_label, short_id, str(date), title)

        c.print(table)
        c.print()

    return ordered_labels


def render_detail(ctx: MeshContext, label: str) -> None:
    """Render a single artifact's detail view."""
    from synix.build.artifacts import ArtifactStore
    from synix.build.provenance import ProvenanceTracker

    c = ctx.console
    store = ArtifactStore(ctx.build_dir)

    # Resolve prefix
    try:
        resolved = store.resolve_prefix(label)
    except ValueError as exc:
        c.print(f"[red]Ambiguous:[/red] {exc}")
        return

    if resolved is None:
        c.print(f"[red]Artifact not found:[/red] {label}")
        return

    artifact = store.load_artifact(resolved)
    if artifact is None:
        c.print(f"[red]Artifact not found:[/red] {resolved}")
        return

    info = store._manifest.get(resolved, {})
    level = info.get("level", 0)
    layer_name = info.get("layer", "unknown")
    style = get_layer_style(level)

    # Metadata header
    meta_parts = [f"[dim]Layer:[/dim] [{style}]{layer_name}[/{style}] (L{level})"]
    meta_parts.append(f"[dim]Type:[/dim] {artifact.artifact_type}")
    meta_parts.append(f"[dim]Label:[/dim] {resolved}")
    if artifact.metadata.get("date"):
        meta_parts.append(f"[dim]Date:[/dim] {artifact.metadata['date']}")
    if artifact.metadata.get("month"):
        meta_parts.append(f"[dim]Month:[/dim] {artifact.metadata['month']}")
    if artifact.metadata.get("title"):
        meta_parts.append(f"[dim]Title:[/dim] {artifact.metadata['title']}")

    c.print("  ".join(meta_parts))
    c.print()

    # Provenance chain (depth-limited to avoid stack overflow on deep DAGs)
    provenance = ProvenanceTracker(ctx.build_dir)
    chain = provenance.get_chain(resolved)
    if chain:
        prov_tree = Tree("[bold]Provenance:[/bold]")
        max_depth = 10

        def _add_parents(node, lbl, visited=None, depth=0):
            if visited is None:
                visited = set()
            if lbl in visited or depth >= max_depth:
                if depth >= max_depth:
                    node.add("[dim]... (truncated)[/dim]")
                return
            visited.add(lbl)
            rec = provenance.get_record(lbl)
            if rec:
                for parent_label in sorted(rec.parent_labels):
                    child = node.add(f"[dim]{parent_label}[/dim]")
                    _add_parents(child, parent_label, visited, depth + 1)

        _add_parents(prov_tree, resolved)
        c.print(prov_tree)
        c.print()

    # Content
    c.print(Markdown(artifact.content))
    c.print()


def render_search(ctx: MeshContext, query: str) -> list[dict]:
    """Render search results. Returns list of result dicts for navigation."""
    c = ctx.console

    if not query:
        c.print("[dim]Usage: s <query>[/dim]")
        return []

    # Try server search first
    results_data = _post_json(ctx, "/api/v1/search", {"query": query, "mode": "hybrid", "limit": 10})

    results = []
    if results_data and "results" in results_data:
        results = results_data["results"]
    elif results_data and "error" in results_data:
        # Fall back to local keyword search
        results = _local_search(ctx, query)
    else:
        # Server unreachable, try local
        results = _local_search(ctx, query)

    if not results:
        c.print(f'[dim]No results for:[/dim] "{query}"')
        return []

    c.print(f'\n[bold]Search results for:[/bold] "{query}"\n')

    table = Table(box=box.ROUNDED, show_header=True)
    table.add_column("#", style="dim", no_wrap=True, width=4)
    table.add_column("Label", style="bold", no_wrap=True)
    table.add_column("Layer", no_wrap=True)
    table.add_column("Score", no_wrap=True, justify="right")
    table.add_column("Snippet", max_width=60)

    for i, result in enumerate(results, 1):
        label = result.get("label", "")
        layer = result.get("layer_name", "")
        score = result.get("score", 0)
        content = result.get("content", "")
        level = result.get("layer_level", 0)
        style = get_layer_style(level)

        # Snippet: first 200 chars
        snippet = content[:200].replace("\n", " ").strip()
        if len(content) > 200:
            snippet += "..."

        table.add_row(str(i), label, f"[{style}]{layer}[/{style}]", f"{score:.2f}", snippet)

    c.print(table)
    c.print()
    return results


def _local_search(ctx: MeshContext, query: str) -> list[dict]:
    """Fall back to local FTS5 search when server is unreachable."""
    search_db = ctx.build_dir / "search.db"
    if not search_db.exists():
        return []

    try:
        from synix.search.indexer import SearchIndexProjection

        projection = SearchIndexProjection(str(ctx.build_dir))
        results = projection.query(query)
        out = []
        for r in results[:10]:
            out.append(
                {
                    "label": r.label,
                    "layer_name": r.layer_name,
                    "layer_level": r.layer_level,
                    "score": r.score,
                    "content": r.content,
                }
            )
        return out
    except Exception as exc:
        logger.debug("Local search failed: %s", exc)
        return []


def render_config(ctx: MeshContext) -> None:
    """Render mesh config as Rich panels per section."""
    c = ctx.console

    if not ctx.config_path.exists():
        c.print(f"[red]Config not found:[/red] {ctx.config_path}")
        return

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    raw = tomllib.loads(ctx.config_path.read_text())

    c.print(f"[bold]Config:[/bold] {ctx.config_path}\n")

    for section, values in raw.items():
        if isinstance(values, dict):
            lines = []
            for key, val in values.items():
                if isinstance(val, dict):
                    # Nested section (e.g., deploy.server)
                    for sub_key, sub_val in val.items():
                        lines.append(f"  {sub_key} = {_format_toml_value(sub_val)}")
                else:
                    lines.append(f"  {key} = {_format_toml_value(val)}")
            content = "\n".join(lines) if lines else "[dim]empty[/dim]"
            c.print(Panel(content, title=f"[bold]{section}[/bold]", border_style="cyan"))
        else:
            c.print(f"  [bold]{section}[/bold] = {_format_toml_value(values)}")

    c.print()


def _format_toml_value(val: object) -> str:
    """Format a TOML value for display."""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, str):
        return f'"{val}"' if val else '[dim]""[/dim]'
    if isinstance(val, list):
        if not val:
            return "[dim][][/dim]"
        return str(val)
    return str(val)


def render_builds(ctx: MeshContext) -> None:
    """Render build information."""
    from synix.build.artifacts import ArtifactStore

    c = ctx.console

    # Scheduler status from API
    build_status = _fetch_json(ctx, "/api/v1/builds/status")
    server_status = _fetch_json(ctx, "/api/v1/status")

    # Scheduler info
    if build_status:
        c.print(
            Panel(
                _format_scheduler_info(build_status),
                title="[bold]Scheduler[/bold]",
                border_style="green",
            )
        )
    elif server_status and "scheduler" in server_status:
        c.print(
            Panel(
                _format_scheduler_info(server_status["scheduler"]),
                title="[bold]Scheduler[/bold]",
                border_style="green",
            )
        )
    else:
        c.print("[dim](scheduler status unavailable)[/dim]")

    c.print()

    # Build counts from server
    if server_status:
        builds = server_status.get("build_count", 0)
        sessions = server_status.get("sessions", {})
        c.print(f"[bold]Total builds:[/bold] {builds}")
        c.print(f"[bold]Sessions:[/bold] {sessions.get('total', 0)} total, {sessions.get('pending', 0)} pending")
        c.print()

    # Per-layer artifact counts from manifest
    if ctx.build_dir.exists():
        store = ArtifactStore(ctx.build_dir)
        manifest = store._manifest

        if manifest:
            layer_info: dict[str, tuple[int, int, str]] = {}  # name -> (level, count, newest_date)
            for _label, info in manifest.items():
                layer_name = info.get("layer", "unknown")
                level = info.get("level", 0)
                prev = layer_info.get(layer_name, (level, 0, ""))
                layer_info[layer_name] = (level, prev[1] + 1, prev[2])

            table = Table(
                title="[bold]Artifacts by Layer[/bold]",
                box=box.ROUNDED,
                show_header=True,
            )
            table.add_column("Layer", style="bold")
            table.add_column("Level", style="dim", justify="center")
            table.add_column("Count", justify="right")

            for layer_name, (level, count, _) in sorted(layer_info.items(), key=lambda x: x[1][0]):
                style = get_layer_style(level)
                table.add_row(
                    f"[{style}]{layer_name}[/{style}]",
                    f"L{level}",
                    str(count),
                )

            c.print(table)
    else:
        c.print("[dim]No build directory found.[/dim]")

    c.print()


def _format_scheduler_info(data: dict) -> str:
    """Format scheduler status dict into display lines."""
    lines = []
    state = data.get("state", "unknown")
    lines.append(f"  State: [bold]{state}[/bold]")

    pending = data.get("pending_count", data.get("pending"))
    if pending is not None:
        lines.append(f"  Pending: {pending}")

    last_build = data.get("last_build_secs_ago")
    if last_build is not None:
        from synix.mesh.dashboard import _format_secs_ago

        lines.append(f"  Last build: {_format_secs_ago(last_build)}")

    next_build = data.get("next_build_in_secs")
    if next_build is not None:
        lines.append(f"  Next build in: {_format_uptime(next_build)}")

    return "\n".join(lines)


def render_pipeline(ctx: MeshContext) -> None:
    """Render the pipeline definition: metadata, layer DAG, and projections."""
    from synix.build.dag import resolve_build_order
    from synix.build.pipeline import load_pipeline
    from synix.core.models import FlatFile, SearchIndex, Source, Transform

    c = ctx.console

    # Resolve pipeline path relative to config directory
    pipeline_path = ctx.pipeline_path
    if pipeline_path and not Path(pipeline_path).is_absolute():
        pipeline_path = str(ctx.config_path.parent / pipeline_path)

    if not pipeline_path:
        c.print("[red]No pipeline path configured.[/red]")
        return

    try:
        pipeline = load_pipeline(pipeline_path)
    except FileNotFoundError:
        c.print(f"[red]Pipeline file not found:[/red] {pipeline_path}")
        return
    except (ImportError, ValueError, TypeError) as exc:
        c.print(f"[red]Failed to load pipeline:[/red] {exc}")
        return

    # -- Section 1: Pipeline metadata --
    meta_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    meta_table.add_column("Key", style="bold")
    meta_table.add_column("Value")

    meta_table.add_row("Name", pipeline.name)
    meta_table.add_row("Source dir", pipeline.source_dir)
    meta_table.add_row("Build dir", pipeline.build_dir)
    meta_table.add_row("Concurrency", str(pipeline.concurrency))

    if pipeline.llm_config:
        model = pipeline.llm_config.get("model", "")
        provider = pipeline.llm_config.get("provider", "")
        if model:
            meta_table.add_row("Model", model)
        if provider:
            meta_table.add_row("Provider", provider)

    c.print(Panel(meta_table, title="[bold]Pipeline[/bold]", border_style="green"))
    c.print()

    # -- Section 2: Layer DAG --
    build_order = resolve_build_order(pipeline)
    dag_tree = Tree("[bold]Layer DAG[/bold]")

    for layer in build_order:
        type_name = "Source" if isinstance(layer, Source) else type(layer).__name__
        style = get_layer_style(layer._level)
        node_label = f"[{style}]{layer.name}[/{style}] (L{layer._level}) \u2014 {type_name}"
        node = dag_tree.add(node_label)

        for dep in layer.depends_on:
            node.add(f"[dim]\u2190 {dep.name}[/dim]")

        if isinstance(layer, Transform):
            if layer.prompt_name:
                node.add(f"[dim]prompt: {layer.prompt_name}[/dim]")
            if layer.context_budget:
                node.add(f"[dim]context_budget: {layer.context_budget}[/dim]")

    c.print(dag_tree)
    c.print()

    # -- Section 3: Projections --
    if pipeline.projections:
        proj_table = Table(
            title="[bold]Projections[/bold]",
            box=box.ROUNDED,
            show_header=True,
        )
        proj_table.add_column("Name", style="bold")
        proj_table.add_column("Type")
        proj_table.add_column("Sources")
        proj_table.add_column("Details")

        for proj in pipeline.projections:
            source_names = ", ".join(s.name for s in proj.sources)
            if isinstance(proj, SearchIndex):
                proj_type = "SearchIndex"
                details_parts = [f"search: {proj.search}"]
                if proj.embedding_config:
                    details_parts.append(f"embeddings: {proj.embedding_config}")
                details = "  ".join(details_parts)
            elif isinstance(proj, FlatFile):
                proj_type = "FlatFile"
                details = f"output: {proj.output_path}"
            else:
                proj_type = type(proj).__name__
                details = ""
            proj_table.add_row(proj.name, proj_type, source_names, details)

        c.print(proj_table)
        c.print()

    # -- Validators/Fixers summary --
    if pipeline.validators or pipeline.fixers:
        counts = []
        if pipeline.validators:
            counts.append(f"{len(pipeline.validators)} validator(s)")
        if pipeline.fixers:
            counts.append(f"{len(pipeline.fixers)} fixer(s)")
        c.print(f"[dim]{', '.join(counts)} configured[/dim]")
        c.print()


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


def print_nav_hints(console: Console, view: str) -> None:
    """Print navigation hints for the current view."""
    hints = {
        "overview": r"  \[a]rtifacts  \[s]earch  \[c]onfig  \[b]uilds  \[p]ipeline  \[q]uit",
        "artifacts": r"  \[#] view artifact  \[a <layer>] filter  \[enter] back  \[q]uit",
        "detail": r"  \[enter] back  \[q]uit",
        "search": r"  \[#] view artifact  \[s <query>] new search  \[enter] back  \[q]uit",
        "config": r"  \[enter] back  \[q]uit",
        "builds": r"  \[enter] back  \[q]uit",
        "pipeline": r"  \[enter] back  \[q]uit",
    }
    console.print(hints.get(view, "  [enter] back  [q]uit"))


def dispatch(cmd: str, current: ViewState) -> ViewState:
    """Parse user input and return updated ViewState."""
    new = replace(current)

    if not cmd:
        # Enter with no input → back to overview
        new.view = "overview"
        new.layer_filter = ""
        new.artifact_label = ""
        return new

    parts = cmd.split(None, 1)
    action = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if action in ("q", "quit", "exit"):
        new.view = "quit"
        return new

    if action in ("a", "artifacts"):
        new.view = "artifacts"
        new.layer_filter = arg
        new.artifact_labels = []
        return new

    if action in ("s", "search"):
        new.view = "search"
        new.search_query = arg
        return new

    if action in ("c", "config"):
        new.view = "config"
        return new

    if action in ("b", "builds"):
        new.view = "builds"
        return new

    if action in ("p", "pipeline"):
        new.view = "pipeline"
        return new

    # Number → drill into artifact (from artifacts or search view)
    if action.isdigit():
        idx = int(action) - 1
        if current.view == "artifacts" and 0 <= idx < len(current.artifact_labels):
            new.view = "detail"
            new.artifact_label = current.artifact_labels[idx]
            return new
        if current.view == "search" and 0 <= idx < len(current.search_results):
            result = current.search_results[idx]
            new.view = "detail"
            new.artifact_label = result.get("label", "")
            return new

    return new


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_viewer(name: str) -> None:
    """Launch the interactive memory viewer."""
    console = Console()
    ctx = load_mesh_context(name, console=console)
    vs = ViewState()

    while True:
        console.clear()

        if vs.view == "overview":
            render_overview(ctx)
        elif vs.view == "artifacts":
            vs.artifact_labels = render_artifacts(ctx, layer_filter=vs.layer_filter)
        elif vs.view == "detail":
            render_detail(ctx, label=vs.artifact_label)
        elif vs.view == "search":
            results = render_search(ctx, query=vs.search_query)
            vs.search_results = results
        elif vs.view == "config":
            render_config(ctx)
        elif vs.view == "builds":
            render_builds(ctx)
        elif vs.view == "pipeline":
            render_pipeline(ctx)

        print_nav_hints(console, vs.view)

        try:
            cmd = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        vs = dispatch(cmd, vs)
        if vs.view == "quit":
            break

    console.print("\n[dim]Viewer closed.[/dim]")
