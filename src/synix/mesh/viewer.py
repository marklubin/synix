"""Interactive memory viewer — Textual TUI for browsing artifacts, search, config."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx
from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Markdown,
    Static,
    TabbedContent,
    TabPane,
)

from synix.cli.main import get_layer_style
from synix.mesh.auth import auth_headers
from synix.mesh.dashboard import _format_uptime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Textual screens
# ---------------------------------------------------------------------------


class DetailScreen(Screen):
    """Full-screen artifact detail view with metadata, provenance, and rendered content."""

    BINDINGS = [
        Binding("escape", "dismiss", "Back"),
        Binding("q", "quit", "Quit"),
    ]

    DEFAULT_CSS = """
    DetailScreen {
        layout: vertical;
    }
    #detail-meta {
        padding: 0 2;
        height: auto;
        max-height: 40%;
    }
    #detail-provenance {
        padding: 0 2;
        height: auto;
        max-height: 20%;
    }
    #detail-markdown {
        padding: 0 2;
        height: 1fr;
    }
    #detail-error {
        padding: 1 2;
        color: $error;
    }
    """

    def __init__(self, ctx: MeshContext, label: str) -> None:
        super().__init__()
        self.ctx = ctx
        self.label = label

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll():
            yield Static(id="detail-meta")
            yield Static(id="detail-provenance")
            yield Markdown(id="detail-markdown")
            yield Static(id="detail-error")
        yield Footer()

    def on_mount(self) -> None:
        self._load_artifact()

    def _load_artifact(self) -> None:
        from synix.build.artifacts import ArtifactStore
        from synix.build.provenance import ProvenanceTracker

        store = ArtifactStore(self.ctx.build_dir)

        try:
            resolved = store.resolve_prefix(self.label)
        except ValueError as exc:
            self.query_one("#detail-error", Static).update(f"Ambiguous: {exc}")
            return

        if resolved is None:
            self.query_one("#detail-error", Static).update(f"Artifact not found: {self.label}")
            return

        artifact = store.load_artifact(resolved)
        if artifact is None:
            self.query_one("#detail-error", Static).update(f"Artifact not found: {resolved}")
            return

        info = store._manifest.get(resolved, {})
        level = info.get("level", 0)
        layer_name = info.get("layer", "unknown")
        style = get_layer_style(level)

        self.sub_title = resolved

        # -- Metadata panel --
        meta_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        meta_table.add_column("Key", style="bold")
        meta_table.add_column("Value")
        meta_table.add_row("Layer", Text(f"{layer_name} (L{level})", style=style))
        meta_table.add_row("Type", artifact.artifact_type)
        meta_table.add_row("Label", resolved)
        if artifact.artifact_id:
            meta_table.add_row("ID", artifact.artifact_id)
        if artifact.metadata.get("date"):
            meta_table.add_row("Date", artifact.metadata["date"])
        if artifact.metadata.get("month"):
            meta_table.add_row("Month", artifact.metadata["month"])
        if artifact.metadata.get("title"):
            meta_table.add_row("Title", artifact.metadata["title"])
        # Show any extra metadata keys
        shown = {"date", "month", "title", "source_path"}
        for key, val in sorted(artifact.metadata.items()):
            if key not in shown and val:
                meta_table.add_row(key.replace("_", " ").title(), str(val))
        self.query_one("#detail-meta", Static).update(
            Panel(meta_table, title="[bold]Metadata[/bold]", border_style="cyan")
        )

        # -- Provenance tree --
        provenance = ProvenanceTracker(self.ctx.build_dir)
        chain = provenance.get_chain(resolved)
        if chain:
            prov_tree = Tree(f"[bold]Provenance:[/bold] {resolved}")
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
                        child = node.add(f"[dim]\u2190 {parent_label}[/dim]")
                        _add_parents(child, parent_label, visited, depth + 1)

            _add_parents(prov_tree, resolved)
            self.query_one("#detail-provenance", Static).update(prov_tree)

        # -- Content (rendered markdown) --
        self.query_one("#detail-markdown", Markdown).update(artifact.content)


def _manifest_search(ctx: MeshContext, query: str) -> list[dict]:
    """Search artifacts by scanning the manifest and content (no index needed)."""
    from synix.build.artifacts import ArtifactStore

    if not ctx.build_dir.exists():
        return []

    store = ArtifactStore(ctx.build_dir)
    if not store._manifest:
        return []

    query_lower = query.lower()
    results = []

    for label, info in store._manifest.items():
        # Check label and layer name
        score = 0.0
        if query_lower in label.lower():
            score = 0.8
        elif query_lower in info.get("layer", "").lower():
            score = 0.5

        # Check content
        if score == 0.0:
            artifact = store.load_artifact(label)
            if artifact and query_lower in artifact.content.lower():
                score = 0.6
            elif artifact and query_lower in str(artifact.metadata).lower():
                score = 0.4

        if score > 0:
            artifact = artifact if score >= 0.6 else store.load_artifact(label)
            content_snippet = ""
            if artifact:
                content_snippet = artifact.content[:200]
            results.append(
                {
                    "label": label,
                    "layer_name": info.get("layer", "unknown"),
                    "layer_level": info.get("level", 0),
                    "score": score,
                    "content": content_snippet,
                }
            )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:20]


class SearchScreen(Screen):
    """Search screen with input and results table."""

    BINDINGS = [
        Binding("escape", "dismiss_search", "Back"),
    ]

    DEFAULT_CSS = """
    SearchScreen {
        layout: vertical;
    }
    #search-input {
        margin: 0 0 1 0;
    }
    #search-results {
        height: 1fr;
    }
    #search-status {
        padding: 0 1;
        color: $text-muted;
        height: auto;
    }
    """

    def __init__(self, ctx: MeshContext) -> None:
        super().__init__()
        self.ctx = ctx
        self._results: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield Input(placeholder="Search artifacts...", id="search-input")
        yield Static(id="search-status")
        yield DataTable(id="search-results")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#search-results", DataTable)
        table.add_columns("#", "Label", "Layer", "Score", "Snippet")
        table.cursor_type = "row"
        self.query_one("#search-input", Input).focus()
        self.query_one("#search-status", Static).update("Type a query and press Enter to search.")

    def action_dismiss_search(self) -> None:
        """Dismiss search: if results table is focused, go back to input; otherwise pop."""
        results = self.query_one("#search-results", DataTable)
        if results.has_focus:
            self.query_one("#search-input", Input).focus()
        else:
            self.dismiss()

    @on(Input.Submitted, "#search-input")
    def _on_search_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return

        status = self.query_one("#search-status", Static)
        status.update(f"Searching for: {query}...")

        # Try server search first, then local FTS5, then manifest scan
        results_data = _post_json(self.ctx, "/api/v1/search", {"query": query, "mode": "hybrid", "limit": 10})

        results: list[dict] = []
        source = ""
        if results_data and "results" in results_data:
            results = results_data["results"]
            source = "server"
        else:
            results = _local_search(self.ctx, query)
            if results:
                source = "index"
            else:
                results = _manifest_search(self.ctx, query)
                source = "scan"

        self._results = results

        table = self.query_one("#search-results", DataTable)
        table.clear()

        if not results:
            status.update(f'No results for: "{query}"')
            return

        status.update(f'{len(results)} result(s) for: "{query}" (via {source})')

        for i, result in enumerate(results, 1):
            label = result.get("label", "")
            layer = result.get("layer_name", "")
            score = result.get("score", 0)
            content = result.get("content", "")
            level = result.get("layer_level", 0)
            row_style = get_layer_style(level)

            snippet = content[:100].replace("\n", " ").strip()
            if len(content) > 100:
                snippet += "..."

            table.add_row(
                str(i),
                label,
                Text(layer, style=row_style),
                f"{score:.2f}",
                snippet,
                key=label,
            )

        table.focus()

    @on(DataTable.RowSelected, "#search-results")
    def _on_result_selected(self, event: DataTable.RowSelected) -> None:
        label = str(event.row_key.value)
        if label:
            self.app.push_screen(DetailScreen(self.ctx, label))


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


class MeshViewerApp(App):
    """Interactive Synix mesh memory viewer."""

    CSS = """
    TabbedContent {
        height: 1fr;
    }
    TabPane {
        padding: 1 2;
    }
    #artifacts-table {
        height: 1fr;
    }
    #search-results {
        height: 1fr;
    }
    #detail-content {
        padding: 1 2;
    }
    #search-input {
        margin: 0 0 1 0;
    }
    .empty-message {
        color: $text-muted;
        padding: 1 0;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("slash", "search", "Search"),
    ]

    def __init__(self, ctx: MeshContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.title = f"Synix Mesh: {ctx.name}"

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Overview", id="tab-overview"):
                yield VerticalScroll(Static(id="overview-content"))
            with TabPane("Artifacts", id="tab-artifacts"):
                yield Static(id="artifacts-empty", classes="empty-message")
                yield DataTable(id="artifacts-table")
            with TabPane("Pipeline", id="tab-pipeline"):
                yield VerticalScroll(Static(id="pipeline-content"))
            with TabPane("Config", id="tab-config"):
                yield VerticalScroll(Static(id="config-content"))
            with TabPane("Builds", id="tab-builds"):
                yield VerticalScroll(Static(id="builds-content"))
        yield Footer()

    def on_mount(self) -> None:
        self._populate_overview()
        self._populate_artifacts()
        self._populate_pipeline()
        self._populate_config()
        self._populate_builds()

    def action_search(self) -> None:
        """Push the search screen."""
        self.push_screen(SearchScreen(self.ctx))

    @on(DataTable.RowSelected, "#artifacts-table")
    def _on_artifact_selected(self, event: DataTable.RowSelected) -> None:
        label = str(event.row_key.value)
        if label:
            self.push_screen(DetailScreen(self.ctx, label))

    # -- Tab population --

    def _populate_overview(self) -> None:
        content = self._build_overview()
        self.query_one("#overview-content", Static).update(content)

    def _populate_artifacts(self) -> None:
        from synix.build.artifacts import ArtifactStore

        table = self.query_one("#artifacts-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("#", "Layer", "Label", "ID", "Date", "Title")

        if not self.ctx.build_dir.exists():
            self.query_one("#artifacts-empty", Static).update("No build directory found.")
            return

        store = ArtifactStore(self.ctx.build_dir)
        manifest = store._manifest

        if not manifest:
            self.query_one("#artifacts-empty", Static).update("No artifacts found.")
            return

        by_layer: dict[str, list[tuple[str, dict]]] = {}
        for label, info in manifest.items():
            layer_name = info.get("layer", "unknown")
            by_layer.setdefault(layer_name, []).append((label, info))

        sorted_layers = sorted(
            by_layer.items(),
            key=lambda x: x[1][0][1].get("level", 0) if x[1] else 0,
        )

        row_num = 0
        for layer_name, entries in sorted_layers:
            level = entries[0][1].get("level", 0) if entries else 0
            style = get_layer_style(level)

            for label, _info in sorted(entries, key=lambda x: x[0]):
                row_num += 1
                artifact = store.load_artifact(label)

                if artifact is None:
                    table.add_row(
                        str(row_num),
                        Text(layer_name, style=style),
                        label,
                        "-",
                        "-",
                        Text("<missing>", style="dim"),
                        key=label,
                    )
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

                table.add_row(
                    str(row_num),
                    Text(layer_name, style=style),
                    label,
                    short_id,
                    str(date),
                    title,
                    key=label,
                )

    def _populate_pipeline(self) -> None:
        content = self._build_pipeline()
        self.query_one("#pipeline-content", Static).update(content)

    def _populate_config(self) -> None:
        content = self._build_config()
        self.query_one("#config-content", Static).update(content)

    def _populate_builds(self) -> None:
        content = self._build_builds()
        self.query_one("#builds-content", Static).update(content)

    # -- Renderable builders --

    def _build_overview(self) -> Panel:
        """Build the overview Rich renderable."""
        from synix.build.artifacts import ArtifactStore

        tree = Tree("[bold]Memory Tree[/bold]")
        layer_counts: dict[str, tuple[int, int]] = {}

        if self.ctx.build_dir.exists():
            store = ArtifactStore(self.ctx.build_dir)
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

        status_lines: list[str] = []
        server_status = _fetch_json(self.ctx, "/api/v1/status")

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

        leader = self.ctx.state.get("term", {}).get("leader_id", "unknown")
        hostname = self.ctx.state.get("my_hostname", "unknown")
        status_lines.append(f"[bold]Cluster:[/bold] {hostname}")
        if leader:
            status_lines.append(f"  Leader: {leader}")
        if server_status:
            status_lines.append(f"  Uptime: {_format_uptime(server_status.get('uptime_seconds', 0))}")

        status_text = "\n".join(status_lines)

        left_panel = Panel(tree, title="Memory", border_style="green", expand=True)
        right_panel = Panel(status_text, title="Status", border_style="blue", expand=True)

        return Panel(
            Columns([left_panel, right_panel], equal=True, expand=True),
            title=f"[bold]Synix Mesh: {self.ctx.name}[/bold]",
            border_style="white",
        )

    def _build_pipeline(self) -> Group | Text:
        """Build the pipeline Rich renderable."""
        from synix.build.dag import resolve_build_order
        from synix.build.pipeline import load_pipeline
        from synix.core.models import FlatFile, SearchIndex, Source, Transform

        pipeline_path = self.ctx.pipeline_path
        if pipeline_path and not Path(pipeline_path).is_absolute():
            pipeline_path = str(self.ctx.config_path.parent / pipeline_path)

        if not pipeline_path:
            return Text("No pipeline path configured.", style="red")

        try:
            pipeline = load_pipeline(pipeline_path)
        except FileNotFoundError:
            return Text(f"Pipeline file not found: {pipeline_path}", style="red")
        except (ImportError, ValueError, TypeError) as exc:
            return Text(f"Failed to load pipeline: {exc}", style="red")

        parts: list = []

        # Metadata
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

        parts.append(Panel(meta_table, title="[bold]Pipeline[/bold]", border_style="green"))

        # DAG
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

        parts.append(dag_tree)

        # Projections
        if pipeline.projections:
            proj_table = Table(title="[bold]Projections[/bold]", box=box.ROUNDED, show_header=True)
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

            parts.append(proj_table)

        # Validators/fixers
        if pipeline.validators or pipeline.fixers:
            counts = []
            if pipeline.validators:
                counts.append(f"{len(pipeline.validators)} validator(s)")
            if pipeline.fixers:
                counts.append(f"{len(pipeline.fixers)} fixer(s)")
            parts.append(Text(f"{', '.join(counts)} configured", style="dim"))

        return Group(*parts)

    def _build_config(self) -> Group | Text:
        """Build the config Rich renderable."""
        if not self.ctx.config_path.exists():
            return Text(f"Config not found: {self.ctx.config_path}", style="red")

        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        raw = tomllib.loads(self.ctx.config_path.read_text())

        parts: list = []
        parts.append(Text(f"Config: {self.ctx.config_path}", style="bold"))

        for section, values in raw.items():
            if isinstance(values, dict):
                lines = []
                for key, val in values.items():
                    if isinstance(val, dict):
                        for sub_key, sub_val in val.items():
                            lines.append(f"  {sub_key} = {_format_toml_value(sub_val)}")
                    else:
                        lines.append(f"  {key} = {_format_toml_value(val)}")
                content = "\n".join(lines) if lines else "[dim]empty[/dim]"
                parts.append(Panel(content, title=f"[bold]{section}[/bold]", border_style="cyan"))
            else:
                parts.append(Text(f"  {section} = {_format_toml_value(values)}", style="bold"))

        return Group(*parts)

    def _build_builds(self) -> Group:
        """Build the builds Rich renderable."""
        from synix.build.artifacts import ArtifactStore

        parts: list = []

        build_status = _fetch_json(self.ctx, "/api/v1/builds/status")
        server_status = _fetch_json(self.ctx, "/api/v1/status")

        if build_status:
            parts.append(
                Panel(
                    _format_scheduler_info(build_status),
                    title="[bold]Scheduler[/bold]",
                    border_style="green",
                )
            )
        elif server_status and "scheduler" in server_status:
            parts.append(
                Panel(
                    _format_scheduler_info(server_status["scheduler"]),
                    title="[bold]Scheduler[/bold]",
                    border_style="green",
                )
            )
        else:
            parts.append(Text("(scheduler status unavailable)", style="dim"))

        if server_status:
            builds = server_status.get("build_count", 0)
            sessions = server_status.get("sessions", {})
            parts.append(Text(f"Total builds: {builds}"))
            parts.append(Text(f"Sessions: {sessions.get('total', 0)} total, {sessions.get('pending', 0)} pending"))

        if self.ctx.build_dir.exists():
            store = ArtifactStore(self.ctx.build_dir)
            manifest = store._manifest

            if manifest:
                table = Table(title="[bold]Artifacts by Layer[/bold]", box=box.ROUNDED, show_header=True)
                table.add_column("Layer", style="bold")
                table.add_column("Level", style="dim", justify="center")
                table.add_column("Count", justify="right")

                layer_info: dict[str, tuple[int, int]] = {}
                for _label, info in manifest.items():
                    layer_name = info.get("layer", "unknown")
                    level = info.get("level", 0)
                    prev = layer_info.get(layer_name, (level, 0))
                    layer_info[layer_name] = (level, prev[1] + 1)

                for layer_name, (level, count) in sorted(layer_info.items(), key=lambda x: x[1][0]):
                    style = get_layer_style(level)
                    table.add_row(
                        f"[{style}]{layer_name}[/{style}]",
                        f"L{level}",
                        str(count),
                    )

                parts.append(table)
            else:
                parts.append(Text("No artifacts found.", style="dim"))
        else:
            parts.append(Text("No build directory found.", style="dim"))

        return Group(*parts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_viewer(name: str) -> None:
    """Launch the interactive memory viewer."""
    ctx = load_mesh_context(name, console=Console())
    app = MeshViewerApp(ctx)
    app.run()
