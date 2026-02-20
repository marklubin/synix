"""CLI dashboard — live Rich display of mesh health, members, and activity."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import httpx
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from synix.mesh.auth import auth_headers

logger = logging.getLogger(__name__)


def tail_log_file(log_path: Path, lines: int = 20) -> list[dict]:
    """Read the last N lines from a JSON Lines log file.

    Returns parsed dicts for valid JSON lines, skips malformed lines.
    """
    if not log_path.exists():
        return []

    try:
        raw_lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    except Exception:
        logger.debug("Failed to read log file %s", log_path, exc_info=True)
        return []

    entries = []
    for line in raw_lines[-lines:]:
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _format_age(timestamp: float) -> str:
    """Format a timestamp as human-readable age (e.g., '2s ago', '5m ago')."""
    if timestamp <= 0:
        return "never"
    age = time.time() - timestamp
    if age < 0:
        return "just now"
    if age < 60:
        return f"{int(age)}s ago"
    if age < 3600:
        return f"{int(age / 60)}m ago"
    return f"{int(age / 3600)}h {int((age % 3600) / 60)}m ago"


def _format_secs_ago(secs: float | None) -> str:
    """Format a 'seconds ago' value as human-readable age."""
    if secs is None:
        return "never"
    if secs < 0:
        return "just now"
    if secs < 60:
        return f"{int(secs)}s ago"
    if secs < 3600:
        return f"{int(secs / 60)}m ago"
    return f"{int(secs / 3600)}h {int((secs % 3600) / 60)}m ago"


def _format_uptime(seconds: float) -> str:
    """Format uptime as Xh Ym."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m {int(seconds % 60)}s"
    hours = int(seconds / 3600)
    mins = int((seconds % 3600) / 60)
    return f"{hours}h {mins}m"


def _level_style(level: str) -> str:
    """Return Rich style for a log level."""
    level = level.upper()
    if level == "ERROR":
        return "bold red"
    if level == "WARNING":
        return "yellow"
    if level == "DEBUG":
        return "dim"
    return ""


class MeshDashboard:
    """Rich-renderable mesh dashboard.

    Data sources (read-only):
    1. state.json — local role, hostname, term, leader
    2. logs/{role}.log — tail last N lines, parse JSON
    3. GET /api/v1/status — build count, sessions, uptime
    4. GET /api/v1/cluster — member list with heartbeat times
    """

    def __init__(
        self,
        mesh_dir: Path,
        name: str,
        token: str = "",
        server_url: str = "",
        log_lines: int = 20,
    ):
        self.mesh_dir = mesh_dir
        self.name = name
        self.token = token
        self.server_url = server_url
        self.log_lines = log_lines
        self._server_status: dict | None = None
        self._cluster_info: dict | None = None
        self._server_reachable = False

    def refresh(self) -> None:
        """Fetch fresh data from server and local files."""
        if not self.server_url:
            self._load_server_url()

        if self.server_url:
            self._fetch_server_data()

    def _load_server_url(self) -> None:
        """Load server URL from state.json."""
        state_path = self.mesh_dir / "state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
                self.server_url = state.get("server_url", "")
            except Exception:
                pass

    def _fetch_server_data(self) -> None:
        """Fetch status and cluster info from server."""
        headers = auth_headers(self.token) if self.token else {}
        try:
            status_resp = httpx.get(
                f"{self.server_url}/api/v1/status",
                headers=headers,
                timeout=5,
            )
            if status_resp.status_code == 200:
                self._server_status = status_resp.json()
                self._server_reachable = True
            else:
                self._server_reachable = False
        except Exception:
            self._server_reachable = False
            self._server_status = None

        try:
            cluster_resp = httpx.get(
                f"{self.server_url}/api/v1/cluster",
                headers=headers,
                timeout=5,
            )
            if cluster_resp.status_code == 200:
                self._cluster_info = cluster_resp.json()
        except Exception:
            self._cluster_info = None

    def _get_local_state(self) -> dict:
        """Read state.json for local role/hostname/term."""
        state_path = self.mesh_dir / "state.json"
        if not state_path.exists():
            return {}
        try:
            return json.loads(state_path.read_text())
        except Exception:
            return {}

    def _build_health_panel(self) -> Panel:
        """Build the Health summary panel."""
        state = self._get_local_state()
        role = state.get("role", "unknown")
        hostname = state.get("my_hostname", "unknown")
        term = state.get("term", {})
        term_counter = term.get("counter", 0)
        leader = term.get("leader_id", "none")

        lines = []
        lines.append(
            f"Role: [bold]{role}[/bold]    Leader: [bold]{leader}[/bold]    Term: {term_counter}    Host: {hostname}"
        )

        if self._server_reachable and self._server_status:
            s = self._server_status
            uptime = _format_uptime(s.get("uptime_seconds", 0))
            sessions = s.get("sessions", {})
            total = sessions.get("total", 0)
            pending = sessions.get("pending", 0)
            scheduler = s.get("scheduler", {})
            sched_state = scheduler.get("state", "unknown")
            last_build_ago = scheduler.get("last_build_secs_ago")
            last_build_str = _format_secs_ago(last_build_ago) if last_build_ago else "never"

            builds = s.get("build_count", 0)
            lines.append(f"Builds: {builds}       Sessions: {total} total, {pending} pending    Uptime: {uptime}")
            lines.append(f"Scheduler: {sched_state}  Last build: {last_build_str}")
        elif not self._server_reachable:
            lines.append("[dim]\\[server unreachable][/dim]")

        return Panel("\n".join(lines), title="Health", border_style="green")

    def _build_members_panel(self) -> Panel:
        """Build the Members table panel."""
        table = Table(show_header=True, header_style="bold", expand=True, box=None)
        table.add_column("Hostname")
        table.add_column("Status")
        table.add_column("Last Heartbeat")
        table.add_column("Term")
        table.add_column("Config Hash")

        if self._cluster_info and "members" in self._cluster_info:
            members = self._cluster_info["members"]
            for hostname, info in members.items():
                last_hb = info.get("last_heartbeat", 0)
                age = time.time() - last_hb if last_hb > 0 else float("inf")

                if age < 180:
                    status = Text("healthy", style="green")
                elif age < 360:
                    status = Text("suspect", style="yellow")
                else:
                    status = Text("dead", style="red")

                term = info.get("term", {})
                term_counter = term.get("counter", 0)

                table.add_row(
                    hostname,
                    status,
                    _format_age(last_hb),
                    str(term_counter),
                    self._cluster_info.get("config_hash", "")[:12] + "...",
                )
        elif not self._server_reachable:
            table.add_row("[dim]\\[server unreachable][/dim]", "", "", "", "")

        return Panel(table, title="Members", border_style="blue")

    def _build_activity_panel(self) -> Panel:
        """Build the Recent Activity panel from log files."""
        state = self._get_local_state()
        role = state.get("role", "client")
        log_path = self.mesh_dir / "logs" / f"{role}.log"
        entries = tail_log_file(log_path, self.log_lines)

        lines = []
        for entry in entries:
            ts = entry.get("ts", "")
            # Extract HH:MM:SS from ISO timestamp
            if "T" in ts:
                time_part = ts.split("T")[1][:8]
            else:
                time_part = ts[:8] if len(ts) >= 8 else ts

            level = entry.get("level", "INFO")
            msg = entry.get("msg", "")
            style = _level_style(level)
            level_str = f"[{style}]{level:<5s}[/{style}]" if style else f"{level:<5s}"
            lines.append(f"{time_part} {level_str} {msg}")

        if not lines:
            lines.append("[dim]No log entries yet[/dim]")

        content = "\n".join(lines[-self.log_lines :])
        return Panel(content, title="Recent Activity", border_style="cyan")

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """Render the full dashboard."""
        yield Text(f" Synix Mesh: {self.name} ", style="bold white on blue")
        yield Text("")
        yield self._build_health_panel()
        yield Text("")
        yield self._build_members_panel()
        yield Text("")
        yield self._build_activity_panel()


def run_dashboard(
    mesh_dir: Path,
    name: str,
    token: str = "",
    server_url: str = "",
    refresh_interval: int = 3,
    log_lines: int = 20,
) -> None:
    """Run the live dashboard in a terminal."""
    console = Console()
    dashboard = MeshDashboard(
        mesh_dir=mesh_dir,
        name=name,
        token=token,
        server_url=server_url,
        log_lines=log_lines,
    )

    console.print(f"[dim]Dashboard for mesh '{name}' — refreshing every {refresh_interval}s (Ctrl+C to exit)[/dim]")

    try:
        with Live(dashboard, console=console, refresh_per_second=1, screen=True) as live:
            while True:
                dashboard.refresh()
                live.update(dashboard)
                time.sleep(refresh_interval)
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard stopped.[/dim]")
