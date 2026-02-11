"""Live progress display for pipeline builds."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Any

from rich.console import Console, ConsoleOptions, RenderResult
from rich.text import Text


@dataclass
class _ArtifactState:
    name: str
    status: str = "pending"  # pending, running, done, cached
    start_time: float = 0.0
    elapsed: float = 0.0


class BuildProgress:
    """Thread-safe live progress tracker for pipeline builds.

    Implements Rich's console protocol for rendering with Live.
    Updated from worker threads via the SynixLogger callbacks.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._layer_name = ""
        self._layer_level = 0
        self._layer_start = 0.0
        self._artifacts: dict[str, _ArtifactState] = {}
        self._order: list[str] = []
        self._completed_layers: list[dict[str, Any]] = []
        self._projection_states: list[dict[str, str]] = []
        # Map: layer_name → list of projection states filed under that layer
        self._layer_projections: dict[str, list[dict[str, str]]] = {}
        self._last_completed_layer: str = ""
        self._embedding_state: dict | None = None

    def layer_start(self, name: str, level: int) -> None:
        with self._lock:
            self._layer_name = name
            self._layer_level = level
            self._layer_start = time.time()
            self._artifacts.clear()
            self._order.clear()

    def layer_finish(self, name: str, built: int, cached: int) -> None:
        with self._lock:
            elapsed = time.time() - self._layer_start
            self._completed_layers.append({
                "name": name,
                "level": self._layer_level,
                "built": built,
                "cached": cached,
                "elapsed": elapsed,
            })
            self._last_completed_layer = name
            self._layer_name = ""
            self._artifacts.clear()
            self._order.clear()

    def artifact_start(self, name: str) -> None:
        with self._lock:
            if name not in self._artifacts:
                self._artifacts[name] = _ArtifactState(name, "running")
                self._order.append(name)
            else:
                self._artifacts[name].status = "running"
            self._artifacts[name].start_time = time.time()

    def artifact_finish(self, name: str, elapsed: float = 0.0) -> None:
        with self._lock:
            if name in self._artifacts:
                art = self._artifacts[name]
                art.status = "done"
                art.elapsed = elapsed or (time.time() - art.start_time)
            else:
                self._artifacts[name] = _ArtifactState(name, "done")
                self._artifacts[name].elapsed = elapsed
                self._order.append(name)

    def artifact_cached(self, name: str) -> None:
        with self._lock:
            if name not in self._artifacts:
                self._artifacts[name] = _ArtifactState(name, "cached")
                self._order.append(name)
            else:
                self._artifacts[name].status = "cached"

    def projection_start(self, name: str, triggered_by: str | None = None) -> None:
        with self._lock:
            entry = {"name": name, "status": "running"}
            self._projection_states.append(entry)
            layer = triggered_by or self._last_completed_layer
            if layer:
                self._layer_projections.setdefault(layer, []).append(entry)

    def projection_finish(self, name: str, triggered_by: str | None = None) -> None:
        with self._lock:
            for ps in self._projection_states:
                if ps["name"] == name:
                    ps["status"] = "done"
                    break

    def projection_cached(self, name: str, triggered_by: str | None = None) -> None:
        with self._lock:
            entry = {"name": name, "status": "cached"}
            self._projection_states.append(entry)
            layer = triggered_by or self._last_completed_layer
            if layer:
                self._layer_projections.setdefault(layer, []).append(entry)

    # -- Embedding events --

    def embedding_start(self, count: int, provider: str) -> None:
        with self._lock:
            self._embedding_state = {
                "count": count,
                "provider": provider,
                "completed": 0,
                "status": "running",
                "start_time": time.time(),
            }

    def embedding_progress(self, completed: int, total: int) -> None:
        with self._lock:
            if self._embedding_state:
                self._embedding_state["completed"] = completed

    def embedding_finish(self, count: int, cached: int, generated: int) -> None:
        with self._lock:
            if self._embedding_state:
                elapsed = time.time() - self._embedding_state["start_time"]
                self._embedding_state.update({
                    "status": "done",
                    "cached": cached,
                    "generated": generated,
                    "elapsed": elapsed,
                })
                # File it under the last projection
                layer = self._last_completed_layer
                if layer and layer in self._layer_projections:
                    projs = self._layer_projections[layer]
                    if projs:
                        projs[-1]["embedding_state"] = dict(self._embedding_state)
                self._embedding_state = None

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        with self._lock:
            # Completed layers (with inline projections)
            for layer in self._completed_layers:
                style = _level_style(layer["level"])
                if layer["cached"] > 0 and layer["built"] == 0:
                    detail = f"{layer['cached']} cached"
                elif layer["cached"] > 0:
                    detail = f"{layer['built']} built, {layer['cached']} cached"
                else:
                    detail = f"{layer['built']} built"
                yield Text.from_markup(
                    f"  [green]✓[/green] [{style}]{layer['name']}[/{style}] "
                    f"(level {layer['level']})  "
                    f"{detail}  "
                    f"[dim]{layer['elapsed']:.1f}s[/dim]"
                )

                # Projections filed under this layer
                for ps in self._layer_projections.get(layer["name"], []):
                    yield from self._render_projection_inline(ps)

            # Active embedding state (shown during materialization, between layers)
            if self._embedding_state and self._embedding_state["status"] == "running":
                now_emb = time.time()
                emb = self._embedding_state
                emb_elapsed = now_emb - emb["start_time"]
                completed = emb["completed"]
                count = emb["count"]
                provider = emb["provider"]
                yield Text.from_markup(
                    f"       └─ embeddings  "
                    f"[yellow]{completed}/{count}[/yellow]  "
                    f"[dim]{emb_elapsed:.1f}s[/dim]  "
                    f"[dim]({provider})[/dim]"
                )

            if not self._layer_name:
                return

            # Current layer header
            now = time.time()
            layer_elapsed = now - self._layer_start
            done_count = sum(
                1 for a in self._artifacts.values()
                if a.status in ("done", "cached")
            )
            running_count = sum(
                1 for a in self._artifacts.values() if a.status == "running"
            )
            total = len(self._order)

            style = _level_style(self._layer_level)
            parts = [
                f"  [{style}]{self._layer_name}[/{style}] "
                f"(level {self._layer_level})"
            ]
            if total > 0:
                parts.append(f"  {done_count}/{total}")
            if running_count > 0:
                parts.append(
                    f"  [yellow]⟳ {running_count} in flight[/yellow]"
                )
            parts.append(f"  [dim]{layer_elapsed:.1f}s[/dim]")
            yield Text.from_markup("".join(parts))

            # Artifact list — one per line
            if self._order:
                yield Text("")
                for name in self._order:
                    art = self._artifacts[name]
                    short = _short_name(name)
                    if art.status == "done":
                        yield Text.from_markup(
                            f"    [green]✓[/green] {short}  "
                            f"[dim]{art.elapsed:.1f}s[/dim]"
                        )
                    elif art.status == "cached":
                        yield Text.from_markup(
                            f"    [cyan]=[/cyan] {short}  [dim]cached[/dim]"
                        )
                    elif art.status == "running":
                        rt = now - art.start_time
                        yield Text.from_markup(
                            f"    [yellow]⟳[/yellow] {short}  "
                            f"[yellow]{rt:.1f}s[/yellow]"
                        )
                    else:
                        yield Text.from_markup(f"    [dim]· {short}[/dim]")

    @staticmethod
    def _render_projection_inline(ps: dict[str, str]) -> RenderResult:
        """Render a single projection state indented under its layer."""
        name = ps["name"]
        status = ps["status"]
        if status == "done":
            yield Text.from_markup(
                f"    └─ [magenta]{name}[/magenta]  [dim]materialized[/dim]"
            )
        elif status == "cached":
            yield Text.from_markup(
                f"    └─ [magenta]{name}[/magenta]  [dim]cached[/dim]"
            )
        elif status == "running":
            yield Text.from_markup(
                f"    └─ [magenta]{name}[/magenta]  [yellow]materializing...[/yellow]"
            )

        # Embedding state inline under projection
        emb = ps.get("embedding_state")
        if emb and emb.get("status") == "done":
            count = emb.get("count", 0)
            cached = emb.get("cached", 0)
            elapsed = emb.get("elapsed", 0.0)
            provider = emb.get("provider", "")
            yield Text.from_markup(
                f"       └─ embeddings  {count}/{count}  "
                f"[dim]{elapsed:.1f}s[/dim]  "
                f"[dim]({provider}, cached: {cached})[/dim]"
            )


def _short_name(name: str) -> str:
    """Shorten artifact description for display."""
    for prefix in (
        "episode ", "monthly rollup ", "topical rollup ",
        "core memory ",
    ):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    # Truncate long UUIDs: ep-chatgpt-67b96eaf-6884-800c-... -> ep-chatgpt-67b96eaf
    parts = name.split("-")
    if len(parts) > 3 and parts[1] in ("chatgpt", "claude"):
        return "-".join(parts[:3])
    if len(name) > 35:
        return name[:32] + "..."
    return name


def _level_style(level: int) -> str:
    return {0: "dim", 1: "blue", 2: "green", 3: "yellow"}.get(level, "white")
