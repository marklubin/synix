"""Info command — synix info."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import click
from rich import box
from rich.table import Table
from rich.text import Text

from synix.cli.main import console, get_layer_style

SYNIX_LOGO = r"""
 ███████╗██╗   ██╗███╗   ██╗██╗██╗  ██╗
 ██╔════╝╚██╗ ██╔╝████╗  ██║██║╚██╗██╔╝
 ███████╗ ╚████╔╝ ██╔██╗ ██║██║ ╚███╔╝
 ╚════██║  ╚██╔╝  ██║╚██╗██║██║ ██╔██╗
 ███████║   ██║   ██║ ╚████║██║██╔╝ ██╗
 ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝
"""


def _get_version() -> str:
    """Get the synix package version from metadata."""
    try:
        from importlib.metadata import version

        return version("synix")
    except Exception:
        return "unknown"


def _get_python_version() -> str:
    """Get the Python version (first line only)."""
    return sys.version.split("\n")[0]


def _get_platform_info() -> str:
    """Get platform system and machine architecture."""
    return f"{platform.system()} {platform.machine()}"


def _show_system_info() -> None:
    """Display the system info table."""
    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Version", _get_version())
    table.add_row("Python", _get_python_version())
    table.add_row("Platform", _get_platform_info())

    console.print(table)


def _show_pipeline_info() -> None:
    """Try to load pipeline.py from the current directory and display config summary."""
    pipeline_path = Path.cwd() / "pipeline.py"
    if not pipeline_path.exists():
        console.print("[dim]No pipeline.py in current directory[/dim]")
        return

    try:
        from synix.build.pipeline import load_pipeline

        pipeline = load_pipeline(str(pipeline_path))
    except Exception as e:
        console.print(f"[yellow]Could not load pipeline.py:[/yellow] {e}")
        return

    table = Table(
        title="Pipeline Configuration",
        box=box.ROUNDED,
        show_header=False,
        padding=(0, 2),
    )
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Pipeline", pipeline.name)
    table.add_row("Layers", str(len(pipeline.layers)))

    # Layer names with levels
    layer_parts = []
    for layer in sorted(pipeline.layers, key=lambda l: l.level):
        style = get_layer_style(layer.level)
        layer_parts.append(f"[{style}]{layer.name}[/{style}] (L{layer.level})")
    if layer_parts:
        table.add_row("Layer Detail", ", ".join(layer_parts))

    table.add_row("Projections", str(len(pipeline.projections)))
    table.add_row("Validators", str(len(pipeline.validators)))
    table.add_row("Fixers", str(len(pipeline.fixers)))

    # LLM config
    llm_config = pipeline.llm_config
    if llm_config:
        model = llm_config.get("model", "-")
        provider = llm_config.get("provider", "-")
        table.add_row("LLM", f"{model} ({provider})")

    console.print(table)


def _show_build_status() -> None:
    """Show build status if a build directory exists."""
    import json
    import sqlite3

    build_path = Path.cwd() / "build"
    manifest_path = build_path / "manifest.json"

    if not manifest_path.exists():
        console.print("[dim]No build directory with manifest found[/dim]")
        return

    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        console.print("[yellow]Could not read manifest.json[/yellow]")
        return

    table = Table(
        title="Build Status",
        box=box.ROUNDED,
        show_header=False,
        padding=(0, 2),
    )
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Artifacts", str(len(manifest)))

    # Last build time — find the most recent artifact
    last_modified = None
    for _aid, info in manifest.items():
        path_str = info.get("path")
        if path_str:
            artifact_path = build_path / path_str
            if artifact_path.exists():
                mtime = artifact_path.stat().st_mtime
                if last_modified is None or mtime > last_modified:
                    last_modified = mtime

    if last_modified is not None:
        from datetime import datetime

        last_dt = datetime.fromtimestamp(last_modified)
        table.add_row("Last Build", last_dt.strftime("%Y-%m-%d %H:%M:%S"))

    # Search index status
    search_db = build_path / "search.db"
    if search_db.exists():
        try:
            conn = sqlite3.connect(str(search_db))
            cursor = conn.execute("SELECT COUNT(*) FROM search_index")
            count = cursor.fetchone()[0]
            conn.close()
            table.add_row("Search Index", f"{count} entries")
        except Exception:
            table.add_row("Search Index", "exists (could not read)")
    else:
        table.add_row("Search Index", "[dim]not built[/dim]")

    console.print(table)


@click.command()
def info():
    """Display Synix system information and configuration overview."""
    # Logo
    console.print(Text(SYNIX_LOGO, style="bold cyan"))
    console.print("[dim]A build system for agent memory[/dim]\n")

    # System info
    _show_system_info()
    console.print()

    # Pipeline configuration
    _show_pipeline_info()
    console.print()

    # Build status
    _show_build_status()
