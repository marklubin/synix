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
    table.add_row("Surfaces", str(len(pipeline.surfaces)))

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


def _discover_synix_dir(build_dir: str | None = None, synix_dir_opt: str | None = None) -> Path | None:
    """Discover the .synix directory from explicit options or convention.

    Priority:
    1. Explicit --synix-dir option
    2. Explicit --build-dir option (resolves via synix_dir_for_build_dir)
    3. pipeline.py in cwd (uses pipeline.build_dir)
    4. Convention: cwd/build → sibling .synix
    5. Direct: cwd/.synix
    """
    from synix.build.refs import synix_dir_for_build_dir

    # 1. Explicit --synix-dir
    if synix_dir_opt:
        resolved = Path(synix_dir_opt)
        if resolved.exists():
            return resolved
        return None

    # 2. Explicit --build-dir
    if build_dir:
        try:
            candidate = synix_dir_for_build_dir(Path(build_dir))
            if candidate.exists():
                return candidate
        except (ValueError, OSError):
            pass
        return None

    # 3. pipeline.py in cwd
    pipeline_file = Path.cwd() / "pipeline.py"
    if pipeline_file.exists():
        try:
            from synix.build.pipeline import load_pipeline

            p = load_pipeline(str(pipeline_file))
            candidate = synix_dir_for_build_dir(Path(p.build_dir))
            if candidate.exists():
                return candidate
        except Exception:
            pass

    # 4. Convention: cwd/build
    try:
        candidate = synix_dir_for_build_dir(Path.cwd() / "build")
        if candidate.exists():
            return candidate
    except (ValueError, OSError):
        pass

    # 5. Direct: cwd/.synix
    candidate = Path.cwd() / ".synix"
    if candidate.exists():
        return candidate

    return None


def _show_build_status(build_dir: str | None = None, synix_dir_opt: str | None = None) -> None:
    """Show build status if a .synix snapshot store exists."""
    from synix.build.snapshot_view import SnapshotView

    synix_dir = _discover_synix_dir(build_dir, synix_dir_opt)

    if synix_dir is None:
        console.print("[dim]No build state found[/dim]")
        return

    # Open snapshot view and list artifacts grouped by layer
    try:
        view = SnapshotView.open(synix_dir)
    except (ValueError, KeyError):
        console.print("[dim]No build state found[/dim]")
        return

    artifacts = view.list_artifacts()

    table = Table(
        title="Build Status",
        box=box.ROUNDED,
        show_header=False,
        padding=(0, 2),
    )
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Artifacts", str(len(artifacts)))

    # Group by layer
    layers: dict[str, int] = {}
    for art in artifacts:
        meta = art.get("metadata", {})
        layer = meta.get("layer_name", "unknown")
        layers[layer] = layers.get(layer, 0) + 1

    if layers:
        layer_parts = []
        for layer_name, count in sorted(layers.items()):
            layer_parts.append(f"{layer_name} ({count})")
        table.add_row("Layers", ", ".join(layer_parts))

    # Check for releases — read from receipt.json for accurate target info
    releases_dir = synix_dir / "releases"
    if releases_dir.exists():
        release_parts = []
        for release_path in sorted(releases_dir.iterdir()):
            if not release_path.is_dir():
                continue
            receipt_file = release_path / "receipt.json"
            if not receipt_file.exists():
                continue
            import json as _json

            receipt_data = _json.loads(receipt_file.read_text(encoding="utf-8"))
            adapters = receipt_data.get("adapters", {})
            if adapters:
                outputs = []
                for _adapter_name, adapter_info in adapters.items():
                    target = adapter_info.get("target", "")
                    adapter_type = adapter_info.get("adapter", "")
                    count = adapter_info.get("artifacts_applied", 0)
                    # Show short path if inside .synix/, full path if external
                    if target and str(synix_dir) in target:
                        short = Path(target).name
                    else:
                        short = target
                    outputs.append(f"{adapter_type}:{short} ({count})")
                release_parts.append(f"[green]{release_path.name}[/green]: {', '.join(outputs)}")
        if release_parts:
            table.add_row("Releases", "  ".join(release_parts))

    console.print(table)


@click.command()
@click.option("--build-dir", default=None, help="Build directory (used to locate .synix)")
@click.option("--synix-dir", default=None, help="Explicit .synix directory")
def info(build_dir: str | None, synix_dir: str | None):
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
    _show_build_status(build_dir, synix_dir)
