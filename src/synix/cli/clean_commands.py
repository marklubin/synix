"""Clean command — remove release targets and build-time work files."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import click

from synix.build.refs import synix_dir_for_build_dir
from synix.cli.main import console

logger = logging.getLogger(__name__)


def _warn_external_targets(release_path: Path) -> None:
    """Check release receipt for external target paths and warn the user."""
    receipt_path = release_path / "receipt.json"
    if not receipt_path.exists():
        return
    try:
        data = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    for adapter_name, adapter_data in data.get("adapters", {}).items():
        target = adapter_data.get("target", "")
        if target and str(release_path) not in target:
            console.print(
                f"[yellow]Note:[/yellow] release '{release_path.name}' adapter "
                f"'{adapter_name}' wrote to external target: {target}\n"
                f"  Clean will remove the receipt but not the external files."
            )


@click.command()
@click.option("--build-dir", default="./build", help="Build directory (used to locate .synix)")
@click.option("--synix-dir", default=None, help="Explicit .synix directory")
@click.option("--release", "release_name", default=None, help="Clean a specific release target")
@click.option("-y", "--yes", is_flag=True, help="Skip confirmation prompt")
def clean(build_dir: str, synix_dir: str | None, release_name: str | None, yes: bool):
    """Remove release targets and build-time work files.

    By default cleans all releases and the work directory under .synix/.
    Use --release to clean a specific release target only.
    """
    try:
        sd = synix_dir_for_build_dir(build_dir, configured_synix_dir=synix_dir)
    except ValueError:
        # Also clean legacy build/ if it exists
        build_path = Path(build_dir)
        if build_path.exists():
            if not yes:
                console.print(f"This will delete [bold]{build_path}[/bold] and all its contents.")
                if not click.confirm("Continue?"):
                    console.print("[dim]Aborted.[/dim]")
                    return
            shutil.rmtree(build_path)
            console.print(f"[green]Cleaned:[/green] {build_path}")
        else:
            console.print("[dim]Nothing to clean — no .synix directory found.[/dim]")
        return

    targets: list[tuple[str, Path]] = []

    if release_name:
        release_path = sd / "releases" / release_name
        if release_path.exists():
            targets.append((f"release '{release_name}'", release_path))
            _warn_external_targets(release_path)
        else:
            console.print(f"[dim]Release '{release_name}' does not exist.[/dim]")
            return
    else:
        releases_dir = sd / "releases"
        if releases_dir.exists():
            targets.append(("releases", releases_dir))
            # Warn about external targets in any release
            for rp in sorted(releases_dir.iterdir()):
                if rp.is_dir():
                    _warn_external_targets(rp)
        work_dir = sd / "work"
        if work_dir.exists():
            targets.append(("work", work_dir))
        # Also clean legacy build/ if it exists
        build_path = Path(build_dir)
        if build_path.exists():
            targets.append(("legacy build", build_path))

    if not targets:
        console.print("[dim]Nothing to clean.[/dim]")
        return

    if not yes:
        names = ", ".join(name for name, _ in targets)
        console.print(f"This will clean: [bold]{names}[/bold]")
        if not click.confirm("Continue?"):
            console.print("[dim]Aborted.[/dim]")
            return

    for name, path in targets:
        shutil.rmtree(path)
        console.print(f"[green]Cleaned:[/green] {name} ({path})")

    # Release refs are preserved — they still point at valid snapshots
    # and can be used to re-materialize releases. Use `refs delete` to
    # explicitly remove refs.
