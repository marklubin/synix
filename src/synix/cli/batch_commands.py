"""Batch build commands — synix batch-build run/resume/list/status/plan."""

from __future__ import annotations

import secrets
from datetime import datetime
from pathlib import Path

import click
from rich.table import Table

from synix.cli.main import console, pipeline_argument


def _experimental_warning() -> None:
    console.print("[yellow][Experimental][/yellow] Batch build uses OpenAI Batch API. This feature may change.")


@click.group("batch-build")
def batch_build():
    """[Experimental] Build pipeline using OpenAI Batch API."""
    pass


@batch_build.command()
@pipeline_argument
@click.option("--poll", is_flag=True, help="Stay alive and poll until the build completes.")
@click.option("--poll-interval", type=int, default=60, help="Seconds between status checks (default: 60).")
def run(pipeline_path: str, poll: bool, poll_interval: int):
    """Create a build instance and submit the first batch."""
    _experimental_warning()

    from synix.build.batch_runner import batch_run
    from synix.build.pipeline import load_pipeline

    pipeline = load_pipeline(pipeline_path)
    build_id = f"batch-{secrets.token_hex(4)}"

    console.print(f"  Build ID: [bold]{build_id}[/bold]")
    console.print(f"  Pipeline: {pipeline_path}")
    console.print()

    with console.status("[bold]Collecting requests...[/bold]"):
        result = batch_run(
            pipeline,
            build_id,
            poll=poll,
            poll_interval=poll_interval,
        )

    _print_result(result)


@batch_build.command()
@click.argument("build_id")
@pipeline_argument
@click.option("--poll", is_flag=True, help="Stay alive and poll until the build completes.")
@click.option("--poll-interval", type=int, default=60, help="Seconds between status checks.")
@click.option("--allow-pipeline-mismatch", is_flag=True, help="Resume despite pipeline fingerprint changes.")
@click.option("--reset-state", is_flag=True, help="Restart current layer after state corruption.")
def resume(
    build_id: str, pipeline_path: str, poll: bool, poll_interval: int, allow_pipeline_mismatch: bool, reset_state: bool
):
    """Resume an existing build instance."""
    _experimental_warning()

    from synix.build.batch_runner import batch_run
    from synix.build.pipeline import load_pipeline

    pipeline = load_pipeline(pipeline_path)

    console.print(f"  Resuming build: [bold]{build_id}[/bold]")
    console.print()

    with console.status("[bold]Checking batch status...[/bold]"):
        result = batch_run(
            pipeline,
            build_id,
            poll=poll,
            poll_interval=poll_interval,
            allow_pipeline_mismatch=allow_pipeline_mismatch,
            reset_state=reset_state,
        )

    _print_result(result)


@batch_build.command(name="list")
@click.option("--build-dir", type=click.Path(), default="./build", help="Build directory.")
def list_builds(build_dir: str):
    """Show all build instances and their status."""
    _experimental_warning()

    from synix.build.batch_state import BatchState

    builds = BatchState.list_builds(Path(build_dir))
    if not builds:
        console.print("  No batch builds found.")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Build ID")
    table.add_column("Status")
    table.add_column("Pipeline Hash")
    table.add_column("Created")
    table.add_column("Layers Done")
    table.add_column("Current")
    table.add_column("Errors")

    for b in builds:
        created = datetime.fromtimestamp(b.created_at).strftime("%Y-%m-%d %H:%M")
        layers_done = str(len(b.layers_completed))

        status_style = {
            "completed": "green",
            "completed_with_errors": "yellow",
            "failed": "red",
            "submitted": "cyan",
            "collecting": "blue",
            "pending": "dim",
        }.get(b.status, "white")

        table.add_row(
            b.build_id,
            f"[{status_style}]{b.status}[/{status_style}]",
            b.pipeline_hash[:12],
            created,
            layers_done,
            b.current_layer or "—",
            str(b.failed_requests) if b.failed_requests else "—",
        )

    console.print(table)


@batch_build.command()
@click.argument("build_id", required=False, default=None)
@click.option("--latest", is_flag=True, help="Show status for the most recent build.")
@click.option("--build-dir", type=click.Path(), default="./build", help="Build directory.")
def status(build_id: str | None, latest: bool, build_dir: str):
    """Detailed status for a specific build instance."""
    _experimental_warning()

    from synix.build.batch_state import BatchState

    if latest:
        builds = BatchState.list_builds(Path(build_dir))
        if not builds:
            console.print("[red]No batch builds found.[/red]")
            raise SystemExit(1)
        # Sort by creation time, pick most recent
        builds.sort(key=lambda b: b.created_at, reverse=True)
        build_id = builds[0].build_id
    elif build_id is None:
        console.print("[red]Provide a BUILD_ID or use --latest.[/red]")
        raise SystemExit(1)

    try:
        state = BatchState(Path(build_dir), build_id)
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise SystemExit(1) from exc

    manifest = state.load_manifest()
    if manifest is None:
        console.print(f"[red]Build {build_id!r} not found.[/red]")
        raise SystemExit(1)

    console.print(f"[bold]Build:[/bold] {manifest.build_id}")
    console.print(f"[bold]Status:[/bold] {manifest.status}")
    console.print(f"[bold]Pipeline Hash:[/bold] {manifest.pipeline_hash}")
    console.print(f"[bold]Created:[/bold] {datetime.fromtimestamp(manifest.created_at).strftime('%Y-%m-%d %H:%M:%S')}")
    console.print(f"[bold]Layers Completed:[/bold] {', '.join(manifest.layers_completed) or '—'}")
    console.print(f"[bold]Current Layer:[/bold] {manifest.current_layer or '—'}")
    if manifest.failed_requests:
        console.print(f"[bold]Failed Requests:[/bold] [yellow]{manifest.failed_requests}[/yellow]")
    if manifest.error:
        console.print(f"[bold]Error:[/bold] [red]{manifest.error}[/red]")
    console.print()

    # Per-batch details
    batches = state.get_batches()
    if batches:
        console.print("[bold]Batches:[/bold]")
        batch_table = Table(show_header=True, header_style="bold")
        batch_table.add_column("Batch ID")
        batch_table.add_column("Layer")
        batch_table.add_column("Requests")
        batch_table.add_column("Status")
        batch_table.add_column("OpenAI Dashboard")

        for bid, b in batches.items():
            dashboard_url = f"https://platform.openai.com/batches/{bid}" if bid.startswith("batch_") else "—"
            batch_table.add_row(
                bid[:20],
                b["layer"],
                str(len(b.get("keys", []))),
                b["status"],
                f"[link={dashboard_url}]{dashboard_url}[/link]" if dashboard_url != "—" else "—",
            )
        console.print(batch_table)
        console.print()

    # Error summary
    errors = state.get_errors()
    if errors:
        console.print(f"[bold]Errors:[/bold] {len(errors)} failed requests")
        for key, err in list(errors.items())[:10]:
            console.print(f"  {key[:16]}... [{err['code']}] {err['message']}")
        if len(errors) > 10:
            console.print(f"  ... and {len(errors) - 10} more")


@batch_build.command()
@pipeline_argument
def plan(pipeline_path: str):
    """Dry-run showing which layers would batch vs sync."""
    _experimental_warning()

    from synix.build.batch_runner import plan_batch
    from synix.build.pipeline import load_pipeline

    pipeline = load_pipeline(pipeline_path)
    layers_info = plan_batch(pipeline)

    table = Table(show_header=True, header_style="bold", title="Batch Build Plan")
    table.add_column("Level")
    table.add_column("Layer")
    table.add_column("Type")
    table.add_column("Mode")
    table.add_column("Est. Requests")

    for info in layers_info:
        mode = info["mode"]
        mode_style = {
            "batch": "cyan",
            "sync": "dim",
            "source": "green",
        }.get(mode, "white")

        est = str(info.get("estimated_requests", "—"))

        table.add_row(
            str(info["level"]),
            info["name"],
            info["type"],
            f"[{mode_style}]{mode}[/{mode_style}]",
            est if mode == "batch" else "—",
        )

    console.print(table)

    batch_count = sum(1 for i in layers_info if i["mode"] == "batch")
    sync_count = sum(1 for i in layers_info if i["mode"] == "sync")
    console.print(f"\n  {batch_count} layer(s) will use Batch API, {sync_count} will run synchronously.")


def _print_result(result) -> None:
    """Print a BatchRunResult summary."""
    status_style = {
        "completed": "green",
        "completed_with_errors": "yellow",
        "failed": "red",
        "submitted": "cyan",
        "polling": "blue",
    }.get(result.status, "white")

    console.print(f"\n[bold]Status:[/bold] [{status_style}]{result.status}[/{status_style}]")
    console.print(f"[bold]Build ID:[/bold] {result.build_id}")

    if result.layers_completed:
        console.print(f"[bold]Layers Completed:[/bold] {', '.join(result.layers_completed)}")
    if result.layers_pending:
        console.print(f"[bold]Layers Pending:[/bold] {', '.join(result.layers_pending)}")
    if result.batches_submitted:
        console.print(f"[bold]Batches:[/bold] {len(result.batches_submitted)}")
        for bid in result.batches_submitted:
            if bid.startswith("batch_"):
                console.print(
                    f"  [link=https://platform.openai.com/batches/{bid}]https://platform.openai.com/batches/{bid}[/link]"
                )
    if result.total_time:
        console.print(f"[bold]Time:[/bold] {result.total_time:.1f}s")

    if result.status == "submitted":
        console.print(f"\n  Resume with: [bold]synix batch-build resume {result.build_id} pipeline.py --poll[/bold]")

    if result.errors:
        console.print(f"\n[yellow]{len(result.errors)} request(s) failed.[/yellow]")
