"""Build commands — synix build, synix plan."""

from __future__ import annotations

import json as json_module
import sys
import time

import click
from rich import box
from rich.panel import Panel
from rich.table import Table

from synix.cli.main import console, get_layer_style


@click.command()
@click.argument("pipeline_path", type=click.Path(exists=True))
@click.option("--source-dir", default=None, help="Override source directory")
@click.option("--build-dir", default=None, help="Override build directory")
@click.option("--verbose", "-v", count=True,
              help="Verbosity level: -v per-artifact, -vv debug/LLM details")
@click.option("--concurrency", "-j", default=1, type=int,
              help="Number of concurrent LLM requests (default 1 = sequential)")
def build(pipeline_path: str, source_dir: str | None, build_dir: str | None,
          verbose: int, concurrency: int):
    """Build memory artifacts from a pipeline definition.

    PIPELINE_PATH is the Python file defining the pipeline (e.g., pipeline.py).
    """
    from synix.build.pipeline import load_pipeline
    from synix.build.runner import run as run_pipeline

    # Trigger search projection registration
    import synix.search.indexer  # noqa: F401

    try:
        pipeline = load_pipeline(pipeline_path)
    except Exception as e:
        console.print(f"[red]Error loading pipeline:[/red] {e}")
        sys.exit(1)

    if source_dir:
        pipeline.source_dir = source_dir
    if build_dir:
        pipeline.build_dir = build_dir

    concurrency_label = f"{concurrency} threads" if concurrency > 1 else "sequential"
    console.print(Panel(
        f"[bold]Pipeline:[/bold] {pipeline.name}\n"
        f"[bold]Source:[/bold] {pipeline.source_dir}\n"
        f"[bold]Build:[/bold] {pipeline.build_dir}\n"
        f"[bold]Layers:[/bold] {len(pipeline.layers)}\n"
        f"[bold]Concurrency:[/bold] {concurrency_label}",
        title="[bold cyan]Synix Build[/bold cyan]",
        border_style="cyan",
    ))

    start_time = time.time()

    try:
        result = run_pipeline(pipeline, source_dir=source_dir, verbosity=verbose,
                              concurrency=concurrency)
    except Exception as e:
        console.print(f"\n[red]Pipeline failed:[/red] {e}")
        sys.exit(1)

    elapsed = time.time() - start_time

    # Summary table
    table = Table(title="Build Summary", box=box.ROUNDED)
    table.add_column("Layer", style="bold")
    table.add_column("Level", justify="center")
    table.add_column("Built", justify="right", style="green")
    table.add_column("Cached", justify="right", style="cyan")
    table.add_column("Skipped", justify="right", style="dim")

    for stats in result.layer_stats:
        style = get_layer_style(stats.level)
        table.add_row(
            f"[{style}]{stats.name}[/{style}]",
            str(stats.level),
            str(stats.built),
            str(stats.cached),
            str(stats.skipped),
        )

    console.print()
    console.print(table)
    console.print(
        f"\n[bold]Total:[/bold] {result.built} built, "
        f"{result.cached} cached, {result.skipped} skipped"
    )
    console.print(f"[bold]Time:[/bold] {elapsed:.1f}s")

    # Show run log summary when verbose
    run_log = result.run_log
    if run_log and run_log.get("total_llm_calls", 0) > 0:
        console.print(
            f"[bold]LLM calls:[/bold] {run_log['total_llm_calls']}, "
            f"[bold]Tokens:[/bold] {run_log.get('total_tokens', 0):,}, "
            f"[bold]Est. cost:[/bold] ${run_log.get('total_cost_estimate', 0):.4f}"
        )


# Hidden alias for backward compatibility
@click.command(hidden=True)
@click.argument("pipeline_path", type=click.Path(exists=True))
@click.option("--source-dir", default=None, help="Override source directory")
@click.option("--build-dir", default=None, help="Override build directory")
@click.option("--verbose", "-v", count=True, help="Verbosity level")
@click.option("--concurrency", "-j", default=1, type=int,
              help="Number of concurrent LLM requests (default 1 = sequential)")
def run_alias(pipeline_path: str, source_dir: str | None, build_dir: str | None,
              verbose: int, concurrency: int):
    """Process exports through a memory pipeline (alias for 'build')."""
    # Reuse the build command context
    ctx = click.get_current_context()
    ctx.invoke(build, pipeline_path=pipeline_path, source_dir=source_dir,
               build_dir=build_dir, verbose=verbose, concurrency=concurrency)


@click.command()
@click.argument("pipeline_path", type=click.Path(exists=True))
@click.option("--source-dir", default=None, help="Override source directory")
@click.option("--build-dir", default=None, help="Override build directory")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--save", is_flag=True, help="Save plan as artifact in the build directory")
def plan(
    pipeline_path: str,
    source_dir: str | None,
    build_dir: str | None,
    output_json: bool,
    save: bool,
):
    """Show what a pipeline build would do without executing.

    PIPELINE_PATH is the Python file defining the pipeline (e.g., pipeline.py).
    Analyzes cache state and estimates LLM calls, tokens, and cost.
    """
    from synix.build.pipeline import load_pipeline
    from synix.build.plan import plan_build

    try:
        pipeline = load_pipeline(pipeline_path)
    except Exception as e:
        console.print(f"[red]Error loading pipeline:[/red] {e}")
        sys.exit(1)

    if source_dir:
        pipeline.source_dir = source_dir
    if build_dir:
        pipeline.build_dir = build_dir

    try:
        build_plan = plan_build(pipeline, source_dir=source_dir)
    except Exception as e:
        console.print(f"[red]Error planning build:[/red] {e}")
        sys.exit(1)

    if output_json:
        click.echo(build_plan.to_json())
        return

    # Rich table output
    console.print(Panel(
        f"[bold]Pipeline:[/bold] {build_plan.pipeline_name}\n"
        f"[bold]Source:[/bold] {pipeline.source_dir}\n"
        f"[bold]Build:[/bold] {pipeline.build_dir}\n"
        f"[bold]Layers:[/bold] {len(build_plan.steps)}",
        title="[bold cyan]Synix Build Plan[/bold cyan]",
        border_style="cyan",
    ))

    table = Table(title="Build Plan", box=box.ROUNDED)
    table.add_column("Layer", style="bold")
    table.add_column("Level", justify="center")
    table.add_column("Status", justify="center")
    table.add_column("Artifacts", justify="right")
    table.add_column("LLM Calls", justify="right")
    table.add_column("Est. Tokens", justify="right")
    table.add_column("Est. Cost", justify="right")
    table.add_column("Reason")

    STATUS_STYLES = {
        "cached": "cyan",
        "rebuild": "yellow",
        "new": "green",
    }

    for step in build_plan.steps:
        layer_style = get_layer_style(step.level)
        status_style = STATUS_STYLES.get(step.status, "white")
        cost_str = f"${step.estimated_cost:.4f}" if step.estimated_cost > 0 else "-"
        tokens_str = f"{step.estimated_tokens:,}" if step.estimated_tokens > 0 else "-"
        calls_str = str(step.estimated_llm_calls) if step.estimated_llm_calls > 0 else "-"

        table.add_row(
            f"[{layer_style}]{step.name}[/{layer_style}]",
            str(step.level),
            f"[{status_style}]{step.status}[/{status_style}]",
            str(step.artifact_count),
            calls_str,
            tokens_str,
            cost_str,
            step.reason,
        )

    console.print()
    console.print(table)

    # Summary
    console.print(
        f"\n[bold]Summary:[/bold] {build_plan.total_rebuild} layer(s) to build, "
        f"{build_plan.total_cached} layer(s) cached"
    )
    if build_plan.total_estimated_llm_calls > 0:
        console.print(
            f"[bold]Estimated:[/bold] {build_plan.total_estimated_llm_calls} LLM calls, "
            f"{build_plan.total_estimated_tokens:,} tokens, "
            f"${build_plan.total_estimated_cost:.4f}"
        )
    else:
        console.print("[bold]Estimated:[/bold] no LLM calls needed (fully cached)")

    if save:
        _save_plan_artifact(build_plan, pipeline)


def _save_plan_artifact(build_plan, pipeline):
    """Save the build plan as an artifact in the build directory."""
    from pathlib import Path

    from synix.build.artifacts import ArtifactStore
    from synix.core.models import Artifact

    build_dir = Path(pipeline.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(build_dir)

    content = build_plan.to_json()
    artifact = Artifact(
        artifact_id="build-plan",
        artifact_type="build_plan",
        content=content,
        metadata={
            "pipeline_name": build_plan.pipeline_name,
            "total_rebuild": build_plan.total_rebuild,
            "total_cached": build_plan.total_cached,
            "total_estimated_llm_calls": build_plan.total_estimated_llm_calls,
            "total_estimated_cost": build_plan.total_estimated_cost,
        },
    )
    store.save_artifact(artifact, layer_name="plans", layer_level=99)
    console.print(f"\n[dim]Plan saved as artifact 'build-plan' in {build_dir}[/dim]")
