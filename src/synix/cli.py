"""Synix command-line interface."""

import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Load .env file before importing config
load_dotenv()


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Synix - Declarative pipeline for AI conversation exports."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    setup_logging(verbose)


@cli.command()
@click.argument("name")
@click.option("--agent", "-a", default="default", help="Agent name")
def init(name: str, agent: str) -> None:
    """Initialize a new pipeline.

    Creates the pipeline state in the database.
    """
    from synix.config import get_settings
    from synix.db.engine import get_control_session, init_databases
    from synix.services.pipelines import get_pipeline, save_pipeline

    console = Console()
    settings = get_settings()

    # Initialize databases
    init_databases(settings)

    with get_control_session(settings) as session:
        existing = get_pipeline(session, name)
        if existing:
            console.print(f"[yellow]Pipeline '{name}' already exists[/]")
            return

        save_pipeline(session, name, agent, {})
        console.print(f"[green]Created pipeline:[/] {name}")
        console.print(f"  Agent: {agent}")
        console.print(f"  Storage: {settings.storage_dir}")


@cli.command()
@click.argument("pipeline_file", type=click.Path(exists=True))
@click.option("--step", "-s", help="Run only this step (and dependencies)")
@click.option("--full", is_flag=True, help="Force full reprocess (ignore cache)")
def run(pipeline_file: str, step: str | None, full: bool) -> None:
    """Execute a pipeline from a Python file.

    The file should define a Pipeline and call its methods.

    Example:
        synix run my_pipeline.py
        synix run my_pipeline.py --step summaries
        synix run my_pipeline.py --full
    """
    console = Console()

    # Execute the pipeline file
    pipeline_path = Path(pipeline_file).resolve()
    if not pipeline_path.exists():
        console.print(f"[red]File not found:[/] {pipeline_file}")
        sys.exit(1)

    # Import and execute the file
    import importlib.util

    spec = importlib.util.spec_from_file_location("pipeline_module", pipeline_path)
    if spec is None or spec.loader is None:
        console.print(f"[red]Cannot load:[/] {pipeline_file}")
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_module"] = module
    spec.loader.exec_module(module)

    # Find the Pipeline instance
    from synix.pipeline import Pipeline

    pipeline = None
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, Pipeline):
            pipeline = obj
            break

    if pipeline is None:
        console.print("[red]No Pipeline instance found in file[/]")
        sys.exit(1)

    console.print(f"\n[bold]Running pipeline:[/] {pipeline.name}")
    console.print(f"  Agent: {pipeline.agent}")
    if step:
        console.print(f"  Step: {step}")
    if full:
        console.print("  Mode: [yellow]full reprocess[/]")
    console.print()

    result = pipeline.run(step=step, full=full)

    if result.status == "completed":
        console.print("\n[bold green]Run completed:[/]")
    else:
        console.print(f"\n[bold red]Run failed:[/] {result.error}")

    console.print(f"  Run ID:  {result.run_id}")
    console.print(f"  Input:   {result.stats.get('input', 0)}")
    console.print(f"  Output:  {result.stats.get('output', 0)}")
    console.print(f"  Skipped: {result.stats.get('skipped', 0)}")
    console.print(f"  Errors:  {result.stats.get('errors', 0)}")
    console.print(f"  Tokens:  {result.stats.get('tokens', 0)}")


@cli.command()
def status() -> None:
    """Show pipeline status and record counts."""
    from synix.config import get_settings
    from synix.db.engine import get_artifact_session, get_control_session, init_databases
    from synix.services.pipelines import list_pipelines
    from synix.services.records import count_records_by_step
    from synix.services.runs import get_latest_run

    console = Console()
    settings = get_settings()

    if not settings.control_db_path.exists():
        console.print("[yellow]No pipelines found. Run 'synix init' first.[/]")
        return

    init_databases(settings)

    with get_control_session(settings) as control_session:
        pipelines = list_pipelines(control_session)

        if not pipelines:
            console.print("[yellow]No pipelines found.[/]")
            return

        for pipeline in pipelines:
            console.print(f"\n[bold]Pipeline:[/] {pipeline.name}")
            console.print(f"  Agent: {pipeline.agent}")
            console.print(f"  Updated: {pipeline.updated_at}")

            # Get latest run
            latest = get_latest_run(control_session, pipeline.name)
            if latest:
                status_color = "green" if latest.status == "completed" else "yellow"
                status_text = f"[{status_color}]{latest.status}[/]"
                console.print(f"  Last run: {status_text} ({latest.created_at})")

            # Get step counts
            definition = pipeline.definition
            sources = definition.get("sources", [])
            steps = definition.get("steps", [])

            if sources or steps:
                console.print("  Steps:")
                with get_artifact_session(settings) as artifact_session:
                    for step_name in sources + steps:
                        count = count_records_by_step(artifact_session, step_name)
                        console.print(f"    {step_name}: {count} records")


@cli.command()
@click.argument("query")
@click.option("--step", "-s", help="Filter by step name")
@click.option("--limit", "-n", default=10, help="Max results")
def search(query: str, step: str | None, limit: int) -> None:
    """Search records using FTS.

    Examples:
        synix search "rust ownership"
        synix search "api design" --step summaries
    """
    from synix.config import get_settings
    from synix.db.engine import get_artifact_session, init_databases
    from synix.services.search import search_fts

    console = Console()
    settings = get_settings()

    if not settings.artifact_db_path.exists():
        console.print("[yellow]No records found. Run a pipeline first.[/]")
        return

    init_databases(settings)

    with get_artifact_session(settings) as session:
        hits = search_fts(session, query, step=step, limit=limit)

        if not hits:
            console.print("[yellow]No results found.[/]")
            return

        console.print(f"\n[bold]Search results for:[/] {query}\n")

        for i, hit in enumerate(hits, 1):
            # Show snippet or truncated content
            display = hit.snippet or hit.content[:200]
            if len(hit.content) > 200 and not hit.snippet:
                display += "..."

            console.print(
                Panel(
                    display,
                    title=f"#{i} [bold blue]{hit.step_name}[/] (rank: {-hit.rank:.2f})",
                    subtitle=f"[dim]{hit.record_id}[/]",
                    title_align="left",
                    subtitle_align="left",
                    border_style="blue",
                )
            )
            console.print()


@cli.command()
@click.argument("pipeline_file", type=click.Path(exists=True))
def plan(pipeline_file: str) -> None:
    """Show what would execute without running.

    Dry-run that shows step order, input counts, and cache status.
    """
    console = Console()

    # Load pipeline file
    pipeline_path = Path(pipeline_file).resolve()

    import importlib.util

    spec = importlib.util.spec_from_file_location("pipeline_module", pipeline_path)
    if spec is None or spec.loader is None:
        console.print(f"[red]Cannot load:[/] {pipeline_file}")
        sys.exit(1)

    module = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_module"] = module
    spec.loader.exec_module(module)

    from synix.pipeline import Pipeline

    pipeline = None
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, Pipeline):
            pipeline = obj
            break

    if pipeline is None:
        console.print("[red]No Pipeline instance found in file[/]")
        sys.exit(1)

    console.print(f"\n[bold]Pipeline plan:[/] {pipeline.name}\n")

    result = pipeline.plan()

    table = Table(title="Execution Order")
    table.add_column("Step", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("From", style="yellow")
    table.add_column("Inputs")
    table.add_column("Existing")

    for step_info in result.steps:
        table.add_row(
            step_info["name"],
            step_info["type"],
            step_info.get("from", "-"),
            str(step_info.get("inputs", "-")),
            str(step_info.get("existing", 0)),
        )

    console.print(table)


@cli.command()
def runs() -> None:
    """List recent pipeline runs."""
    from synix.config import get_settings
    from synix.db.engine import get_control_session, init_databases

    console = Console()
    settings = get_settings()

    if not settings.control_db_path.exists():
        console.print("[yellow]No runs found.[/]")
        return

    init_databases(settings)

    with get_control_session(settings) as session:
        from sqlalchemy import select

        from synix.db.control import Run

        stmt = select(Run).order_by(Run.created_at.desc()).limit(20)
        runs_list = list(session.scalars(stmt))

        if not runs_list:
            console.print("[yellow]No runs found.[/]")
            return

        table = Table(title="Recent Runs")
        table.add_column("ID", style="dim")
        table.add_column("Pipeline", style="cyan")
        table.add_column("Status")
        table.add_column("Output")
        table.add_column("Skipped")
        table.add_column("Errors")
        table.add_column("Created")

        for run in runs_list:
            stats = run.stats
            status_style = "green" if run.status == "completed" else "red"
            table.add_row(
                str(run.id)[:8] + "...",
                run.pipeline_name,
                f"[{status_style}]{run.status}[/]",
                str(stats.get("output", 0)),
                str(stats.get("skipped", 0)),
                str(stats.get("errors", 0)),
                str(run.created_at)[:19],
            )

        console.print(table)


@cli.command()
@click.argument("step")
@click.option("--format", "-f", "output_format", default="markdown", help="markdown/json/text")
@click.option("--output", "-o", default="-", help="Output file (- for stdout)")
def export(step: str, output_format: str, output: str) -> None:
    """Export records from a step.

    Examples:
        synix export summaries
        synix export monthly --format json --output monthly.json
        synix export narrative -f text -o report.txt
    """
    import json as json_module

    from synix.config import get_settings
    from synix.db.engine import get_artifact_session, init_databases
    from synix.services.records import get_records_by_step

    console = Console()
    settings = get_settings()

    if not settings.artifact_db_path.exists():
        console.print("[yellow]No records found. Run a pipeline first.[/]")
        return

    init_databases(settings)

    with get_artifact_session(settings) as session:
        records = get_records_by_step(session, step)

        if not records:
            console.print(f"[yellow]No records found for step: {step}[/]")
            return

        # Format output
        if output_format == "json":
            data = []
            for r in records:
                data.append(
                    {
                        "id": str(r.id),
                        "step_name": r.step_name,
                        "content": r.content,
                        "metadata": r.metadata_,
                        "created_at": str(r.created_at),
                    }
                )
            content = json_module.dumps(data, indent=2)
        elif output_format == "markdown":
            parts = []
            for r in records:
                title = r.metadata_.get("meta.source.title", r.step_name)
                parts.append(f"## {title}\n\n{r.content}")
            content = "\n\n---\n\n".join(parts)
        else:  # text
            content = "\n\n".join(r.content for r in records)

        # Output
        if output == "-":
            console.print(content)
        else:
            Path(output).write_text(content)
            console.print(f"[green]Exported {len(records)} records to {output}[/]")


if __name__ == "__main__":
    cli()
