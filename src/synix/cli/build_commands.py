"""Build commands — synix build, synix plan."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from rich import box
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from synix.cli.main import console, get_layer_style, pipeline_argument
from synix.cli.progress import BuildProgress
from synix.core.models import FlatFile, Pipeline, SearchIndex, SearchSurface, SynixSearch


def _print_error(label: str, exc: Exception, verbose: int, con) -> None:
    """Print an error with verbosity-dependent detail.

    - verbose == 0 (default): "Label: TypeError: message"
    - verbose >= 1 (-v): "Label: TypeError at file.py:93" + message on next line
    - verbose >= 2 (-vv): Full traceback via console.print_exception()
    """
    import traceback as _tb

    if verbose >= 2:
        con.print(f"[red]{label}:[/red]")
        con.print_exception(show_locals=False)
    elif verbose >= 1:
        tb = _tb.extract_tb(exc.__traceback__)
        loc = f" at {tb[-1].filename.rsplit('/', 1)[-1]}:{tb[-1].lineno}" if tb else ""
        con.print(f"[red]{label}:[/red] {type(exc).__name__}{loc}")
        con.print(f"  {exc}")
    else:
        con.print(f"[red]{label}:[/red] {type(exc).__name__}: {exc}")


def _print_build_error(exc: Exception, dlq_enabled: bool, verbose: int, con) -> None:
    """Print a build error with DLQ opt-in hint when applicable."""
    _print_error("Pipeline failed", exc, verbose, con)

    if not dlq_enabled:
        # Check if this looks like a content-filter or input-too-large error
        from synix.build.error_classifier import ErrorVerdict, LLMErrorClassifier

        classifier = LLMErrorClassifier()
        verdict = classifier.classify(exc, "")
        if verdict == ErrorVerdict.DLQ:
            con.print(
                "\n[yellow]Hint:[/yellow] This error is recoverable. "
                "Re-run with [bold]--dlq[/bold] to skip failing artifacts "
                "and continue building:\n"
                "  [dim]synix build pipeline.py --dlq[/dim]"
            )


def _projection_triggers(pipeline: Pipeline) -> dict[str, list[tuple[str, str, str]]]:
    """Compute layer_name → [(proj_name, proj_type, trigger_type)] mapping.

    - SearchIndex compatibility projections: every source layer → "progressive"
    - SynixSearch and FlatFile projections: only the last source layer → "complete"
    """
    from synix.build.dag import resolve_build_order

    build_order = resolve_build_order(pipeline)
    layer_order = {layer.name: i for i, layer in enumerate(build_order)}

    triggers: dict[str, list[tuple[str, str, str]]] = {}

    for proj in pipeline.projections:
        source_layer_names = [s.name for s in proj.sources]

        if isinstance(proj, SynixSearch):
            proj_type = "synix_search"
        elif isinstance(proj, SearchIndex):
            proj_type = "search_index"
        elif isinstance(proj, FlatFile):
            proj_type = "flat_file"
        else:
            proj_type = "unknown"

        if proj_type == "search_index":
            # Progressive: every source layer triggers
            for ln in source_layer_names:
                triggers.setdefault(ln, []).append((proj.name, proj_type, "progressive"))
        elif proj_type in {"synix_search", "flat_file"}:
            # Complete: only the last source layer triggers
            if source_layer_names:
                last = max(source_layer_names, key=lambda ln: layer_order.get(ln, 0))
                triggers.setdefault(last, []).append((proj.name, proj_type, "complete"))
        else:
            # Unknown type: last source layer
            if source_layer_names:
                last = max(source_layer_names, key=lambda ln: layer_order.get(ln, 0))
                triggers.setdefault(last, []).append((proj.name, proj_type, "complete"))

    return triggers


def _surface_triggers(pipeline: Pipeline) -> dict[str, list[tuple[str, str, str]]]:
    """Compute layer_name → [(surface_name, surface_type, trigger_type)] mapping."""
    from synix.build.dag import resolve_build_order

    build_order = resolve_build_order(pipeline)
    layer_order = {layer.name: i for i, layer in enumerate(build_order)}
    triggers: dict[str, list[tuple[str, str, str]]] = {}

    for surface in pipeline.surfaces:
        source_layer_names = [s.name for s in surface.sources]
        if not isinstance(surface, SearchSurface):
            continue
        if source_layer_names:
            last = max(source_layer_names, key=lambda ln: layer_order.get(ln, 0))
            triggers.setdefault(last, []).append((surface.name, "search_surface", "complete"))

    return triggers


@click.command()
@pipeline_argument
@click.option("--source-dir", default=None, help="Override source directory")
@click.option("--build-dir", default=None, help="Override build directory")
@click.option("--verbose", "-v", count=True, help="Verbosity level: -v per-artifact, -vv debug/LLM details")
@click.option("--concurrency", "-j", default=5, type=int, help="Number of concurrent LLM requests (default 5)")
@click.option("--validate", is_flag=True, default=False, help="Run domain validators after build")
@click.option("--plain", is_flag=True, default=False, help="Plain text output (no TUI, safe for CI/pipes)")
@click.option(
    "--dlq",
    is_flag=True,
    default=False,
    help="Enable dead letter queue: skip content-filter and input-too-large errors instead of aborting",
)
def build(
    pipeline_path: str,
    source_dir: str | None,
    build_dir: str | None,
    verbose: int,
    concurrency: int,
    validate: bool,
    plain: bool,
    dlq: bool,
):
    """Build memory artifacts from a pipeline definition.

    PIPELINE_PATH defaults to pipeline.py in the current directory.
    """
    # Trigger search projection registration
    import synix.search.indexer  # noqa: F401
    from synix.build.error_classifier import LLMErrorClassifier
    from synix.build.pipeline import load_pipeline
    from synix.build.runner import run as run_pipeline

    classifier = LLMErrorClassifier() if dlq else None

    try:
        pipeline = load_pipeline(pipeline_path)
    except Exception as e:
        _print_error("Error loading pipeline", e, verbose, console)
        sys.exit(1)

    if source_dir:
        pipeline.source_dir = str(Path(source_dir).resolve())
    if build_dir:
        pipeline.build_dir = str(Path(build_dir).resolve())
        pipeline.synix_dir = None  # Force recomputation from overridden build_dir

    concurrency_label = f"{concurrency} threads" if concurrency > 1 else "sequential"
    console.print(
        Panel(
            f"[bold]Pipeline:[/bold] {pipeline.name}\n"
            f"[bold]Source:[/bold] {pipeline.source_dir}\n"
            f"[bold]Build:[/bold] {pipeline.build_dir}\n"
            f"[bold]Layers:[/bold] {len(pipeline.layers)}\n"
            f"[bold]Surfaces:[/bold] {len(pipeline.surfaces)}\n"
            f"[bold]Concurrency:[/bold] {concurrency_label}",
            title="[bold cyan]Synix Build[/bold cyan]",
            border_style="cyan",
        )
    )

    start_time = time.time()

    # Default verbosity to 1 (verbose) so progress is always shown
    effective_verbosity = max(verbose, 1)

    use_plain = plain

    if use_plain:
        from synix.cli.progress import PlainBuildProgress

        progress = PlainBuildProgress(console=console)
        try:
            result = run_pipeline(
                pipeline,
                verbosity=effective_verbosity,
                concurrency=concurrency,
                progress=progress,
                validate=validate,
                error_classifier=classifier,
            )
        except Exception as e:
            console.print()
            _print_build_error(e, dlq, verbose, console)
            sys.exit(1)
    else:
        progress = BuildProgress()
        try:
            with Live(progress, console=console, refresh_per_second=4):
                result = run_pipeline(
                    pipeline,
                    verbosity=effective_verbosity,
                    concurrency=concurrency,
                    progress=progress,
                    validate=validate,
                    error_classifier=classifier,
                )
        except Exception as e:
            console.print()
            _print_build_error(e, dlq, verbose, console)
            sys.exit(1)

    elapsed = time.time() - start_time

    # Compute projection triggers for inline display
    proj_triggers = _projection_triggers(pipeline)

    # Build projection status lookup from result
    proj_status = {ps.name: ps.status for ps in result.projection_stats}

    # Summary table
    has_dlq = any(s.dlq_count > 0 for s in result.layer_stats)

    table = Table(title="Build Summary", box=box.ROUNDED)
    table.add_column("Layer", style="bold", no_wrap=True)
    table.add_column("Level", justify="center")
    table.add_column("Built", justify="right", style="green")
    table.add_column("Cached", justify="right", style="cyan")
    table.add_column("Skipped", justify="right", style="dim")
    if has_dlq:
        table.add_column("DLQ", justify="right", style="yellow")

    # Determine the last trigger layer for each projection (for showing final status)
    last_trigger_layer: dict[str, str] = {}
    for layer_name, trigs in proj_triggers.items():
        for proj_name, _, _ in trigs:
            last_trigger_layer[proj_name] = layer_name

    for stats in result.layer_stats:
        style = get_layer_style(stats.level)
        row = [
            f"[{style}]{stats.name}[/{style}]",
            str(stats.level),
            str(stats.built),
            str(stats.cached),
            str(stats.skipped),
        ]
        if has_dlq:
            row.append(str(stats.dlq_count) if stats.dlq_count > 0 else "")
        table.add_row(*row)

        # Inline projection rows after this layer
        for proj_name, proj_type, trigger_type in proj_triggers.get(stats.name, []):
            # Show actual status only on the last trigger layer for this projection
            if last_trigger_layer.get(proj_name) == stats.name:
                status_label = proj_status.get(proj_name, trigger_type)
            else:
                status_label = trigger_type
            proj_row = [
                f"  [dim]→[/dim] [magenta]{proj_name}[/magenta]",
                f"[dim]{proj_type}[/dim]",
                "",
                "",
                f"[dim]{status_label}[/dim]",
            ]
            if has_dlq:
                proj_row.append("")
            table.add_row(*proj_row)

    console.print()
    console.print(table)
    from synix.cli.main import is_demo_mode

    console.print(f"\n[bold]Total:[/bold] {result.built} built, {result.cached} cached, {result.skipped} skipped")
    if len(result.dlq) > 0:
        console.print(f"[yellow bold]DLQ:[/yellow bold] {result.dlq.summary()}")
    if not is_demo_mode():
        console.print(f"[bold]Time:[/bold] {elapsed:.1f}s")
    if not is_demo_mode() and result.snapshot_oid and result.run_ref:
        run_id = result.run_ref.rsplit("/", 1)[-1]
        console.print(f"[bold]Artifact Snapshot:[/bold] {result.snapshot_oid[:12]}")
        console.print(f"[bold]Run ID:[/bold] {run_id}")
        console.print(f"[bold]Run Ref:[/bold] {result.run_ref}")
        console.print()
        console.print("[dim]Next: synix release HEAD --to local[/dim]")

    # Show run log summary when verbose
    run_log = result.run_log
    if run_log and run_log.get("total_llm_calls", 0) > 0:
        console.print(
            f"[bold]LLM calls:[/bold] {run_log['total_llm_calls']}, "
            f"[bold]Tokens:[/bold] {run_log.get('total_tokens', 0):,}, "
            f"[bold]Est. cost:[/bold] ${run_log.get('total_cost_estimate', 0):.4f}"
        )

    # Display domain validation results
    if result.validation is not None:
        _display_validation_results(result.validation)
        if not result.validation.passed:
            sys.exit(1)


def _display_validation_results(validation):
    """Display domain validation results as a Rich table with violation details."""
    console.print()

    # Summary table
    vtable = Table(title="Domain Validations", box=box.ROUNDED)
    vtable.add_column("Validator", style="bold")
    vtable.add_column("Status", justify="center")
    vtable.add_column("Message")

    # Group violations by validator
    violations_by_validator: dict[str, list] = {}
    for v in validation.violations:
        violations_by_validator.setdefault(v.violation_type, []).append(v)

    for name in validation.validators_run:
        viol_list = violations_by_validator.get(name, [])
        errors = [v for v in viol_list if v.severity == "error"]
        warnings = [v for v in viol_list if v.severity == "warning"]

        if errors:
            status_str = "[red]FAIL[/red]"
            msg = f"{len(errors)} {name} violation(s)"
        elif warnings:
            status_str = "[yellow]WARN[/yellow]"
            msg = f"{len(warnings)} warning(s)"
        else:
            status_str = "[green]PASS[/green]"
            msg = "All artifacts passed"

        vtable.add_row(name, status_str, msg)

    console.print(vtable)

    # Show violation details
    for name in validation.validators_run:
        viol_list = violations_by_validator.get(name, [])
        if not viol_list:
            continue

        console.print(f"\n[bold]{name}[/bold] violations:")
        for v in viol_list:
            severity_style = "red" if v.severity == "error" else "yellow"
            console.print(f"  [{severity_style}]{v.label}[/{severity_style}]: {v.message}")

            if v.provenance_trace:
                console.print("    [dim]Provenance:[/dim]")
                for step in v.provenance_trace:
                    val_str = f"  {v.field}: {step.field_value}" if step.field_value else ""
                    console.print(f"      {step.label} [dim]({step.layer})[/dim]{val_str}")


# Hidden alias for backward compatibility
@click.command(hidden=True)
@pipeline_argument
@click.option("--source-dir", default=None, help="Override source directory")
@click.option("--build-dir", default=None, help="Override build directory")
@click.option("--verbose", "-v", count=True, help="Verbosity level")
@click.option("--concurrency", "-j", default=5, type=int, help="Number of concurrent LLM requests (default 5)")
@click.option("--validate", is_flag=True, default=False, help="Run domain validators after build")
@click.option("--plain", is_flag=True, default=False, help="Plain text output (no TUI, safe for CI/pipes)")
def run_alias(
    pipeline_path: str,
    source_dir: str | None,
    build_dir: str | None,
    verbose: int,
    concurrency: int,
    validate: bool,
    plain: bool,
):
    """Process exports through a memory pipeline (alias for 'build')."""
    # Reuse the build command context
    ctx = click.get_current_context()
    ctx.invoke(
        build,
        pipeline_path=pipeline_path,
        source_dir=source_dir,
        build_dir=build_dir,
        verbose=verbose,
        concurrency=concurrency,
        validate=validate,
        plain=plain,
    )


@click.command()
@pipeline_argument
@click.option("--source-dir", default=None, help="Override source directory")
@click.option("--build-dir", default=None, help="Override build directory")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--save", is_flag=True, help="Save plan as artifact in the build directory")
@click.option("--explain-cache", is_flag=True, help="Show per-layer cache decision breakdown")
def plan(
    pipeline_path: str,
    source_dir: str | None,
    build_dir: str | None,
    output_json: bool,
    save: bool,
    explain_cache: bool,
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
        pipeline.source_dir = str(Path(source_dir).resolve())
    if build_dir:
        pipeline.build_dir = str(Path(build_dir).resolve())
        pipeline.synix_dir = None  # Force recomputation from overridden build_dir

    try:
        build_plan = plan_build(pipeline)
    except Exception as e:
        console.print(f"[red]Error planning build:[/red] {e}")
        sys.exit(1)

    if output_json:
        click.echo(build_plan.to_json())
        return

    # ── Header panel ─────────────────────────────────────────────
    console.print(
        Panel(
            f"[bold]Pipeline:[/bold] {build_plan.pipeline_name}\n"
            f"[bold]Source:[/bold] {pipeline.source_dir}\n"
            f"[bold]Build:[/bold] {pipeline.build_dir}\n"
            f"[bold]Layers:[/bold] {len(build_plan.steps)}\n"
            f"[bold]Surfaces:[/bold] {len(build_plan.surfaces)}\n"
            f"[bold]Projections:[/bold] {len(build_plan.projections)}",
            title="[bold cyan]Synix Build Plan[/bold cyan]",
            border_style="cyan",
        )
    )

    # LLM Configuration section
    _display_llm_config(build_plan)

    # ── Tree view ──────────────────────────────────────────────────
    STATUS_STYLES = {
        "cached": "cyan",
        "rebuild": "yellow",
        "new": "green",
        "error": "red",
    }

    # Model label for root
    global_cfg = build_plan.global_llm_config
    model_label = global_cfg.get("model", "?") if global_cfg else "?"
    provider_label = global_cfg.get("provider", "") if global_cfg else ""

    tree = Tree(f"[bold cyan]{build_plan.pipeline_name}[/bold cyan]  [dim]{model_label} ({provider_label})[/dim]")

    # Build lookup: which projections source from which layers
    surface_plan_map = {sp.name: sp for sp in build_plan.surfaces}
    proj_plan_map = {pp.name: pp for pp in build_plan.projections}

    surface_triggers = _surface_triggers(pipeline)
    # Track which projections we've already rendered (show once, on last source layer)
    proj_triggers = _projection_triggers(pipeline)
    last_trigger_layer_surface: dict[str, str] = {}
    for layer_name, trigs in surface_triggers.items():
        for surface_name, _, _ in trigs:
            last_trigger_layer_surface[surface_name] = layer_name
    last_trigger_layer_plan: dict[str, str] = {}
    for layer_name, trigs in proj_triggers.items():
        for proj_name, _, _ in trigs:
            last_trigger_layer_plan[proj_name] = layer_name

    # Lookups
    step_lookup = {s.name: s for s in build_plan.steps}
    layer_lookup = {l.name: l for l in pipeline.layers}

    MATERIALIZATION_TYPE_LABELS = {
        "search_surface": "search_surface (sqlite)",
        "synix_search": "synix_search (sqlite)",
        "search_index": "search_index (legacy sqlite)",
        "flat_file": "flat_file (markdown)",
    }

    for step in build_plan.steps:
        layer_style = get_layer_style(step.level)
        status_style = STATUS_STYLES.get(step.status, "white")
        layer_obj = layer_lookup.get(step.name)

        # Entity type label: source:parse or transform:ClassName
        kind = "source" if step.level == 0 else "transform"
        if layer_obj is None:
            transform_name = "?"
        elif kind == "source":
            transform_name = "parse"
        else:
            transform_name = type(layer_obj).__name__
        type_label = f"[dim]{kind}:{transform_name}[/dim]"

        # Status counts
        status_parts = []
        if step.status in ("new", "rebuild"):
            parallel_label = ""
            if step.parallel_units > 1:
                parallel_label = f" ({step.parallel_units} parallel)"
            if step.cached_count > 0:
                # Mixed: some cached, some need rebuild
                rebuild_str = f"{step.rebuild_count} {step.status}{parallel_label}"
                status_parts.append(f"[{status_style}]{rebuild_str}[/{status_style}]")
                status_parts.append(f"[cyan]{step.cached_count} cached[/cyan]")
            else:
                count_str = f"{step.artifact_count} {step.status}{parallel_label}"
                status_parts.append(f"[{status_style}]{count_str}[/{status_style}]")
        elif step.status == "cached":
            status_parts.append(f"[{status_style}]{step.artifact_count} cached[/{status_style}]")
        elif step.status == "error":
            status_parts.append(f"[{status_style}]error: {step.reason}[/{status_style}]")
        status_str = "  ".join(status_parts)

        # Inline cache reason when --explain-cache
        if explain_cache and step.status in ("new", "rebuild") and step.reason not in ("new",):
            status_str += f" [dim]— {step.reason}[/dim]"

        layer_node = tree.add(f"[{layer_style}][bold]{step.name}[/bold][/{layer_style}]  {type_label}  {status_str}")

        # Show source info for parse layers
        if step.source_info:
            layer_node.add(f"[dim]← {step.source_info}[/dim]")

        # Show inputs (← depends_on)
        if layer_obj and layer_obj.depends_on:
            for dep in layer_obj.depends_on:
                dep_name = dep.name
                dep_step = step_lookup.get(dep_name)
                dep_style = get_layer_style(dep_step.level) if dep_step else "dim"
                layer_node.add(f"[dim]← [{dep_style}]{dep_name}[/{dep_style}][/dim]")

        # Inline cache fingerprint breakdown when --explain-cache and cached
        if explain_cache and step.status == "cached" and step.fingerprint:
            from synix.build.fingerprint import Fingerprint

            fp = Fingerprint.from_dict(step.fingerprint)
            if fp:
                scheme_ver = fp.scheme.split(":")[-1]
                component_names = " ".join(sorted(fp.components.keys()))
                layer_node.add(f"[dim]cache: all components match ({scheme_ver}: {component_names})[/dim]")

        # Show build-time search surfaces on their final trigger layer
        for surface_name, _, _ in surface_triggers.get(step.name, []):
            if last_trigger_layer_surface.get(surface_name) != step.name:
                continue
            sp = surface_plan_map.get(surface_name)
            if not sp:
                continue

            surface_status_style = STATUS_STYLES.get(sp.status, "dim")
            surface_type_label = MATERIALIZATION_TYPE_LABELS.get(sp.projection_type, sp.projection_type)
            surface_label = (
                f"[magenta][bold]{sp.name}[/bold][/magenta]  "
                f"[dim]{surface_type_label}[/dim]  "
                f"[{surface_status_style}]{sp.status}[/{surface_status_style}]  "
                f"{sp.artifact_count} indexed"
            )
            surface_node = layer_node.add(f"⇢ {surface_label}")

            for src_name in sp.source_layers:
                src_style = get_layer_style(step_lookup[src_name].level) if src_name in step_lookup else "dim"
                surface_node.add(f"[dim]← [{src_style}]{src_name}[/{src_style}][/dim]")

            if sp.embedding_config:
                ec = sp.embedding_config
                model = ec.get("model", "?")
                if "/" in model:
                    model = model.rsplit("/", 1)[-1]
                dims = ec.get("dimensions", "")
                emb_provider = ec.get("provider", "")
                surface_node.add(f"[dim]embeddings  {model}  {dims}d  ({emb_provider})[/dim]")

        # Show projections on their final trigger layer
        for proj_name, _, _ in proj_triggers.get(step.name, []):
            if last_trigger_layer_plan.get(proj_name) != step.name:
                continue
            pp = proj_plan_map.get(proj_name)
            if not pp:
                continue

            proj_status_style = STATUS_STYLES.get(pp.status, "dim")
            proj_type_label = MATERIALIZATION_TYPE_LABELS.get(pp.projection_type, pp.projection_type)
            proj_label = (
                f"[magenta][bold]{pp.name}[/bold][/magenta]  "
                f"[dim]{proj_type_label}[/dim]  "
                f"[{proj_status_style}]{pp.status}[/{proj_status_style}]  "
                f"{pp.artifact_count} indexed"
            )
            proj_node = layer_node.add(f"→ {proj_label}")

            # Show source layers feeding this projection
            for src_name in pp.source_layers:
                src_style = get_layer_style(step_lookup[src_name].level) if src_name in step_lookup else "dim"
                proj_node.add(f"[dim]← [{src_style}]{src_name}[/{src_style}][/dim]")

            # Show embedding config
            if pp.embedding_config:
                ec = pp.embedding_config
                model = ec.get("model", "?")
                if "/" in model:
                    model = model.rsplit("/", 1)[-1]
                dims = ec.get("dimensions", "")
                emb_provider = ec.get("provider", "")
                proj_node.add(f"[dim]embeddings  {model}  {dims}d  ({emb_provider})[/dim]")

    console.print()
    console.print(tree)

    # ── Summary line ──────────────────────────────────────────────
    summary_parts = []

    # Layers
    summary_parts.append(f"{build_plan.total_rebuild} layer(s) to build, {build_plan.total_cached} cached")
    summary_parts.append(f"{len(build_plan.surfaces)} surface(s), {len(build_plan.projections)} projection(s)")

    # Cost estimate
    if build_plan.total_estimated_llm_calls > 0:
        summary_parts.append(
            f"{build_plan.total_estimated_llm_calls} LLM calls, "
            f"{build_plan.total_estimated_tokens:,} tokens, "
            f"${build_plan.total_estimated_cost:.4f}"
        )
    else:
        summary_parts.append("no LLM calls needed (fully cached)")

    console.print()
    console.print(f"[bold]Estimated:[/bold] {' · '.join(summary_parts)}")

    # ── Source-change warnings ───────────────────────────────────
    _display_source_change_warnings(build_plan, pipeline)

    if save:
        _save_plan_artifact(build_plan, pipeline)


def _display_source_change_warnings(build_plan, pipeline):
    """Emit warnings when transform source code changed (detected via fingerprint)."""
    from synix.build.fingerprint import Fingerprint

    build_dir = pipeline.build_dir
    if not build_dir:
        return

    from synix.build.refs import synix_dir_for_build_dir
    from synix.build.snapshot_view import SnapshotArtifactCache

    build_path = Path(build_dir)
    if not build_path.exists():
        return

    synix_dir = synix_dir_for_build_dir(build_path)
    store = SnapshotArtifactCache(synix_dir)
    layer_lookup = {layer.name: layer for layer in pipeline.layers}

    for step in build_plan.steps:
        if step.level == 0 or step.fingerprint is None:
            continue
        layer_obj = layer_lookup.get(step.name)
        if layer_obj is None:
            continue

        current_fp = Fingerprint.from_dict(step.fingerprint)
        if current_fp is None:
            continue

        # Check if existing artifacts have a different source component
        existing = store.list_artifacts(step.name)
        for art in existing:
            stored_tfp_data = art.metadata.get("transform_fingerprint")
            if stored_tfp_data is None:
                continue
            stored_fp = Fingerprint.from_dict(stored_tfp_data)
            if stored_fp is None:
                continue
            stored_source = stored_fp.components.get("source")
            current_source = current_fp.components.get("source")
            if stored_source and current_source and stored_source != current_source:
                console.print(
                    f"\n[yellow]Warning:[/yellow] Transform source changed for "
                    f"[bold]{type(layer_obj).__name__}[/bold] (layer '{step.name}') — rebuild required"
                )
                break  # One warning per layer


def _display_llm_config(build_plan):
    """Display LLM configuration: global config and per-layer overrides."""
    global_cfg = build_plan.global_llm_config
    if not global_cfg:
        return

    # Global LLM config table
    config_table = Table(
        title="LLM Configuration",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
    )
    config_table.add_column("Setting", style="bold")
    config_table.add_column("Value")

    config_table.add_row("Provider", str(global_cfg.get("provider", "-")))
    config_table.add_row("Model", str(global_cfg.get("model", "-")))
    if global_cfg.get("base_url"):
        config_table.add_row("Base URL", str(global_cfg["base_url"]))
    config_table.add_row("Temperature", str(global_cfg.get("temperature", "-")))
    config_table.add_row("Max Tokens", str(global_cfg.get("max_tokens", "-")))
    if global_cfg.get("api_key"):
        config_table.add_row("API Key", f"[dim]{global_cfg['api_key']}[/dim]")

    console.print()
    console.print(config_table)

    # Per-layer overrides — only show layers that differ from global
    overrides: list[tuple[str, dict]] = []
    for step in build_plan.steps:
        if step.resolved_llm_config is None:
            continue  # parse layer
        # Check if the resolved config differs from global on any key
        # (ignoring api_key which may resolve differently per-layer)
        differs = False
        for key in ("provider", "model", "temperature", "max_tokens", "base_url"):
            global_val = global_cfg.get(key)
            layer_val = step.resolved_llm_config.get(key)
            if global_val != layer_val:
                differs = True
                break
        if differs:
            overrides.append((step.name, step.resolved_llm_config))

    if overrides:
        override_table = Table(
            title="Per-Layer LLM Overrides",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold",
        )
        override_table.add_column("Layer", style="bold")
        override_table.add_column("Provider")
        override_table.add_column("Model")
        override_table.add_column("Base URL")
        override_table.add_column("Temperature", justify="right")
        override_table.add_column("Max Tokens", justify="right")

        for layer_name, cfg in overrides:
            layer_style = "white"
            for step in build_plan.steps:
                if step.name == layer_name:
                    layer_style = get_layer_style(step.level)
                    break
            override_table.add_row(
                f"[{layer_style}]{layer_name}[/{layer_style}]",
                str(cfg.get("provider", "-")),
                str(cfg.get("model", "-")),
                str(cfg.get("base_url", "-")) if cfg.get("base_url") else "-",
                str(cfg.get("temperature", "-")),
                str(cfg.get("max_tokens", "-")),
            )

        console.print()
        console.print(override_table)


def _save_plan_artifact(build_plan, pipeline):
    """Save the build plan as a snapshot in the .synix object store."""
    import hashlib
    from datetime import UTC, datetime
    from pathlib import Path

    from synix.build.object_store import SCHEMA_VERSION, ObjectStore
    from synix.build.refs import RefStore, synix_dir_for_build_dir

    build_dir = Path(pipeline.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    synix_dir = synix_dir_for_build_dir(build_dir, configured_synix_dir=pipeline.synix_dir)
    synix_dir.mkdir(parents=True, exist_ok=True)
    object_store = ObjectStore(synix_dir)
    ref_store = RefStore(synix_dir)

    content = build_plan.to_json()
    content_oid, _ = object_store.put_text(content)
    content_hash = f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"

    artifact_payload = {
        "type": "artifact",
        "schema_version": SCHEMA_VERSION,
        "label": "build-plan",
        "artifact_type": "build_plan",
        "artifact_id": content_hash,
        "content_oid": content_oid,
        "input_ids": [],
        "prompt_id": None,
        "model_config": None,
        "metadata": {
            "pipeline_name": build_plan.pipeline_name,
            "total_rebuild": build_plan.total_rebuild,
            "total_cached": build_plan.total_cached,
            "total_estimated_llm_calls": build_plan.total_estimated_llm_calls,
            "total_estimated_cost": build_plan.total_estimated_cost,
            "layer_name": "plans",
            "layer_level": 99,
        },
        "parent_labels": [],
    }
    artifact_oid = object_store.put_json(artifact_payload)

    # Build manifest and snapshot so SnapshotArtifactCache can read it back
    manifest_payload = {
        "type": "manifest",
        "schema_version": SCHEMA_VERSION,
        "pipeline_name": build_plan.pipeline_name,
        "pipeline_fingerprint": "sha256:plan-save",
        "artifacts": [{"label": "build-plan", "oid": artifact_oid}],
        "projections": {},
    }
    manifest_oid = object_store.put_json(manifest_payload)

    snapshot_payload = {
        "type": "snapshot",
        "schema_version": SCHEMA_VERSION,
        "manifest_oid": manifest_oid,
        "parent_snapshot_oids": [],
        "created_at": datetime.now(UTC).isoformat(),
        "pipeline_name": build_plan.pipeline_name,
        "run_id": "plan-save",
    }
    snapshot_oid = object_store.put_json(snapshot_payload)

    # Write to a separate plans ref — never clobber refs/heads/main which
    # holds the real build HEAD. This avoids destroying build history.
    ref_store.write_ref("refs/plans/latest", snapshot_oid)

    console.print(f"\n[dim]Plan saved as artifact 'build-plan' ({artifact_oid[:12]}) in {synix_dir}[/dim]")
