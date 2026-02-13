"""Build commands — synix build, synix plan."""

from __future__ import annotations

import sys
import time

import click
from rich import box
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from synix.cli.main import console, get_layer_style, pipeline_argument
from synix.cli.progress import BuildProgress
from synix.core.models import Pipeline


def _projection_triggers(pipeline: Pipeline) -> dict[str, list[tuple[str, str, str]]]:
    """Compute layer_name → [(proj_name, proj_type, trigger_type)] mapping.

    - search_index projections: every source layer → "progressive"
    - flat_file projections: only the last source layer (by build order) → "complete"
    """
    from synix.build.dag import resolve_build_order

    build_order = resolve_build_order(pipeline)
    layer_order = {layer.name: i for i, layer in enumerate(build_order)}

    triggers: dict[str, list[tuple[str, str, str]]] = {}

    for proj in pipeline.projections:
        source_layers = [s["layer"] for s in proj.sources]

        if proj.projection_type == "search_index":
            # Progressive: every source layer triggers
            for ln in source_layers:
                triggers.setdefault(ln, []).append((proj.name, proj.projection_type, "progressive"))
        elif proj.projection_type == "flat_file":
            # Complete: only the last source layer triggers
            if source_layers:
                last = max(source_layers, key=lambda ln: layer_order.get(ln, 0))
                triggers.setdefault(last, []).append((proj.name, proj.projection_type, "complete"))
        else:
            # Unknown type: last source layer
            if source_layers:
                last = max(source_layers, key=lambda ln: layer_order.get(ln, 0))
                triggers.setdefault(last, []).append((proj.name, proj.projection_type, "complete"))

    return triggers


@click.command()
@pipeline_argument
@click.option("--source-dir", default=None, help="Override source directory")
@click.option("--build-dir", default=None, help="Override build directory")
@click.option("--verbose", "-v", count=True, help="Verbosity level: -v per-artifact, -vv debug/LLM details")
@click.option("--concurrency", "-j", default=5, type=int, help="Number of concurrent LLM requests (default 5)")
@click.option("--validate", is_flag=True, default=False, help="Run domain validators after build")
def build(
    pipeline_path: str, source_dir: str | None, build_dir: str | None, verbose: int, concurrency: int, validate: bool
):
    """Build memory artifacts from a pipeline definition.

    PIPELINE_PATH defaults to pipeline.py in the current directory.
    """
    # Trigger search projection registration
    import synix.search.indexer  # noqa: F401
    from synix.build.pipeline import load_pipeline
    from synix.build.runner import run as run_pipeline

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
    console.print(
        Panel(
            f"[bold]Pipeline:[/bold] {pipeline.name}\n"
            f"[bold]Source:[/bold] {pipeline.source_dir}\n"
            f"[bold]Build:[/bold] {pipeline.build_dir}\n"
            f"[bold]Layers:[/bold] {len(pipeline.layers)}\n"
            f"[bold]Concurrency:[/bold] {concurrency_label}",
            title="[bold cyan]Synix Build[/bold cyan]",
            border_style="cyan",
        )
    )

    start_time = time.time()

    # Default verbosity to 1 (verbose) so progress is always shown
    effective_verbosity = max(verbose, 1)

    progress = BuildProgress()
    try:
        with Live(progress, console=console, refresh_per_second=4):
            result = run_pipeline(
                pipeline,
                source_dir=source_dir,
                verbosity=effective_verbosity,
                concurrency=concurrency,
                progress=progress,
                validate=validate,
            )
    except Exception as e:
        console.print(f"\n[red]Pipeline failed:[/red] {e}")
        sys.exit(1)

    elapsed = time.time() - start_time

    # Compute projection triggers for inline display
    proj_triggers = _projection_triggers(pipeline)

    # Build projection status lookup from result
    proj_status = {ps.name: ps.status for ps in result.projection_stats}

    # Summary table
    table = Table(title="Build Summary", box=box.ROUNDED)
    table.add_column("Layer", style="bold", no_wrap=True)
    table.add_column("Level", justify="center")
    table.add_column("Built", justify="right", style="green")
    table.add_column("Cached", justify="right", style="cyan")
    table.add_column("Skipped", justify="right", style="dim")

    # Determine the last trigger layer for each projection (for showing final status)
    last_trigger_layer: dict[str, str] = {}
    for layer_name, trigs in proj_triggers.items():
        for proj_name, _, _ in trigs:
            last_trigger_layer[proj_name] = layer_name

    for stats in result.layer_stats:
        style = get_layer_style(stats.level)
        table.add_row(
            f"[{style}]{stats.name}[/{style}]",
            str(stats.level),
            str(stats.built),
            str(stats.cached),
            str(stats.skipped),
        )

        # Inline projection rows after this layer
        for proj_name, proj_type, trigger_type in proj_triggers.get(stats.name, []):
            # Show actual status only on the last trigger layer for this projection
            if last_trigger_layer.get(proj_name) == stats.name:
                status_label = proj_status.get(proj_name, trigger_type)
            else:
                status_label = trigger_type
            table.add_row(
                f"  [dim]→[/dim] [magenta]{proj_name}[/magenta]",
                f"[dim]{proj_type}[/dim]",
                "",
                "",
                f"[dim]{status_label}[/dim]",
            )

    console.print()
    console.print(table)
    from synix.cli.main import is_demo_mode

    console.print(f"\n[bold]Total:[/bold] {result.built} built, {result.cached} cached, {result.skipped} skipped")
    if not is_demo_mode():
        console.print(f"[bold]Time:[/bold] {elapsed:.1f}s")

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
def run_alias(
    pipeline_path: str, source_dir: str | None, build_dir: str | None, verbose: int, concurrency: int, validate: bool
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

    # ── Header panel ─────────────────────────────────────────────
    console.print(
        Panel(
            f"[bold]Pipeline:[/bold] {build_plan.pipeline_name}\n"
            f"[bold]Source:[/bold] {pipeline.source_dir}\n"
            f"[bold]Build:[/bold] {pipeline.build_dir}\n"
            f"[bold]Layers:[/bold] {len(build_plan.steps)}",
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
    }

    # Model label for root
    global_cfg = build_plan.global_llm_config
    model_label = global_cfg.get("model", "?") if global_cfg else "?"
    provider_label = global_cfg.get("provider", "") if global_cfg else ""

    tree = Tree(f"[bold cyan]{build_plan.pipeline_name}[/bold cyan]  [dim]{model_label} ({provider_label})[/dim]")

    # Build lookup: which projections source from which layers
    proj_plan_map = {pp.name: pp for pp in build_plan.projections}
    # layer_name → list of (proj, source_config)
    proj_by_source_layer: dict[str, list] = {}
    for proj in pipeline.projections:
        for src in proj.sources:
            proj_by_source_layer.setdefault(src["layer"], []).append(proj)

    # Track which projections we've already rendered (show once, on last source layer)
    proj_triggers = _projection_triggers(pipeline)
    last_trigger_layer_plan: dict[str, str] = {}
    for layer_name, trigs in proj_triggers.items():
        for proj_name, _, _ in trigs:
            last_trigger_layer_plan[proj_name] = layer_name

    # Lookups
    step_lookup = {s.name: s for s in build_plan.steps}
    layer_lookup = {l.name: l for l in pipeline.layers}

    PROJECTION_TYPE_LABELS = {
        "search_index": "synix_search_index (sqlite)",
        "flat_file": "flat_file (markdown)",
    }

    for step in build_plan.steps:
        layer_style = get_layer_style(step.level)
        status_style = STATUS_STYLES.get(step.status, "white")
        layer_obj = layer_lookup.get(step.name)

        # Entity type label: source:transform_name or transform:transform_name
        kind = "source" if step.level == 0 else "transform"
        transform_name = layer_obj.transform if layer_obj else "?"
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
                dep_step = step_lookup.get(dep)
                dep_style = get_layer_style(dep_step.level) if dep_step else "dim"
                layer_node.add(f"[dim]← [{dep_style}]{dep}[/{dep_style}][/dim]")

        # Inline cache fingerprint breakdown when --explain-cache and cached
        if explain_cache and step.status == "cached" and step.fingerprint:
            from synix.build.fingerprint import Fingerprint

            fp = Fingerprint.from_dict(step.fingerprint)
            if fp:
                scheme_ver = fp.scheme.split(":")[-1]
                component_names = " ".join(sorted(fp.components.keys()))
                layer_node.add(f"[dim]cache: all components match ({scheme_ver}: {component_names})[/dim]")

        # Show projections on their final trigger layer
        for proj_name, _, _ in proj_triggers.get(step.name, []):
            if last_trigger_layer_plan.get(proj_name) != step.name:
                continue
            pp = proj_plan_map.get(proj_name)
            if not pp:
                continue

            proj_status_style = STATUS_STYLES.get(pp.status, "dim")
            proj_type_label = PROJECTION_TYPE_LABELS.get(pp.projection_type, pp.projection_type)
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

    from pathlib import Path

    from synix.build.artifacts import ArtifactStore

    build_path = Path(build_dir)
    if not build_path.exists():
        return

    store = ArtifactStore(build_path)
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
                    f"[bold]{layer_obj.transform}[/bold] (layer '{step.name}') — rebuild required"
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
    """Save the build plan as an artifact in the build directory."""
    from pathlib import Path

    from synix.build.artifacts import ArtifactStore
    from synix.core.models import Artifact

    build_dir = Path(pipeline.build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(build_dir)

    content = build_plan.to_json()
    artifact = Artifact(
        label="build-plan",
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
