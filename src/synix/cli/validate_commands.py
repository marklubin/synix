"""Validate command — detect violations in built artifacts."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.tree import Tree

from synix.cli.main import console, pipeline_argument


def _build_provenance_tree(label, store, provenance):
    """Build a Rich Tree showing the actual provenance DAG for an artifact."""
    root_art = store.load_artifact(label)
    root_layer = root_art.metadata.get("layer_name", root_art.artifact_type) if root_art else ""
    tree = Tree("[dim]Provenance[/dim]")

    def _add_node(parent_tree, node_label, visited):
        if node_label in visited:
            parent_tree.add(f"[dim]{node_label} (cycle)[/dim]")
            return
        visited = visited | {node_label}

        art = store.load_artifact(node_label)
        layer = art.metadata.get("layer_name", art.artifact_type) if art else ""
        display = f"[bold]{node_label}[/bold] [dim]({layer})[/dim]"
        node = parent_tree.add(display)

        parents = provenance.get_parents(node_label)
        for pid in sorted(parents):
            _add_node(node, pid, visited)

    # Start from the violation artifact
    root_node = tree.add(f"[bold]{label}[/bold] [dim]({root_layer})[/dim]")
    parents = provenance.get_parents(label)
    for pid in sorted(parents):
        _add_node(root_node, pid, {label})

    return tree


@click.command()
@pipeline_argument
@click.option("--build-dir", default=None, help="Override build directory")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def validate(pipeline_path: str, build_dir: str | None, output_json: bool):
    """Validate built artifacts for contradictions, PII, and other issues.

    PIPELINE_PATH defaults to pipeline.py in the current directory.

    Runs validators and reports violations. Use `synix fix` to resolve them.
    """
    from synix.build.pipeline import load_pipeline

    try:
        pipeline = load_pipeline(pipeline_path)
    except Exception as e:
        console.print(f"[red]Error loading pipeline:[/red] {e}")
        sys.exit(1)

    if build_dir:
        pipeline.build_dir = build_dir

    build_path = Path(pipeline.build_dir)
    if not build_path.exists():
        console.print(f"[red]Build directory not found:[/red] {build_path}\nRun [bold]synix build[/bold] first.")
        sys.exit(1)

    _run_validate_mode(pipeline, build_path, output_json)


def _run_validators_with_progress(pipeline, store, provenance, output_json: bool):
    """Run validators one by one with live status output.

    Returns the aggregated ValidationResult.
    """
    from synix.build.validators import (
        ValidationContext,
        ValidationResult,
        _gather_artifacts,
    )

    ctx = ValidationContext(store, provenance, pipeline)
    result = ValidationResult()
    result.violations_by_validator = {}

    for validator in pipeline.validators:
        config = validator.to_config_dict()
        validator_name = validator.name or type(validator).__name__

        artifacts = _gather_artifacts(store, config)

        if not output_json:
            label = f"[bold]{validator_name}[/bold] [dim]({len(artifacts)} artifact(s))[/dim]"
            spinner_msg = f"Running [bold]{validator_name}[/bold] on {len(artifacts)} artifact(s)..."

        start = time.monotonic()

        if output_json:
            violations = validator.validate(artifacts, ctx)
        else:
            with Status(spinner_msg, console=console, spinner="dots"):
                violations = validator.validate(artifacts, ctx)

        elapsed = time.monotonic() - start

        # Auto-resolve provenance for violations without traces
        for v in violations:
            if not v.provenance_trace:
                v.provenance_trace = ctx.trace_field_origin(v.label, v.field)

        result.violations.extend(violations)
        result.validators_run.append(validator_name)
        result.violations_by_validator[validator_name] = violations

        if not output_json:
            errors = sum(1 for v in violations if v.severity == "error")
            warnings = sum(1 for v in violations if v.severity == "warning")
            if errors:
                status = f"[red]{errors} error(s)[/red]"
            elif warnings:
                status = f"[yellow]{warnings} warning(s)[/yellow]"
            else:
                status = "[green]passed[/green]"
            console.print(f"  {label}  {status}  [dim]{elapsed:.1f}s[/dim]")

    return result


def _run_validate_mode(pipeline, build_path: Path, output_json: bool):
    """Run validators and report violations."""
    from synix.build.artifacts import ArtifactStore
    from synix.build.provenance import ProvenanceTracker
    from synix.build.validators import ViolationQueue

    store = ArtifactStore(build_path)
    provenance = ProvenanceTracker(build_path)

    if not output_json:
        console.print(
            Panel(
                f"[bold]Pipeline:[/bold] {pipeline.name}\n"
                f"[bold]Build:[/bold] {build_path}\n"
                f"[bold]Validators:[/bold] {len(pipeline.validators)}",
                title="[bold cyan]Synix Validate[/bold cyan]",
                border_style="cyan",
            )
        )

    if not pipeline.validators:
        if not output_json:
            console.print("\n[yellow]No validators declared in pipeline.[/yellow]")
        return

    if not output_json:
        console.print()

    result = _run_validators_with_progress(pipeline, store, provenance, output_json)

    # Persist violations to queue
    queue = ViolationQueue.load(build_path)
    for v in result.violations:
        queue.upsert(v)
    queue.save_state()

    if output_json:
        out = result.to_dict()
        # Add violation_ids to JSON output
        for i, v in enumerate(result.violations):
            out["violations"][i]["violation_id"] = v.violation_id
        # Use print() not console.print() to avoid Rich word-wrapping inside JSON strings
        print(json.dumps(out, indent=2))
        sys.exit(0 if result.passed else 1)

    # Rich display
    if not result.violations:
        console.print("\n[green]All validators passed.[/green] No violations found.")
        return

    # Summary table (use per-validator violation counts from progress loop)
    by_validator = getattr(result, "violations_by_validator", {})

    table = Table(title="Validation Results", box=box.ROUNDED)
    table.add_column("Validator", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Count", justify="right")

    for name in result.validators_run:
        viols = by_validator.get(name, [])
        errors = [v for v in viols if v.severity == "error"]
        warnings = [v for v in viols if v.severity == "warning"]
        if errors:
            status_str = "[red]FAIL[/red]"
            count = str(len(errors))
        elif warnings:
            status_str = "[yellow]WARN[/yellow]"
            count = str(len(warnings))
        else:
            status_str = "[green]PASS[/green]"
            count = "0"
        table.add_row(name, status_str, count)

    console.print()
    console.print(table)

    # Split into errors (full panels) and warnings (compact list)
    errors = [v for v in result.violations if v.severity == "error"]
    warnings = [v for v in result.violations if v.severity == "warning"]

    # Errors — full panel display
    for v in errors:
        type_label = v.violation_type.replace("_", " ")

        # Build panel body
        lines: list[str] = []
        lines.append(f"[bold]{v.message}[/bold]")

        # Semantic conflict details
        if v.violation_type == "semantic_conflict":
            claim_a = v.metadata.get("claim_a", "")
            claim_b = v.metadata.get("claim_b", "")
            explanation = v.metadata.get("explanation", "")
            if claim_a and claim_b:
                lines.append("")
                lines.append(f'[dim]Claim A:[/dim]  "{claim_a}"')
                lines.append(f'[dim]Claim B:[/dim]  "{claim_b}"')
            if explanation:
                lines.append(f"[dim]Reasoning:[/dim]  {explanation}")

        body = "\n".join(lines)

        # Build panel content — text body + optional provenance tree
        if v.provenance_trace:
            provenance_tree = _build_provenance_tree(v.label, store, provenance)
            panel_content = Group(body, "", provenance_tree)
        else:
            panel_content = body

        console.print()
        console.print(
            Panel(
                panel_content,
                title=(
                    f"[red bold]ERROR[/red bold] [dim]|[/dim] [red]{type_label}[/red] [dim]|[/dim] [red]{v.label}[/red]"
                ),
                border_style="red",
                padding=(0, 1),
            )
        )

    # Warnings — compact bullet list
    if warnings:
        console.print()
        console.print("[yellow bold]Warnings[/yellow bold]")
        for v in warnings:
            type_label = v.violation_type.replace("_", " ")
            console.print(f"  [yellow]\u2022[/yellow] [dim]{type_label}[/dim]  {v.label}: {v.message}")

    error_count = sum(1 for v in result.violations if v.severity == "error")
    warn_count = sum(1 for v in result.violations if v.severity == "warning")
    console.print(
        f"\n[bold]Total:[/bold] {len(result.violations)} violation(s) ({error_count} errors, {warn_count} warnings)"
    )
    console.print(f"[dim]Violations saved to {build_path / 'violations_state.json'}[/dim]")

    if any(v.severity == "error" for v in result.violations):
        sys.exit(1)
