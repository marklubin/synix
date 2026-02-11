"""Verify, diff, lineage, and status commands."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich import box
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from synix.cli.main import console, get_layer_style


@click.command()
@click.argument("artifact_id")
@click.option("--build-dir", default="./build", help="Build directory")
def lineage(artifact_id: str, build_dir: str):
    """Show provenance chain for an artifact.

    ARTIFACT_ID is the artifact to trace.
    """
    from synix.build.artifacts import ArtifactStore
    from synix.build.provenance import ProvenanceTracker

    provenance = ProvenanceTracker(build_dir)
    store = ArtifactStore(build_dir)

    chain = provenance.get_chain(artifact_id)

    if not chain:
        console.print(f"[red]No provenance found for:[/red] {artifact_id}")
        sys.exit(1)

    console.print(f"\n[bold]Lineage for:[/bold] {artifact_id}\n")

    tree = Tree(f"[bold]{artifact_id}[/bold]")

    def add_parents(node, aid):
        record = next((r for r in chain if r.artifact_id == aid), None)
        if record:
            for parent_id in record.parent_artifact_ids:
                artifact = store.load_artifact(parent_id)
                label = parent_id
                if artifact:
                    label += f" [dim]({artifact.artifact_type})[/dim]"
                child_node = node.add(label)
                add_parents(child_node, parent_id)

    add_parents(tree, artifact_id)
    console.print(tree)


@click.command()
@click.option("--build-dir", default="./build", help="Build directory")
def status(build_dir: str):
    """Show build status summary."""
    from synix.build.artifacts import ArtifactStore

    build_path = Path(build_dir)
    if not build_path.exists():
        console.print("[red]No build directory found.[/red] Run [bold]synix build[/bold] first.")
        sys.exit(1)

    store = ArtifactStore(build_dir)

    table = Table(title="Build Status", box=box.ROUNDED)
    table.add_column("Layer", style="bold")
    table.add_column("Artifacts", justify="right")
    table.add_column("Last Build", justify="center")

    # Group artifacts by layer from manifest
    manifest = store._manifest
    layers: dict[str, dict] = {}
    for _aid, info in manifest.items():
        layer = info.get("layer", "unknown")
        level = info.get("level", 0)
        if layer not in layers:
            layers[layer] = {"count": 0, "level": level}
        layers[layer]["count"] += 1

    # Sort by level
    for layer_name, info in sorted(layers.items(), key=lambda x: x[1]["level"]):
        style = get_layer_style(info["level"])
        table.add_row(
            f"[{style}]{layer_name}[/{style}]",
            str(info["count"]),
            "-",
        )

    console.print()
    console.print(table)

    # Search index status
    search_db = build_path / "search.db"
    if search_db.exists():
        console.print(f"\n[green]Search index:[/green] {search_db} exists")
    else:
        console.print("\n[yellow]Search index:[/yellow] not built yet")

    # Context doc status
    context_doc = build_path / "context.md"
    if context_doc.exists():
        size = context_doc.stat().st_size
        console.print(f"[green]Context doc:[/green] {context_doc} ({size} bytes)")
    else:
        console.print("[yellow]Context doc:[/yellow] not built yet")


@click.command()
@click.option("--build-dir", default="./build", help="Build directory")
@click.option("--check", "checks", multiple=True, help="Run specific checks only")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--pipeline", "pipeline_path", default=None,
              type=click.Path(exists=True),
              help="Pipeline file to run domain validators from")
def verify(build_dir: str, checks: tuple[str, ...], output_json: bool,
           pipeline_path: str | None):
    """Verify integrity of a completed build.

    Checks: build_exists, manifest_valid, artifacts_exist,
    provenance_complete, search_index, content_hashes, no_orphans,
    merge_integrity.

    Use --pipeline to also run domain-specific validators declared in the pipeline.
    """
    from synix.build.verify import verify_build

    check_list = list(checks) if checks else None
    result = verify_build(build_dir, checks=check_list)

    # Run pipeline domain validators if requested
    validation_result = None
    if pipeline_path:
        try:
            from synix.build.artifacts import ArtifactStore
            from synix.build.pipeline import load_pipeline
            from synix.build.provenance import ProvenanceTracker
            from synix.build.validators import run_validators

            pipeline = load_pipeline(pipeline_path)
            if build_dir:
                pipeline.build_dir = build_dir

            if pipeline.validators:
                store = ArtifactStore(pipeline.build_dir)
                provenance = ProvenanceTracker(pipeline.build_dir)
                validation_result = run_validators(pipeline, store, provenance)
        except Exception as e:
            console.print(f"[yellow]Warning: could not run pipeline validators:[/yellow] {e}")

    if output_json:
        out = result.to_dict()
        if validation_result is not None:
            out["domain_validations"] = validation_result.to_dict()
        console.print(json.dumps(out, indent=2))
        all_passed = result.passed and (validation_result is None or validation_result.passed)
        sys.exit(0 if all_passed else 1)

    console.print()

    table = Table(title="Build Verification", box=box.ROUNDED)
    table.add_column("Check", style="bold")
    table.add_column("Status", justify="center")
    table.add_column("Message")

    for check in result.checks:
        status_str = "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]"
        table.add_row(check.name, status_str, check.message)

    console.print(table)

    # Show details for failed checks
    for check in result.failed_checks:
        if check.details:
            console.print(f"\n[red bold]{check.name}[/red bold] details:")
            for detail in check.details:
                console.print(f"  [dim]{detail}[/dim]")

    console.print(f"\n[bold]{result.summary}[/bold]")

    # Display domain validation results
    if validation_result is not None:
        _display_domain_validations(validation_result)

    all_passed = result.passed and (validation_result is None or validation_result.passed)
    sys.exit(0 if all_passed else 1)


def _display_domain_validations(validation):
    """Display domain validation results in verify output."""

    console.print()

    vtable = Table(title="Domain Validations", box=box.ROUNDED)
    vtable.add_column("Validator", style="bold")
    vtable.add_column("Status", justify="center")
    vtable.add_column("Message")

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

    for name in validation.validators_run:
        viol_list = violations_by_validator.get(name, [])
        if not viol_list:
            continue

        console.print(f"\n[bold]{name}[/bold] violations:")
        for v in viol_list:
            severity_style = "red" if v.severity == "error" else "yellow"
            console.print(f"  [{severity_style}]{v.artifact_id}[/{severity_style}]: {v.message}")

            if v.provenance_trace:
                console.print("    [dim]Provenance:[/dim]")
                for step in v.provenance_trace:
                    val_str = f"  {v.field}: {step.field_value}" if step.field_value else ""
                    console.print(
                        f"      {step.artifact_id} [dim]({step.layer})[/dim]{val_str}"
                    )


@click.command()
@click.argument("artifact_id", required=False)
@click.option("--build-dir", default="./build", help="Current build directory")
@click.option("--old-build-dir", default=None, help="Previous build directory to compare against")
@click.option("--layer", default=None, help="Filter diff to a specific layer")
def diff(artifact_id: str | None, build_dir: str, old_build_dir: str | None, layer: str | None):
    """Show differences between artifact versions.

    If ARTIFACT_ID is given, shows diff for that artifact.
    Otherwise, diffs all artifacts between two build directories.
    """
    from synix.build.diff import diff_artifact_by_id, diff_builds

    if artifact_id:
        # Single artifact diff
        result = diff_artifact_by_id(build_dir, artifact_id, previous_build_dir=old_build_dir)
        if result is None:
            console.print(f"[red]Cannot diff artifact:[/red] {artifact_id}")
            console.print("[dim]Either artifact not found or no previous version available.[/dim]")
            sys.exit(1)

        if not result.has_changes:
            console.print(f"[green]No changes[/green] for {artifact_id}")
            return

        console.print(f"\n[bold]Diff for:[/bold] {artifact_id}")
        if result.old_prompt_id != result.new_prompt_id:
            console.print(
                f"[yellow]Prompt changed:[/yellow] {result.old_prompt_id} → {result.new_prompt_id}"
            )
        if result.content_diff:
            console.print(Panel(
                Syntax(result.content_diff, "diff", theme="monokai"),
                title="Content diff",
                border_style="yellow",
            ))
        if result.metadata_diff:
            console.print("\n[bold]Metadata changes:[/bold]")
            for key, changes in result.metadata_diff.items():
                console.print(f"  {key}: [red]{changes['old']}[/red] → [green]{changes['new']}[/green]")
    else:
        # Full build diff
        if not old_build_dir:
            console.print("[red]--old-build-dir required[/red] when diffing all artifacts")
            sys.exit(1)

        result = diff_builds(old_build_dir, build_dir, layer=layer)

        if not result.has_changes:
            console.print("[green]No differences[/green] between builds")
            return

        console.print("\n[bold]Build diff[/bold]")
        if layer:
            console.print(f"[dim]Layer filter: {layer}[/dim]")

        if result.added:
            console.print(f"\n[green]+{len(result.added)} added:[/green]")
            for aid in result.added:
                console.print(f"  [green]+ {aid}[/green]")

        if result.removed:
            console.print(f"\n[red]-{len(result.removed)} removed:[/red]")
            for aid in result.removed:
                console.print(f"  [red]- {aid}[/red]")

        if result.diffs:
            console.print(f"\n[yellow]~{len(result.diffs)} modified:[/yellow]")
            for d in result.diffs:
                console.print(f"  [yellow]~ {d.artifact_id}[/yellow]")
                if d.content_diff:
                    lines = d.content_diff.count("\n")
                    console.print(f"    [dim]{lines} lines changed[/dim]")
