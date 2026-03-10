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


def _resolve_synix_dir(build_dir: str, synix_dir: str | None) -> Path | None:
    """Resolve the .synix directory from explicit option or build-dir fallback.

    Returns the resolved Path, or None if no snapshot store can be found.
    """
    from synix.build.refs import synix_dir_for_build_dir

    if synix_dir:
        resolved = Path(synix_dir)
        if resolved.exists():
            return resolved
        return None

    build_path = Path(build_dir)
    try:
        resolved = synix_dir_for_build_dir(build_path)
    except ValueError:
        return None

    if resolved.exists():
        return resolved
    return None


@click.command()
@click.argument("artifact_id")
@click.option("--build-dir", default="./build", help="Build directory")
@click.option("--synix-dir", default=None, help="Path to .synix directory")
@click.option("--ref", default="HEAD", help="Snapshot ref to read (default: HEAD)")
def lineage(artifact_id: str, build_dir: str, synix_dir: str | None, ref: str):
    """Show provenance chain for an artifact.

    ARTIFACT_ID is the artifact to trace.
    """
    from synix.build.snapshot_view import SnapshotView

    resolved_synix_dir = _resolve_synix_dir(build_dir, synix_dir)
    if resolved_synix_dir is None:
        console.print(f"[red]No provenance found for:[/red] {artifact_id}")
        sys.exit(1)

    try:
        view = SnapshotView.open(resolved_synix_dir, ref=ref)
    except ValueError as e:
        console.print(f"[red]Cannot open snapshot:[/red] {e}")
        sys.exit(1)

    # Resolve prefix (git-like)
    try:
        resolved_label = view.resolve_prefix(artifact_id)
    except ValueError as e:
        console.print(f"[red]Ambiguous:[/red] {e}")
        sys.exit(1)

    if resolved_label is None:
        console.print(f"[red]No provenance found for:[/red] {artifact_id}")
        sys.exit(1)

    try:
        chain = view.get_provenance(resolved_label)
    except KeyError:
        console.print(f"[red]No provenance found for:[/red] {artifact_id}")
        sys.exit(1)

    if len(chain) <= 1:
        console.print(f"[red]No provenance found for:[/red] {artifact_id}")
        sys.exit(1)

    console.print(f"\n[bold]Lineage for:[/bold] {resolved_label}\n")

    tree = Tree(f"[bold]{resolved_label}[/bold]")

    def add_parents(node, label):
        try:
            art = view.get_artifact(label)
        except KeyError:
            return
        parent_labels = art.get("parent_labels", [])
        for parent_label in parent_labels:
            display = parent_label
            try:
                parent_art = view.get_artifact(parent_label)
                display += f" [dim]({parent_art.get('artifact_type', 'unknown')})[/dim]"
            except KeyError:
                pass
            child_node = node.add(display)
            add_parents(child_node, parent_label)

    add_parents(tree, resolved_label)
    console.print(tree)


@click.command()
@click.option("--build-dir", default="./build", help="Build directory")
@click.option("--resolved", is_flag=True, help="Show resolved violation details")
def status(build_dir: str, resolved: bool):
    """Show build status summary."""
    import json as _json
    from datetime import datetime

    from rich.tree import Tree

    from synix.build.refs import synix_dir_for_build_dir
    from synix.build.snapshot_view import SnapshotArtifactCache

    build_path = Path(build_dir)
    try:
        synix_dir = synix_dir_for_build_dir(build_path)
    except (ValueError, OSError):
        console.print("[red]No build directory found.[/red] Run [bold]synix build[/bold] first.")
        sys.exit(1)

    if not synix_dir.exists():
        console.print("[red]No build directory found.[/red] Run [bold]synix build[/bold] first.")
        sys.exit(1)
    store = SnapshotArtifactCache(synix_dir)

    # ── Build layers table ──────────────────────────────────────────────
    table = Table(title="Build Status", box=box.ROUNDED)
    table.add_column("Layer", style="bold")
    table.add_column("Artifacts", justify="right")
    table.add_column("Last Built", justify="center")

    # Group artifacts by layer, track newest created_at per layer
    manifest = store.iter_entries()
    layers: dict[str, dict] = {}
    for aid, info in manifest.items():
        layer = info.get("layer", "unknown")
        level = info.get("level", 0)
        if layer not in layers:
            layers[layer] = {"count": 0, "level": level, "newest": None}
        layers[layer]["count"] += 1

        # Read created_at from artifact file for last-built timestamp
        art = store.load_artifact(aid)
        if art and art.created_at:
            ts = art.created_at if isinstance(art.created_at, str) else art.created_at.isoformat()
            current = layers[layer]["newest"]
            if current is None or ts > current:
                layers[layer]["newest"] = ts

    for layer_name, info in sorted(layers.items(), key=lambda x: x[1]["level"]):
        if layer_name == "traces":
            continue  # skip system artifacts
        style = get_layer_style(info["level"])
        ts = info.get("newest")
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                time_str = dt.strftime("%b %d %H:%M")
            except (ValueError, TypeError):
                time_str = "-"
        else:
            time_str = "-"
        table.add_row(
            f"[{style}]{layer_name}[/{style}]",
            str(info["count"]),
            time_str,
        )

    console.print()
    console.print(table)

    # ── Releases / Projections ─────────────────────────────────────────
    releases_dir = synix_dir / "releases"
    has_releases = False
    if releases_dir.exists():
        for release_path in sorted(releases_dir.iterdir()):
            if not release_path.is_dir():
                continue
            name = release_path.name
            # Enumerate all non-hidden files in the release directory
            outputs = [
                f.name for f in sorted(release_path.iterdir())
                if f.is_file() and not f.name.startswith(".")
            ]
            if outputs:
                if not has_releases:
                    console.print("\n[bold]Releases:[/bold]")
                    has_releases = True
                console.print(f"  [green]{name}[/green]: {', '.join(outputs)}")

    # ── Stale artifacts ─────────────────────────────────────────────────
    stale = _find_stale_artifacts(store)
    if stale:
        stale_tree = Tree(f"[yellow bold]Stale Artifacts: {len(stale)} artifact(s) need rebuild[/yellow bold]")
        for label, changed_parents in sorted(stale.items()):
            node = stale_tree.add(f"[bold]{label}[/bold]")
            for parent in changed_parents:
                node.add(f"[yellow]\u21b3 parent changed: {parent}[/yellow]")
        console.print()
        console.print(stale_tree)

    # ── Violations ──────────────────────────────────────────────────────
    violations_state = build_path / "violations_state.json"
    by_status: dict[str, int] = {}
    state: dict = {}

    if violations_state.exists():
        state = _json.loads(violations_state.read_text())

    if state:
        # Counts by status
        for v in state.values():
            s = v.get("status", "unknown")
            by_status[s] = by_status.get(s, 0) + 1

        parts = []
        if by_status.get("active"):
            parts.append(f"[red]{by_status['active']} active[/red]")
        if by_status.get("resolved"):
            parts.append(f"[green]{by_status['resolved']} resolved[/green]")
        if by_status.get("ignored"):
            parts.append(f"[dim]{by_status['ignored']} ignored[/dim]")
        if by_status.get("expired"):
            parts.append(f"[dim]{by_status['expired']} expired[/dim]")
        console.print(f"\n[bold]Violations:[/bold] {', '.join(parts)}")

        # Group active violations by artifact label for tree display
        active_by_label: dict[str, list[dict]] = {}
        for entry in state.values():
            if entry.get("status") != "active":
                continue
            viol = entry.get("violation", {})
            label = viol.get("label", "?")
            active_by_label.setdefault(label, []).append(viol)

        if active_by_label:
            tree = Tree(f"[red bold]{by_status.get('active', 0)} active violation(s)[/red bold]")
            for label, viols in sorted(active_by_label.items()):
                node = tree.add(f"[bold]{label}[/bold] ({len(viols)})")
                for viol in viols:
                    msg = viol.get("message", "")
                    if len(msg) > 90:
                        msg = msg[:87] + "..."
                    sev = viol.get("severity", "warning")
                    sev_style = "red" if sev == "error" else "yellow"
                    node.add(f"[{sev_style}]{sev.upper()}[/{sev_style}] {msg}")
            console.print(tree)

        # Show resolved details when --resolved flag is set
        if resolved and by_status.get("resolved"):
            _show_resolved_details(state)

    # ── Next steps ──────────────────────────────────────────────────────
    active_count = by_status.get("active", 0)
    stale_count = len(stale)
    _print_next_steps(active_count, stale_count)


def _find_stale_artifacts(store) -> dict[str, list[str]]:
    """Find artifacts whose parents have changed since they were built.

    Returns a dict mapping stale artifact labels to the list of parent labels
    whose artifact_id no longer matches the child's input_ids.
    """
    stale: dict[str, list[str]] = {}

    for label in store.iter_entries():
        parent_labels = store.get_parents(label)
        if not parent_labels:
            continue

        # Load the artifact to get its input_ids
        artifact = store.load_artifact(label)
        if artifact is None:
            continue

        input_ids = set(artifact.input_ids or [])
        changed_parents: list[str] = []

        for parent_label in parent_labels:
            parent_hash = store.get_artifact_id(parent_label)
            if parent_hash and parent_hash not in input_ids:
                changed_parents.append(parent_label)

        if changed_parents:
            stale[label] = changed_parents

    return stale


def _show_resolved_details(state: dict) -> None:
    """Show resolved violation details grouped by label."""
    from datetime import datetime

    resolved_by_label: dict[str, list[dict]] = {}
    for entry in state.values():
        if entry.get("status") != "resolved":
            continue
        viol = entry.get("violation", {})
        label = viol.get("label", "?")
        resolved_by_label.setdefault(label, []).append(entry)

    if not resolved_by_label:
        return

    total = sum(len(v) for v in resolved_by_label.values())
    tree = Tree(f"[green bold]Resolved Violations: {total}[/green bold]")
    for label, entries in sorted(resolved_by_label.items()):
        node = tree.add(f"[bold]{label}[/bold] ({len(entries)})")
        for entry in entries:
            viol = entry.get("violation", {})
            fix_action = entry.get("fix_action", "")
            resolved_at = entry.get("resolved_at", "")
            msg = viol.get("message", "")
            if len(msg) > 80:
                msg = msg[:77] + "..."

            ts_str = ""
            if resolved_at:
                try:
                    dt = datetime.fromisoformat(resolved_at)
                    ts_str = f"  ({dt.strftime('%b %d %H:%M')})"
                except (ValueError, TypeError):
                    pass

            action_str = f"  {fix_action}" if fix_action else ""
            node.add(f"[green]\u2713{action_str}[/green]  {msg}{ts_str}")

    console.print()
    console.print(tree)


def _print_next_steps(active_count: int, stale_count: int) -> None:
    """Print actionable next-step guidance based on build state."""
    if active_count and stale_count:
        console.print(
            "\n[bold]Next:[/bold] Run [bold]synix fix <pipeline>[/bold] to resolve violations, "
            f"then [bold]synix build <pipeline>[/bold] to rebuild {stale_count} stale artifact(s)."
        )
    elif active_count:
        console.print("\n[bold]Next:[/bold] Run [bold]synix fix <pipeline>[/bold] to resolve violations.")
    elif stale_count:
        console.print(
            f"\n[bold]Next:[/bold] Run [bold]synix build <pipeline>[/bold] to rebuild {stale_count} stale artifact(s)."
        )
    else:
        console.print("\n[green]Build is clean.[/green]")


@click.command()
@click.option("--build-dir", default="./build", help="Build directory")
@click.option("--check", "checks", multiple=True, help="Run specific checks only")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option(
    "--pipeline",
    "pipeline_path",
    default=None,
    type=click.Path(exists=True),
    help="Pipeline file to run domain validators from",
)
def verify(build_dir: str, checks: tuple[str, ...], output_json: bool, pipeline_path: str | None):
    """Verify integrity of a completed build.

    Checks: build_exists, manifest_valid, artifacts_exist,
    provenance_complete, synix_search, content_hashes, no_orphans,
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
            from synix.build.pipeline import load_pipeline
            from synix.build.refs import synix_dir_for_build_dir
            from synix.build.snapshot_view import SnapshotArtifactCache
            from synix.build.validators import run_validators

            pipeline = load_pipeline(pipeline_path)
            if build_dir:
                pipeline.build_dir = build_dir

            if pipeline.validators:
                v_synix_dir = synix_dir_for_build_dir(Path(pipeline.build_dir))
                v_store = SnapshotArtifactCache(v_synix_dir)
                validation_result = run_validators(pipeline, v_store)
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
            console.print(f"  [{severity_style}]{v.label}[/{severity_style}]: {v.message}")

            if v.provenance_trace:
                console.print("    [dim]Provenance:[/dim]")
                for step in v.provenance_trace:
                    val_str = f"  {v.field}: {step.field_value}" if step.field_value else ""
                    console.print(f"      {step.label} [dim]({step.layer})[/dim]{val_str}")


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
    from synix.build.diff import diff_artifact_by_label, diff_builds

    if artifact_id:
        # Single artifact diff
        result = diff_artifact_by_label(build_dir, artifact_id, previous_build_dir=old_build_dir)
        if result is None:
            console.print(f"[red]Cannot diff artifact:[/red] {artifact_id}")
            console.print("[dim]Either artifact not found or no previous version available.[/dim]")
            sys.exit(1)

        if not result.has_changes:
            console.print(f"[green]No changes[/green] for {artifact_id}")
            return

        console.print(f"\n[bold]Diff for:[/bold] {artifact_id}")
        if result.old_prompt_id != result.new_prompt_id:
            console.print(f"[yellow]Prompt changed:[/yellow] {result.old_prompt_id} → {result.new_prompt_id}")
        if result.content_diff:
            console.print(
                Panel(
                    Syntax(result.content_diff, "diff", theme="monokai"),
                    title="Content diff",
                    border_style="yellow",
                )
            )
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
                console.print(f"  [yellow]~ {d.label}[/yellow]")
                if d.content_diff:
                    lines = d.content_diff.count("\n")
                    console.print(f"    [dim]{lines} lines changed[/dim]")
