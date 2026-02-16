"""Fix command — resolve violations found by validate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Group
from rich.panel import Panel

from synix.cli.main import console, pipeline_argument


@click.command()
@pipeline_argument
@click.option("--build-dir", default=None, help="Override build directory")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
@click.option("--dry-run", is_flag=True, help="Show fix proposals without applying")
def fix(pipeline_path: str, build_dir: str | None, output_json: bool, dry_run: bool):
    """Fix violations found by validate.

    Reads pre-saved violations from the violation queue and runs
    pipeline fixers to propose resolutions. Stale violations
    (where the artifact has been rebuilt) are automatically expired.
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

    _run_fix_mode(pipeline, build_path, output_json, dry_run, pipeline_path)


def _run_fix_mode(pipeline, build_path: Path, output_json: bool, dry_run: bool, pipeline_path: str = "pipeline.py"):
    """Load persisted violations from queue and propose/apply fixes."""
    from synix.build.artifacts import ArtifactStore
    from synix.build.fixers import apply_fix, run_fixers
    from synix.build.provenance import ProvenanceTracker
    from synix.build.validators import (
        ValidationResult,
        Violation,
        ViolationQueue,
    )

    store = ArtifactStore(build_path)
    provenance = ProvenanceTracker(build_path)

    mode_label = "dry-run" if dry_run else "fix"

    # Load persisted violations from queue (written by previous validate run)
    # Pass store to auto-expire stale violations where artifact was rebuilt
    queue = ViolationQueue.load(build_path)
    active_dicts = queue.active(store=store)
    queue.save_state()  # persist any expiry changes

    if not output_json:
        console.print(
            Panel(
                f"[bold]Pipeline:[/bold] {pipeline.name}\n"
                f"[bold]Build:[/bold] {build_path}\n"
                f"[bold]Active violations:[/bold] {len(active_dicts)}\n"
                f"[bold]Mode:[/bold] {mode_label}",
                title="[bold cyan]Synix Fix[/bold cyan]",
                border_style="cyan",
            )
        )

    if not active_dicts:
        if not output_json:
            console.print(
                "\n[green]No active violations.[/green] Run [bold]synix validate[/bold] first to detect violations."
            )
        return

    # Reconstruct Violation objects from persisted state
    result = ValidationResult()
    for vd in active_dicts:
        result.violations.append(
            Violation(
                violation_type=vd.get("violation_type", ""),
                severity=vd.get("severity", "warning"),
                message=vd.get("message", ""),
                label=vd.get("label", vd.get("artifact_id", "")),
                field=vd.get("field", ""),
                metadata=vd.get("metadata", {}),
                violation_id=vd.get("violation_id", ""),
            )
        )

    if not output_json:
        errors = sum(1 for v in result.violations if v.severity == "error")
        warnings = sum(1 for v in result.violations if v.severity == "warning")
        console.print(
            f"\n  [bold]{len(result.violations)}[/bold] active violation(s) ({errors} errors, {warnings} warnings)"
        )

    if not pipeline.fixers:
        console.print("\n[yellow]No fixers declared in pipeline.[/yellow] Add fixers to pipeline.fixers to enable fix.")
        return

    # Set up LLM client for fixers
    llm_client = None
    try:
        from synix.build.cassette import maybe_wrap_client
        from synix.build.llm_client import LLMClient
        from synix.core.config import LLMConfig

        llm_cfg = LLMConfig.from_dict(pipeline.llm_config)
        llm_client = maybe_wrap_client(LLMClient(llm_cfg))
    except Exception:
        pass

    # Set up search index
    search_index = None
    try:
        from synix.search.indexer import SearchIndex

        search_db = build_path / "search.db"
        if search_db.exists():
            search_index = SearchIndex(search_db)
    except Exception:
        pass

    # Run fixers with live progress
    if not output_json:
        status = console.status("[bold cyan]Running fixers...[/bold cyan]", spinner="dots")
        status.start()

        def _on_progress(msg: str, current: int, total: int) -> None:
            status.update(f"[bold cyan]Fixing[/bold cyan] [{current}/{total}] {msg}")

        fix_result = run_fixers(
            result,
            pipeline,
            store,
            provenance,
            search_index=search_index,
            llm_client=llm_client,
            on_progress=_on_progress,
        )
        status.stop()
    else:
        fix_result = run_fixers(
            result,
            pipeline,
            store,
            provenance,
            search_index=search_index,
            llm_client=llm_client,
        )

    if output_json:
        out = {
            "fixers_run": fix_result.fixers_run,
            "actions": [
                {
                    "label": a.label,
                    "action": a.action,
                    "description": a.description,
                    "interactive": a.interactive,
                    "llm_explanation": a.llm_explanation,
                    "downstream_invalidated": a.downstream_invalidated,
                    "evidence_source_ids": a.evidence_source_ids,
                }
                for a in fix_result.actions
            ],
            "errors": fix_result.errors,
            "fixed_count": fix_result.fixed_count,
            "skipped_count": fix_result.skipped_count,
        }
        # Use print() not console.print() to avoid Rich word-wrapping inside JSON strings
        print(json.dumps(out, indent=2))
        return

    if not fix_result.actions:
        console.print("\n[yellow]No fixable violations found.[/yellow]")
        return

    # Index ALL violations by label (batched — one fix action per artifact)
    violations_by_label: dict[str, list[Violation]] = {}
    for v in result.violations:
        violations_by_label.setdefault(v.label, []).append(v)

    # Track decisions per action for summary
    decisions: list[dict] = []

    for action in fix_result.actions:
        if action.action == "skip":
            console.print(f"\n[dim]Skipping {action.label}: {action.description}[/dim]")
            decisions.append({"action": action, "choice": "skip", "violations": []})
            continue

        label_violations = violations_by_label.get(action.label, [])
        first_violation = label_violations[0] if label_violations else None
        if action.action == "rewrite":
            _display_rewrite_proposal(action, first_violation, store, provenance)
        elif action.action == "unresolved":
            _display_unresolved(action, first_violation, store, provenance)

        if dry_run:
            console.print("[dim](dry-run: no changes applied)[/dim]")
            decisions.append({"action": action, "choice": "dry-run", "violations": label_violations})
            continue

        # Interactive prompt
        choice = _prompt_choice()

        if choice == "a":
            if action.action == "rewrite":
                apply_fix(action, store, provenance)
                for v in label_violations:
                    queue.resolve(v.violation_id, fix_action="rewrite")
            else:
                for v in label_violations:
                    queue.resolve(v.violation_id, fix_action="accept_original")
            decisions.append({"action": action, "choice": "accept", "violations": label_violations})
        elif choice == "d":
            decisions.append({"action": action, "choice": "deny", "violations": label_violations})
        elif choice == "i":
            for v in label_violations:
                queue.ignore(v.violation_id)
            decisions.append({"action": action, "choice": "ignore", "violations": label_violations})

    queue.save_state()

    # Remaining active violations in queue
    remaining = queue.active()

    _print_fix_summary(decisions, remaining, fix_result, pipeline_path)

    if search_index is not None:
        search_index.close()


def _print_fix_summary(decisions, remaining, fix_result, pipeline_path):
    """Print a detailed fix summary showing what happened to each artifact."""
    from rich.tree import Tree

    tree = Tree("[bold]Fix Summary[/bold]")

    for d in decisions:
        action = d["action"]
        choice = d["choice"]
        violations = d["violations"]
        n_viols = len(violations)

        if choice == "accept" and action.action == "rewrite":
            icon = "[green]\u2713[/green]"
            label = f"{icon} [bold]{action.label}[/bold]  [green]rewritten[/green] ({n_viols} violations resolved)"
        elif choice == "accept" and action.action == "unresolved":
            icon = "[green]\u2713[/green]"
            label = (
                f"{icon} [bold]{action.label}[/bold]  [green]accepted original[/green] ({n_viols} violations resolved)"
            )
        elif choice == "deny":
            icon = "[yellow]\u2717[/yellow]"
            label = f"{icon} [bold]{action.label}[/bold]  [yellow]denied[/yellow] ({n_viols} violations remain)"
        elif choice == "ignore":
            icon = "[dim]\u2500[/dim]"
            label = f"{icon} [bold]{action.label}[/bold]  [dim]ignored[/dim] ({n_viols} violations suppressed)"
        elif choice == "skip":
            icon = "[dim]\u2500[/dim]"
            label = f"{icon} [bold]{action.label}[/bold]  [dim]skipped[/dim]"
        else:
            label = f"  [bold]{action.label}[/bold]  {choice}"

        node = tree.add(label)

        # Show the violations that were addressed
        for v in violations:
            msg = v.message
            if len(msg) > 90:
                msg = msg[:87] + "..."
            node.add(f"[dim]{msg}[/dim]")

        # Show downstream artifacts that need rebuilding
        if choice == "accept" and action.downstream_invalidated:
            for ds in action.downstream_invalidated:
                node.add(f"[yellow]\u21b3 rebuild needed: {ds}[/yellow]")

    console.print()
    console.print(tree)

    # Next steps
    unique_downstream: set[str] = set()
    for d in decisions:
        if d["choice"] == "accept":
            unique_downstream.update(d["action"].downstream_invalidated)

    if unique_downstream:
        console.print(
            f"\n[yellow]Next:[/yellow] [bold]synix build {pipeline_path}[/bold] "
            f"to rebuild {len(unique_downstream)} downstream artifact(s)."
        )
        console.print("[dim]Run [bold]synix status[/bold] to see full build health.[/dim]")

    if remaining:
        console.print(
            f"\n[dim]{len(remaining)} violation(s) still active. "
            f"Run [bold]synix validate {pipeline_path}[/bold] to re-check.[/dim]"
        )
    elif not unique_downstream:
        console.print("\n[green]All violations resolved.[/green]")


def _render_diff(original: str, proposed: str) -> str:
    """Render a colorized unified diff between original and proposed content."""
    import difflib

    orig_lines = original.splitlines(keepends=True)
    prop_lines = proposed.splitlines(keepends=True)
    diff = difflib.unified_diff(orig_lines, prop_lines, fromfile="original", tofile="proposed", lineterm="")

    out: list[str] = []
    for line in diff:
        line = line.rstrip("\n")
        # Escape Rich markup in diff content
        escaped = line.replace("[", "\\[")
        if line.startswith("+++") or line.startswith("---"):
            out.append(f"[bold]{escaped}[/bold]")
        elif line.startswith("@@"):
            out.append(f"[cyan]{escaped}[/cyan]")
        elif line.startswith("+"):
            out.append(f"[green]{escaped}[/green]")
        elif line.startswith("-"):
            out.append(f"[red]{escaped}[/red]")
        else:
            out.append(f"[dim]{escaped}[/dim]")
    return "\n".join(out)


def _display_rewrite_proposal(action, violation=None, store=None, provenance=None):
    """Display a rewrite proposal with diff."""
    # Header: explanation of what the fix does
    lines: list[str] = []
    if action.llm_explanation:
        lines.append(f"[bold]{action.llm_explanation}[/bold]")
    elif action.description:
        lines.append(f"[bold]{action.description}[/bold]")

    body = "\n".join(lines)

    # Build diff between original and proposed content
    diff_text = ""
    if action.new_content and store:
        original_art = store.load_artifact(action.label)
        if original_art:
            diff_text = _render_diff(original_art.content, action.new_content)

    if diff_text:
        panel_content = Group(body, "", diff_text) if body else diff_text
    else:
        panel_content = body

    console.print(
        Panel(
            panel_content,
            title=f"[cyan]Fix: {action.label}[/cyan]",
            border_style="cyan",
            padding=(0, 1),
        )
    )


def _display_unresolved(action, violation=None, store=None, provenance=None):
    """Display an unresolved contradiction."""
    lines: list[str] = []
    if action.llm_explanation:
        lines.append(f"[bold]{action.llm_explanation}[/bold]")
    elif action.description:
        lines.append(f"[bold]{action.description}[/bold]")
    else:
        lines.append("[bold]Cannot auto-resolve[/bold]")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"[yellow]Unresolved: {action.label}[/yellow]",
            border_style="yellow",
            padding=(0, 1),
        )
    )


def _prompt_choice() -> str:
    """Prompt user for accept/deny/ignore. Returns 'a', 'd', or 'i'."""
    while True:
        try:
            choice = input("\n  [a]ccept / [d]eny / [i]gnore > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "d"
        if choice in ("a", "d", "i"):
            return choice
        console.print("  [dim]Please enter 'a', 'd', or 'i'[/dim]")
