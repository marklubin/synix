"""Fix command — resolve violations found by validate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Group
from rich.panel import Panel
from rich.tree import Tree

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

    _run_fix_mode(pipeline, build_path, output_json, dry_run)


def _run_fix_mode(pipeline, build_path: Path, output_json: bool, dry_run: bool):
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
                artifact_id=vd.get("artifact_id", ""),
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
                    "artifact_id": a.artifact_id,
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
        console.print(json.dumps(out, indent=2))
        return

    if not fix_result.actions:
        console.print("\n[yellow]No fixable violations found.[/yellow]")
        return

    applied_count = 0
    denied_count = 0
    ignored_count = 0

    # Index violations by artifact_id for lookup
    violations_by_artifact: dict[str, Violation] = {}
    for v in result.violations:
        violations_by_artifact.setdefault(v.artifact_id, v)

    for action in fix_result.actions:
        if action.action == "skip":
            console.print(f"\n[dim]Skipping {action.artifact_id}: {action.description}[/dim]")
            continue

        violation = violations_by_artifact.get(action.artifact_id)
        if action.action == "rewrite":
            _display_rewrite_proposal(action, violation, store, provenance)
        elif action.action == "unresolved":
            _display_unresolved(action, violation, store, provenance)

        if dry_run:
            console.print("[dim](dry-run: no changes applied)[/dim]")
            continue

        # Interactive prompt
        choice = _prompt_choice()

        if choice == "a":
            if action.action == "rewrite":
                apply_fix(action, store, provenance)
                # Find violation_id for this action
                for v in result.violations:
                    if v.artifact_id == action.artifact_id:
                        queue.resolve(v.violation_id, fix_action="rewrite")
                        break
                console.print(f"[green]Applied fix to {action.artifact_id}[/green]")
            else:
                # Unresolved: accept original as-is
                for v in result.violations:
                    if v.artifact_id == action.artifact_id:
                        queue.resolve(v.violation_id, fix_action="accept_original")
                        break
                console.print(f"[green]Accepted original for {action.artifact_id}[/green]")
            applied_count += 1
        elif choice == "d":
            console.print(f"[yellow]Denied fix for {action.artifact_id}[/yellow]")
            denied_count += 1
        elif choice == "i":
            for v in result.violations:
                if v.artifact_id == action.artifact_id:
                    queue.ignore(v.violation_id)
                    break
            console.print(f"[dim]Ignored {action.artifact_id} (won't resurface for same content)[/dim]")
            ignored_count += 1

    queue.save_state()

    # Summary
    console.print("\n[bold]Fix summary:[/bold]")
    console.print(f"  Applied: {applied_count}")
    console.print(f"  Denied: {denied_count}")
    console.print(f"  Ignored: {ignored_count}")

    if applied_count > 0 and fix_result.rebuild_required:
        unique_downstream = set()
        for a in fix_result.actions:
            unique_downstream.update(a.downstream_invalidated)
        console.print(
            f"\n[yellow]Run [bold]synix build {sys.argv[-1] if len(sys.argv) > 1 else 'pipeline.py'}[/bold] "
            f"to rebuild {len(unique_downstream)} downstream artifact(s).[/yellow]"
        )

    if search_index is not None:
        search_index.close()


def _build_fix_investigation_tree(action, violation, store, provenance):
    """Build a tree showing the investigation path from conflict to sources."""
    tree = Tree(f"[bold]{action.artifact_id}[/bold] [dim](conflict detected)[/dim]")

    # Walk provenance to show parents
    parent_ids = provenance.get_parents(action.artifact_id)
    if parent_ids:
        parents_branch = tree.add("[dim]parents[/dim]")
        for pid in parent_ids:
            parent_art = store.load_artifact(pid)
            if parent_art:
                label = f"[bold]{pid}[/bold] [dim]({parent_art.artifact_type})[/dim]"
            else:
                label = f"[bold]{pid}[/bold]"
            parents_branch.add(label)

    # Evidence sources the fixer found
    if action.evidence_source_ids:
        evidence_branch = tree.add("[dim]evidence sources[/dim]")
        for eid in action.evidence_source_ids:
            ev_art = store.load_artifact(eid)
            if ev_art:
                label = f"[bold]{eid}[/bold] [dim]({ev_art.artifact_type})[/dim]"
            else:
                label = f"[bold]{eid}[/bold]"
            evidence_branch.add(label)

    return tree


def _display_rewrite_proposal(action, violation=None, store=None, provenance=None):
    """Display a rewrite proposal with investigation tree and diff."""
    severity_style = "red"
    lines: list[str] = []
    if violation:
        lines.append(
            f"[{severity_style} bold]{violation.severity.upper()}[/{severity_style} bold]  {violation.message}"
        )
        if violation.violation_type == "semantic_conflict":
            claim_a = violation.metadata.get("claim_a", "")
            claim_b = violation.metadata.get("claim_b", "")
            explanation = violation.metadata.get("explanation", "")
            if claim_a and claim_b:
                lines.append("")
                lines.append(f'[dim]Claim A:[/dim]  "{claim_a}"')
                lines.append(f'[dim]Claim B:[/dim]  "{claim_b}"')
            if explanation:
                lines.append(f"[dim]Reasoning:[/dim]  {explanation}")
    else:
        lines.append(f"[bold]Fix proposal for {action.artifact_id}[/bold]")

    body = "\n".join(lines)

    if store and provenance:
        inv_tree = _build_fix_investigation_tree(
            action,
            violation,
            store,
            provenance,
        )
        panel_content = Group(body, "", inv_tree)
    else:
        panel_content = body

    console.print(
        Panel(
            panel_content,
            title=f"[cyan]Fix: {action.artifact_id}[/cyan]",
            border_style="cyan",
            padding=(0, 1),
        )
    )

    if action.new_content:
        fix_lines: list[str] = []
        fix_lines.append(f"[dim]Description:[/dim]  {action.description}")
        if action.llm_explanation:
            fix_lines.append(f"[dim]Explanation:[/dim]  {action.llm_explanation}")
        fix_lines.append("")
        preview = action.new_content[:800]
        if len(action.new_content) > 800:
            preview += "\n..."
        fix_lines.append(preview)
        console.print(
            Panel(
                "\n".join(fix_lines),
                title="[green]Proposed rewrite[/green]",
                border_style="green",
                padding=(0, 1),
            )
        )


def _display_unresolved(action, violation=None, store=None, provenance=None):
    """Display an unresolved contradiction with investigation tree."""
    lines: list[str] = []
    if violation:
        lines.append(f"[yellow bold]WARNING[/yellow bold]  {violation.message}")
        if violation.violation_type == "semantic_conflict":
            claim_a = violation.metadata.get("claim_a", "")
            claim_b = violation.metadata.get("claim_b", "")
            if claim_a and claim_b:
                lines.append("")
                lines.append(f'[dim]Claim A:[/dim]  "{claim_a}"')
                lines.append(f'[dim]Claim B:[/dim]  "{claim_b}"')
    else:
        lines.append(f"[bold yellow]Cannot auto-resolve: {action.artifact_id}[/bold yellow]")

    if action.llm_explanation:
        lines.append("")
        lines.append(f"[dim]Explanation:[/dim]  {action.llm_explanation}")

    body = "\n".join(lines)

    if store and provenance:
        inv_tree = _build_fix_investigation_tree(
            action,
            violation,
            store,
            provenance,
        )
        panel_content = Group(body, "", inv_tree)
    else:
        panel_content = body

    console.print(
        Panel(
            panel_content,
            title=f"[yellow]Unresolved: {action.artifact_id}[/yellow]",
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
