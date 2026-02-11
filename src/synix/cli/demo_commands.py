"""Demo commands — synix demo note, synix demo run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import click
from rich.panel import Panel

from synix.cli.main import console


@click.group()
def demo():
    """Demo tools for deterministic recordings."""
    pass


@demo.command()
@click.argument("message")
def note(message: str):
    """Print a narrative note panel for VHS recordings.

    MESSAGE is the text to display. No timestamps or dynamic content —
    purely deterministic output for reproducible recordings.
    """
    console.print()
    console.print(Panel(
        f"[bold]{message}[/bold]",
        border_style="cyan",
        padding=(0, 2),
    ))
    console.print()


@demo.command(name="run")
@click.argument("case_dir", type=click.Path(exists=True))
@click.option("--update-goldens", is_flag=True, help="Regenerate golden output files")
def run_case(case_dir: str, update_goldens: bool):
    """Run a demo case and compare against golden outputs.

    CASE_DIR is the path to a demo case directory containing case.py.
    Sets SYNIX_CASSETTE_MODE=replay and SYNIX_DEMO=1 for deterministic output.
    """
    case_path = Path(case_dir).resolve()
    case_module_path = case_path / "case.py"

    if not case_module_path.exists():
        console.print(f"[red]No case.py found in {case_dir}[/red]")
        sys.exit(1)

    # Load case definition
    case = _load_case(case_module_path)
    if case is None:
        console.print(f"[red]Failed to load case from {case_module_path}[/red]")
        sys.exit(1)

    pipeline_file = case.get("pipeline", "pipeline.py")
    steps = case.get("steps", [])
    goldens = case.get("goldens", {})
    case_name = case.get("name", case_path.name)

    console.print(Panel(
        f"[bold]Case:[/bold] {case_name}\n"
        f"[bold]Dir:[/bold] {case_path}\n"
        f"[bold]Steps:[/bold] {len(steps)}\n"
        f"[bold]Goldens:[/bold] {len(goldens)}",
        title="[bold cyan]Synix Demo Run[/bold cyan]",
        border_style="cyan",
    ))

    # Set up environment for deterministic replay
    env = dict(os.environ)
    cassette_dir = str(case_path / "cassettes")
    env["SYNIX_CASSETTE_MODE"] = "replay"
    env["SYNIX_CASSETTE_DIR"] = cassette_dir
    env["SYNIX_DEMO"] = "1"

    golden_dir = case_path / "golden"
    golden_dir.mkdir(exist_ok=True)

    captured: dict[str, str] = {}
    failed = False

    for step in steps:
        step_name = step.get("name", "unnamed")
        command = list(step.get("command", []))
        stdin_data = step.get("stdin")
        capture_json = step.get("capture_json", False)

        # Replace PIPELINE placeholder with actual pipeline path
        command = [
            pipeline_file if c == "PIPELINE" else c
            for c in command
        ]

        # Replace bare "synix" with the venv script path
        # so the demo runs in the same Python environment as the runner.
        if command and command[0] == "synix":
            venv_synix = Path(sys.executable).parent / "synix"
            if venv_synix.exists():
                command[0] = str(venv_synix)
            # If no venv script, try sys.executable -m synix.cli.main
            else:
                command = [sys.executable, "-m", "synix.cli.main"] + command[1:]

        console.print(f"\n  [dim]step:[/dim] [bold]{step_name}[/bold]  →  {' '.join(command)}")

        try:
            result = subprocess.run(
                command,
                cwd=str(case_path),
                env=env,
                capture_output=True,
                text=True,
                input=stdin_data,
                timeout=120,
            )

            if result.returncode != 0:
                console.print(f"    [yellow]exit {result.returncode}[/yellow]")
                if result.stderr:
                    for line in result.stderr.strip().splitlines()[:5]:
                        console.print(f"    [dim]{line}[/dim]")

            # Show stdout (abbreviated)
            if result.stdout:
                lines = result.stdout.strip().splitlines()
                for line in lines[:10]:
                    console.print(f"    {line}")
                if len(lines) > 10:
                    console.print(f"    [dim]... ({len(lines) - 10} more lines)[/dim]")

            if capture_json and result.stdout:
                captured[step_name] = result.stdout.strip()

        except subprocess.TimeoutExpired:
            console.print("    [red]TIMEOUT[/red]")
            failed = True
        except FileNotFoundError:
            console.print(f"    [red]Command not found: {command[0]}[/red]")
            failed = True

    # Golden comparison
    if goldens:
        console.print("\n[bold]Golden comparison:[/bold]")

        for step_name, golden_file in goldens.items():
            golden_path = golden_dir / golden_file

            if step_name not in captured:
                console.print(f"  {step_name}: [yellow]no captured output[/yellow]")
                continue

            actual_output = captured[step_name]

            if update_goldens:
                golden_path.write_text(actual_output)
                console.print(f"  {step_name}: [cyan]updated[/cyan] → {golden_file}")
                continue

            if not golden_path.exists():
                console.print(
                    f"  {step_name}: [yellow]no golden file[/yellow] "
                    f"(run with --update-goldens)"
                )
                failed = True
                continue

            expected = golden_path.read_text().strip()
            actual = actual_output.strip()

            # Compare JSON structurally if both parse
            try:
                expected_json = json.loads(expected)
                actual_json = json.loads(actual)
                match = expected_json == actual_json
            except (json.JSONDecodeError, ValueError):
                match = expected == actual

            if match:
                console.print(f"  {step_name}: [green]PASS[/green]")
            else:
                console.print(f"  {step_name}: [red]FAIL[/red]")
                failed = True
                # Show diff snippet
                expected_lines = expected.splitlines()
                actual_lines = actual.splitlines()
                for i, (e, a) in enumerate(zip(expected_lines, actual_lines)):
                    if e != a:
                        console.print(f"    [dim]line {i+1}:[/dim]")
                        console.print(f"      [red]expected:[/red] {e[:100]}")
                        console.print(f"      [green]actual:[/green]  {a[:100]}")
                        break

    if failed:
        console.print("\n[red]Demo case failed.[/red]")
        sys.exit(1)
    else:
        console.print("\n[green]Demo case passed.[/green]")


def _load_case(case_path: Path) -> dict | None:
    """Load a case.py module and return its `case` dict."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("case", str(case_path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        console.print(f"[red]Error loading case.py:[/red] {e}")
        return None
    return getattr(module, "case", None)
