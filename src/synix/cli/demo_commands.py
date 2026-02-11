"""Demo commands — synix demo note, synix demo run."""

from __future__ import annotations

import difflib
import json
import os
import re
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
    console.print(
        Panel(
            f"[bold]{message}[/bold]",
            border_style="cyan",
            padding=(0, 2),
        )
    )
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

    console.print(
        Panel(
            f"[bold]Case:[/bold] {case_name}\n"
            f"[bold]Dir:[/bold] {case_path}\n"
            f"[bold]Steps:[/bold] {len(steps)}\n"
            f"[bold]Goldens:[/bold] {len(goldens)}",
            title="[bold cyan]Synix Demo Run[/bold cyan]",
            border_style="cyan",
        )
    )

    # Set up environment for deterministic replay
    env = dict(os.environ)
    cassette_dir = str(case_path / "cassettes")
    env["SYNIX_CASSETTE_MODE"] = "replay"
    env["SYNIX_CASSETTE_DIR"] = cassette_dir
    env["SYNIX_DEMO"] = "1"
    env["COLUMNS"] = "120"
    env["NO_COLOR"] = "1"

    golden_dir = case_path / "golden"
    golden_dir.mkdir(exist_ok=True)

    captured_json: dict[str, str] = {}
    captured_stdout: dict[str, str] = {}
    captured_stderr: dict[str, str] = {}
    failed = False

    for step in steps:
        step_name = step.get("name", "unnamed")
        command = list(step.get("command", []))
        stdin_data = step.get("stdin")
        capture_json = step.get("capture_json", False)

        # Replace PIPELINE placeholder with actual pipeline path
        command = [pipeline_file if c == "PIPELINE" else c for c in command]

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

            # Capture stdout/stderr for every step (text golden comparison)
            if result.stdout:
                captured_stdout[step_name] = result.stdout
            if result.stderr:
                captured_stderr[step_name] = result.stderr

            if capture_json and result.stdout:
                captured_json[step_name] = result.stdout.strip()

        except subprocess.TimeoutExpired:
            console.print("    [red]TIMEOUT[/red]")
            failed = True
        except FileNotFoundError:
            console.print(f"    [red]Command not found: {command[0]}[/red]")
            failed = True

    # JSON golden comparison (existing behavior)
    if goldens:
        console.print("\n[bold]Golden comparison (JSON):[/bold]")

        for step_name, golden_file in goldens.items():
            golden_path = golden_dir / golden_file

            if step_name not in captured_json:
                console.print(f"  {step_name}: [yellow]no captured output[/yellow]")
                continue

            actual_output = captured_json[step_name]

            if update_goldens:
                golden_path.write_text(actual_output)
                console.print(f"  {step_name}: [cyan]updated[/cyan] → {golden_file}")
                continue

            if not golden_path.exists():
                console.print(f"  {step_name}: [yellow]no golden file[/yellow] (run with --update-goldens)")
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
                _show_text_diff(expected, actual, golden_file)

    # Text golden comparison (stdout/stderr for every step)
    all_step_names = [step.get("name", "unnamed") for step in steps]
    has_text_goldens = any((golden_dir / f"{name}.stdout.txt").exists() for name in all_step_names)

    if captured_stdout or has_text_goldens or update_goldens:
        console.print("\n[bold]Golden comparison (text):[/bold]")

        for step_name in all_step_names:
            # stdout golden
            stdout_golden_file = f"{step_name}.stdout.txt"
            stdout_golden_path = golden_dir / stdout_golden_file

            if step_name in captured_stdout:
                actual_normalized = _normalize_output(captured_stdout[step_name], case_path)

                if update_goldens:
                    stdout_golden_path.write_text(actual_normalized)
                    console.print(f"  {step_name} stdout: [cyan]updated[/cyan] → {stdout_golden_file}")
                elif stdout_golden_path.exists():
                    expected = stdout_golden_path.read_text()
                    if expected == actual_normalized:
                        console.print(f"  {step_name} stdout: [green]PASS[/green]")
                    else:
                        console.print(f"  {step_name} stdout: [red]FAIL[/red]")
                        failed = True
                        _show_text_diff(expected, actual_normalized, stdout_golden_file)
                else:
                    console.print(f"  {step_name} stdout: [yellow]no golden[/yellow] (run with --update-goldens)")

            # stderr golden (only if non-empty)
            stderr_golden_file = f"{step_name}.stderr.txt"
            stderr_golden_path = golden_dir / stderr_golden_file

            if step_name in captured_stderr:
                actual_normalized = _normalize_output(captured_stderr[step_name], case_path)

                if update_goldens:
                    stderr_golden_path.write_text(actual_normalized)
                    console.print(f"  {step_name} stderr: [cyan]updated[/cyan] → {stderr_golden_file}")
                elif stderr_golden_path.exists():
                    expected = stderr_golden_path.read_text()
                    if expected == actual_normalized:
                        console.print(f"  {step_name} stderr: [green]PASS[/green]")
                    else:
                        console.print(f"  {step_name} stderr: [red]FAIL[/red]")
                        failed = True
                        _show_text_diff(expected, actual_normalized, stderr_golden_file)

    if failed:
        console.print("\n[red]Demo case failed.[/red]")
        sys.exit(1)
    else:
        console.print("\n[green]Demo case passed.[/green]")


def _normalize_output(text: str, case_path: Path) -> str:
    """Replace dynamic values with stable placeholders for golden comparison."""
    case_path_str = str(case_path)

    lines = text.splitlines()
    normalized = []
    for line in lines:
        # Replace case directory path with placeholder
        line = line.replace(case_path_str, "<CASE_DIR>")
        # Replace timing values like "1.2s" or "0.05s"
        line = re.sub(r"\b\d+\.\d+s\b", "<TIME>", line)
        # Replace full LLM stats lines
        line = re.sub(
            r"LLM calls: \d+, Tokens: [\d,]+, Est\. cost: \$[\d.]+",
            "LLM calls: <N>, Tokens: <N>, Est. cost: $<COST>",
            line,
        )
        # Replace inline token/cost fragments
        line = re.sub(r"[\d,]+ tokens, \$[\d.]+", "<N> tokens, $<COST>", line)
        # Remove API key display line (depends on env — may or may not be present)
        if re.search(r"│\s*API Key\s*│", line):
            continue
        # Replace verify output counts (artifact/provenance/hash counts grow across runs)
        line = re.sub(r"(\bManifest valid with )\d+( artifacts\b)", r"\g<1><N>\2", line)
        line = re.sub(r"(\bAll )\d+( artifact files\b)", r"\g<1><N>\2", line)
        line = re.sub(r"\b\d+( non-root artifacts lack provenance\b)", r"<N>\1", line)
        line = re.sub(r"(\bAll )\d+( content hashes\b)", r"\g<1><N>\2", line)
        line = re.sub(r"(\bSearch index has )\d+( entries\b)", r"\g<1><N>\2", line)
        # Strip trailing whitespace
        line = line.rstrip()
        normalized.append(line)
    return "\n".join(normalized)


def _show_text_diff(expected: str, actual: str, label: str) -> None:
    """Show a unified diff between expected and actual text, capped at 15 lines."""
    diff_lines = list(
        difflib.unified_diff(
            expected.splitlines(),
            actual.splitlines(),
            fromfile=f"golden/{label}",
            tofile="actual",
            lineterm="",
        )
    )
    if not diff_lines:
        return
    for line in diff_lines[:15]:
        if line.startswith("+") and not line.startswith("+++"):
            console.print(f"    [green]{line}[/green]")
        elif line.startswith("-") and not line.startswith("---"):
            console.print(f"    [red]{line}[/red]")
        else:
            console.print(f"    [dim]{line}[/dim]")
    if len(diff_lines) > 15:
        console.print(f"    [dim]... ({len(diff_lines) - 15} more diff lines)[/dim]")


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
