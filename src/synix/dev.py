"""Dev workflow commands — release checks and demo verification."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _run(cmd: list[str], label: str, *, check: bool = True) -> int:
    """Run a command with a header, return exit code."""
    print(f"\n=== {label} ===")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if check and result.returncode != 0:
        print(f"\nFAILED: {label}")
        sys.exit(result.returncode)
    return result.returncode


def _find_demo_cases() -> list[Path]:
    """Find all example directories that have both case.py and cassettes."""
    examples_dir = REPO_ROOT / "examples"
    cases = []
    for case_dir in sorted(examples_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        if (case_dir / "case.py").exists() and (case_dir / "cassettes" / "llm.yaml").exists():
            cases.append(case_dir)
    return cases


def verify_demos() -> None:
    """Run demo verification for all example cases."""
    synix_bin = Path(sys.executable).parent / "synix"
    cases = _find_demo_cases()

    if not cases:
        print("No demo cases found.")
        sys.exit(1)

    failed = []
    for case_dir in cases:
        label = case_dir.name
        print(f"\n--- {label} ---")
        result = subprocess.run(
            [str(synix_bin), "demo", "run", str(case_dir)],
            cwd=str(REPO_ROOT),
        )
        if result.returncode != 0:
            failed.append(label)

    print()
    if failed:
        print(f"FAILED demos: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"All {len(cases)} demos passed.")


def release() -> None:
    """Full pre-commit check suite: lint → sync → test → demos."""
    _run([sys.executable, "-m", "ruff", "check", "--fix", "."], "ruff fix")
    _run([sys.executable, "-m", "ruff", "format", "."], "ruff format")
    _run([str(REPO_ROOT / "scripts" / "sync-templates")], "sync templates")
    _run([sys.executable, "-m", "ruff", "check", "."], "ruff check")
    _run([sys.executable, "-m", "pytest", "tests/", "-v"], "pytest")

    print("\n=== verify demos ===")
    verify_demos()

    print("\n=== all checks passed ===")
