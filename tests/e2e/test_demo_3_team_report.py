"""Demo 3: Team Report — E2E cassette replay test.

Verifies that `synix demo run examples/03-team-report` replays from cassettes
and matches golden outputs, with no real LLM calls required.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

CASE_DIR = Path(__file__).parent.parent.parent / "examples" / "03-team-report"


@pytest.fixture
def synix_bin():
    """Return path to the synix CLI binary in the current venv."""
    return str(Path(sys.executable).parent / "synix")


def test_demo_run_replays_from_cassettes(tmp_path, synix_bin):
    """synix demo run replays the team_report case from cassettes and passes goldens."""
    # Run in a subprocess so env vars are isolated
    result = subprocess.run(
        [synix_bin, "demo", "run", str(CASE_DIR)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(CASE_DIR),
        env={
            "PATH": str(Path(sys.executable).parent),
            "HOME": str(Path.home()),
        },
    )

    assert "Demo case passed" in result.stdout, (
        f"Demo replay failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.returncode == 0


def test_cassette_files_exist():
    """Cassette directory has the recorded LLM calls."""
    cassette_file = CASE_DIR / "cassettes" / "llm.yaml"
    assert cassette_file.exists(), f"Missing cassette file: {cassette_file}"


def test_golden_files_exist():
    """Golden directory has the expected output files."""
    golden_file = CASE_DIR / "golden" / "validate.json"
    assert golden_file.exists(), f"Missing golden file: {golden_file}"
