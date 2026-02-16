"""E2E test — verify `synix init` scaffolds projects that load without import errors.

For each available template, runs `synix init` in a temp directory, then uses
`synix plan` to load the generated pipeline module(s) and asserts no ImportError
or ModuleNotFoundError occurs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYNIX_BIN = Path(sys.executable).parent / "synix"

TEMPLATES = [
    "01-chatbot-export-synthesis",
    "02-tv-returns",
    "03-team-report",
    "04-sales-deal-room",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(args: list[str], cwd: Path, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a synix CLI command as subprocess."""
    return subprocess.run(
        [str(SYNIX_BIN)] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInitScaffoldLoads:
    """Verify every template scaffolds a project whose pipeline.py loads cleanly."""

    @pytest.mark.parametrize("template", TEMPLATES)
    def test_init_then_plan_loads(self, tmp_path: Path, template: str):
        """synix init --template <T> produces pipeline files that synix plan can load."""
        project_dir = tmp_path / "test-project"

        # Step 1: scaffold the project
        result = _run(["init", "test-project", "--template", template], cwd=tmp_path)
        assert result.returncode == 0, (
            f"synix init failed for template {template}:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert project_dir.is_dir(), f"Project directory not created for template {template}"

        # Step 2: find all pipeline*.py files in the scaffolded directory
        pipeline_files = sorted(project_dir.glob("pipeline*.py"))
        assert len(pipeline_files) > 0, f"No pipeline*.py found in scaffolded project for template {template}"

        # Step 3: run synix plan on each pipeline file to verify it loads
        for pf in pipeline_files:
            plan_result = _run(["plan", str(pf)], cwd=project_dir)

            # Check no import errors in stderr
            combined_output = (plan_result.stdout or "") + (plan_result.stderr or "")
            assert "ImportError" not in combined_output, (
                f"ImportError when loading {pf.name} from template {template}:\n{combined_output}"
            )
            assert "ModuleNotFoundError" not in combined_output, (
                f"ModuleNotFoundError when loading {pf.name} from template {template}:\n{combined_output}"
            )

            assert plan_result.returncode == 0, (
                f"synix plan failed for {pf.name} from template {template}:\n"
                f"stdout: {plan_result.stdout}\nstderr: {plan_result.stderr}"
            )

    def test_init_list_shows_all_templates(self, tmp_path: Path):
        """synix init --list returns all known templates."""
        result = _run(["init", "--list"], cwd=tmp_path)
        assert result.returncode == 0, f"synix init --list failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

        for template in TEMPLATES:
            assert template in result.stdout, f"Template {template} not found in --list output:\n{result.stdout}"
