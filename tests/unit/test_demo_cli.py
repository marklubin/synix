"""Tests for the demo CLI commands."""

from __future__ import annotations

import os
from unittest.mock import patch

from click.testing import CliRunner

from synix.cli.main import is_demo_mode, main


class TestDemoNote:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["demo", "note", "--help"])
        assert result.exit_code == 0
        assert "narrative note" in result.output.lower() or "MESSAGE" in result.output

    def test_prints_message(self):
        runner = CliRunner()
        result = runner.invoke(main, ["demo", "note", "Step 1: Building..."])
        assert result.exit_code == 0
        assert "Step 1: Building..." in result.output


class TestDemoRun:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["demo", "run", "--help"])
        assert result.exit_code == 0
        assert "CASE_DIR" in result.output

    def test_missing_case_dir(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["demo", "run", str(tmp_path / "nonexistent")])
        assert result.exit_code != 0

    def test_missing_case_py(self, tmp_path):
        """Empty directory with no case.py."""
        runner = CliRunner()
        result = runner.invoke(main, ["demo", "run", str(tmp_path)])
        assert result.exit_code != 0
        assert "case.py" in result.output.lower()


class TestIsDemoMode:
    def test_off_by_default(self):
        env = dict(os.environ)
        env.pop("SYNIX_DEMO", None)
        with patch.dict(os.environ, env, clear=True):
            assert is_demo_mode() is False

    def test_on_when_set(self):
        with patch.dict(os.environ, {"SYNIX_DEMO": "1"}):
            assert is_demo_mode() is True

    def test_off_for_other_values(self):
        with patch.dict(os.environ, {"SYNIX_DEMO": "0"}):
            assert is_demo_mode() is False


class TestDemoGroup:
    def test_demo_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["demo", "--help"])
        assert result.exit_code == 0
        assert "note" in result.output
        assert "run" in result.output
