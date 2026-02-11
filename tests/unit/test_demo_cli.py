"""Tests for the demo CLI commands."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from synix.cli.demo_commands import _normalize_output
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


class TestNormalizeOutput:
    """Tests for _normalize_output."""

    def test_normalize_timing(self):
        text = "Build completed in 1.2s"
        result = _normalize_output(text, Path("/tmp/case"))
        assert result == "Build completed in <TIME>"

    def test_normalize_timing_fractional(self):
        text = "Step took 0.05s, total 12.34s"
        result = _normalize_output(text, Path("/tmp/case"))
        assert result == "Step took <TIME>, total <TIME>"

    def test_normalize_paths(self):
        case_path = Path("/home/user/synix/examples/01-chatbot")
        text = f"Loading pipeline from {case_path}/pipeline.py"
        result = _normalize_output(text, case_path)
        assert result == "Loading pipeline from <CASE_DIR>/pipeline.py"

    def test_normalize_llm_stats(self):
        text = "LLM calls: 5, Tokens: 12,345, Est. cost: $0.42"
        result = _normalize_output(text, Path("/tmp/case"))
        assert result == "LLM calls: <N>, Tokens: <N>, Est. cost: $<COST>"

    def test_normalize_token_cost_fragment(self):
        text = "Used 1,234 tokens, $0.05 for this step"
        result = _normalize_output(text, Path("/tmp/case"))
        assert result == "Used <N> tokens, $<COST> for this step"

    def test_normalize_verify_counts(self):
        text = "│ manifest_valid      │  PASS  │ Manifest valid with 27 artifacts     │"
        result = _normalize_output(text, Path("/tmp/case"))
        assert "<N> artifacts" in result
        assert "27" not in result

    def test_normalize_plan_tree_stats(self):
        cached = "├── bios  source:parse  3 cached"
        fresh = "├── bios  source:parse  3 new"
        result_cached = _normalize_output(cached, Path("/tmp/case"))
        result_fresh = _normalize_output(fresh, Path("/tmp/case"))
        assert result_cached == result_fresh
        assert "source:parse  <STATS>" in result_cached

    def test_normalize_build_status(self):
        built = "  ✓ bios (level 0)  3 built  1.2s"
        cached = "  ✓ bios (level 0)  3 cached  0.0s"
        result_built = _normalize_output(built, Path("/tmp/case"))
        result_cached = _normalize_output(cached, Path("/tmp/case"))
        assert result_built == result_cached
        assert "<N> <STATUS>" in result_built

    def test_normalize_materialization(self):
        text1 = "    └─ search  materialized"
        text2 = "    └─ search  materializing..."
        text3 = "    └─ search  cached"
        result1 = _normalize_output(text1, Path("/tmp/case"))
        result2 = _normalize_output(text2, Path("/tmp/case"))
        result3 = _normalize_output(text3, Path("/tmp/case"))
        assert result1 == result2 == result3
        assert "<MATERIALIZED>" in result1

    def test_normalize_search_projection_status(self):
        cached = "    └── → search  synix_search_index (sqlite)  cached  9 indexed"
        fresh = "    └── → search  synix_search_index (sqlite)  new  14 indexed"
        result_cached = _normalize_output(cached, Path("/tmp/case"))
        result_fresh = _normalize_output(fresh, Path("/tmp/case"))
        assert result_cached == result_fresh
        assert "<MATERIALIZED>  <N> indexed" in result_cached

    def test_normalize_standalone_built(self):
        text = "│   → search    │ search_index │       │        │       built │"
        result = _normalize_output(text, Path("/tmp/case"))
        assert "<MATERIALIZED>" in result
        assert "built" not in result

    def test_normalize_cassette_miss_key(self):
        text = "Cassette miss for key 8c0f2d755f48... (prompt: You are)"
        result = _normalize_output(text, Path("/tmp/case"))
        assert result == "Cassette miss for key <HASH>... (prompt: You are)"

    def test_normalize_passthrough(self):
        text = "No dynamic content here\nJust plain text"
        result = _normalize_output(text, Path("/tmp/case"))
        assert result == "No dynamic content here\nJust plain text"

    def test_normalize_trailing_whitespace(self):
        text = "line one   \nline two\t\nline three"
        result = _normalize_output(text, Path("/tmp/case"))
        assert result == "line one\nline two\nline three"


class TestDemoGroup:
    def test_demo_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["demo", "--help"])
        assert result.exit_code == 0
        assert "note" in result.output
        assert "run" in result.output
