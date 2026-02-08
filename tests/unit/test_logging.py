"""Unit tests for Synix structured logging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synix.core.logging import (
    RunLog,
    RunSummary,
    StepLog,
    SynixLogger,
    Verbosity,
)


class TestStepLog:
    def test_creation_defaults(self):
        """StepLog has sensible defaults."""
        step = StepLog(name="episodes")
        assert step.name == "episodes"
        assert step.llm_calls == 0
        assert step.cache_hits == 0
        assert step.rebuilt_ids == []
        assert step.cached_ids == []
        assert step.time_seconds == 0.0
        assert step.tokens_used == 0

    def test_to_dict(self):
        """StepLog serializes to dict."""
        step = StepLog(
            name="episodes",
            llm_calls=3,
            cache_hits=2,
            rebuilt_ids=["ep-001", "ep-002", "ep-003"],
            cached_ids=["ep-004", "ep-005"],
            time_seconds=1.5,
            tokens_used=900,
        )
        d = step.to_dict()
        assert d["name"] == "episodes"
        assert d["llm_calls"] == 3
        assert d["cache_hits"] == 2
        assert d["rebuilt_ids"] == ["ep-001", "ep-002", "ep-003"]
        assert d["cached_ids"] == ["ep-004", "ep-005"]
        assert d["time_seconds"] == 1.5
        assert d["tokens_used"] == 900


class TestRunLog:
    def test_creation_defaults(self):
        """RunLog starts empty."""
        log = RunLog(run_id="test-run")
        assert log.run_id == "test-run"
        assert log.steps == {}
        assert log.total_llm_calls == 0
        assert log.total_cache_hits == 0

    def test_get_or_create_step(self):
        """get_or_create_step creates on first call, returns same on second."""
        log = RunLog(run_id="test")
        step1 = log.get_or_create_step("episodes")
        step1.llm_calls = 5
        step2 = log.get_or_create_step("episodes")
        assert step2.llm_calls == 5
        assert step1 is step2

    def test_finalize_computes_totals(self):
        """finalize() aggregates step data into totals."""
        log = RunLog(run_id="test")
        ep = log.get_or_create_step("episodes")
        ep.llm_calls = 8
        ep.cache_hits = 0
        ep.tokens_used = 1200

        monthly = log.get_or_create_step("monthly")
        monthly.llm_calls = 2
        monthly.cache_hits = 0
        monthly.tokens_used = 800

        core = log.get_or_create_step("core")
        core.llm_calls = 1
        core.cache_hits = 0
        core.tokens_used = 500

        log.finalize()
        assert log.total_llm_calls == 11
        assert log.total_cache_hits == 0
        assert log.total_tokens == 2500
        assert log.total_cost_estimate > 0

    def test_to_dict_matches_assertion_format(self):
        """to_dict() produces the format expected by tests/helpers/assertions.py."""
        log = RunLog(run_id="20260207T120000Z")
        ep = log.get_or_create_step("episodes")
        ep.llm_calls = 8
        ep.cache_hits = 2
        ep.rebuilt_ids = ["ep-001", "ep-002"]
        ep.cached_ids = ["ep-003"]
        ep.tokens_used = 1200

        log.total_time = 5.0
        log.finalize()

        d = log.to_dict()

        # Top-level keys
        assert "steps" in d
        assert "total_llm_calls" in d
        assert "total_cache_hits" in d
        assert "total_time" in d
        assert "total_tokens" in d
        assert "total_cost_estimate" in d

        # Step-level keys (the format assertions.py expects)
        assert "episodes" in d["steps"]
        step_data = d["steps"]["episodes"]
        assert step_data["llm_calls"] == 8
        assert step_data["cache_hits"] == 2
        assert step_data["rebuilt_ids"] == ["ep-001", "ep-002"]
        assert step_data["cached_ids"] == ["ep-003"]

    def test_to_dict_works_with_assertion_helpers(self):
        """RunLog.to_dict() works with the assertion helper functions."""
        from tests.helpers.assertions import (
            count_cache_hits,
            count_llm_calls,
            count_llm_calls_for_step,
            count_cache_hits_for_step,
            assert_artifact_rebuilt,
            assert_artifact_cached,
        )

        log = RunLog(run_id="test")
        ep = log.get_or_create_step("episodes")
        ep.llm_calls = 3
        ep.cache_hits = 5
        ep.rebuilt_ids = ["ep-001", "ep-002", "ep-003"]
        ep.cached_ids = ["ep-004", "ep-005", "ep-006", "ep-007", "ep-008"]
        ep.tokens_used = 600

        core = log.get_or_create_step("core")
        core.llm_calls = 1
        core.cache_hits = 0
        core.rebuilt_ids = ["core-memory"]
        core.cached_ids = []
        core.tokens_used = 200

        log.finalize()
        d = log.to_dict()

        assert count_llm_calls(d) == 4
        assert count_cache_hits(d) == 5
        assert count_llm_calls_for_step(d, "episodes") == 3
        assert count_cache_hits_for_step(d, "core") == 0

        assert_artifact_rebuilt(d, "episodes", "ep-001")
        assert_artifact_cached(d, "episodes", "ep-004")
        assert_artifact_rebuilt(d, "core", "core-memory")


class TestSynixLogger:
    def test_logger_creation_no_build_dir(self):
        """Logger works without a build_dir (no file logging)."""
        logger = SynixLogger(verbosity=Verbosity.DEFAULT)
        assert logger.run_log is not None
        assert logger._log_file is None
        logger.close()

    def test_logger_creates_log_file(self, tmp_path):
        """Logger creates JSONL log file when build_dir is provided."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        logger = SynixLogger(verbosity=Verbosity.DEFAULT, build_dir=build_dir)
        assert logger._log_file is not None
        assert logger._log_path is not None
        assert (build_dir / "logs").exists()

        logger.run_start("test-pipeline", 4)
        logger.close()

        # Verify log file was created
        log_files = list((build_dir / "logs").glob("*.jsonl"))
        assert len(log_files) == 1

    def test_jsonl_format(self, tmp_path):
        """Log entries are valid JSONL."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        logger = SynixLogger(verbosity=Verbosity.DEFAULT, build_dir=build_dir)
        logger.run_start("test", 2)
        logger.layer_start("episodes", 1)
        logger.artifact_built("episodes", "ep-001")
        logger.artifact_cached("episodes", "ep-002")
        logger.layer_finish("episodes", 1, 1)
        logger.run_finish(1.0)

        log_path = logger._log_path
        assert log_path is not None

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) >= 5  # run_start, layer_start, artifact_built, artifact_cached, layer_finish, run_finish

        for line in lines:
            entry = json.loads(line)
            assert "timestamp" in entry
            assert "event" in entry

    def test_jsonl_event_types(self, tmp_path):
        """All event types are correctly logged."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        logger = SynixLogger(verbosity=Verbosity.DEFAULT, build_dir=build_dir)
        logger.run_start("test", 2)
        logger.layer_start("episodes", 1)
        start = logger.llm_call_start("episodes", "ep-001", "claude-sonnet")
        logger.llm_call_finish("episodes", "ep-001", start, input_tokens=100, output_tokens=50)
        logger.artifact_built("episodes", "ep-001")
        logger.artifact_cached("episodes", "ep-002")
        logger.layer_finish("episodes", 1, 1)
        logger.run_finish(2.5)

        lines = logger._log_path.read_text().strip().split("\n")
        events = [json.loads(line)["event"] for line in lines]

        assert "run_start" in events
        assert "layer_start" in events
        assert "llm_call_start" in events
        assert "llm_call_finish" in events
        assert "artifact_built" in events
        assert "artifact_cached" in events
        assert "layer_finish" in events
        assert "run_finish" in events

    def test_layer_tracking(self):
        """Layer events update the RunLog step data."""
        logger = SynixLogger(verbosity=Verbosity.DEFAULT)
        logger.layer_start("episodes", 1)
        logger.artifact_built("episodes", "ep-001")
        logger.artifact_built("episodes", "ep-002")
        logger.artifact_cached("episodes", "ep-003")
        logger.layer_finish("episodes", 2, 1)

        step = logger.run_log.steps["episodes"]
        assert step.rebuilt_ids == ["ep-001", "ep-002"]
        assert step.cached_ids == ["ep-003"]
        assert step.cache_hits == 1
        assert step.time_seconds > 0

        logger.close()

    def test_llm_call_tracking(self):
        """LLM call events update step statistics."""
        logger = SynixLogger(verbosity=Verbosity.DEFAULT)
        logger.layer_start("episodes", 1)

        start = logger.llm_call_start("episodes", "ep-001", "claude-sonnet")
        logger.llm_call_finish("episodes", "ep-001", start, input_tokens=100, output_tokens=50)

        start2 = logger.llm_call_start("episodes", "ep-002", "claude-sonnet")
        logger.llm_call_finish("episodes", "ep-002", start2, input_tokens=80, output_tokens=40)

        step = logger.run_log.steps["episodes"]
        assert step.llm_calls == 2
        assert step.tokens_used == 270  # 100+50 + 80+40

        logger.close()

    def test_get_summary(self):
        """get_summary() returns correct RunSummary."""
        logger = SynixLogger(verbosity=Verbosity.DEFAULT)
        logger.layer_start("episodes", 1)
        logger.artifact_built("episodes", "ep-001")
        logger.artifact_cached("episodes", "ep-002")
        logger.artifact_cached("episodes", "ep-003")
        logger.layer_finish("episodes", 1, 2)
        logger.run_finish(3.0)

        summary = logger.get_summary()
        assert summary.total_time == 3.0
        assert summary.built == 1
        assert summary.cached == 2
        assert summary.cache_hit_rate == pytest.approx(2 / 3)

        logger.close()

    def test_run_finish_finalizes(self):
        """run_finish() calls finalize() on the RunLog."""
        logger = SynixLogger(verbosity=Verbosity.DEFAULT)
        logger.layer_start("episodes", 1)
        start = logger.llm_call_start("episodes", "ep-001", "model")
        logger.llm_call_finish("episodes", "ep-001", start, input_tokens=100, output_tokens=50)
        logger.artifact_built("episodes", "ep-001")
        logger.layer_finish("episodes", 1, 0)
        logger.run_finish(2.0)

        assert logger.run_log.total_llm_calls == 1
        assert logger.run_log.total_tokens == 150
        assert logger.run_log.total_time == 2.0

        logger.close()

    def test_close_is_idempotent(self, tmp_path):
        """Calling close() multiple times is safe."""
        build_dir = tmp_path / "build"
        build_dir.mkdir()

        logger = SynixLogger(verbosity=Verbosity.DEFAULT, build_dir=build_dir)
        logger.run_start("test", 1)
        logger.close()
        logger.close()  # should not raise


class TestVerbosity:
    def test_verbosity_enum_values(self):
        """Verbosity levels have correct integer values."""
        assert Verbosity.DEFAULT == 0
        assert Verbosity.VERBOSE == 1
        assert Verbosity.DEBUG == 2

    def test_verbosity_from_int(self):
        """Verbosity can be created from int."""
        assert Verbosity(0) == Verbosity.DEFAULT
        assert Verbosity(1) == Verbosity.VERBOSE
        assert Verbosity(2) == Verbosity.DEBUG


class TestRunLogWithRunner:
    """Tests that verify RunLog integrates correctly with the runner's RunResult."""

    @pytest.fixture
    def pipeline_and_sources(self, tmp_path, mock_llm):
        """Create a pipeline and source dir that don't conflict."""
        import shutil
        from synix import Layer, Pipeline, Projection

        fixtures_dir = Path(__file__).parent.parent / "synix" / "fixtures"
        source_dir = tmp_path / "src_exports"
        source_dir.mkdir()
        shutil.copy(fixtures_dir / "chatgpt_export.json", source_dir / "chatgpt_export.json")
        shutil.copy(fixtures_dir / "claude_export.json", source_dir / "claude_export.json")

        build_dir = tmp_path / "build"

        pipeline = Pipeline("test-logging")
        pipeline.source_dir = str(source_dir)
        pipeline.build_dir = str(build_dir)
        pipeline.llm_config = {
            "model": "claude-sonnet-4-20250514",
            "temperature": 0.3,
            "max_tokens": 1024,
        }
        pipeline.add_layer(Layer(name="transcripts", level=0, transform="parse"))
        pipeline.add_layer(Layer(
            name="episodes", level=1, depends_on=["transcripts"],
            transform="episode_summary", grouping="by_conversation",
        ))
        pipeline.add_layer(Layer(
            name="monthly", level=2, depends_on=["episodes"],
            transform="monthly_rollup", grouping="by_month",
        ))
        pipeline.add_layer(Layer(
            name="core", level=3, depends_on=["monthly"],
            transform="core_synthesis", grouping="single", context_budget=10000,
        ))
        pipeline.add_projection(Projection(
            name="memory-index",
            projection_type="search_index",
            sources=[
                {"layer": "episodes", "search": ["fulltext"]},
                {"layer": "monthly", "search": ["fulltext"]},
                {"layer": "core", "search": ["fulltext"]},
            ],
        ))
        pipeline.add_projection(Projection(
            name="context-doc",
            projection_type="flat_file",
            sources=[{"layer": "core"}],
            config={"output_path": str(build_dir / "context.md")},
        ))

        return pipeline, source_dir, build_dir

    def test_run_result_has_run_log(self, pipeline_and_sources):
        """RunResult from runner includes run_log dict."""
        from synix.build.runner import run

        pipeline, source_dir, build_dir = pipeline_and_sources
        result = run(pipeline, source_dir=str(source_dir))

        assert result.run_log is not None
        assert isinstance(result.run_log, dict)
        assert "steps" in result.run_log
        assert "total_llm_calls" in result.run_log

    def test_run_log_has_all_layers(self, pipeline_and_sources):
        """RunLog tracks all pipeline layers."""
        from synix.build.runner import run

        pipeline, source_dir, build_dir = pipeline_and_sources
        result = run(pipeline, source_dir=str(source_dir))

        steps = result.run_log["steps"]
        assert "transcripts" in steps
        assert "episodes" in steps
        assert "monthly" in steps
        assert "core" in steps

    def test_run_log_llm_calls_nonzero_for_llm_layers(self, pipeline_and_sources):
        """LLM layers should have llm_calls > 0 on first run."""
        from synix.build.runner import run

        pipeline, source_dir, build_dir = pipeline_and_sources
        result = run(pipeline, source_dir=str(source_dir))

        # Episodes are LLM-powered
        episodes = result.run_log["steps"]["episodes"]
        assert episodes["llm_calls"] > 0

        # Transcripts are parse-only, no LLM
        transcripts = result.run_log["steps"]["transcripts"]
        assert transcripts["llm_calls"] == 0

    def test_run_log_cached_on_second_run(self, pipeline_and_sources):
        """Second run should show cache hits and zero LLM calls."""
        from synix.build.runner import run

        pipeline, source_dir, build_dir = pipeline_and_sources
        run(pipeline, source_dir=str(source_dir))

        result2 = run(pipeline, source_dir=str(source_dir))
        assert result2.run_log["total_llm_calls"] == 0
        assert result2.run_log["total_cache_hits"] > 0

    def test_run_log_creates_jsonl_file(self, pipeline_and_sources):
        """Pipeline run creates a JSONL log file in build_dir/logs/."""
        from synix.build.runner import run

        pipeline, source_dir, build_dir = pipeline_and_sources
        result = run(pipeline, source_dir=str(source_dir))

        logs_dir = build_dir / "logs"
        assert logs_dir.exists()

        log_files = list(logs_dir.glob("*.jsonl"))
        assert len(log_files) >= 1

        # Verify JSONL is parseable
        content = log_files[0].read_text().strip()
        for line in content.split("\n"):
            entry = json.loads(line)
            assert "event" in entry
            assert "timestamp" in entry

    def test_run_log_rebuilt_ids_match_built_count(self, pipeline_and_sources):
        """rebuilt_ids length should match the layer's built count."""
        from synix.build.runner import run

        pipeline, source_dir, build_dir = pipeline_and_sources
        result = run(pipeline, source_dir=str(source_dir))

        for layer_stats in result.layer_stats:
            step = result.run_log["steps"].get(layer_stats.name)
            if step:
                assert len(step["rebuilt_ids"]) == layer_stats.built, (
                    f"Layer {layer_stats.name}: rebuilt_ids count "
                    f"({len(step['rebuilt_ids'])}) != built count ({layer_stats.built})"
                )

    def test_run_log_verbosity_passed_through(self, pipeline_and_sources):
        """Verbosity parameter is accepted by run()."""
        from synix.build.runner import run

        pipeline, source_dir, build_dir = pipeline_and_sources
        # Should not raise
        result = run(pipeline, source_dir=str(source_dir), verbosity=1)
        assert result.run_log is not None
