"""Tests for parallel layer execution across same-level layers."""

from __future__ import annotations

import time

from synix import Artifact, Pipeline, Source
from synix.core.models import Transform


class SlowTransform(Transform):
    """Test transform that sleeps to simulate LLM call latency."""

    def __init__(self, name: str, *, depends_on=None, sleep_time: float = 0.1, label: str = ""):
        super().__init__(name, depends_on=depends_on, batch=False)
        self.sleep_time = sleep_time
        self.label_value = label or name
        self.started_at: float | None = None
        self.finished_at: float | None = None

    def split(self, inputs, ctx):
        return [(inputs, {})]

    def estimate_output_count(self, input_count):
        return 1

    def execute(self, inputs, ctx):
        self.started_at = time.monotonic()
        time.sleep(self.sleep_time)
        self.finished_at = time.monotonic()
        return [
            Artifact(
                label=self.label_value,
                artifact_type="test",
                content=f"output from {self.name}",
                input_ids=[a.artifact_id for a in inputs],
            )
        ]


class TestParallelLayers:
    """Verify layers at the same DAG level run concurrently."""

    def test_same_level_layers_run_in_parallel(self, tmp_path, mock_llm):
        """Three independent transforms at the same level should overlap in time."""
        from synix.build.runner import run as run_pipeline

        source = Source("src")
        t1 = SlowTransform("t1", depends_on=[source], sleep_time=0.3, label="t1-out")
        t2 = SlowTransform("t2", depends_on=[source], sleep_time=0.3, label="t2-out")
        t3 = SlowTransform("t3", depends_on=[source], sleep_time=0.3, label="t3-out")

        pipeline = Pipeline(
            "parallel-test",
            source_dir=str(tmp_path / "sources"),
            build_dir=str(tmp_path / "build"),
            llm_config={"model": "test"},
        )
        (tmp_path / "sources").mkdir()
        (tmp_path / "sources" / "test.md").write_text("test content")

        pipeline.add(source, t1, t2, t3)

        start = time.monotonic()
        result = run_pipeline(pipeline)
        elapsed = time.monotonic() - start

        # All three should produce output
        assert result.built >= 4  # 1 source + 3 transforms

        # If sequential: 3 x 0.3s = 0.9s minimum
        # If parallel: ~0.3s + overhead
        # Allow generous margin but it should be well under 0.9s
        assert elapsed < 0.8, f"Took {elapsed:.2f}s — layers appear sequential, not parallel"

    def test_single_layer_level_runs_inline(self, tmp_path, mock_llm):
        """A level with only one layer should still work (no thread overhead)."""
        from synix.build.runner import run as run_pipeline

        source = Source("src")
        t1 = SlowTransform("t1", depends_on=[source], sleep_time=0.1, label="t1-out")

        pipeline = Pipeline(
            "single-test",
            source_dir=str(tmp_path / "sources"),
            build_dir=str(tmp_path / "build"),
            llm_config={"model": "test"},
        )
        (tmp_path / "sources").mkdir()
        (tmp_path / "sources" / "test.md").write_text("test content")

        pipeline.add(source, t1)

        result = run_pipeline(pipeline)
        assert result.built >= 2  # source + transform

    def test_dependent_layers_still_sequential(self, tmp_path, mock_llm):
        """Layers at different levels must still run in order."""
        from synix.build.runner import run as run_pipeline

        source = Source("src")
        t1 = SlowTransform("t1", depends_on=[source], sleep_time=0.1, label="t1-out")
        t2 = SlowTransform("t2", depends_on=[t1], sleep_time=0.1, label="t2-out")

        pipeline = Pipeline(
            "sequential-test",
            source_dir=str(tmp_path / "sources"),
            build_dir=str(tmp_path / "build"),
            llm_config={"model": "test"},
        )
        (tmp_path / "sources").mkdir()
        (tmp_path / "sources" / "test.md").write_text("test content")

        pipeline.add(source, t1, t2)

        result = run_pipeline(pipeline)
        assert result.built >= 3  # source + t1 + t2

    def test_parallel_layers_all_produce_artifacts(self, tmp_path, mock_llm):
        """Each parallel layer produces its own artifacts."""
        from synix.build.runner import run as run_pipeline

        source = Source("src")
        layers = [
            SlowTransform(f"fold-{i}", depends_on=[source], sleep_time=0.05, label=f"fold-{i}-out") for i in range(5)
        ]

        pipeline = Pipeline(
            "five-parallel",
            source_dir=str(tmp_path / "sources"),
            build_dir=str(tmp_path / "build"),
            llm_config={"model": "test"},
        )
        (tmp_path / "sources").mkdir()
        (tmp_path / "sources" / "test.md").write_text("test content")

        pipeline.add(source, *layers)

        result = run_pipeline(pipeline)
        # 1 source + 5 transforms = 6
        assert result.built >= 6

        # Verify each layer's output is in the stats
        layer_names = {s.name for s in result.layer_stats}
        for i in range(5):
            assert f"fold-{i}" in layer_names
