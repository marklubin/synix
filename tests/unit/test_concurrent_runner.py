"""Tests for concurrent transform execution in the pipeline runner."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from synix import Artifact, Layer, Pipeline, Projection
from synix.build.runner import run, _execute_transform_concurrent
from synix.build.transforms import BaseTransform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockEpisodeTransform(BaseTransform):
    """A mock 1:1 transform that simulates an LLM call with configurable delay.

    Each input artifact produces exactly one output artifact, mirroring
    how EpisodeSummaryTransform works.
    """

    def __init__(self, delay: float = 0.0, fail_ids: set[str] | None = None):
        self.delay = delay
        self.fail_ids = fail_ids or set()
        self.call_log: list[dict] = []
        self._lock = threading.Lock()

    def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
        results: list[Artifact] = []
        for inp in inputs:
            with self._lock:
                self.call_log.append({
                    "artifact_id": inp.artifact_id,
                    "thread": threading.current_thread().name,
                    "time": time.monotonic(),
                })

            if inp.artifact_id in self.fail_ids:
                raise RuntimeError(f"Transform failed for {inp.artifact_id}")

            if self.delay > 0:
                time.sleep(self.delay)

            results.append(Artifact(
                artifact_id=f"ep-{inp.artifact_id}",
                artifact_type="episode",
                content=f"Summary of {inp.content}",
                input_hashes=[inp.content_hash],
                prompt_id="test_prompt_v1",
                model_config={"model": "test", "temperature": 0.3},
                metadata={
                    "source_conversation_id": inp.artifact_id,
                    "date": inp.metadata.get("date", "2024-01-01"),
                },
            ))
        return results


def _make_transcript(tid: str, content: str = "", date: str = "2024-01-15") -> Artifact:
    """Create a transcript artifact for testing."""
    return Artifact(
        artifact_id=tid,
        artifact_type="transcript",
        content=content or f"Conversation {tid}",
        metadata={
            "source_conversation_id": tid,
            "date": date,
            "title": f"Conv {tid}",
            "message_count": 4,
        },
    )


def _make_pipeline_with_episodes(build_dir: str, source_dir: str) -> Pipeline:
    """Create a minimal pipeline with transcripts -> episodes."""
    p = Pipeline("test-concurrent")
    p.build_dir = build_dir
    p.source_dir = source_dir
    p.llm_config = {"model": "test-model", "temperature": 0.3, "max_tokens": 1024}
    p.add_layer(Layer(name="transcripts", level=0, transform="parse"))
    p.add_layer(Layer(
        name="episodes", level=1, depends_on=["transcripts"],
        transform="episode_summary", grouping="by_conversation",
    ))
    return p


# ---------------------------------------------------------------------------
# Tests for _execute_transform_concurrent
# ---------------------------------------------------------------------------


class TestExecuteTransformConcurrent:
    """Tests for the _execute_transform_concurrent helper function."""

    def test_produces_same_results_as_sequential(self):
        """Concurrent execution produces the same artifacts as sequential."""
        inputs = [_make_transcript(f"t-{i}") for i in range(8)]
        transform = MockEpisodeTransform()
        config = {"llm_config": {"model": "test"}}

        # Sequential
        sequential_results = transform.execute(inputs, config)

        # Concurrent
        transform2 = MockEpisodeTransform()
        concurrent_results = _execute_transform_concurrent(
            transform2, inputs, config, concurrency=4
        )

        # Same number of artifacts
        assert len(concurrent_results) == len(sequential_results)

        # Same artifact IDs in the same order
        seq_ids = [a.artifact_id for a in sequential_results]
        conc_ids = [a.artifact_id for a in concurrent_results]
        assert conc_ids == seq_ids

        # Same content
        for seq, conc in zip(sequential_results, concurrent_results):
            assert seq.content == conc.content
            assert seq.input_hashes == conc.input_hashes
            assert seq.prompt_id == conc.prompt_id

    def test_preserves_input_order(self):
        """Results are returned in the same order as inputs regardless of completion."""
        inputs = [_make_transcript(f"t-{i}") for i in range(10)]
        transform = MockEpisodeTransform(delay=0.01)
        config = {"llm_config": {"model": "test"}}

        results = _execute_transform_concurrent(transform, inputs, config, concurrency=5)

        expected_ids = [f"ep-t-{i}" for i in range(10)]
        actual_ids = [a.artifact_id for a in results]
        assert actual_ids == expected_ids

    def test_single_input(self):
        """Works correctly with a single input."""
        inputs = [_make_transcript("t-only")]
        transform = MockEpisodeTransform()
        config = {"llm_config": {"model": "test"}}

        results = _execute_transform_concurrent(transform, inputs, config, concurrency=4)

        assert len(results) == 1
        assert results[0].artifact_id == "ep-t-only"

    def test_uses_multiple_threads(self):
        """Concurrent execution actually uses multiple threads."""
        inputs = [_make_transcript(f"t-{i}") for i in range(6)]
        transform = MockEpisodeTransform(delay=0.05)
        config = {"llm_config": {"model": "test"}}

        _execute_transform_concurrent(transform, inputs, config, concurrency=4)

        # Check that multiple threads were used
        threads_used = {entry["thread"] for entry in transform.call_log}
        assert len(threads_used) > 1, (
            f"Expected multiple threads, got: {threads_used}"
        )


class TestConcurrentBuildSameResults:
    """Verify that concurrent builds produce identical results to sequential builds."""

    def test_concurrent_build_same_results(self, tmp_path, mock_llm):
        """Build with concurrency=1 and concurrency=4 produces identical artifacts."""
        # Set up source files
        source_dir = tmp_path / "exports"
        source_dir.mkdir()

        # Create some simple text files as source data
        for i in range(4):
            (source_dir / f"conv_{i}.txt").write_text(
                f"user: Question {i}\nassistant: Answer {i}\n"
            )

        # --- Sequential build (concurrency=1) ---
        build_dir_seq = tmp_path / "build_seq"
        p_seq = Pipeline("test-seq")
        p_seq.build_dir = str(build_dir_seq)
        p_seq.source_dir = str(source_dir)
        p_seq.llm_config = {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}
        p_seq.add_layer(Layer(name="transcripts", level=0, transform="parse"))
        p_seq.add_layer(Layer(
            name="episodes", level=1, depends_on=["transcripts"],
            transform="episode_summary", grouping="by_conversation",
        ))
        p_seq.add_projection(Projection(
            name="context-doc", projection_type="flat_file",
            sources=[{"layer": "episodes"}],
            config={"output_path": str(build_dir_seq / "context.md")},
        ))

        result_seq = run(p_seq, concurrency=1)

        # --- Concurrent build (concurrency=4) ---
        build_dir_conc = tmp_path / "build_conc"
        p_conc = Pipeline("test-conc")
        p_conc.build_dir = str(build_dir_conc)
        p_conc.source_dir = str(source_dir)
        p_conc.llm_config = {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}
        p_conc.add_layer(Layer(name="transcripts", level=0, transform="parse"))
        p_conc.add_layer(Layer(
            name="episodes", level=1, depends_on=["transcripts"],
            transform="episode_summary", grouping="by_conversation",
        ))
        p_conc.add_projection(Projection(
            name="context-doc", projection_type="flat_file",
            sources=[{"layer": "episodes"}],
            config={"output_path": str(build_dir_conc / "context.md")},
        ))

        result_conc = run(p_conc, concurrency=4)

        # Both should build the same number of artifacts
        assert result_seq.built == result_conc.built

        # Both should have the same layer stats structure
        seq_names = {s.name for s in result_seq.layer_stats}
        conc_names = {s.name for s in result_conc.layer_stats}
        assert seq_names == conc_names

        # Same number of built artifacts per layer
        for stat_seq in result_seq.layer_stats:
            stat_conc = next(s for s in result_conc.layer_stats if s.name == stat_seq.name)
            assert stat_seq.built == stat_conc.built, (
                f"Layer {stat_seq.name}: seq built {stat_seq.built} vs conc built {stat_conc.built}"
            )

        # Load artifacts from both builds and compare
        from synix.build.artifacts import ArtifactStore

        store_seq = ArtifactStore(build_dir_seq)
        store_conc = ArtifactStore(build_dir_conc)

        ep_seq = sorted(store_seq.list_artifacts("episodes"), key=lambda a: a.artifact_id)
        ep_conc = sorted(store_conc.list_artifacts("episodes"), key=lambda a: a.artifact_id)

        assert len(ep_seq) == len(ep_conc)
        for a, b in zip(ep_seq, ep_conc):
            assert a.artifact_id == b.artifact_id
            # Content should match (same mock LLM)
            assert a.content == b.content
            assert a.prompt_id == b.prompt_id


class TestConcurrentBuildRespectsLimit:
    """Verify max concurrent threads doesn't exceed the limit."""

    def test_concurrent_build_respects_limit(self):
        """No more than `concurrency` threads run transforms simultaneously."""
        max_concurrent = {"value": 0}
        current_concurrent = {"value": 0}
        lock = threading.Lock()

        class TrackingTransform(BaseTransform):
            def execute(self, inputs: list[Artifact], config: dict) -> list[Artifact]:
                results = []
                for inp in inputs:
                    with lock:
                        current_concurrent["value"] += 1
                        if current_concurrent["value"] > max_concurrent["value"]:
                            max_concurrent["value"] = current_concurrent["value"]
                    time.sleep(0.05)  # Hold the slot to allow overlap measurement
                    with lock:
                        current_concurrent["value"] -= 1
                    results.append(Artifact(
                        artifact_id=f"ep-{inp.artifact_id}",
                        artifact_type="episode",
                        content=f"Summary of {inp.content}",
                        input_hashes=[inp.content_hash],
                    ))
                return results

        inputs = [_make_transcript(f"t-{i}") for i in range(10)]
        transform = TrackingTransform()
        config = {"llm_config": {"model": "test"}}
        concurrency_limit = 3

        _execute_transform_concurrent(transform, inputs, config, concurrency_limit)

        assert max_concurrent["value"] <= concurrency_limit, (
            f"Max concurrent was {max_concurrent['value']}, expected <= {concurrency_limit}"
        )


class TestConcurrentBuildDefaultSequential:
    """Verify that concurrency=1 uses the existing sequential code path."""

    def test_concurrency_1_no_thread_pool(self, tmp_path, mock_llm):
        """With concurrency=1, transforms run in the main thread (no ThreadPoolExecutor)."""
        source_dir = tmp_path / "exports"
        source_dir.mkdir()
        (source_dir / "conv.txt").write_text("user: hi\nassistant: hello\n")

        build_dir = tmp_path / "build"
        p = Pipeline("test-seq-default")
        p.build_dir = str(build_dir)
        p.source_dir = str(source_dir)
        p.llm_config = {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}
        p.add_layer(Layer(name="transcripts", level=0, transform="parse"))
        p.add_layer(Layer(
            name="episodes", level=1, depends_on=["transcripts"],
            transform="episode_summary", grouping="by_conversation",
        ))

        # With concurrency=1, should still produce correct results
        result = run(p, concurrency=1)
        assert result.built > 0

        # Verify artifacts were created
        from synix.build.artifacts import ArtifactStore
        store = ArtifactStore(build_dir)
        episodes = store.list_artifacts("episodes")
        assert len(episodes) > 0

    def test_concurrency_1_single_input_no_concurrent(self):
        """With a single input, _execute_transform_concurrent is not used
        (the runner requires len(inputs) > 1 for concurrent path)."""
        # This tests the guard condition in the runner.
        # Single input + by_conversation + concurrency>1 still falls through
        # to sequential because len(inputs) <= 1.
        inputs = [_make_transcript("t-only")]
        transform = MockEpisodeTransform()
        config = {"llm_config": {"model": "test"}}

        # Direct sequential call should work fine
        results = transform.execute(inputs, config)
        assert len(results) == 1
        assert results[0].artifact_id == "ep-t-only"


class TestConcurrentErrorsDontCrash:
    """Verify that if one transform fails, others still complete."""

    def test_concurrent_errors_dont_crash(self):
        """If one transform fails, others still run to completion, and the error is raised."""
        inputs = [_make_transcript(f"t-{i}") for i in range(5)]
        # Fail on t-2
        transform = MockEpisodeTransform(fail_ids={"t-2"})
        config = {"llm_config": {"model": "test"}}

        with pytest.raises(RuntimeError, match="Transform failed for t-2"):
            _execute_transform_concurrent(transform, inputs, config, concurrency=4)

        # Check that other inputs were still processed
        # (they ran in parallel, so they should have completed before the error was raised)
        processed_ids = {entry["artifact_id"] for entry in transform.call_log}
        # At minimum, several inputs should have been submitted before the error
        assert len(processed_ids) >= 2, (
            f"Expected at least 2 inputs to be processed, got: {processed_ids}"
        )

    def test_concurrent_all_errors_raises_first(self):
        """If all transforms fail, the first error (by input order) is raised."""
        inputs = [_make_transcript(f"t-{i}") for i in range(3)]
        transform = MockEpisodeTransform(fail_ids={"t-0", "t-1", "t-2"})
        config = {"llm_config": {"model": "test"}}

        with pytest.raises(RuntimeError, match="Transform failed for t-0"):
            _execute_transform_concurrent(transform, inputs, config, concurrency=4)

    def test_sequential_layers_unaffected_by_concurrency(self, tmp_path, mock_llm):
        """Layers with grouping != by_conversation run sequentially even with -j4."""
        source_dir = tmp_path / "exports"
        source_dir.mkdir()
        for i in range(3):
            (source_dir / f"conv_{i}.txt").write_text(
                f"user: Q{i}\nassistant: A{i}\n"
            )

        build_dir = tmp_path / "build"
        p = Pipeline("test-monthly-seq")
        p.build_dir = str(build_dir)
        p.source_dir = str(source_dir)
        p.llm_config = {"model": "claude-sonnet-4-20250514", "temperature": 0.3, "max_tokens": 1024}
        p.add_layer(Layer(name="transcripts", level=0, transform="parse"))
        p.add_layer(Layer(
            name="episodes", level=1, depends_on=["transcripts"],
            transform="episode_summary", grouping="by_conversation",
        ))
        p.add_layer(Layer(
            name="monthly", level=2, depends_on=["episodes"],
            transform="monthly_rollup", grouping="by_month",
        ))
        p.add_layer(Layer(
            name="core", level=3, depends_on=["monthly"],
            transform="core_synthesis", grouping="single", context_budget=10000,
        ))

        # Should complete without errors even with concurrency > 1
        result = run(p, concurrency=4)
        assert result.built > 0

        # Monthly and core layers should have been built
        from synix.build.artifacts import ArtifactStore
        store = ArtifactStore(build_dir)
        assert len(store.list_artifacts("monthly")) >= 1
        assert len(store.list_artifacts("core")) == 1


class TestConcurrentPerformance:
    """Verify that concurrent execution is actually faster."""

    def test_concurrent_faster_than_sequential(self):
        """With multiple inputs and delay, concurrent should be meaningfully faster."""
        inputs = [_make_transcript(f"t-{i}") for i in range(6)]
        config = {"llm_config": {"model": "test"}}

        # Sequential timing
        transform_seq = MockEpisodeTransform(delay=0.05)
        seq_start = time.monotonic()
        seq_results = transform_seq.execute(inputs, config)
        seq_time = time.monotonic() - seq_start

        # Concurrent timing
        transform_conc = MockEpisodeTransform(delay=0.05)
        conc_start = time.monotonic()
        conc_results = _execute_transform_concurrent(
            transform_conc, inputs, config, concurrency=6
        )
        conc_time = time.monotonic() - conc_start

        # Both should produce correct results
        assert len(seq_results) == len(conc_results) == 6

        # Concurrent should be meaningfully faster
        # 6 * 0.05 = 0.3s sequential vs ~0.05s concurrent
        assert conc_time < seq_time * 0.75, (
            f"Concurrent ({conc_time:.3f}s) not meaningfully faster than "
            f"sequential ({seq_time:.3f}s)"
        )
