"""Unit tests for database models."""

from uuid import uuid4

import pytest

from synix.db.artifacts import Record, RecordSource
from synix.db.control import PipelineState, Run, StepConfig


class TestRecord:
    """Tests for Record model."""

    def test_create_record(self):
        """Record can be created with required fields."""
        record = Record(
            id=str(uuid4()),
            content="Test content",
            content_fingerprint=Record.compute_fingerprint("Test content"),
            step_name="test_step",
            branch="main",
            materialization_key="test:key",
            run_id=str(uuid4()),
        )

        assert record.content == "Test content"
        assert record.step_name == "test_step"
        assert record.branch == "main"

    def test_compute_fingerprint(self):
        """Fingerprint is deterministic SHA-256."""
        fp1 = Record.compute_fingerprint("test")
        fp2 = Record.compute_fingerprint("test")
        fp3 = Record.compute_fingerprint("different")

        assert fp1 == fp2
        assert fp1 != fp3
        assert len(fp1) == 64  # SHA-256 hex

    def test_metadata_json_roundtrip(self):
        """Metadata can be set and retrieved."""
        record = Record(
            id=str(uuid4()),
            content="Test",
            content_fingerprint="abc",
            step_name="step",
            branch="main",
            materialization_key="key",
            run_id=str(uuid4()),
        )

        record.metadata_ = {"key": "value", "nested": {"a": 1}}
        assert record.metadata_ == {"key": "value", "nested": {"a": 1}}

    def test_audit_json_roundtrip(self):
        """Audit info can be set and retrieved."""
        record = Record(
            id=str(uuid4()),
            content="Test",
            content_fingerprint="abc",
            step_name="step",
            branch="main",
            materialization_key="key",
            run_id=str(uuid4()),
        )

        record.audit = {"model": "gpt-4", "tokens": 100}
        assert record.audit == {"model": "gpt-4", "tokens": 100}


class TestPipelineState:
    """Tests for PipelineState model."""

    def test_create_pipeline(self):
        """PipelineState can be created."""
        pipeline = PipelineState(
            name="test-pipeline",
            agent="test-agent",
        )

        assert pipeline.name == "test-pipeline"
        assert pipeline.agent == "test-agent"

    def test_definition_json_roundtrip(self):
        """Definition dict is serialized/deserialized correctly."""
        pipeline = PipelineState(
            name="test",
            agent="agent",
        )

        pipeline.definition = {"steps": ["a", "b"], "sources": ["source1"]}
        assert pipeline.definition == {"steps": ["a", "b"], "sources": ["source1"]}


class TestRun:
    """Tests for Run model."""

    def test_create_run(self):
        """Run can be created with default status."""
        run = Run(
            id=str(uuid4()),
            pipeline_name="test-pipeline",
            status="pending",  # Status doesn't have a Python default, only DB default
            branch="main",
        )

        assert run.status == "pending"
        assert run.branch == "main"

    def test_stats_json_roundtrip(self):
        """Stats dict is serialized/deserialized correctly."""
        run = Run(
            id=str(uuid4()),
            pipeline_name="test",
        )

        run.stats = {"input": 10, "output": 5, "skipped": 3}
        assert run.stats == {"input": 10, "output": 5, "skipped": 3}


class TestStepConfig:
    """Tests for StepConfig model."""

    def test_create_step_config(self):
        """StepConfig can be created."""
        config = StepConfig(
            id=str(uuid4()),
            pipeline_name="test-pipeline",
            step_name="summarize",
            step_type="transform",
            from_step="source",
            version_hash="abc123",
        )

        assert config.step_name == "summarize"
        assert config.step_type == "transform"
        assert config.from_step == "source"

    def test_config_json_roundtrip(self):
        """Config dict is serialized/deserialized correctly."""
        step = StepConfig(
            id=str(uuid4()),
            pipeline_name="test",
            step_name="step",
            step_type="transform",
            from_step=None,
            version_hash="abc",
        )

        step.config = {"model": "gpt-4", "temperature": 0.7}
        assert step.config == {"model": "gpt-4", "temperature": 0.7}
