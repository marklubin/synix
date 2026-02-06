"""End-to-end tests for artifact publishing pipelines."""

import json
from pathlib import Path

import pytest


class TestArtifactPipeline:
    """E2E tests for pipelines with artifact publishing."""

    def test_artifact_publishes_to_file(
        self, initialized_db, claude_export_file, mock_llm, tmp_path
    ):
        """Artifact step publishes records to file."""
        from synix import Pipeline

        def summarize(record) -> str:
            return f"Summarize: {record.content[:50]}"

        output_path = tmp_path / "output" / "summaries.md"

        pipeline = Pipeline("artifact-test", agent="test", settings=initialized_db)
        pipeline.llm = mock_llm
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.transform("summaries", from_="claude", prompt=summarize)
        pipeline.artifact("report", from_="summaries", surface=f"file://{output_path}")

        result = pipeline.run()

        assert result.status == "completed"
        assert output_path.exists()
        content = output_path.read_text()
        # Should contain markdown sections
        assert "##" in content

    def test_artifact_publishes_json_format(
        self, initialized_db, claude_export_file, mock_llm, tmp_path
    ):
        """Artifact step can publish as JSON."""
        from synix import Pipeline

        output_path = tmp_path / "records.json"

        pipeline = Pipeline("json-artifact-test", agent="test", settings=initialized_db)
        pipeline.llm = mock_llm
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.artifact("data", from_="claude", surface=f"file://{output_path}")

        result = pipeline.run()

        assert result.status == "completed"
        assert output_path.exists()

        data = json.loads(output_path.read_text())
        assert isinstance(data, list)
        assert len(data) == 3  # 3 conversations

    def test_artifact_publishes_text_format(
        self, initialized_db, claude_export_file, mock_llm, tmp_path
    ):
        """Artifact step can publish as plain text."""
        from synix import Pipeline

        output_path = tmp_path / "output.txt"

        pipeline = Pipeline("text-artifact-test", agent="test", settings=initialized_db)
        pipeline.llm = mock_llm
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.artifact("output", from_="claude", surface=f"file://{output_path}")

        result = pipeline.run()

        assert result.status == "completed"
        assert output_path.exists()
        content = output_path.read_text()
        # Should contain raw content
        assert len(content) > 0

    def test_artifact_from_multiple_steps(
        self, initialized_db, claude_export_file, chatgpt_export_file, mock_llm, tmp_path
    ):
        """Artifact can publish from multiple upstream steps."""
        from synix import Pipeline

        output_path = tmp_path / "all.md"

        pipeline = Pipeline("multi-artifact-test", agent="test", settings=initialized_db)
        pipeline.llm = mock_llm
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.source("chatgpt", file=str(chatgpt_export_file), format="chatgpt-export")
        pipeline.artifact("all", from_=["claude", "chatgpt"], surface=f"file://{output_path}")

        result = pipeline.run()

        assert result.status == "completed"
        assert output_path.exists()
        content = output_path.read_text()
        # Should contain content from both sources
        assert len(content) > 100

    def test_artifact_with_template_path(
        self, initialized_db, claude_export_file, mock_llm, tmp_path
    ):
        """Artifact path can use template variables."""
        from synix import Pipeline

        def summarize(record) -> str:
            return "summary"

        # Use template with step_name
        output_template = tmp_path / "{step_name}_output.md"

        pipeline = Pipeline("template-test", agent="test", settings=initialized_db)
        pipeline.llm = mock_llm
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.transform("summaries", from_="claude", prompt=summarize)
        pipeline.artifact("report", from_="summaries", surface=f"file://{output_template}")

        result = pipeline.run()

        assert result.status == "completed"
        # File should be created with expanded step_name
        expected_path = tmp_path / "summaries_output.md"
        assert expected_path.exists()

    def test_full_pipeline_with_fold_merge_artifact(
        self, initialized_db, claude_export_file, chatgpt_export_file, mock_llm, tmp_path
    ):
        """Full pipeline with all new step types and artifact output."""
        from synix import Pipeline

        def summarize(record) -> str:
            return f"Summary of: {record.content[:30]}"

        def combine(sources: dict) -> str:
            parts = []
            for step, records in sources.items():
                parts.append(f"From {step}: {len(records)} items")
            return "\n".join(parts)

        def evolve(state: str, record) -> str:
            return f"{state}\n- {record.content[:50]}"

        output_path = tmp_path / "narrative.md"

        pipeline = Pipeline("full-pipeline", agent="test", settings=initialized_db)
        pipeline.llm = mock_llm

        # Sources
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.source("chatgpt", file=str(chatgpt_export_file), format="chatgpt-export")

        # Parallel transforms
        pipeline.transform("claude-sum", from_="claude", prompt=summarize)
        pipeline.transform("gpt-sum", from_="chatgpt", prompt=summarize)

        # Merge
        pipeline.merge("all-summaries", sources=["claude-sum", "gpt-sum"], prompt=combine)

        # Fold
        pipeline.fold("narrative", from_="all-summaries", prompt=evolve, initial_state="# My Story")

        # Artifact
        pipeline.artifact("report", from_="narrative", surface=f"file://{output_path}")

        result = pipeline.run()

        assert result.status == "completed"
        assert output_path.exists()

        # Check output content
        content = output_path.read_text()
        assert len(content) > 0

    def test_artifact_creates_directories(
        self, initialized_db, claude_export_file, mock_llm, tmp_path
    ):
        """Artifact step creates parent directories if needed."""
        from synix import Pipeline

        output_path = tmp_path / "deep" / "nested" / "path" / "report.md"

        pipeline = Pipeline("dir-test", agent="test", settings=initialized_db)
        pipeline.llm = mock_llm
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.artifact("report", from_="claude", surface=f"file://{output_path}")

        result = pipeline.run()

        assert result.status == "completed"
        assert output_path.exists()

    def test_artifact_does_not_affect_step_execution(
        self, initialized_db, claude_export_file, mock_llm, tmp_path
    ):
        """Artifact definition doesn't change step execution order."""
        from synix import Pipeline

        def summarize(record) -> str:
            return "summary"

        output_path = tmp_path / "report.md"

        pipeline = Pipeline("artifact-order-test", agent="test", settings=initialized_db)
        pipeline.llm = mock_llm
        pipeline.source("claude", file=str(claude_export_file), format="claude-export")
        pipeline.transform("summaries", from_="claude", prompt=summarize)
        pipeline.artifact("report", from_="summaries", surface=f"file://{output_path}")

        # Plan should show normal step order
        plan = pipeline.plan()
        step_names = [s["name"] for s in plan.steps]
        assert step_names == ["claude", "summaries"]  # Artifact not in step order

        result = pipeline.run()
        assert result.status == "completed"
