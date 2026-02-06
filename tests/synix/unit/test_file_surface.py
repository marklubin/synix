"""Unit tests for File surface."""

import json
from pathlib import Path
from uuid import uuid4

import pytest


class TestFileSurface:
    """Tests for FileSurface."""

    def test_create_file_surface(self):
        """FileSurface can be created."""
        from synix.surfaces.file import FileSurface

        surface = FileSurface(
            name="report",
            path_template="output/report.md",
            format="markdown",
        )

        assert surface.name == "report"
        assert surface.path_template == "output/report.md"
        assert surface.format == "markdown"

    def test_name_defaults_from_path(self):
        """Name defaults to filename stem if not provided."""
        from synix.surfaces.file import FileSurface

        surface = FileSurface(
            path_template="output/my_report.md",
            format="markdown",
        )

        assert surface.name == "my_report"

    def test_publish_creates_file(self, tmp_path):
        """Publish creates the file."""
        from synix.db.artifacts import Record
        from synix.surfaces.file import FileSurface

        output_path = tmp_path / "output" / "report.md"

        surface = FileSurface(
            name="report",
            path_template=str(output_path),
            format="markdown",
        )

        record = Record(
            id=str(uuid4()),
            content="Hello world",
            content_fingerprint="fp1",
            step_name="summaries",
            branch="main",
            materialization_key="key1",
            run_id="run",
        )
        record.metadata_ = {}

        result = surface.publish([record], "run-123")

        assert result.success is True
        assert output_path.exists()
        assert "Hello world" in output_path.read_text()

    def test_publish_creates_parent_directories(self, tmp_path):
        """Publish creates parent directories if needed."""
        from synix.db.artifacts import Record
        from synix.surfaces.file import FileSurface

        output_path = tmp_path / "deep" / "nested" / "path" / "report.md"

        surface = FileSurface(
            name="report",
            path_template=str(output_path),
            format="text",
        )

        record = Record(
            id=str(uuid4()),
            content="Content",
            content_fingerprint="fp1",
            step_name="test",
            branch="main",
            materialization_key="key1",
            run_id="run",
        )

        result = surface.publish([record], "run-123")

        assert result.success is True
        assert output_path.exists()

    def test_publish_markdown_format(self, tmp_path):
        """Publish formats as markdown with sections."""
        from synix.db.artifacts import Record
        from synix.surfaces.file import FileSurface

        output_path = tmp_path / "report.md"

        surface = FileSurface(
            name="report",
            path_template=str(output_path),
            format="markdown",
        )

        records = []
        for i in range(2):
            record = Record(
                id=str(uuid4()),
                content=f"Content {i}",
                content_fingerprint=f"fp{i}",
                step_name=f"step_{i}",
                branch="main",
                materialization_key=f"key{i}",
                run_id="run",
            )
            record.metadata_ = {"meta.source.title": f"Title {i}"}
            records.append(record)

        result = surface.publish(records, "run-123")

        assert result.success is True
        content = output_path.read_text()
        assert "## Title 0" in content
        assert "## Title 1" in content
        assert "Content 0" in content
        assert "Content 1" in content
        assert "---" in content  # Separator

    def test_publish_json_format(self, tmp_path):
        """Publish formats as JSON array."""
        from synix.db.artifacts import Record
        from synix.surfaces.file import FileSurface

        output_path = tmp_path / "records.json"

        surface = FileSurface(
            name="records",
            path_template=str(output_path),
            format="json",
        )

        records = []
        for i in range(2):
            record = Record(
                id=str(uuid4()),
                content=f"Content {i}",
                content_fingerprint=f"fp{i}",
                step_name=f"step_{i}",
                branch="main",
                materialization_key=f"key{i}",
                run_id="run",
            )
            record.metadata_ = {"key": f"value_{i}"}
            records.append(record)

        result = surface.publish(records, "run-123")

        assert result.success is True
        data = json.loads(output_path.read_text())
        assert len(data) == 2
        assert data[0]["step_name"] == "step_0"
        assert data[0]["content"] == "Content 0"
        assert data[1]["metadata"]["key"] == "value_1"

    def test_publish_text_format(self, tmp_path):
        """Publish formats as plain text."""
        from synix.db.artifacts import Record
        from synix.surfaces.file import FileSurface

        output_path = tmp_path / "output.txt"

        surface = FileSurface(
            name="output",
            path_template=str(output_path),
            format="text",
        )

        records = []
        for i in range(2):
            record = Record(
                id=str(uuid4()),
                content=f"Line {i}",
                content_fingerprint=f"fp{i}",
                step_name=f"step_{i}",
                branch="main",
                materialization_key=f"key{i}",
                run_id="run",
            )
            records.append(record)

        result = surface.publish(records, "run-123")

        assert result.success is True
        content = output_path.read_text()
        assert content == "Line 0\n\nLine 1"

    def test_template_expansion_step_name(self, tmp_path):
        """Template expands {step_name} variable."""
        from synix.db.artifacts import Record
        from synix.surfaces.file import FileSurface

        surface = FileSurface(
            name="output",
            path_template=str(tmp_path / "{step_name}.md"),
            format="markdown",
        )

        record = Record(
            id=str(uuid4()),
            content="Content",
            content_fingerprint="fp1",
            step_name="my_summaries",
            branch="main",
            materialization_key="key1",
            run_id="run",
        )
        record.metadata_ = {}

        result = surface.publish([record], "run-123")

        assert result.success is True
        expected_path = tmp_path / "my_summaries.md"
        assert expected_path.exists()

    def test_template_expansion_run_id(self, tmp_path):
        """Template expands {run_id} variable (truncated to 8 chars)."""
        from synix.db.artifacts import Record
        from synix.surfaces.file import FileSurface

        surface = FileSurface(
            name="output",
            path_template=str(tmp_path / "run_{run_id}.txt"),
            format="text",
        )

        record = Record(
            id=str(uuid4()),
            content="Content",
            content_fingerprint="fp1",
            step_name="test",
            branch="main",
            materialization_key="key1",
            run_id="run",
        )

        result = surface.publish([record], "abc12345-6789-0000-1111")

        assert result.success is True
        expected_path = tmp_path / "run_abc12345.txt"
        assert expected_path.exists()

    def test_template_expansion_date(self, tmp_path):
        """Template expands {date} variable."""
        from synix.db.artifacts import Record
        from synix.surfaces.file import FileSurface

        surface = FileSurface(
            name="output",
            path_template=str(tmp_path / "report_{date}.txt"),
            format="text",
        )

        record = Record(
            id=str(uuid4()),
            content="Content",
            content_fingerprint="fp1",
            step_name="test",
            branch="main",
            materialization_key="key1",
            run_id="run",
        )

        result = surface.publish([record], "run-123")

        assert result.success is True
        # Check that a file with date pattern exists
        files = list(tmp_path.glob("report_*.txt"))
        assert len(files) == 1
        # Date should be in YYYY-MM-DD format
        filename = files[0].name
        assert filename.startswith("report_")
        assert len(filename) == len("report_YYYY-MM-DD.txt")

    def test_publish_empty_records(self, tmp_path):
        """Publish with empty records returns early."""
        from synix.surfaces.file import FileSurface

        output_path = tmp_path / "report.md"

        surface = FileSurface(
            name="report",
            path_template=str(output_path),
            format="markdown",
        )

        result = surface.publish([], "run-123")

        assert result.success is True
        assert result.count == 0
        assert result.error == "No records to publish"
        assert not output_path.exists()

    def test_publish_result_location(self, tmp_path):
        """Publish result includes absolute location."""
        from synix.db.artifacts import Record
        from synix.surfaces.file import FileSurface

        output_path = tmp_path / "report.md"

        surface = FileSurface(
            name="report",
            path_template=str(output_path),
            format="markdown",
        )

        record = Record(
            id=str(uuid4()),
            content="Content",
            content_fingerprint="fp1",
            step_name="test",
            branch="main",
            materialization_key="key1",
            run_id="run",
        )
        record.metadata_ = {}

        result = surface.publish([record], "run-123")

        assert result.success is True
        assert result.location == str(output_path.absolute())
        assert result.count == 1


class TestParseFileSurface:
    """Tests for parse_file_surface helper."""

    def test_parse_markdown_uri(self):
        """Parse file:// URI with .md extension."""
        from synix.surfaces.file import parse_file_surface

        surface = parse_file_surface("report", "file://output/report.md")

        assert surface.name == "report"
        assert surface.path_template == "output/report.md"
        assert surface.format == "markdown"

    def test_parse_json_uri(self):
        """Parse file:// URI with .json extension."""
        from synix.surfaces.file import parse_file_surface

        surface = parse_file_surface("data", "file://output/data.json")

        assert surface.format == "json"

    def test_parse_text_uri(self):
        """Parse file:// URI with other extension."""
        from synix.surfaces.file import parse_file_surface

        surface = parse_file_surface("log", "file://output/log.txt")

        assert surface.format == "text"

    def test_parse_invalid_uri(self):
        """Parse raises error for non-file URI."""
        from synix.surfaces.file import parse_file_surface

        with pytest.raises(ValueError, match="Expected file:// URI"):
            parse_file_surface("test", "https://example.com/file")
