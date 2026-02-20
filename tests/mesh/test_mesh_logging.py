"""Tests for synix.mesh.logging — structured logging formatters and setup."""

from __future__ import annotations

import json
import logging

import pytest

from synix.mesh.logging import HumanFormatter, JSONFormatter, mesh_event, setup_mesh_logging

pytestmark = pytest.mark.mesh


class TestJSONFormatter:
    def test_basic_format(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="synix.mesh.server",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = json.loads(formatter.format(record))
        assert result["level"] == "INFO"
        assert result["logger"] == "synix.mesh.server"
        assert result["msg"] == "Test message"
        assert "ts" in result

    def test_event_fields(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="synix.mesh.server",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Session submitted",
            args=(),
            exc_info=None,
        )
        record.event = "session_submitted"
        record.detail = {"hostname": "obispo", "session_id": "abc"}
        result = json.loads(formatter.format(record))
        assert result["event"] == "session_submitted"
        assert result["detail"]["hostname"] == "obispo"

    def test_exception_included(self):
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Failed",
            args=(),
            exc_info=exc_info,
        )
        result = json.loads(formatter.format(record))
        assert "exception" in result
        assert "ValueError" in result["exception"]

    def test_no_event_fields_when_absent(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="plain",
            args=(),
            exc_info=None,
        )
        result = json.loads(formatter.format(record))
        assert "event" not in result
        assert "detail" not in result


class TestHumanFormatter:
    def test_basic_format(self):
        formatter = HumanFormatter()
        record = logging.LogRecord(
            name="synix.mesh.server",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "INFO" in result
        assert "[server]" in result
        assert "Test message" in result

    def test_short_logger_name(self):
        formatter = HumanFormatter()
        record = logging.LogRecord(
            name="synix.mesh.client",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Warning",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "[client]" in result


class TestSetupMeshLogging:
    def test_creates_log_file(self, tmp_path):
        setup_mesh_logging(tmp_path, "server")
        log = logging.getLogger("synix.mesh.test_setup")
        log.info("test message")
        log_path = tmp_path / "logs" / "server.log"
        assert log_path.exists()
        content = log_path.read_text()
        assert "test message" in content

    def test_json_format_in_file(self, tmp_path):
        setup_mesh_logging(tmp_path, "server")
        log = logging.getLogger("synix.mesh.test_json")
        log.info("json test")
        log_path = tmp_path / "logs" / "server.log"
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) >= 1
        entry = json.loads(lines[-1])
        assert entry["msg"] == "json test"

    def test_custom_levels(self, tmp_path):
        setup_mesh_logging(tmp_path, "client", file_level=logging.WARNING, stderr_level=logging.ERROR)
        mesh_logger = logging.getLogger("synix.mesh")
        # File handler should be WARNING level
        file_handlers = [h for h in mesh_logger.handlers if hasattr(h, "baseFilename")]
        assert any(h.level == logging.WARNING for h in file_handlers)

    def test_logs_dir_created(self, tmp_path):
        mesh_dir = tmp_path / "nonexistent"
        setup_mesh_logging(mesh_dir, "server")
        assert (mesh_dir / "logs").is_dir()

    def test_no_duplicate_handlers_on_reinit(self, tmp_path):
        setup_mesh_logging(tmp_path, "server")
        setup_mesh_logging(tmp_path, "server")
        mesh_logger = logging.getLogger("synix.mesh")
        # Should have exactly 2 handlers (file + stderr), not 4
        assert len(mesh_logger.handlers) == 2


class TestMeshEvent:
    def test_mesh_event_logs_with_extra(self, tmp_path):
        setup_mesh_logging(tmp_path, "server")
        log = logging.getLogger("synix.mesh.test_event")
        mesh_event(
            log,
            logging.INFO,
            "Session submitted",
            "session_submitted",
            {
                "hostname": "obispo",
                "session_id": "abc",
            },
        )
        log_path = tmp_path / "logs" / "server.log"
        lines = log_path.read_text().strip().splitlines()
        entry = json.loads(lines[-1])
        assert entry["event"] == "session_submitted"
        assert entry["detail"]["hostname"] == "obispo"

    def test_mesh_event_without_detail(self, tmp_path):
        setup_mesh_logging(tmp_path, "server")
        log = logging.getLogger("synix.mesh.test_no_detail")
        mesh_event(log, logging.WARNING, "Something happened", "generic_event")
        log_path = tmp_path / "logs" / "server.log"
        lines = log_path.read_text().strip().splitlines()
        entry = json.loads(lines[-1])
        assert entry["event"] == "generic_event"
        assert entry["detail"] == {}
