"""Structured logging for mesh server and client.

Two formatters:
- JSONFormatter — JSON Lines for log files (machine-parseable by dashboard)
- HumanFormatter — HH:MM:SS LEVEL [logger] message for stderr/systemd journal

Setup function attaches both handlers to the synix.mesh root logger.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Format log records as JSON Lines for file output."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="seconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Structured event fields from mesh_event()
        if hasattr(record, "event"):
            entry["event"] = record.event
        if hasattr(record, "detail"):
            entry["detail"] = record.detail
        if record.exc_info and record.exc_info[1] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


class HumanFormatter(logging.Formatter):
    """Human-readable format: HH:MM:SS LEVEL [logger] message."""

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        # Shorten logger name: synix.mesh.server -> server
        short_name = record.name.rsplit(".", 1)[-1] if "." in record.name else record.name
        msg = record.getMessage()
        base = f"{ts} {record.levelname:<7s} [{short_name}] {msg}"
        if record.exc_info and record.exc_info[1] is not None:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_mesh_logging(
    mesh_dir: Path,
    role: str,
    file_level: int = logging.DEBUG,
    stderr_level: int = logging.INFO,
) -> None:
    """Configure mesh logging with file (JSON) and stderr (human) handlers.

    Args:
        mesh_dir: Root mesh directory (e.g., ~/.synix-mesh/my-mesh/)
        role: "server" or "client" — determines log file name
        file_level: Logging level for file handler (default DEBUG)
        stderr_level: Logging level for stderr handler (default INFO)
    """
    logs_dir = mesh_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Root mesh logger
    mesh_logger = logging.getLogger("synix.mesh")
    mesh_logger.setLevel(min(file_level, stderr_level))
    # Remove existing handlers to avoid duplicates on re-init
    mesh_logger.handlers.clear()

    # File handler — JSON Lines, rotating 5MB x 3 backups (~20MB per role)
    log_path = logs_dir / f"{role}.log"
    file_handler = RotatingFileHandler(
        str(log_path),
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(JSONFormatter())
    mesh_logger.addHandler(file_handler)

    # Stderr handler — human-readable
    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(stderr_level)
    stderr_handler.setFormatter(HumanFormatter())
    mesh_logger.addHandler(stderr_handler)

    # Capture uvicorn logs into the same handlers
    for uvicorn_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv_logger = logging.getLogger(uvicorn_name)
        uv_logger.handlers.clear()
        uv_logger.addHandler(file_handler)
        uv_logger.addHandler(stderr_handler)
        uv_logger.setLevel(min(file_level, stderr_level))


def mesh_event(
    log: logging.Logger,
    level: int,
    msg: str,
    event: str,
    detail: dict | None = None,
) -> None:
    """Log a structured mesh event with extra fields for JSON serialization.

    Args:
        log: Logger instance
        level: Logging level (e.g., logging.INFO)
        msg: Human-readable message
        event: Machine-readable event name (e.g., "session_submitted")
        detail: Structured detail dict serialized by JSONFormatter
    """
    log.log(level, msg, extra={"event": event, "detail": detail or {}})
