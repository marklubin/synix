"""Error classification for build failures — determines retry, skip (DLQ), or abort."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

logger = logging.getLogger(__name__)


class ErrorVerdict(Enum):
    """How the runner should handle a failed work unit."""

    FATAL = "fatal"  # Abort the entire build immediately
    RETRYABLE = "retryable"  # Retry (handled by LLM client already; runner sees this post-retry)
    DLQ = "dlq"  # Skip this artifact, log to dead letter queue, continue building


class ErrorClassifier(Protocol):
    """Interface for classifying build errors into verdicts."""

    def classify(self, exc: Exception, artifact_desc: str) -> ErrorVerdict:
        """Classify an exception into a verdict.

        Args:
            exc: The exception that was raised.
            artifact_desc: Human-readable label for the failing artifact.

        Returns:
            ErrorVerdict indicating how the runner should handle the error.
        """
        ...


@dataclass
class DeadLetterEntry:
    """A single failed artifact in the dead letter queue."""

    artifact_desc: str
    error_type: str
    error_message: str
    layer_name: str = ""


@dataclass
class DeadLetterQueue:
    """Collects artifacts that were skipped due to non-fatal errors.

    When a structured logger (SynixLogger) is attached, DLQ entries are
    written to the JSONL build log at `.synix/logs/{run_id}.jsonl` and
    surfaced in the CLI build summary.
    """

    entries: list[DeadLetterEntry] = field(default_factory=list)
    slogger: object | None = field(default=None, repr=False)

    def add(self, artifact_desc: str, exc: Exception, layer_name: str = "") -> None:
        """Record a skipped artifact."""
        error_type = type(exc).__name__
        error_message = str(exc)
        entry = DeadLetterEntry(
            artifact_desc=artifact_desc,
            error_type=error_type,
            error_message=error_message,
            layer_name=layer_name,
        )
        self.entries.append(entry)
        logger.warning("DLQ: skipped %s — %s: %s", artifact_desc, error_type, exc)

        # Write to structured build log if available
        if self.slogger is not None:
            self.slogger.artifact_dlq(layer_name, artifact_desc, error_type, error_message)

    def __len__(self) -> int:
        return len(self.entries)

    def summary(self) -> str:
        """Human-readable summary of DLQ contents."""
        if not self.entries:
            return "No errors"
        by_type: dict[str, int] = {}
        for e in self.entries:
            by_type[e.error_type] = by_type.get(e.error_type, 0) + 1
        parts = [f"{count} {etype}" for etype, count in sorted(by_type.items())]
        return f"{len(self.entries)} artifacts skipped ({', '.join(parts)})"


# --- Built-in classifiers ---

# Patterns that indicate content filtering / moderation rejection
_CONTENT_FILTER_PATTERNS = [
    re.compile(r"content.?filter", re.IGNORECASE),
    re.compile(r"high.?risk", re.IGNORECASE),
    re.compile(r"content.?moderation", re.IGNORECASE),
    re.compile(r"safety.?system", re.IGNORECASE),
    re.compile(r"rejected.*prompt", re.IGNORECASE),
    re.compile(r"flagged.*content", re.IGNORECASE),
]

# Patterns that indicate auth failures — always fatal
_AUTH_PATTERNS = [
    re.compile(r"invalid.*auth", re.IGNORECASE),
    re.compile(r"invalid.*api.?key", re.IGNORECASE),
    re.compile(r"authentication.*failed", re.IGNORECASE),
    re.compile(r"unauthorized", re.IGNORECASE),
    re.compile(r"error code: 401", re.IGNORECASE),
]

# Patterns that indicate the input is too large for the model
_INPUT_TOO_LARGE_PATTERNS = [
    re.compile(r"context.?length.*exceeded", re.IGNORECASE),
    re.compile(r"maximum.*tokens", re.IGNORECASE),
    re.compile(r"too.?many.?tokens", re.IGNORECASE),
    re.compile(r"input.*too.*long", re.IGNORECASE),
    re.compile(r"prompt.*too.*long", re.IGNORECASE),
]


class LLMErrorClassifier:
    """Default error classifier for LLM API errors.

    Classification rules (checked in order):
    1. Auth errors (401, invalid key) → FATAL (no point continuing)
    2. Content filter / moderation → DLQ (skip this input, continue)
    3. Input too large → DLQ (this specific input can't be processed)
    4. Everything else → FATAL (unknown error, fail safe)
    """

    def classify(self, exc: Exception, artifact_desc: str) -> ErrorVerdict:
        msg = str(exc)

        # Check the full exception chain
        cause = exc.__cause__
        cause_msg = str(cause) if cause else ""
        full_msg = f"{msg} {cause_msg}"

        # Auth errors — always fatal, no point processing more artifacts
        for pattern in _AUTH_PATTERNS:
            if pattern.search(full_msg):
                logger.debug("Classified as FATAL (auth): %s", artifact_desc)
                return ErrorVerdict.FATAL

        # Content filter — skip this artifact, continue with others
        for pattern in _CONTENT_FILTER_PATTERNS:
            if pattern.search(full_msg):
                logger.debug("Classified as DLQ (content filter): %s", artifact_desc)
                return ErrorVerdict.DLQ

        # Input too large — skip this artifact, continue
        for pattern in _INPUT_TOO_LARGE_PATTERNS:
            if pattern.search(full_msg):
                logger.debug("Classified as DLQ (input too large): %s", artifact_desc)
                return ErrorVerdict.DLQ

        # Default: fatal — fail safe on unknown errors
        logger.debug("Classified as FATAL (unknown): %s", artifact_desc)
        return ErrorVerdict.FATAL
