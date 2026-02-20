"""Build scheduler with state machine, debounce, and single-flight enforcement."""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)


class BuildState(Enum):
    IDLE = "idle"
    QUEUED = "queued"
    RUNNING = "running"


class BuildScheduler:
    """Manages build timing with debounce and single-flight enforcement.

    State machine: IDLE -> QUEUED -> RUNNING -> IDLE
    (or back to QUEUED if new sessions arrived during the build)
    """

    def __init__(
        self,
        min_interval: int = 300,
        batch_threshold: int = 5,
        max_delay: int = 900,
    ):
        self.state = BuildState.IDLE
        self.min_interval = min_interval
        self.batch_threshold = batch_threshold
        self.max_delay = max_delay
        self.pending_count = 0
        self.first_pending_at: float | None = None
        self.last_build_at: float = 0.0
        self._force_rebuild = False
        self._lock = asyncio.Lock()

    async def notify_new_session(self) -> None:
        """Called when a new session is submitted. Increments pending count."""
        async with self._lock:
            self.pending_count += 1
            if self.first_pending_at is None:
                self.first_pending_at = time.monotonic()
            logger.debug(
                "New session notified, pending_count=%d, state=%s",
                self.pending_count,
                self.state.value,
            )
            if self.state == BuildState.IDLE:
                self.state = BuildState.QUEUED

    async def should_build(self) -> bool:
        """Check if a build should start now based on thresholds and timers."""
        async with self._lock:
            if self.state == BuildState.RUNNING:
                return False

            if self._force_rebuild:
                return True

            if self.pending_count <= 0:
                return False

            # Respect minimum interval between builds
            now = time.monotonic()
            if self.last_build_at > 0 and (now - self.last_build_at) < self.min_interval:
                return False

            # Batch threshold: enough pending sessions to justify a build
            if self.pending_count >= self.batch_threshold:
                return True

            # Max delay: waited too long since first pending session
            if self.first_pending_at is not None and (now - self.first_pending_at) >= self.max_delay:
                return True

            return False

    async def start_build(self) -> None:
        """Transition to RUNNING state. Raises if already RUNNING."""
        async with self._lock:
            if self.state == BuildState.RUNNING:
                raise RuntimeError("Build already running — single-flight violation")
            self.state = BuildState.RUNNING
            self._force_rebuild = False
            logger.info(
                "Build started (pending_count=%d)",
                self.pending_count,
            )

    async def complete_build(self) -> bool:
        """Transition from RUNNING. Returns True if another build is needed."""
        async with self._lock:
            self.last_build_at = time.monotonic()
            self.pending_count = 0
            self.first_pending_at = None

            if self._force_rebuild:
                self.state = BuildState.QUEUED
                logger.info("Build completed, force rebuild queued")
                return True

            self.state = BuildState.IDLE
            logger.info("Build completed, state -> IDLE")
            return False

    async def force_rebuild(self) -> str:
        """Request a forced rebuild. Returns 'started' if idle, 'queued' if running."""
        async with self._lock:
            if self.state == BuildState.RUNNING:
                self._force_rebuild = True
                logger.info("Force rebuild queued (build in progress)")
                return "queued"
            # IDLE or QUEUED -> mark force and transition to QUEUED
            self._force_rebuild = True
            self.state = BuildState.QUEUED
            logger.info("Force rebuild requested, state -> QUEUED")
            return "started"

    def get_status(self) -> dict:
        """Return current scheduler state as dict."""
        return {
            "state": self.state.value,
            "pending_count": self.pending_count,
            "first_pending_at": self.first_pending_at,
            "last_build_at": self.last_build_at,
            "force_rebuild": self._force_rebuild,
        }
