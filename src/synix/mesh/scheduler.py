"""Build scheduler with quiet-period debounce and single-flight enforcement.

The scheduler waits for submissions to stop arriving before triggering a build.
This naturally handles both backfill (wait for the flood to finish, then one big
build) and steady-state (build shortly after a few sessions trickle in).

Trigger logic:
  1. Each submission resets the quiet timer.
  2. Build fires when `quiet_period` elapses with no new submissions.
  3. `max_delay` is a safety net — forces a build even if submissions never stop.
  4. `min_interval` prevents back-to-back builds.
  5. Manual `force_rebuild()` bypasses all timers.
"""

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
    """Manages build timing with quiet-period debounce and single-flight enforcement.

    State machine: IDLE -> QUEUED -> RUNNING -> IDLE
    (or back to QUEUED if new sessions arrived during the build)
    """

    def __init__(
        self,
        min_interval: int = 300,
        quiet_period: int = 60,
        max_delay: int = 1800,
    ):
        self.state = BuildState.IDLE
        self.min_interval = min_interval
        self.quiet_period = quiet_period
        self.max_delay = max_delay
        self.pending_count = 0
        self.first_pending_at: float | None = None
        self.last_submission_at: float | None = None
        self.last_build_at: float = 0.0
        self._force_rebuild = False
        self._pending_at_build_start: int = 0
        self._lock = asyncio.Lock()

    async def notify_new_session(self) -> None:
        """Called when a new session is submitted. Resets the quiet timer."""
        async with self._lock:
            self.pending_count += 1
            now = time.monotonic()
            self.last_submission_at = now
            if self.first_pending_at is None:
                self.first_pending_at = now
            logger.debug(
                "New session notified, pending_count=%d, state=%s",
                self.pending_count,
                self.state.value,
            )
            if self.state == BuildState.IDLE:
                self.state = BuildState.QUEUED

    async def should_build(self) -> bool:
        """Check if a build should start now.

        Returns True when:
        - Force rebuild was requested, OR
        - There are pending sessions AND the quiet period has elapsed
          (no new submissions for `quiet_period` seconds), OR
        - There are pending sessions AND `max_delay` has been exceeded
          (safety net for continuous submission streams).

        Always respects `min_interval` between builds.
        """
        async with self._lock:
            if self.state == BuildState.RUNNING:
                return False

            if self._force_rebuild:
                return True

            if self.pending_count <= 0:
                return False

            now = time.monotonic()

            # Respect minimum interval between builds
            if self.last_build_at > 0 and (now - self.last_build_at) < self.min_interval:
                return False

            # Quiet period: no new submissions for quiet_period seconds
            if self.last_submission_at is not None and (now - self.last_submission_at) >= self.quiet_period:
                return True

            # Max delay safety net: don't wait forever if submissions keep trickling
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
            self._pending_at_build_start = self.pending_count
            logger.info(
                "Build started (pending_count=%d)",
                self.pending_count,
            )

    async def complete_build(self) -> bool:
        """Transition from RUNNING. Returns True if another build is needed.

        Sessions that arrived *during* the build (i.e. pending_count grew beyond
        what it was at start_build) are preserved as pending for the next cycle.
        """
        async with self._lock:
            self.last_build_at = time.monotonic()

            # Sessions arriving during the build weren't included in this build
            arrived_during_build = max(0, self.pending_count - self._pending_at_build_start)

            if self._force_rebuild or arrived_during_build > 0:
                self.pending_count = arrived_during_build
                # Keep first_pending_at and last_submission_at so the quiet-period
                # timer tracks the new arrivals correctly
                if arrived_during_build == 0:
                    self.first_pending_at = None
                    self.last_submission_at = None
                self.state = BuildState.QUEUED
                if self._force_rebuild:
                    reason = "force rebuild"
                else:
                    reason = f"{arrived_during_build} sessions arrived during build"
                logger.info("Build completed, queuing another (%s)", reason)
                self._force_rebuild = False
                return True

            self.pending_count = 0
            self.first_pending_at = None
            self.last_submission_at = None
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
            "last_submission_at": self.last_submission_at,
            "last_build_at": self.last_build_at,
            "force_rebuild": self._force_rebuild,
        }
