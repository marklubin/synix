"""Tests for synix.mesh.scheduler — build scheduler with quiet-period debounce."""

from __future__ import annotations

import time

import pytest

from synix.mesh.scheduler import BuildScheduler, BuildState

pytestmark = pytest.mark.mesh


@pytest.fixture
def scheduler():
    return BuildScheduler(min_interval=10, quiet_period=5, max_delay=60)


class TestStateTransitions:
    @pytest.mark.asyncio
    async def test_starts_idle(self, scheduler):
        assert scheduler.state == BuildState.IDLE

    @pytest.mark.asyncio
    async def test_new_session_transitions_to_queued(self, scheduler):
        await scheduler.notify_new_session()
        assert scheduler.state == BuildState.QUEUED
        assert scheduler.pending_count == 1

    @pytest.mark.asyncio
    async def test_start_build_transitions_to_running(self, scheduler):
        await scheduler.notify_new_session()
        await scheduler.start_build()
        assert scheduler.state == BuildState.RUNNING

    @pytest.mark.asyncio
    async def test_complete_build_transitions_to_idle(self, scheduler):
        await scheduler.notify_new_session()
        await scheduler.start_build()
        needs_another = await scheduler.complete_build()
        assert scheduler.state == BuildState.IDLE
        assert needs_another is False

    @pytest.mark.asyncio
    async def test_complete_build_resets_pending(self, scheduler):
        for _ in range(5):
            await scheduler.notify_new_session()
        await scheduler.start_build()
        # No new sessions during build
        await scheduler.complete_build()
        assert scheduler.pending_count == 0
        assert scheduler.first_pending_at is None
        assert scheduler.last_submission_at is None


class TestQuietPeriod:
    @pytest.mark.asyncio
    async def test_no_build_during_active_submissions(self, scheduler):
        """Submissions keep arriving — quiet period hasn't elapsed."""
        await scheduler.notify_new_session()
        scheduler.last_build_at = 0.0
        # last_submission_at is ~now, quiet_period=5 hasn't passed
        assert await scheduler.should_build() is False

    @pytest.mark.asyncio
    async def test_build_after_quiet_period(self, scheduler):
        """Submissions stopped, quiet period elapsed — build fires."""
        await scheduler.notify_new_session()
        scheduler.last_build_at = 0.0
        # Simulate quiet period elapsed
        scheduler.last_submission_at = time.monotonic() - scheduler.quiet_period - 1
        assert await scheduler.should_build() is True

    @pytest.mark.asyncio
    async def test_new_submission_resets_quiet_timer(self, scheduler):
        """A new submission arriving resets the quiet timer."""
        await scheduler.notify_new_session()
        scheduler.last_build_at = 0.0
        # Simulate quiet period almost elapsed
        scheduler.last_submission_at = time.monotonic() - scheduler.quiet_period + 1
        # New submission arrives — resets the timer
        await scheduler.notify_new_session()
        assert await scheduler.should_build() is False

    @pytest.mark.asyncio
    async def test_backfill_waits_for_all_submissions(self, scheduler):
        """Simulates a backfill: many submissions, build only after they stop."""
        scheduler.last_build_at = 0.0
        for _ in range(100):
            await scheduler.notify_new_session()
        # All 100 submitted ~now — quiet period not elapsed
        assert scheduler.pending_count == 100
        assert await scheduler.should_build() is False
        # Simulate quiet period passing
        scheduler.last_submission_at = time.monotonic() - scheduler.quiet_period - 1
        assert await scheduler.should_build() is True


class TestMaxDelay:
    @pytest.mark.asyncio
    async def test_max_delay_forces_build(self, scheduler):
        """Even if submissions keep coming, max_delay forces a build."""
        await scheduler.notify_new_session()
        scheduler.last_build_at = 0.0
        # last_submission_at is recent (quiet period NOT elapsed)
        # but first_pending_at was long ago (max_delay exceeded)
        scheduler.first_pending_at = time.monotonic() - scheduler.max_delay - 1
        assert await scheduler.should_build() is True

    @pytest.mark.asyncio
    async def test_max_delay_not_reached(self, scheduler):
        """Within max_delay, submissions still active — no build."""
        await scheduler.notify_new_session()
        scheduler.last_build_at = 0.0
        # Neither quiet period nor max_delay exceeded
        assert await scheduler.should_build() is False


class TestMinInterval:
    @pytest.mark.asyncio
    async def test_respects_min_interval(self, scheduler):
        for _ in range(5):
            await scheduler.notify_new_session()
        # Quiet period elapsed but min_interval not met
        scheduler.last_submission_at = time.monotonic() - scheduler.quiet_period - 1
        scheduler.last_build_at = time.monotonic()  # just built
        assert await scheduler.should_build() is False


class TestForceRebuild:
    @pytest.mark.asyncio
    async def test_force_when_idle_returns_started(self, scheduler):
        result = await scheduler.force_rebuild()
        assert result == "started"
        assert scheduler.state == BuildState.QUEUED

    @pytest.mark.asyncio
    async def test_force_when_running_returns_queued(self, scheduler):
        await scheduler.notify_new_session()
        await scheduler.start_build()
        result = await scheduler.force_rebuild()
        assert result == "queued"
        assert scheduler.state == BuildState.RUNNING

    @pytest.mark.asyncio
    async def test_force_queued_triggers_another_build(self, scheduler):
        await scheduler.notify_new_session()
        await scheduler.start_build()
        await scheduler.force_rebuild()
        needs_another = await scheduler.complete_build()
        assert needs_another is True
        assert scheduler.state == BuildState.QUEUED


class TestSingleFlight:
    @pytest.mark.asyncio
    async def test_cannot_start_when_running(self, scheduler):
        await scheduler.notify_new_session()
        await scheduler.start_build()
        with pytest.raises(RuntimeError, match="Build already running"):
            await scheduler.start_build()

    @pytest.mark.asyncio
    async def test_should_build_false_when_running(self, scheduler):
        await scheduler.notify_new_session()
        await scheduler.start_build()
        assert await scheduler.should_build() is False


class TestCoalescedSessions:
    """Sessions arriving during a running build should not be forgotten."""

    @pytest.mark.asyncio
    async def test_sessions_during_build_trigger_requeue(self, scheduler):
        """If sessions arrive while a build is running, complete_build returns True."""
        await scheduler.notify_new_session()
        await scheduler.start_build()
        # 3 new sessions arrive during the build
        for _ in range(3):
            await scheduler.notify_new_session()
        needs_another = await scheduler.complete_build()
        assert needs_another is True
        assert scheduler.state == BuildState.QUEUED
        assert scheduler.pending_count == 3

    @pytest.mark.asyncio
    async def test_no_sessions_during_build_goes_idle(self, scheduler):
        """No sessions during build → IDLE, pending_count=0."""
        for _ in range(5):
            await scheduler.notify_new_session()
        await scheduler.start_build()
        needs_another = await scheduler.complete_build()
        assert needs_another is False
        assert scheduler.state == BuildState.IDLE
        assert scheduler.pending_count == 0

    @pytest.mark.asyncio
    async def test_coalesced_preserves_submission_timing(self, scheduler):
        """Coalesced sessions keep their last_submission_at for quiet-period tracking."""
        await scheduler.notify_new_session()
        await scheduler.start_build()
        await scheduler.notify_new_session()
        await scheduler.complete_build()
        # last_submission_at should still be set (from the mid-build arrival)
        assert scheduler.last_submission_at is not None
        assert scheduler.first_pending_at is not None

    @pytest.mark.asyncio
    async def test_force_plus_coalesced(self, scheduler):
        """Force rebuild + coalesced sessions: requeued with pending from mid-build."""
        await scheduler.notify_new_session()
        await scheduler.start_build()
        await scheduler.force_rebuild()
        await scheduler.notify_new_session()
        needs_another = await scheduler.complete_build()
        assert needs_another is True
        assert scheduler.state == BuildState.QUEUED
        # 1 session arrived during build
        assert scheduler.pending_count == 1


class TestGetStatus:
    @pytest.mark.asyncio
    async def test_status_dict(self, scheduler):
        status = scheduler.get_status()
        assert status["state"] == "idle"
        assert status["pending_count"] == 0
        assert status["force_rebuild"] is False
        assert status["last_submission_at"] is None
