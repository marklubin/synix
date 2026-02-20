"""Tests for synix.mesh.scheduler — build scheduler with state machine."""

from __future__ import annotations

import pytest

from synix.mesh.scheduler import BuildScheduler, BuildState

pytestmark = pytest.mark.mesh


@pytest.fixture
def scheduler():
    return BuildScheduler(min_interval=10, batch_threshold=3, max_delay=60)


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
        await scheduler.complete_build()
        assert scheduler.pending_count == 0
        assert scheduler.first_pending_at is None


class TestBatchThreshold:
    @pytest.mark.asyncio
    async def test_below_threshold_no_build(self, scheduler):
        # threshold=3, submit 2
        await scheduler.notify_new_session()
        await scheduler.notify_new_session()
        # Need to have enough time since last build
        scheduler.last_build_at = 0.0
        assert await scheduler.should_build() is False

    @pytest.mark.asyncio
    async def test_at_threshold_triggers_build(self, scheduler):
        for _ in range(3):
            await scheduler.notify_new_session()
        scheduler.last_build_at = 0.0
        assert await scheduler.should_build() is True


class TestMaxDelay:
    @pytest.mark.asyncio
    async def test_max_delay_triggers_build(self, scheduler):
        await scheduler.notify_new_session()
        scheduler.last_build_at = 0.0
        # Simulate that first_pending_at was long ago
        scheduler.first_pending_at = scheduler.first_pending_at - scheduler.max_delay - 1
        assert await scheduler.should_build() is True


class TestMinInterval:
    @pytest.mark.asyncio
    async def test_respects_min_interval(self, scheduler):
        for _ in range(5):
            await scheduler.notify_new_session()
        # Simulate a very recent build
        import time

        scheduler.last_build_at = time.monotonic()
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


class TestGetStatus:
    @pytest.mark.asyncio
    async def test_status_dict(self, scheduler):
        status = scheduler.get_status()
        assert status["state"] == "idle"
        assert status["pending_count"] == 0
        assert status["force_rebuild"] is False
