"""Tests for vLLM subprocess manager — all subprocess/network calls mocked."""

from __future__ import annotations

import asyncio
import json
import signal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from synix.server.vllm_manager import VLLMConfig, VLLMManager


@pytest.fixture
def config():
    """VLLMConfig with a short timeout for fast tests."""
    return VLLMConfig(enabled=True, startup_timeout=2)


@pytest.fixture
def manager(config):
    return VLLMManager(config)


# ------------------------------------------------------------------
# Config defaults
# ------------------------------------------------------------------


class TestVLLMConfig:
    def test_config_defaults(self):
        cfg = VLLMConfig()
        assert cfg.enabled is False
        assert cfg.model == "Qwen/Qwen2.5-3B-Instruct"
        assert cfg.gpu_device == 0
        assert cfg.port == 8100
        assert cfg.max_model_len == 4096
        assert cfg.gpu_memory_utilization == 0.90
        assert cfg.extra_args == []
        assert cfg.startup_timeout == 120

    def test_config_base_url(self, manager):
        assert manager.base_url == "http://localhost:8100/v1"

    def test_config_base_url_custom_port(self):
        mgr = VLLMManager(VLLMConfig(port=9999))
        assert mgr.base_url == "http://localhost:9999/v1"


# ------------------------------------------------------------------
# start()
# ------------------------------------------------------------------


class TestStart:
    @pytest.mark.asyncio
    async def test_start_builds_correct_command(self, manager):
        """Verify the vllm command includes model, port, thinking disabled, prefix caching."""
        captured_args = {}

        async def fake_exec(*args, **kwargs):
            captured_args["cmd"] = args
            captured_args["env"] = kwargs.get("env", {})
            proc = AsyncMock()
            proc.returncode = None
            proc.stdout = _fake_stream()
            proc.stderr = _fake_stream()
            proc.wait = AsyncMock(return_value=0)
            return proc

        with (
            patch("asyncio.create_subprocess_exec", side_effect=fake_exec),
            patch.object(manager, "health_check", return_value=True),
        ):
            await manager.start()

        cmd = captured_args["cmd"]
        assert cmd[0] == "vllm"
        assert cmd[1] == "serve"
        assert manager.config.model in cmd
        assert "--port" in cmd
        assert str(manager.config.port) in cmd
        assert "--enable-prefix-caching" in cmd
        # Thinking disabled via default-chat-template-kwargs
        assert "--default-chat-template-kwargs" in cmd
        kwargs_idx = cmd.index("--default-chat-template-kwargs") + 1
        parsed = json.loads(cmd[kwargs_idx])
        assert parsed["enable_thinking"] is False

    @pytest.mark.asyncio
    async def test_start_sets_cuda_env(self, manager):
        """Verify CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER are set."""
        captured_env = {}

        async def fake_exec(*args, **kwargs):
            captured_env.update(kwargs.get("env", {}))
            proc = AsyncMock()
            proc.returncode = None
            proc.stdout = _fake_stream()
            proc.stderr = _fake_stream()
            proc.wait = AsyncMock(return_value=0)
            return proc

        with (
            patch("asyncio.create_subprocess_exec", side_effect=fake_exec),
            patch.object(manager, "health_check", return_value=True),
        ):
            await manager.start()

        assert captured_env["CUDA_VISIBLE_DEVICES"] == "0"
        assert captured_env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"

    @pytest.mark.asyncio
    async def test_start_health_check_polling(self, config):
        """health_check returns False twice, then True — start() succeeds."""
        mgr = VLLMManager(config)
        call_count = 0

        async def health_sequence():
            nonlocal call_count
            call_count += 1
            return call_count >= 3

        with (
            patch("asyncio.create_subprocess_exec", side_effect=_make_fake_exec()),
            patch.object(mgr, "health_check", side_effect=health_sequence),
            patch("asyncio.sleep", new_callable=AsyncMock),  # skip real sleeps
        ):
            await mgr.start()

        assert call_count >= 3

    @pytest.mark.asyncio
    async def test_start_timeout_raises(self):
        """health_check always returns False — RuntimeError after timeout."""
        cfg = VLLMConfig(startup_timeout=1)
        mgr = VLLMManager(cfg)

        proc = AsyncMock()
        proc.returncode = None
        proc.stdout = _fake_stream()
        proc.stderr = _fake_stream()
        proc.wait = AsyncMock(return_value=1)
        proc.kill = MagicMock()

        async def fake_exec(*args, **kwargs):
            return proc

        with (
            patch("asyncio.create_subprocess_exec", side_effect=fake_exec),
            patch.object(mgr, "health_check", return_value=False),
            patch("asyncio.sleep", new_callable=AsyncMock),
            # Fake time.monotonic to advance past the deadline immediately
            patch("synix.server.vllm_manager.time") as mock_time,
        ):
            # First call sets the deadline, subsequent calls exceed it
            mock_time.monotonic = MagicMock(side_effect=[0, 0, 100])

            with pytest.raises(RuntimeError, match="failed to start"):
                await mgr.start()

    @pytest.mark.asyncio
    async def test_extra_args_included(self):
        """extra_args from config appear in the command."""
        cfg = VLLMConfig(extra_args=["--foo", "bar"])
        mgr = VLLMManager(cfg)
        captured_args = {}

        async def fake_exec(*args, **kwargs):
            captured_args["cmd"] = args
            proc = AsyncMock()
            proc.returncode = None
            proc.stdout = _fake_stream()
            proc.stderr = _fake_stream()
            proc.wait = AsyncMock(return_value=0)
            return proc

        with (
            patch("asyncio.create_subprocess_exec", side_effect=fake_exec),
            patch.object(mgr, "health_check", return_value=True),
        ):
            await mgr.start()

        cmd = captured_args["cmd"]
        assert "--foo" in cmd
        assert "bar" in cmd


# ------------------------------------------------------------------
# stop()
# ------------------------------------------------------------------


class TestStop:
    @pytest.mark.asyncio
    async def test_stop_sends_sigterm(self, manager):
        """SIGTERM is sent when stopping a running process."""
        proc = AsyncMock()
        proc.returncode = None
        proc.send_signal = MagicMock()
        proc.wait = AsyncMock(return_value=0)
        manager._process = proc

        await manager.stop()

        proc.send_signal.assert_called_once_with(signal.SIGTERM)
        assert manager._process is None

    @pytest.mark.asyncio
    async def test_stop_sigkill_on_timeout(self, manager):
        """If process doesn't exit after SIGTERM, SIGKILL is sent."""
        proc = AsyncMock()
        proc.returncode = None
        proc.send_signal = MagicMock()
        proc.kill = MagicMock()
        # After SIGKILL, wait() should succeed normally
        proc.wait = AsyncMock(return_value=0)

        manager._process = proc

        # Patch wait_for to always timeout (simulating SIGTERM grace period expiry)
        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            await manager.stop()

        proc.send_signal.assert_called_once_with(signal.SIGTERM)
        proc.kill.assert_called_once()
        assert manager._process is None

    @pytest.mark.asyncio
    async def test_stop_noop_when_not_started(self, manager):
        """stop() on a fresh manager does not raise."""
        assert manager._process is None
        await manager.stop()  # should not raise


# ------------------------------------------------------------------
# measure_throughput()
# ------------------------------------------------------------------


class TestMeasureThroughput:
    @pytest.mark.asyncio
    async def test_measure_throughput_success(self, manager):
        """Successful throughput measurement returns tok_per_sec and timing."""
        response_body = json.dumps({
            "usage": {"completion_tokens": 50, "prompt_tokens": 10, "total_tokens": 60},
            "choices": [{"message": {"content": "1 2 3 ..."}}],
        }).encode()

        fake_resp = MagicMock()
        fake_resp.read.return_value = response_body
        fake_resp.status = 200

        with patch("synix.server.vllm_manager.urllib.request.urlopen", return_value=fake_resp):
            result = await manager.measure_throughput()

        assert result["completion_tokens"] == 50
        assert result["elapsed_seconds"] > 0
        assert result["tok_per_sec"] > 0
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_measure_throughput_failure(self, manager):
        """Network failure returns error dict with tok_per_sec=0."""
        with patch(
            "synix.server.vllm_manager.urllib.request.urlopen",
            side_effect=ConnectionRefusedError("Connection refused"),
        ):
            result = await manager.measure_throughput()

        assert result["tok_per_sec"] == 0
        assert "error" in result
        assert "Connection refused" in result["error"]


# ------------------------------------------------------------------
# health_check()
# ------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check_returns_true_on_200(self, manager):
        fake_resp = MagicMock()
        fake_resp.status = 200

        with patch("synix.server.vllm_manager.urllib.request.urlopen", return_value=fake_resp):
            assert await manager.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_error(self, manager):
        with patch(
            "synix.server.vllm_manager.urllib.request.urlopen",
            side_effect=ConnectionRefusedError("refused"),
        ):
            assert await manager.health_check() is False


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _fake_stream():
    """Return an AsyncMock that behaves like an empty asyncio.StreamReader."""
    stream = AsyncMock()
    stream.readline = AsyncMock(return_value=b"")
    return stream


def _make_fake_exec():
    """Factory returning a fake create_subprocess_exec coroutine."""

    async def fake_exec(*args, **kwargs):
        proc = AsyncMock()
        proc.returncode = None
        proc.stdout = _fake_stream()
        proc.stderr = _fake_stream()
        proc.wait = AsyncMock(return_value=0)
        return proc

    return fake_exec
