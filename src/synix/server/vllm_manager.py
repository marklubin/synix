"""vLLM subprocess manager — start, health-check, stop, throughput measurement."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
import urllib.request
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VLLMConfig:
    """vLLM server configuration."""

    enabled: bool = False
    model: str = "Qwen/Qwen3.5-2B"
    gpu_device: int = 0
    port: int = 8100
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    extra_args: list[str] = field(default_factory=list)
    startup_timeout: int = 120  # seconds to wait for health check


class VLLMManager:
    """Async lifecycle manager for a vLLM subprocess."""

    def __init__(self, config: VLLMConfig) -> None:
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._drain_task: asyncio.Task | None = None

    @property
    def base_url(self) -> str:
        """OpenAI-compatible API base URL."""
        return f"http://localhost:{self.config.port}/v1"

    async def start(self) -> None:
        """Start the vLLM server subprocess and wait until it is healthy.

        Raises RuntimeError if the server does not pass health checks within
        ``startup_timeout`` seconds.
        """
        cfg = self.config

        cmd = [
            "vllm", "serve", cfg.model,
            "--port", str(cfg.port),
            "--tensor-parallel-size", "1",
            "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
            "--max-model-len", str(cfg.max_model_len),
            "--default-chat-template-kwargs", json.dumps({"enable_thinking": False}),
            "--enable-prefix-caching",
            *cfg.extra_args,
        ]

        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(cfg.gpu_device), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}

        logger.debug("Starting vLLM: %s", " ".join(cmd))
        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        # Background task to drain stdout/stderr so the pipe buffers don't fill up
        self._drain_task = asyncio.get_event_loop().create_task(self._drain_output())

        # Poll health check until ready or timeout
        deadline = time.monotonic() + cfg.startup_timeout
        while time.monotonic() < deadline:
            if self._process.returncode is not None:
                raise RuntimeError(
                    f"vLLM process exited unexpectedly with code {self._process.returncode}"
                )
            if await self.health_check():
                logger.info(
                    "vLLM started: %s on GPU %d, port %d",
                    cfg.model, cfg.gpu_device, cfg.port,
                )
                return
            await asyncio.sleep(2)

        # Timeout — kill the process
        await self._kill_process()
        raise RuntimeError(f"vLLM failed to start within {cfg.startup_timeout}s")

    async def health_check(self) -> bool:
        """Return True if the vLLM server responds 200 on /health."""
        loop = asyncio.get_event_loop()
        try:
            url = f"http://localhost:{self.config.port}/health"
            req = urllib.request.Request(url, method="GET")

            def _check() -> bool:
                resp = urllib.request.urlopen(req, timeout=5)
                return resp.status == 200

            return await loop.run_in_executor(None, _check)
        except Exception:
            return False

    async def stop(self) -> None:
        """Gracefully stop the vLLM subprocess (SIGTERM, then SIGKILL)."""
        if self._process is None or self._process.returncode is not None:
            self._process = None
            return

        try:
            self._process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            logger.debug("vLLM process already exited before SIGTERM")
            self._process = None
            return

        try:
            await asyncio.wait_for(self._process.wait(), timeout=10)
        except TimeoutError:
            logger.warning("vLLM did not exit after SIGTERM, sending SIGKILL")
            self._process.kill()
            await self._process.wait()

        logger.info("vLLM stopped")
        self._process = None

        # Cancel drain task if still running
        if self._drain_task is not None and not self._drain_task.done():
            self._drain_task.cancel()
            self._drain_task = None

    async def measure_throughput(self) -> dict:
        """Send a test prompt and measure tokens-per-second.

        Returns a dict with ``tok_per_sec``, ``completion_tokens``, and
        ``elapsed_seconds`` on success, or ``tok_per_sec: 0`` plus ``error``
        on failure.
        """
        loop = asyncio.get_event_loop()
        try:
            url = f"http://localhost:{self.config.port}/v1/chat/completions"
            payload = json.dumps({
                "model": self.config.model,
                "messages": [{"role": "user", "content": "Count from 1 to 20."}],
                "max_tokens": 100,
                "temperature": 0,
            }).encode()
            req = urllib.request.Request(
                url, data=payload, method="POST",
                headers={"Content-Type": "application/json"},
            )

            def _request() -> tuple[dict, float]:
                t0 = time.monotonic()
                resp = urllib.request.urlopen(req, timeout=30)
                elapsed = time.monotonic() - t0
                body = json.loads(resp.read())
                return body, elapsed

            body, elapsed = await loop.run_in_executor(None, _request)

            completion_tokens = body["usage"]["completion_tokens"]
            tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0.0

            logger.info(
                "vLLM throughput: %.1f tok/s (%d tokens in %.1fs)",
                tok_per_sec, completion_tokens, elapsed,
            )
            return {
                "tok_per_sec": tok_per_sec,
                "completion_tokens": completion_tokens,
                "elapsed_seconds": elapsed,
            }
        except Exception as exc:
            logger.warning("vLLM throughput measurement failed: %s", exc)
            return {"tok_per_sec": 0, "error": str(exc)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _drain_output(self) -> None:
        """Read stdout and stderr from the subprocess and log at DEBUG level."""
        proc = self._process
        if proc is None:
            return

        async def _read_stream(stream: asyncio.StreamReader | None, label: str) -> None:
            if stream is None:
                return
            while True:
                line = await stream.readline()
                if not line:
                    break
                logger.debug("vLLM %s: %s", label, line.decode(errors="replace").rstrip())

        try:
            await asyncio.gather(
                _read_stream(proc.stdout, "stdout"),
                _read_stream(proc.stderr, "stderr"),
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.debug("vLLM drain stopped: %s", exc)

        # Process ended — log if unexpected
        if proc.returncode is not None and proc.returncode != 0:
            logger.error(
                "vLLM process exited unexpectedly with code %d", proc.returncode,
            )

    async def _kill_process(self) -> None:
        """Forcibly kill the subprocess."""
        if self._process is not None and self._process.returncode is None:
            self._process.kill()
            await self._process.wait()
        if self._drain_task is not None and not self._drain_task.done():
            self._drain_task.cancel()
            self._drain_task = None
        self._process = None
