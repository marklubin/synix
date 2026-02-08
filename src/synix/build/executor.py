"""Concurrent LLM execution — sequential and threaded executors."""

from __future__ import annotations

import time
import random
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
from threading import Semaphore

from synix.build.llm_client import LLMClient, LLMResponse


@dataclass
class LLMRequest:
    """A single LLM completion request.

    Wraps the parameters needed for LLMClient.complete() so that
    multiple requests can be batched and executed concurrently.
    """

    messages: list[dict]
    max_tokens: int | None = None
    temperature: float | None = None
    artifact_desc: str = "artifact"


@dataclass
class LLMResult:
    """Result of an LLM request execution.

    Wraps the response along with the original request index
    so results can be matched back to their requests.
    """

    index: int
    response: LLMResponse | None = None
    error: Exception | None = None

    @property
    def success(self) -> bool:
        return self.response is not None and self.error is None


class LLMExecutor(ABC):
    """Abstract base for LLM request execution strategies."""

    @abstractmethod
    def execute(
        self,
        client: LLMClient,
        requests: list[LLMRequest],
    ) -> list[LLMResult]:
        """Execute a batch of LLM requests and return results.

        Results are returned in the same order as the input requests.
        Each result includes the original request index and either a
        response or an error.

        Args:
            client: The LLMClient to use for completions.
            requests: List of LLM requests to execute.

        Returns:
            List of LLMResult in the same order as requests.
        """
        ...


class SequentialExecutor(LLMExecutor):
    """Execute LLM requests one at a time (default, current behavior)."""

    def execute(
        self,
        client: LLMClient,
        requests: list[LLMRequest],
    ) -> list[LLMResult]:
        """Execute requests sequentially, in order."""
        results: list[LLMResult] = []
        for i, req in enumerate(requests):
            try:
                response = client.complete(
                    messages=req.messages,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                    artifact_desc=req.artifact_desc,
                )
                results.append(LLMResult(index=i, response=response))
            except Exception as exc:
                results.append(LLMResult(index=i, error=exc))
        return results


class ConcurrentExecutor(LLMExecutor):
    """Execute LLM requests concurrently using a thread pool.

    Uses concurrent.futures.ThreadPoolExecutor with a configurable
    concurrency limit (semaphore). Includes exponential backoff on
    rate limit (429) errors and per-request timeout.

    Args:
        max_concurrency: Maximum number of concurrent LLM calls (default 10).
        timeout: Per-request timeout in seconds (default 120).
        max_retries: Maximum retries on rate limit errors (default 3).
        base_delay: Base delay in seconds for exponential backoff (default 1.0).
    """

    def __init__(
        self,
        max_concurrency: int = 10,
        timeout: float = 120.0,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        self.max_concurrency = max(1, max_concurrency)
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._semaphore = Semaphore(self.max_concurrency)

    def execute(
        self,
        client: LLMClient,
        requests: list[LLMRequest],
    ) -> list[LLMResult]:
        """Execute requests concurrently with rate limiting and retries."""
        if not requests:
            return []

        # Pre-allocate results list to maintain ordering
        results: list[LLMResult | None] = [None] * len(requests)

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures = {}
            for i, req in enumerate(requests):
                future = pool.submit(self._execute_one, client, req, i)
                futures[future] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result(timeout=self.timeout)
                    results[idx] = result
                except TimeoutError:
                    results[idx] = LLMResult(
                        index=idx,
                        error=TimeoutError(
                            f"Request {idx} ({requests[idx].artifact_desc}) "
                            f"timed out after {self.timeout}s"
                        ),
                    )
                except Exception as exc:
                    results[idx] = LLMResult(index=idx, error=exc)

        # Safety: ensure no None entries
        for i, r in enumerate(results):
            if r is None:
                results[i] = LLMResult(
                    index=i,
                    error=RuntimeError(f"Request {i} produced no result"),
                )

        return results  # type: ignore[return-value]

    def _execute_one(
        self,
        client: LLMClient,
        request: LLMRequest,
        index: int,
    ) -> LLMResult:
        """Execute a single request with semaphore gating and retry logic."""
        self._semaphore.acquire()
        try:
            return self._execute_with_retry(client, request, index)
        finally:
            self._semaphore.release()

    def _execute_with_retry(
        self,
        client: LLMClient,
        request: LLMRequest,
        index: int,
    ) -> LLMResult:
        """Execute with exponential backoff on rate limit errors."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = client.complete(
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    artifact_desc=request.artifact_desc,
                )
                return LLMResult(index=index, response=response)
            except RuntimeError as exc:
                # LLMClient wraps rate limit errors as RuntimeError with
                # "Failed to process ... after 2 attempts" or
                # "LLM API error ...". Check if it's a rate limit case.
                error_msg = str(exc).lower()
                if "rate" in error_msg or "429" in error_msg:
                    last_error = exc
                    if attempt < self.max_retries:
                        delay = self.base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                        time.sleep(delay)
                        continue
                # Non-rate-limit RuntimeError — don't retry
                return LLMResult(index=index, error=exc)
            except Exception as exc:
                return LLMResult(index=index, error=exc)

        return LLMResult(
            index=index,
            error=last_error or RuntimeError(
                f"Request {index} ({request.artifact_desc}) failed after "
                f"{self.max_retries + 1} attempts"
            ),
        )


def create_executor(concurrency: int = 1, **kwargs) -> LLMExecutor:
    """Factory to create the appropriate executor.

    Args:
        concurrency: Number of concurrent requests. 1 = sequential (default).
        **kwargs: Additional arguments passed to ConcurrentExecutor.

    Returns:
        SequentialExecutor if concurrency <= 1, ConcurrentExecutor otherwise.
    """
    if concurrency <= 1:
        return SequentialExecutor()
    return ConcurrentExecutor(max_concurrency=concurrency, **kwargs)
