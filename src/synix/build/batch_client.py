"""Batch LLM client — drop-in replacement for LLMClient using OpenAI Batch API.

Instead of making synchronous LLM calls, queues requests into batch state and
raises control-flow exceptions. The runner catches these to orchestrate the
batch lifecycle (collect → submit → poll → download → resume).
"""

from __future__ import annotations

import io
import json
import logging

from synix.build.batch_state import BatchState
from synix.build.cassette import compute_cassette_key
from synix.build.llm_client import LLMResponse
from synix.core.config import LLMConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Control-flow exceptions
# ---------------------------------------------------------------------------


class BatchCollecting(Exception):
    """Request queued, batch not yet submitted."""


class BatchInProgress(Exception):
    """Batch submitted, awaiting results."""

    def __init__(self, batch_id: str, status: str):
        self.batch_id = batch_id
        self.status = status
        super().__init__(f"Batch {batch_id} is {status}")


class BatchRequestFailed(Exception):
    """Individual request failed in a completed batch."""

    def __init__(self, custom_id: str, error: dict):
        self.custom_id = custom_id
        self.error = error
        super().__init__(f"Request {custom_id[:12]}... failed: {error}")


# ---------------------------------------------------------------------------
# BatchLLMClient
# ---------------------------------------------------------------------------


class BatchLLMClient:
    """Drop-in replacement for LLMClient that queues requests for batch submission.

    Same ``complete()`` signature as ``LLMClient`` so transforms are unaware
    they're running in batch mode. The control-flow exceptions signal the runner
    to manage the batch lifecycle.

    Args:
        config: LLMConfig for this layer (required by ``_logged_complete``).
        batch_state: Shared batch state for this build instance.
        layer_name: Name of the layer being processed.
        cassette_responses: Pre-seeded results from cassettes/batch_responses.json
            for demo replay. Maps request key → response dict.
    """

    def __init__(
        self,
        config: LLMConfig,
        batch_state: BatchState,
        layer_name: str,
        cassette_responses: dict | None = None,
    ):
        self.config = config
        self._batch_state = batch_state
        self._layer_name = layer_name
        self._cassette_responses = cassette_responses

    def complete(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        artifact_desc: str = "artifact",
    ) -> LLMResponse:
        """Queue a request or return a cached result.

        Returns:
            LLMResponse if the result is already available.

        Raises:
            BatchCollecting: Request was queued, batch not yet submitted.
            BatchInProgress: Batch is submitted and awaiting completion.
            BatchRequestFailed: Request failed in a completed batch.
        """
        resolved_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        resolved_temperature = temperature if temperature is not None else self.config.temperature

        key = compute_cassette_key(
            self.config.provider,
            self.config.model,
            messages,
            resolved_max_tokens,
            resolved_temperature,
        )

        # 1. Check batch_state results (from previous download)
        result = self._batch_state.get_result(key)
        if result:
            return LLMResponse(
                content=result["content"],
                model=result["model"],
                input_tokens=result.get("tokens", {}).get("input", 0),
                output_tokens=result.get("tokens", {}).get("output", 0),
                total_tokens=(result.get("tokens", {}).get("input", 0) + result.get("tokens", {}).get("output", 0)),
            )

        # 2. Check cassette responses (demo replay)
        if self._cassette_responses and key in self._cassette_responses:
            r = self._cassette_responses[key]
            tokens = r.get("tokens", {})
            self._batch_state.store_result(key, r["content"], r["model"], tokens)
            return LLMResponse(
                content=r["content"],
                model=r["model"],
                input_tokens=tokens.get("input", 0),
                output_tokens=tokens.get("output", 0),
                total_tokens=tokens.get("input", 0) + tokens.get("output", 0),
            )

        # 3. Check if already in a batch
        batch_id = self._batch_state.get_batch_for_request(key)
        if batch_id:
            batch = self._batch_state.get_batch(batch_id)
            if batch and batch["status"] == "completed":
                # Batch done but key not in results → individual failure
                error = self._batch_state.get_error(key)
                raise BatchRequestFailed(key, error or {"code": "unknown", "message": "Request failed"})
            raise BatchInProgress(batch_id, batch["status"] if batch else "unknown")

        # 4. New request → queue
        body = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": resolved_max_tokens,
            "temperature": resolved_temperature,
        }
        self._batch_state.queue_request(key, self._layer_name, body, artifact_desc)
        raise BatchCollecting()

    def submit_batch(self, layer_name: str) -> str:
        """Submit all pending requests for a layer as an OpenAI batch.

        Returns:
            The batch ID from OpenAI.

        Raises:
            RuntimeError: If no pending requests or API call fails.
        """
        import openai

        pending = self._batch_state.get_pending(layer_name)
        if not pending:
            raise RuntimeError(f"No pending requests for layer {layer_name!r}")

        # Build JSONL content
        lines = []
        keys = list(pending.keys())
        for key in keys:
            entry = pending[key]
            line = {
                "custom_id": key,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": entry["body"],
            }
            lines.append(json.dumps(line, ensure_ascii=False))
        jsonl_content = "\n".join(lines)

        # Upload file
        api_key = self.config.resolve_api_key()
        client = openai.OpenAI(api_key=api_key)

        file_obj = client.files.create(
            file=io.BytesIO(jsonl_content.encode()),
            purpose="batch",
        )

        # Create batch
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        # Record in state
        self._batch_state.record_batch(batch.id, layer_name, keys)
        self._batch_state.save()

        logger.info("Submitted batch %s with %d requests for layer %s", batch.id, len(keys), layer_name)
        return batch.id

    def check_and_download(self, batch_id: str) -> bool:
        """Check batch status and download results if complete.

        Returns:
            True if the batch reached a terminal state (completed/failed/expired).
            False if still in progress.
        """
        import openai

        api_key = self.config.resolve_api_key()
        client = openai.OpenAI(api_key=api_key)

        batch = client.batches.retrieve(batch_id)
        status = batch.status

        if status in ("in_progress", "validating", "finalizing"):
            self._batch_state.update_batch_status(batch_id, status)
            self._batch_state.save()
            return False

        if status == "completed":
            self._batch_state.update_batch_status(batch_id, "completed")
            # Download and parse results
            if batch.output_file_id:
                content = client.files.content(batch.output_file_id)
                self._parse_batch_results(content.text)
            # Check for errors in the error file
            if batch.error_file_id:
                error_content = client.files.content(batch.error_file_id)
                self._parse_batch_errors(error_content.text)
            self._batch_state.save()
            return True

        if status in ("failed", "expired", "cancelled"):
            self._batch_state.update_batch_status(batch_id, status)
            # Mark all keys in this batch as errors
            batch_info = self._batch_state.get_batch(batch_id)
            if batch_info:
                for key in batch_info.get("keys", []):
                    if not self._batch_state.has_result(key):
                        self._batch_state.store_error(key, status, f"Batch {status}")
            self._batch_state.save()
            return True

        # Unknown status — treat as in-progress
        self._batch_state.update_batch_status(batch_id, status)
        self._batch_state.save()
        return False

    def _parse_batch_results(self, jsonl_text: str) -> None:
        """Parse JSONL response and store results."""
        for line in jsonl_text.strip().split("\n"):
            if not line.strip():
                continue
            entry = json.loads(line)
            custom_id = entry.get("custom_id", "")
            response = entry.get("response", {})

            if response.get("status_code") == 200:
                body = response.get("body", {})
                choices = body.get("choices", [])
                usage = body.get("usage", {})
                content = ""
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                model = body.get("model", self.config.model)
                tokens = {
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                }
                self._batch_state.store_result(custom_id, content, model, tokens)
            else:
                error = entry.get("error", {})
                self._batch_state.store_error(
                    custom_id,
                    error.get("code", "unknown"),
                    error.get("message", f"HTTP {response.get('status_code', 'unknown')}"),
                )

    def _parse_batch_errors(self, jsonl_text: str) -> None:
        """Parse error JSONL file and store errors."""
        for line in jsonl_text.strip().split("\n"):
            if not line.strip():
                continue
            entry = json.loads(line)
            custom_id = entry.get("custom_id", "")
            error = entry.get("error", {})
            if custom_id and not self._batch_state.has_result(custom_id):
                self._batch_state.store_error(
                    custom_id,
                    error.get("code", "unknown"),
                    error.get("message", "Unknown error"),
                )
