"""Unified LLM client wrapping both Anthropic and OpenAI SDKs."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

from synix.core.config import LLMConfig


@dataclass
class LLMResponse:
    """Response from an LLM completion call."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


# ---------------------------------------------------------------------------
# OpenAI model family detection
# ---------------------------------------------------------------------------

_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4")
"""Reasoning models: no temperature, use max_completion_tokens."""

_NEW_TOKEN_PARAM_PREFIXES = ("gpt-5",)
"""Non-reasoning models that require max_completion_tokens instead of max_tokens."""


def _is_reasoning_model(model: str) -> bool:
    """Return True for OpenAI reasoning models (o1, o3, o4 family)."""
    return any(model.startswith(p) for p in _REASONING_MODEL_PREFIXES)


def _needs_max_completion_tokens(model: str) -> bool:
    """Return True for models that require max_completion_tokens instead of max_tokens."""
    return _is_reasoning_model(model) or any(model.startswith(p) for p in _NEW_TOKEN_PARAM_PREFIXES)


class LLMClient:
    """Unified LLM client that dispatches to Anthropic or OpenAI SDKs.

    Supports four providers:
    - "anthropic": Uses the anthropic SDK
    - "openai": Uses the openai SDK with OpenAI's default base URL
    - "deepseek": Uses the openai SDK with DeepSeek's base URL
    - "openai-compatible": Uses the openai SDK with a custom base_url
      (for Ollama, vLLM, etc.)

    Includes retry logic: 1 retry on transient errors (rate limit, timeout, connection).
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = self._create_client()

    def _create_client(self):
        """Create the underlying SDK client based on provider."""
        api_key = self.config.resolve_api_key()

        if self.config.provider == "anthropic":
            import anthropic

            kwargs = {}
            if api_key:
                kwargs["api_key"] = api_key
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            return anthropic.Anthropic(**kwargs)

        elif self.config.provider in ("openai", "deepseek", "openai-compatible"):
            import openai

            kwargs = {}
            if api_key:
                kwargs["api_key"] = api_key
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            elif self.config.provider == "deepseek":
                kwargs["base_url"] = "https://api.deepseek.com"
            elif self.config.provider == "openai-compatible" and not self.config.base_url:
                raise ValueError("openai-compatible provider requires base_url to be set")
            return openai.OpenAI(**kwargs)

        else:
            raise ValueError(
                f"Unknown LLM provider: {self.config.provider!r}. "
                f"Supported: 'anthropic', 'openai', 'deepseek', 'openai-compatible'"
            )

    def complete(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        artifact_desc: str = "artifact",
    ) -> LLMResponse:
        """Send a completion request with retry on transient errors.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            max_tokens: Override max_tokens from config.
            temperature: Override temperature from config.
            artifact_desc: Human-readable description for error messages.

        Returns:
            LLMResponse with content and token usage.
        """
        resolved_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        resolved_temperature = temperature if temperature is not None else self.config.temperature

        if self.config.provider == "anthropic":
            return self._complete_anthropic(messages, resolved_max_tokens, resolved_temperature, artifact_desc)
        else:
            return self._complete_openai(messages, resolved_max_tokens, resolved_temperature, artifact_desc)

    def _complete_anthropic(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        artifact_desc: str,
    ) -> LLMResponse:
        """Complete using the Anthropic SDK with retry."""
        import anthropic

        for attempt in range(2):
            try:
                response = self._client.messages.create(
                    model=self.config.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=messages,
                )
                return LLMResponse(
                    content=response.content[0].text,
                    model=response.model if hasattr(response, "model") else self.config.model,
                    input_tokens=getattr(response.usage, "input_tokens", 0),
                    output_tokens=getattr(response.usage, "output_tokens", 0),
                    total_tokens=(
                        getattr(response.usage, "input_tokens", 0) + getattr(response.usage, "output_tokens", 0)
                    ),
                )
            except (
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.APITimeoutError,
            ) as exc:
                if attempt == 0:
                    print(
                        f"[synix] Transient error processing {artifact_desc}, retrying in 5s: {exc}",
                        file=sys.stderr,
                    )
                    time.sleep(5)
                else:
                    raise RuntimeError(f"Failed to process {artifact_desc} after 2 attempts: {exc}") from exc
            except anthropic.APIError as exc:
                raise RuntimeError(f"LLM API error processing {artifact_desc}: {exc}") from exc

        # Unreachable, but satisfies type checker
        raise RuntimeError(f"Failed to process {artifact_desc}")

    def _complete_openai(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        artifact_desc: str,
    ) -> LLMResponse:
        """Complete using the OpenAI SDK with retry."""
        import openai

        for attempt in range(2):
            try:
                kwargs: dict = {
                    "model": self.config.model,
                    "messages": messages,
                }
                if _needs_max_completion_tokens(self.config.model):
                    kwargs["max_completion_tokens"] = max_tokens
                else:
                    kwargs["max_tokens"] = max_tokens
                if not _is_reasoning_model(self.config.model):
                    kwargs["temperature"] = temperature

                response = self._client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                content = choice.message.content or ""
                usage = response.usage
                return LLMResponse(
                    content=content,
                    model=response.model if response.model else self.config.model,
                    input_tokens=usage.prompt_tokens if usage else 0,
                    output_tokens=usage.completion_tokens if usage else 0,
                    total_tokens=usage.total_tokens if usage else 0,
                )
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as exc:
                if attempt == 0:
                    print(
                        f"[synix] Transient error processing {artifact_desc}, retrying in 5s: {exc}",
                        file=sys.stderr,
                    )
                    time.sleep(5)
                else:
                    raise RuntimeError(f"Failed to process {artifact_desc} after 2 attempts: {exc}") from exc
            except openai.APIError as exc:
                raise RuntimeError(f"LLM API error processing {artifact_desc}: {exc}") from exc

        # Unreachable, but satisfies type checker
        raise RuntimeError(f"Failed to process {artifact_desc}")
