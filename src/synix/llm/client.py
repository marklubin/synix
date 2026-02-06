"""OpenAI-compatible LLM client for Synix.

Adapted from KP3's pattern. Works with any OpenAI-compatible API:
- DeepSeek (default)
- OpenAI
- Groq
- Local vLLM instances
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for OpenAI-compatible LLM API."""

    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"

    @classmethod
    def from_env(
        cls,
        api_key_var: str = "LLM_API_KEY",
        base_url_var: str = "LLM_BASE_URL",
        model_var: str = "LLM_MODEL",
    ) -> LLMConfig:
        """Load configuration from environment variables.

        Falls back to SYNIX_LLM_* and DEEPSEEK_API_KEY if primary vars not set.
        """
        # Try multiple env var patterns
        api_key = (
            os.getenv(api_key_var)
            or os.getenv("SYNIX_LLM_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or ""
        )
        if not api_key:
            msg = f"{api_key_var} environment variable is required"
            raise ValueError(msg)

        base_url = (
            os.getenv(base_url_var)
            or os.getenv("SYNIX_LLM_BASE_URL")
            or os.getenv("DEEPSEEK_BASE_URL")
            or cls.base_url
        )

        model = (
            os.getenv(model_var)
            or os.getenv("SYNIX_LLM_MODEL")
            or os.getenv("DEEPSEEK_MODEL")
            or cls.model
        )

        return cls(api_key=api_key, base_url=base_url, model=model)


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens


@dataclass
class LLMClient:
    """Sync client for OpenAI-compatible APIs."""

    config: LLMConfig
    _client: "OpenAI" = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the OpenAI client."""
        from openai import OpenAI

        self._client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    @classmethod
    def from_env(cls) -> LLMClient:
        """Create client from environment variables."""
        return cls(config=LLMConfig.from_env())

    @classmethod
    def from_settings(cls) -> LLMClient:
        """Create client from Synix settings."""
        from synix.config import get_settings

        settings = get_settings()
        if not settings.llm_api_key:
            msg = "SYNIX_LLM_API_KEY not configured"
            raise ValueError(msg)

        config = LLMConfig(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
        )
        return cls(config=config)

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float = 120.0,
    ) -> LLMResponse:
        """Generate a completion.

        Args:
            prompt: User prompt to complete.
            system_prompt: Optional system prompt.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.
            timeout: Request timeout in seconds.

        Returns:
            LLMResponse with content and token counts.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.debug("LLM request: model=%s, messages=%d", self.config.model, len(messages))

        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )

        choice = response.choices[0]
        content = choice.message.content or ""

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        logger.debug(
            "LLM response: tokens=%d+%d, content_len=%d",
            input_tokens,
            output_tokens,
            len(content),
        )

        return LLMResponse(
            content=content,
            model=self.config.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
