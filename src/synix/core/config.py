"""Configuration resolution — CLI > env > config file > pipeline defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass


def redact_api_key(key: str | None) -> str | None:
    """Redact an API key, showing only the first 4 and last 4 characters.

    Returns None if the key is None, or the redacted string otherwise.
    Short keys (8 chars or fewer) are fully redacted as '****'.
    """
    if key is None:
        return None
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


@dataclass
class LLMConfig:
    """Configuration for LLM provider.

    Supports four providers:
    - "anthropic": Anthropic Claude models (default)
    - "openai": OpenAI GPT models
    - "deepseek": DeepSeek models (uses OpenAI SDK with DeepSeek base URL)
    - "openai-compatible": Any OpenAI-compatible API (Ollama, vLLM, etc.)

    Config precedence: explicit config > env vars > defaults.

    API key resolution order:
    1. Explicit ``api_key`` field
    2. Custom env var via ``api_key_env`` (e.g. ``"TOGETHER_API_KEY"``)
    3. ``SYNIX_API_KEY`` (universal fallback for all providers)
    4. Provider-specific env var (ANTHROPIC_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY)

    Other environment variables:
    - SYNIX_LLM_PROVIDER: override provider
    - SYNIX_LLM_MODEL: override model
    - SYNIX_LLM_BASE_URL: override base_url
    """

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    temperature: float | None = 0.3
    max_tokens: int = 1024
    max_completion_tokens: int | None = None  # OpenAI reasoning/newer models; overrides max_tokens
    base_url: str | None = None  # For OpenAI-compatible APIs (Ollama, vLLM, DeepSeek)
    api_key: str | None = None  # Override; defaults to env var
    api_key_env: str | None = None  # Custom env var name (e.g. "TOGETHER_API_KEY")
    default_headers: dict[str, str] | None = None  # Extra HTTP headers for API calls
    timeout: float = 300.0  # Per-request HTTP timeout in seconds (connect + read)

    @classmethod
    def from_dict(cls, data: dict) -> LLMConfig:
        """Create LLMConfig from a dict, applying env var overrides.

        Config precedence: explicit dict values > env vars > class defaults.
        """
        # Start with class defaults
        config = cls()

        # Apply env var overrides (middle precedence)
        env_provider = os.environ.get("SYNIX_LLM_PROVIDER")
        if env_provider:
            config.provider = env_provider
        env_model = os.environ.get("SYNIX_LLM_MODEL")
        if env_model:
            config.model = env_model
        env_base_url = os.environ.get("SYNIX_LLM_BASE_URL")
        if env_base_url:
            config.base_url = env_base_url

        # Apply explicit dict values (highest precedence)
        if "provider" in data:
            config.provider = data["provider"]
        if "model" in data:
            config.model = data["model"]
        if "temperature" in data:
            config.temperature = data["temperature"]
        if "max_tokens" in data:
            config.max_tokens = data["max_tokens"]
        if "max_completion_tokens" in data:
            config.max_completion_tokens = data["max_completion_tokens"]
        if "base_url" in data:
            config.base_url = data["base_url"]
        if "api_key" in data:
            config.api_key = data["api_key"]
        if "api_key_env" in data:
            config.api_key_env = data["api_key_env"]
        if "default_headers" in data:
            config.default_headers = data["default_headers"]
        if "timeout" in data:
            config.timeout = data["timeout"]

        return config

    def resolve_api_key(self) -> str | None:
        """Resolve the API key: explicit > api_key_env > SYNIX_API_KEY > provider default."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            val = os.environ.get(self.api_key_env)
            if val:
                return val
        synix_key = os.environ.get("SYNIX_API_KEY")
        if synix_key:
            return synix_key
        if self.provider == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY")
        if self.provider == "deepseek":
            return os.environ.get("DEEPSEEK_API_KEY")
        # openai and openai-compatible both use OPENAI_API_KEY
        return os.environ.get("OPENAI_API_KEY")


@dataclass
class EmbeddingConfig:
    """Configuration for embedding provider.

    Supports two backends:
    - "fastembed": Local ONNX-based embeddings (default, no API key needed)
    - "openai": OpenAI API embeddings

    API key resolution order:
    1. Explicit ``api_key`` field
    2. Custom env var via ``api_key_env``
    3. ``SYNIX_EMBEDDING_API_KEY``
    4. ``OPENAI_API_KEY``
    """

    provider: str = "fastembed"
    model: str = "BAAI/bge-small-en-v1.5"
    dimensions: int | None = None
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None  # Custom env var name
    batch_size: int = 16
    concurrency: int = 4  # concurrent API calls (OpenAI only; ignored for FastEmbed)

    @classmethod
    def from_dict(cls, data: dict) -> EmbeddingConfig:
        """Create EmbeddingConfig from a dict."""
        config = cls()
        if "provider" in data:
            config.provider = data["provider"]
        if "model" in data:
            config.model = data["model"]
        if "dimensions" in data:
            config.dimensions = data["dimensions"]
        if "base_url" in data:
            config.base_url = data["base_url"]
        if "api_key" in data:
            config.api_key = data["api_key"]
        if "api_key_env" in data:
            config.api_key_env = data["api_key_env"]
        if "batch_size" in data:
            config.batch_size = data["batch_size"]
        if "concurrency" in data:
            config.concurrency = data["concurrency"]
        return config

    def resolve_api_key(self) -> str | None:
        """Resolve the API key: explicit > api_key_env > SYNIX_EMBEDDING_API_KEY > OPENAI_API_KEY."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            val = os.environ.get(self.api_key_env)
            if val:
                return val
        synix_key = os.environ.get("SYNIX_EMBEDDING_API_KEY")
        if synix_key:
            return synix_key
        return os.environ.get("OPENAI_API_KEY")

    def resolve_base_url(self) -> str | None:
        """Resolve base URL: explicit > env var."""
        if self.base_url:
            return self.base_url
        return os.environ.get("OPENAI_BASE_URL")
