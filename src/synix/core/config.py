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

    Environment variables:
    - SYNIX_LLM_PROVIDER: override provider
    - SYNIX_LLM_MODEL: override model
    - SYNIX_LLM_BASE_URL: override base_url
    - ANTHROPIC_API_KEY: API key for Anthropic provider
    - OPENAI_API_KEY: API key for OpenAI / OpenAI-compatible providers
    - DEEPSEEK_API_KEY: API key for DeepSeek provider
    """

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.3
    max_tokens: int = 1024
    base_url: str | None = None  # For OpenAI-compatible APIs (Ollama, vLLM, DeepSeek)
    api_key: str | None = None   # Override; defaults to env var

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
        if "base_url" in data:
            config.base_url = data["base_url"]
        if "api_key" in data:
            config.api_key = data["api_key"]

        return config

    def resolve_api_key(self) -> str | None:
        """Resolve the API key: explicit > env var per provider."""
        if self.api_key:
            return self.api_key
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
    - "openai": OpenAI API embeddings (requires OPENAI_API_KEY)
    """

    provider: str = "fastembed"
    model: str = "BAAI/bge-small-en-v1.5"
    dimensions: int = 384
    base_url: str | None = None
    api_key: str | None = None
    batch_size: int = 64
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
        if "batch_size" in data:
            config.batch_size = data["batch_size"]
        if "concurrency" in data:
            config.concurrency = data["concurrency"]
        return config

    def resolve_api_key(self) -> str | None:
        """Resolve the API key: explicit > env var."""
        if self.api_key:
            return self.api_key
        return os.environ.get("OPENAI_API_KEY")
