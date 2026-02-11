"""Tests for the unified LLM client and LLMConfig."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import synix.build.llm_transforms  # noqa: F401

# Import to trigger @register_transform decorators
import synix.build.parse_transform  # noqa: F401
from synix.build.llm_client import LLMClient, LLMResponse
from synix.core.config import EmbeddingConfig, LLMConfig

# ---------------------------------------------------------------------------
# LLMConfig tests
# ---------------------------------------------------------------------------


class TestLLMConfig:
    """Tests for LLMConfig dataclass and factory methods."""

    def test_defaults_are_anthropic(self):
        """Default config uses Anthropic provider."""
        config = LLMConfig()
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-20250514"
        assert config.temperature == 0.3
        assert config.max_tokens == 1024
        assert config.base_url is None
        assert config.api_key is None

    def test_from_dict_empty_gives_defaults(self):
        """Empty dict produces default config."""
        config = LLMConfig.from_dict({})
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-20250514"

    def test_from_dict_explicit_overrides(self):
        """Explicit dict values override defaults."""
        config = LLMConfig.from_dict(
            {
                "provider": "openai",
                "model": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 2048,
                "base_url": "https://api.example.com/v1",
                "api_key": "sk-test-123",
            }
        )
        assert config.provider == "openai"
        assert config.model == "gpt-4o"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.base_url == "https://api.example.com/v1"
        assert config.api_key == "sk-test-123"

    def test_from_dict_env_var_override(self, monkeypatch):
        """Environment variables override defaults but not explicit values."""
        monkeypatch.setenv("SYNIX_LLM_PROVIDER", "openai")
        monkeypatch.setenv("SYNIX_LLM_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("SYNIX_LLM_BASE_URL", "https://env.example.com/v1")

        # No explicit values — env wins over defaults
        config = LLMConfig.from_dict({})
        assert config.provider == "openai"
        assert config.model == "gpt-4o-mini"
        assert config.base_url == "https://env.example.com/v1"

    def test_from_dict_explicit_beats_env(self, monkeypatch):
        """Explicit dict values take precedence over env vars."""
        monkeypatch.setenv("SYNIX_LLM_PROVIDER", "openai")
        monkeypatch.setenv("SYNIX_LLM_MODEL", "gpt-4o-mini")

        config = LLMConfig.from_dict(
            {
                "provider": "anthropic",
                "model": "claude-haiku-4-5-20251001",
            }
        )
        assert config.provider == "anthropic"
        assert config.model == "claude-haiku-4-5-20251001"

    def test_from_dict_partial_override(self, monkeypatch):
        """Mix of env and explicit — each wins at its level."""
        monkeypatch.setenv("SYNIX_LLM_PROVIDER", "openai")
        # No SYNIX_LLM_MODEL set, but explicit model in dict
        monkeypatch.delenv("SYNIX_LLM_MODEL", raising=False)

        config = LLMConfig.from_dict({"model": "custom-model"})
        assert config.provider == "openai"  # from env
        assert config.model == "custom-model"  # from explicit

    def test_resolve_api_key_explicit(self):
        """Explicit api_key is used if set."""
        config = LLMConfig(api_key="explicit-key")
        assert config.resolve_api_key() == "explicit-key"

    def test_resolve_api_key_anthropic_env(self, monkeypatch):
        """Anthropic provider falls back to ANTHROPIC_API_KEY env var."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key-from-env")
        config = LLMConfig(provider="anthropic")
        assert config.resolve_api_key() == "ant-key-from-env"

    def test_resolve_api_key_openai_env(self, monkeypatch):
        """OpenAI provider falls back to OPENAI_API_KEY env var."""
        monkeypatch.setenv("OPENAI_API_KEY", "oai-key-from-env")
        config = LLMConfig(provider="openai")
        assert config.resolve_api_key() == "oai-key-from-env"

    def test_resolve_api_key_openai_compatible_env(self, monkeypatch):
        """OpenAI-compatible provider uses OPENAI_API_KEY env var."""
        monkeypatch.setenv("OPENAI_API_KEY", "compat-key")
        config = LLMConfig(provider="openai-compatible")
        assert config.resolve_api_key() == "compat-key"

    def test_resolve_api_key_none_when_missing(self, monkeypatch):
        """Returns None when no key is set anywhere."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = LLMConfig(provider="anthropic")
        assert config.resolve_api_key() is None

    def test_resolve_api_key_explicit_beats_env(self, monkeypatch):
        """Explicit api_key takes precedence over env var."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        config = LLMConfig(provider="anthropic", api_key="explicit-key")
        assert config.resolve_api_key() == "explicit-key"

    def test_backward_compat_no_provider_key(self):
        """Dict without 'provider' defaults to anthropic — backward compatible."""
        config = LLMConfig.from_dict(
            {
                "model": "claude-sonnet-4-20250514",
                "temperature": 0.3,
                "max_tokens": 1024,
            }
        )
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-20250514"


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig dataclass."""

    def test_defaults(self):
        config = EmbeddingConfig()
        assert config.provider == "fastembed"
        assert config.model == "BAAI/bge-small-en-v1.5"
        assert config.dimensions == 384
        assert config.batch_size == 64
        assert config.concurrency == 4

    def test_from_dict(self):
        config = EmbeddingConfig.from_dict(
            {
                "provider": "openai",
                "model": "text-embedding-3-large",
                "dimensions": 1024,
                "base_url": "https://embed.example.com/v1",
            }
        )
        assert config.model == "text-embedding-3-large"
        assert config.dimensions == 1024
        assert config.base_url == "https://embed.example.com/v1"

    def test_resolve_api_key_explicit(self):
        config = EmbeddingConfig(api_key="embed-key")
        assert config.resolve_api_key() == "embed-key"

    def test_resolve_api_key_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "oai-key")
        config = EmbeddingConfig()
        assert config.resolve_api_key() == "oai-key"


# ---------------------------------------------------------------------------
# LLMClient tests
# ---------------------------------------------------------------------------


class TestLLMClientInit:
    """Tests for LLMClient initialization with different providers."""

    def test_init_anthropic_provider(self, monkeypatch):
        """Anthropic provider creates an anthropic.Anthropic client."""
        mock_anthropic_cls = MagicMock(return_value=MagicMock())
        monkeypatch.setattr("anthropic.Anthropic", mock_anthropic_cls)

        config = LLMConfig(provider="anthropic", api_key="test-key")
        client = LLMClient(config)

        mock_anthropic_cls.assert_called_once_with(api_key="test-key")
        assert client._client is not None

    def test_init_openai_provider(self, monkeypatch):
        """OpenAI provider creates an openai.OpenAI client."""
        mock_openai_cls = MagicMock(return_value=MagicMock())
        monkeypatch.setattr("openai.OpenAI", mock_openai_cls)

        config = LLMConfig(provider="openai", api_key="oai-test-key")
        client = LLMClient(config)

        mock_openai_cls.assert_called_once_with(api_key="oai-test-key")
        assert client._client is not None

    def test_init_openai_compatible_provider(self, monkeypatch):
        """OpenAI-compatible provider passes base_url to openai.OpenAI."""
        mock_openai_cls = MagicMock(return_value=MagicMock())
        monkeypatch.setattr("openai.OpenAI", mock_openai_cls)

        config = LLMConfig(
            provider="openai-compatible",
            api_key="compat-key",
            base_url="http://localhost:11434/v1",
        )
        client = LLMClient(config)

        mock_openai_cls.assert_called_once_with(
            api_key="compat-key",
            base_url="http://localhost:11434/v1",
        )

    def test_init_openai_compatible_requires_base_url(self, monkeypatch):
        """OpenAI-compatible without base_url raises ValueError."""
        mock_openai_cls = MagicMock(return_value=MagicMock())
        monkeypatch.setattr("openai.OpenAI", mock_openai_cls)

        config = LLMConfig(provider="openai-compatible")
        with pytest.raises(ValueError, match="base_url"):
            LLMClient(config)

    def test_init_unknown_provider_raises(self):
        """Unknown provider raises ValueError."""
        config = LLMConfig(provider="bedrock")
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMClient(config)

    def test_init_anthropic_no_api_key(self, monkeypatch):
        """Anthropic client is created without api_key when none is available."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        mock_anthropic_cls = MagicMock(return_value=MagicMock())
        monkeypatch.setattr("anthropic.Anthropic", mock_anthropic_cls)

        config = LLMConfig(provider="anthropic")
        LLMClient(config)

        # Called with no kwargs (api_key is None, so not passed)
        mock_anthropic_cls.assert_called_once_with()

    def test_init_anthropic_with_base_url(self, monkeypatch):
        """Anthropic provider passes base_url when set."""
        mock_anthropic_cls = MagicMock(return_value=MagicMock())
        monkeypatch.setattr("anthropic.Anthropic", mock_anthropic_cls)

        config = LLMConfig(
            provider="anthropic",
            api_key="test-key",
            base_url="https://custom-anthropic.example.com",
        )
        LLMClient(config)

        mock_anthropic_cls.assert_called_once_with(
            api_key="test-key",
            base_url="https://custom-anthropic.example.com",
        )


class TestLLMClientCompleteAnthropic:
    """Tests for LLMClient.complete() with Anthropic provider."""

    def _make_mock_anthropic_response(self, text="Mock response", model="claude-sonnet-4-20250514"):
        response = MagicMock()
        response.content = [MagicMock(text=text)]
        response.model = model
        response.usage = MagicMock(input_tokens=50, output_tokens=25)
        return response

    def test_complete_anthropic_basic(self, monkeypatch):
        """Basic completion with Anthropic provider."""
        mock_response = self._make_mock_anthropic_response("Hello from Claude")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)

        config = LLMConfig(provider="anthropic")
        client = LLMClient(config)
        result = client.complete(
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello from Claude"
        assert result.model == "claude-sonnet-4-20250514"
        assert result.input_tokens == 50
        assert result.output_tokens == 25
        assert result.total_tokens == 75

    def test_complete_anthropic_passes_params(self, monkeypatch):
        """Verify model, max_tokens, temperature are passed to the SDK."""
        mock_response = self._make_mock_anthropic_response()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)

        config = LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001", temperature=0.5, max_tokens=512)
        client = LLMClient(config)
        client.complete(
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=256,  # override
            temperature=0.9,  # override
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
        assert call_kwargs["max_tokens"] == 256
        assert call_kwargs["temperature"] == 0.9

    def test_complete_anthropic_uses_config_defaults(self, monkeypatch):
        """When no overrides given, config defaults are used."""
        mock_response = self._make_mock_anthropic_response()
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)

        config = LLMConfig(provider="anthropic", model="test-model", temperature=0.1, max_tokens=999)
        client = LLMClient(config)
        client.complete(messages=[{"role": "user", "content": "Test"}])

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["max_tokens"] == 999
        assert call_kwargs["temperature"] == 0.1

    def test_complete_anthropic_retry_on_rate_limit(self, monkeypatch):
        """Retries once on RateLimitError then succeeds."""
        import anthropic

        mock_response = self._make_mock_anthropic_response("Success after retry")
        mock_client = MagicMock()
        call_count = {"n": 0}

        def side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise anthropic.RateLimitError(
                    message="Rate limited",
                    response=MagicMock(status_code=429),
                    body={"error": {"message": "Rate limited"}},
                )
            return mock_response

        mock_client.messages.create.side_effect = side_effect
        monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)
        # Patch time.sleep to avoid actual delay
        monkeypatch.setattr("synix.build.llm_client.time.sleep", lambda _: None)

        config = LLMConfig(provider="anthropic")
        client = LLMClient(config)
        result = client.complete(messages=[{"role": "user", "content": "Test"}])

        assert result.content == "Success after retry"
        assert call_count["n"] == 2

    def test_complete_anthropic_retry_exhausted(self, monkeypatch):
        """Raises RuntimeError after 2 failed attempts."""
        import anthropic

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.RateLimitError(
            message="Rate limited",
            response=MagicMock(status_code=429),
            body={"error": {"message": "Rate limited"}},
        )
        monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)
        monkeypatch.setattr("synix.build.llm_client.time.sleep", lambda _: None)

        config = LLMConfig(provider="anthropic")
        client = LLMClient(config)
        with pytest.raises(RuntimeError, match="Failed to process.*after 2 attempts"):
            client.complete(
                messages=[{"role": "user", "content": "Test"}],
                artifact_desc="test artifact",
            )

    def test_complete_anthropic_api_error_no_retry(self, monkeypatch):
        """Non-transient APIError raises immediately without retry."""
        import anthropic

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.APIError(
            message="Bad request",
            request=MagicMock(),
            body={"error": {"message": "Bad request"}},
        )
        monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)

        config = LLMConfig(provider="anthropic")
        client = LLMClient(config)
        with pytest.raises(RuntimeError, match="LLM API error"):
            client.complete(messages=[{"role": "user", "content": "Test"}])

        # Only called once — no retry for non-transient errors
        assert mock_client.messages.create.call_count == 1


class TestLLMClientCompleteOpenAI:
    """Tests for LLMClient.complete() with OpenAI provider."""

    def _make_mock_openai_response(self, text="Mock OpenAI response", model="gpt-4o"):
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = text
        response.choices = [choice]
        response.model = model
        response.usage = MagicMock(
            prompt_tokens=40,
            completion_tokens=20,
            total_tokens=60,
        )
        return response

    def test_complete_openai_basic(self, monkeypatch):
        """Basic completion with OpenAI provider."""
        mock_response = self._make_mock_openai_response("Hello from GPT")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        monkeypatch.setattr("openai.OpenAI", lambda **kwargs: mock_client)

        config = LLMConfig(provider="openai", model="gpt-4o")
        client = LLMClient(config)
        result = client.complete(
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello from GPT"
        assert result.model == "gpt-4o"
        assert result.input_tokens == 40
        assert result.output_tokens == 20
        assert result.total_tokens == 60

    def test_complete_openai_passes_params(self, monkeypatch):
        """Verify model, max_tokens, temperature are passed to the SDK."""
        mock_response = self._make_mock_openai_response()
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        monkeypatch.setattr("openai.OpenAI", lambda **kwargs: mock_client)

        config = LLMConfig(provider="openai", model="gpt-4o-mini", temperature=0.5, max_tokens=512)
        client = LLMClient(config)
        client.complete(
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=256,
            temperature=0.9,
        )

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["max_tokens"] == 256
        assert call_kwargs["temperature"] == 0.9

    def test_complete_openai_retry_on_rate_limit(self, monkeypatch):
        """Retries once on RateLimitError then succeeds."""
        import openai

        mock_response = self._make_mock_openai_response("Success after retry")
        mock_client = MagicMock()
        call_count = {"n": 0}

        def side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise openai.RateLimitError(
                    message="Rate limited",
                    response=MagicMock(status_code=429),
                    body={"error": {"message": "Rate limited"}},
                )
            return mock_response

        mock_client.chat.completions.create.side_effect = side_effect
        monkeypatch.setattr("openai.OpenAI", lambda **kwargs: mock_client)
        monkeypatch.setattr("synix.build.llm_client.time.sleep", lambda _: None)

        config = LLMConfig(provider="openai", model="gpt-4o")
        client = LLMClient(config)
        result = client.complete(messages=[{"role": "user", "content": "Test"}])

        assert result.content == "Success after retry"
        assert call_count["n"] == 2

    def test_complete_openai_retry_exhausted(self, monkeypatch):
        """Raises RuntimeError after 2 failed attempts."""
        import openai

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = openai.RateLimitError(
            message="Rate limited",
            response=MagicMock(status_code=429),
            body={"error": {"message": "Rate limited"}},
        )
        monkeypatch.setattr("openai.OpenAI", lambda **kwargs: mock_client)
        monkeypatch.setattr("synix.build.llm_client.time.sleep", lambda _: None)

        config = LLMConfig(provider="openai", model="gpt-4o")
        client = LLMClient(config)
        with pytest.raises(RuntimeError, match="Failed to process.*after 2 attempts"):
            client.complete(
                messages=[{"role": "user", "content": "Test"}],
                artifact_desc="test artifact",
            )

    def test_complete_openai_api_error_no_retry(self, monkeypatch):
        """Non-transient APIError raises immediately without retry."""
        import openai

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = openai.APIError(
            message="Bad request",
            request=MagicMock(),
            body={"error": {"message": "Bad request"}},
        )
        monkeypatch.setattr("openai.OpenAI", lambda **kwargs: mock_client)

        config = LLMConfig(provider="openai", model="gpt-4o")
        client = LLMClient(config)
        with pytest.raises(RuntimeError, match="LLM API error"):
            client.complete(messages=[{"role": "user", "content": "Test"}])

        assert mock_client.chat.completions.create.call_count == 1

    def test_complete_openai_compatible(self, monkeypatch):
        """OpenAI-compatible provider uses the OpenAI code path."""
        mock_response = self._make_mock_openai_response("Ollama response", "llama3")
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        monkeypatch.setattr("openai.OpenAI", lambda **kwargs: mock_client)

        config = LLMConfig(
            provider="openai-compatible",
            model="llama3",
            base_url="http://localhost:11434/v1",
        )
        client = LLMClient(config)
        result = client.complete(messages=[{"role": "user", "content": "Hello"}])

        assert result.content == "Ollama response"
        assert result.model == "llama3"


class TestLLMClientTokenUsage:
    """Tests for token usage tracking in LLMResponse."""

    def test_anthropic_token_usage(self, monkeypatch):
        """Anthropic response captures token usage correctly."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = MagicMock(input_tokens=123, output_tokens=456)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)

        config = LLMConfig(provider="anthropic")
        client = LLMClient(config)
        result = client.complete(messages=[{"role": "user", "content": "Test"}])

        assert result.input_tokens == 123
        assert result.output_tokens == 456
        assert result.total_tokens == 579

    def test_openai_token_usage(self, monkeypatch):
        """OpenAI response captures token usage correctly."""
        mock_response = MagicMock()
        choice = MagicMock()
        choice.message.content = "Response"
        mock_response.choices = [choice]
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        monkeypatch.setattr("openai.OpenAI", lambda **kwargs: mock_client)

        config = LLMConfig(provider="openai", model="gpt-4o")
        client = LLMClient(config)
        result = client.complete(messages=[{"role": "user", "content": "Test"}])

        assert result.input_tokens == 200
        assert result.output_tokens == 100
        assert result.total_tokens == 300

    def test_openai_no_usage_returns_zeros(self, monkeypatch):
        """When OpenAI response has no usage, returns zeros."""
        mock_response = MagicMock()
        choice = MagicMock()
        choice.message.content = "Response"
        mock_response.choices = [choice]
        mock_response.model = "gpt-4o"
        mock_response.usage = None

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        monkeypatch.setattr("openai.OpenAI", lambda **kwargs: mock_client)

        config = LLMConfig(provider="openai", model="gpt-4o")
        client = LLMClient(config)
        result = client.complete(messages=[{"role": "user", "content": "Test"}])

        assert result.input_tokens == 0
        assert result.output_tokens == 0
        assert result.total_tokens == 0


class TestLLMClientProviderDispatch:
    """Tests verifying the correct SDK is used per provider."""

    def test_anthropic_uses_anthropic_sdk(self, monkeypatch):
        """Anthropic provider calls anthropic.Anthropic().messages.create."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Result")]
        mock_response.model = "claude-sonnet-4-20250514"
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        monkeypatch.setattr("anthropic.Anthropic", lambda **kwargs: mock_client)

        config = LLMConfig(provider="anthropic")
        client = LLMClient(config)
        client.complete(messages=[{"role": "user", "content": "Test"}])

        mock_client.messages.create.assert_called_once()
        # Ensure the OpenAI path was NOT called
        assert not hasattr(mock_client, "chat") or not mock_client.chat.completions.create.called

    def test_openai_uses_openai_sdk(self, monkeypatch):
        """OpenAI provider calls openai.OpenAI().chat.completions.create."""
        mock_response = MagicMock()
        choice = MagicMock()
        choice.message.content = "Result"
        mock_response.choices = [choice]
        mock_response.model = "gpt-4o"
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        monkeypatch.setattr("openai.OpenAI", lambda **kwargs: mock_client)

        config = LLMConfig(provider="openai", model="gpt-4o")
        client = LLMClient(config)
        client.complete(messages=[{"role": "user", "content": "Test"}])

        mock_client.chat.completions.create.assert_called_once()


class TestLLMClientBackwardCompat:
    """Tests verifying backward compatibility with existing mock_llm fixture."""

    def test_transforms_still_work_with_mock_llm(self, mock_llm, sample_artifacts):
        """Existing mock_llm fixture still works — transforms produce correct output."""
        from synix.build.transforms import get_transform

        transform = get_transform("episode_summary")
        transcripts = [a for a in sample_artifacts if a.artifact_type == "transcript"]

        results = transform.execute(transcripts[:1], {"llm_config": {}})

        assert len(results) == 1
        ep = results[0]
        assert ep.artifact_type == "episode"
        assert ep.content_hash.startswith("sha256:")
        assert len(mock_llm) == 1  # LLM was called exactly once

    def test_pipeline_llm_config_without_provider(self, mock_llm, sample_artifacts):
        """Pipeline configs that omit 'provider' default to anthropic and work."""
        from synix.build.transforms import get_transform

        # Old-style config: no 'provider' key
        old_config = {
            "llm_config": {
                "model": "claude-sonnet-4-20250514",
                "temperature": 0.3,
                "max_tokens": 1024,
            }
        }

        transform = get_transform("episode_summary")
        transcripts = [a for a in sample_artifacts if a.artifact_type == "transcript"]
        results = transform.execute(transcripts[:1], old_config)

        assert len(results) == 1
        assert results[0].artifact_type == "episode"
