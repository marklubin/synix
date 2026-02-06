"""Configuration settings for Synix.

Two-layer storage architecture:
- control.db: Pipeline definitions, run tracking, step configs
- artifacts.db: Records, provenance, FTS index
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="SYNIX_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage directory (default: .synix in current directory)
    storage_dir: Path = Field(default=Path(".synix"))

    # LLM settings
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"

    @property
    def control_db_path(self) -> Path:
        """Path to control plane database."""
        return self.storage_dir / "control.db"

    @property
    def artifact_db_path(self) -> Path:
        """Path to data plane database."""
        return self.storage_dir / "artifacts.db"

    @property
    def control_db_url(self) -> str:
        """SQLAlchemy URL for control plane database."""
        return f"sqlite:///{self.control_db_path}"

    @property
    def artifact_db_url(self) -> str:
        """SQLAlchemy URL for data plane database."""
        return f"sqlite:///{self.artifact_db_path}"

    def ensure_storage_dir(self) -> None:
        """Create storage directory if it doesn't exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset settings cache (useful for testing)."""
    global _settings
    _settings = None


class _SettingsProxy:
    """Lazy proxy for settings that loads on first access."""

    def __getattr__(self, name: str) -> object:
        return getattr(get_settings(), name)


settings: Settings = _SettingsProxy()  # type: ignore[assignment]
