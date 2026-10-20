
"""Centralized application configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Validated settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["development", "testing", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    llm_provider: str = "your_provider_here"
    llm_api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    llm_model: str = "your_model_name_here"

    embedding_model: str = "your_embedding_model_here"

    chroma_persist_dir: Path = Path("data/chroma")
    incoming_resume_dir: Path = Path("data/incoming")

    @property
    def project_root(self) -> Path:
        """Return the absolute project root directory."""
        return PROJECT_ROOT

    def resolve_path(self, path: Path) -> Path:
        """Resolve a relative path from the project root."""
        if path.is_absolute():
            return path
        return self.project_root / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached application settings instance."""
    return Settings()
