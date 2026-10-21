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

    # The only supported provider is "ollama". The placeholders below mean "not configured".
    llm_provider: str = "your_provider_here"
    # Kept for compatibility. The Ollama provider does not use an API key.
    llm_api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    llm_model: str = "your_model_name_here"

    # Local Ollama server. Resume text is sent here when --llm is used.
    ollama_base_url: str = "http://localhost:11434"
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    # JSON for one resume is short. A higher limit only matters for models that think first.
    llm_max_output_tokens: int = Field(default=2048, ge=64, le=32768)

    embedding_model: str = "your_embedding_model_here"
    # Optional task prefixes for models that expect them, such as "search_query:" and
    # "search_document:" for nomic-embed-text. Empty means none.
    embedding_query_prefix: str = Field(default="", max_length=100)
    embedding_document_prefix: str = Field(default="", max_length=100)

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