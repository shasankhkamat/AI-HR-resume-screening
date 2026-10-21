import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from hr_screening.config import Settings, get_settings

_ENV_NAMES = (
    "APP_ENV",
    "LOG_LEVEL",
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "LLM_MODEL",
    "OLLAMA_BASE_URL",
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES",
    "LLM_MAX_OUTPUT_TOKENS",
    "EMBEDDING_MODEL",
    "CHROMA_PERSIST_DIR",
    "INCOMING_RESUME_DIR",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the defaults tests independent of the shell they run in."""
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_default_settings() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_env == "development"
    assert settings.log_level == "INFO"
    assert settings.llm_provider == "your_provider_here"
    assert settings.llm_api_key.get_secret_value() == ""


def test_relative_paths_resolve_from_project_root() -> None:
    settings = Settings(_env_file=None)

    resolved = settings.resolve_path(Path("data/incoming"))

    assert resolved == settings.project_root / "data/incoming"
    assert resolved.is_absolute()


def test_absolute_paths_are_preserved(tmp_path: Path) -> None:
    settings = Settings(_env_file=None)
    # tmp_path is absolute on every platform, unlike a hard coded drive or root path.
    assert tmp_path.is_absolute()

    assert settings.resolve_path(tmp_path) == tmp_path


@pytest.mark.skipif(sys.platform != "win32", reason="Drive letter paths need Windows")
def test_windows_drive_paths_are_preserved_on_windows() -> None:
    settings = Settings(_env_file=None)
    absolute_path = Path("C:/temporary/resumes")

    assert settings.resolve_path(absolute_path) == absolute_path


@pytest.mark.skipif(sys.platform == "win32", reason="A leading slash is not absolute on Windows")
def test_posix_root_paths_are_preserved_on_posix() -> None:
    settings = Settings(_env_file=None)
    absolute_path = Path("/temporary/resumes")

    assert settings.resolve_path(absolute_path) == absolute_path


@pytest.mark.skipif(sys.platform == "win32", reason="Documents POSIX behavior for drive letters")
def test_drive_letter_paths_are_relative_on_posix() -> None:
    settings = Settings(_env_file=None)
    drive_path = Path("C:/temporary/resumes")

    assert not drive_path.is_absolute()
    assert settings.resolve_path(drive_path) == settings.project_root / drive_path


def test_settings_are_cached() -> None:
    get_settings.cache_clear()

    try:
        first = get_settings()
        second = get_settings()

        assert first is second
    finally:
        get_settings.cache_clear()


def test_ollama_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.llm_timeout_seconds == 30
    assert settings.llm_max_retries == 2
    assert settings.llm_max_output_tokens == 2048


def test_ollama_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "my-model")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9999")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "4096")

    settings = Settings(_env_file=None)

    assert settings.llm_provider == "ollama"
    assert settings.llm_model == "my-model"
    assert settings.ollama_base_url == "http://127.0.0.1:9999"
    assert settings.llm_timeout_seconds == 120
    assert settings.llm_max_retries == 0
    assert settings.llm_max_output_tokens == 4096


@pytest.mark.parametrize(
    "values",
    [
        {"llm_timeout_seconds": 0},
        {"llm_timeout_seconds": -1},
        {"llm_timeout_seconds": 601},
        {"llm_max_retries": -1},
        {"llm_max_retries": 6},
        {"llm_max_output_tokens": 63},
        {"llm_max_output_tokens": 40000},
    ],
    ids=[
        "zero_timeout",
        "negative_timeout",
        "long_timeout",
        "negative_retries",
        "many_retries",
        "few_tokens",
        "many_tokens",
    ],
)
def test_invalid_llm_limits_are_rejected(values: dict) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **values)


def test_no_api_key_is_needed_and_a_given_one_is_not_shown() -> None:
    assert Settings(_env_file=None).llm_api_key.get_secret_value() == ""

    settings = Settings(_env_file=None, llm_api_key="sk-secret-value")

    assert "sk-secret-value" not in repr(settings)
    assert "sk-secret-value" not in str(settings)