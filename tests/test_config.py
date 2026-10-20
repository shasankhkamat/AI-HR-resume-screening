
from pathlib import Path

from hr_screening.config import Settings, get_settings


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


def test_absolute_paths_are_preserved() -> None:
    settings = Settings(_env_file=None)
    absolute_path = Path("C:/temporary/resumes")

    assert settings.resolve_path(absolute_path) == absolute_path


def test_settings_are_cached() -> None:
    get_settings.cache_clear()

    try:
        first = get_settings()
        second = get_settings()

        assert first is second
    finally:
        get_settings.cache_clear()
