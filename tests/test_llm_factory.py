"""Provider selection and configuration errors. No network is used."""

import pytest

from hr_screening.config import Settings
from hr_screening.errors import LLMConfigError
from hr_screening.llm import ollama_client
from hr_screening.llm.factory import build_llm_client
from hr_screening.llm.ollama_client import OllamaClient

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
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("the factory must not make a network call")

    monkeypatch.setattr(ollama_client, "_urllib_transport", refuse)


def settings(**values) -> Settings:
    return Settings(_env_file=None, **values)


def test_ollama_provider_builds_an_ollama_client():
    client = build_llm_client(settings(llm_provider="ollama", llm_model="my-model"))
    assert isinstance(client, OllamaClient)
    assert client.model == "my-model"


def test_settings_are_passed_to_the_client():
    client = build_llm_client(
        settings(
            llm_provider="ollama",
            llm_model="my-model",
            ollama_base_url="http://127.0.0.1:9999/",
            llm_timeout_seconds=45,
            llm_max_retries=1,
            llm_max_output_tokens=512,
        )
    )
    assert client.base_url == "http://127.0.0.1:9999"
    assert client.timeout_seconds == 45
    assert client.max_retries == 1
    assert client.max_output_tokens == 512


def test_documented_defaults_are_used():
    client = build_llm_client(settings(llm_provider="ollama", llm_model="m"))
    assert client.base_url == "http://localhost:11434"
    assert client.timeout_seconds == 30
    assert client.max_retries == 2
    assert client.max_output_tokens == 2048


@pytest.mark.parametrize("provider", ["Ollama", " OLLAMA ", "ollama"])
def test_provider_name_ignores_case_and_spaces(provider):
    assert isinstance(
        build_llm_client(settings(llm_provider=provider, llm_model="m")), OllamaClient
    )


def test_no_api_key_is_needed_and_it_is_never_used():
    config = settings(llm_provider="ollama", llm_model="m", llm_api_key="sk-secret-value")
    client = build_llm_client(config)
    assert "sk-secret-value" not in repr(client)
    assert "sk-secret-value" not in repr(vars(client))
    assert build_llm_client(settings(llm_provider="ollama", llm_model="m")) is not None


@pytest.mark.parametrize(
    "provider", ["your_provider_here", "", "   ", "openai", "anthropic", "ollama2"]
)
def test_unsupported_provider_is_a_configuration_error(provider):
    config = settings(llm_provider=provider, llm_model="m", llm_api_key="sk-secret-value")
    with pytest.raises(LLMConfigError, match="LLM_PROVIDER") as caught:
        build_llm_client(config)
    assert "sk-secret-value" not in str(caught.value)
    assert provider.strip() == "" or provider not in str(caught.value)


@pytest.mark.parametrize("model", ["your_model_name_here", "", "   ", "Your_Model_Name_Here"])
def test_missing_model_is_a_configuration_error(model):
    with pytest.raises(LLMConfigError, match="LLM_MODEL"):
        build_llm_client(settings(llm_provider="ollama", llm_model=model))


def test_default_settings_are_not_a_usable_configuration():
    with pytest.raises(LLMConfigError, match="LLM_PROVIDER"):
        build_llm_client(settings())


@pytest.mark.parametrize(
    "url", ["", "localhost:11434", "ftp://localhost", "file:///etc/passwd", "http://u:p@host"]
)
def test_bad_base_url_is_a_configuration_error(url):
    with pytest.raises(LLMConfigError, match="OLLAMA_BASE_URL"):
        build_llm_client(settings(llm_provider="ollama", llm_model="m", ollama_base_url=url))


def test_cloud_model_is_a_configuration_error():
    with pytest.raises(LLMConfigError, match="cloud"):
        build_llm_client(settings(llm_provider="ollama", llm_model="big-model:120b-cloud"))