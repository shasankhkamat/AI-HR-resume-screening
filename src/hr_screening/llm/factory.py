"""Build the LLM client chosen by the settings. Ollama is the only supported provider."""

from hr_screening.config import Settings
from hr_screening.errors import LLMConfigError
from hr_screening.llm.base import LLMClient
from hr_screening.llm.ollama_client import OllamaClient

_PLACEHOLDER_MODELS = frozenset({"", "your_model_name_here"})


def build_llm_client(settings: Settings) -> LLMClient:
    """Create a client from ``settings`` or raise LLMConfigError with a clear message.

    Nothing is sent over the network here. No API key is read: Ollama does not need one.
    """
    if settings.llm_provider.strip().lower() != "ollama":
        raise LLMConfigError(
            "LLM_PROVIDER must be set to 'ollama'. No other provider is supported."
        )
    model = settings.llm_model.strip()
    if model.lower() in _PLACEHOLDER_MODELS:
        raise LLMConfigError(
            "LLM_MODEL is not set. Run 'ollama list' to see installed models, "
            "then set LLM_MODEL to one of them."
        )
    return OllamaClient(
        settings.ollama_base_url,
        model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        max_output_tokens=settings.llm_max_output_tokens,
    )