"""Build the embedder chosen by the settings. Ollama is the only supported source."""

from hr_screening.config import Settings
from hr_screening.errors import LLMConfigError
from hr_screening.llm.ollama_embedder import OllamaEmbedder
from hr_screening.rag.embeddings import Embedder

_PLACEHOLDER_MODELS = frozenset({"", "your_embedding_model_here"})


def build_embedder(settings: Settings) -> Embedder:
    """Create an embedder from ``settings`` or raise LLMConfigError with a clear message.

    Nothing is sent over the network here. No API key is read. The server address, timeout
    and retries are the same settings the chat client uses.
    """
    model = settings.embedding_model.strip()
    if model.lower() in _PLACEHOLDER_MODELS:
        raise LLMConfigError(
            "EMBEDDING_MODEL is not set. Download an embedding model with 'ollama pull <name>', "
            "check its exact name with 'ollama list', then set EMBEDDING_MODEL to it."
        )
    return OllamaEmbedder(
        settings.ollama_base_url,
        model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )