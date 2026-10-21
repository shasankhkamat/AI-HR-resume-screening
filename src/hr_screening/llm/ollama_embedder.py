"""Embeddings from a local Ollama server (POST /api/embed), using only the standard library.

It reuses the chat client's HTTP code (``send_with_retries``), so it has the same privacy
behavior: no proxies, no redirects, only http and https addresses, bounded retries for
transient failures only, fixed error messages, and no texts, vectors or response bodies in
logs or exceptions. The returned vectors are checked with ``validate_vectors``.

The model name comes from the EMBEDDING_MODEL setting. Nothing is hardcoded.
"""

import json
import time
from collections.abc import Sequence

from hr_screening.errors import (
    EmbeddingError,
    LLMConfigError,
    LLMError,
    LLMResponseError,
)
from hr_screening.llm import ollama_client
from hr_screening.llm.ollama_client import (
    MAX_BODY_BYTES,
    MAX_RETRIES_LIMIT,
    Sleep,
    Transport,
    is_cloud_model,
    parse_base_url,
    send_with_retries,
)
from hr_screening.rag.embeddings import normalize_prefix, validate_vectors

EMBED_PATH = "/api/embed"


class OllamaEmbedder:
    """Implements the Embedder interface for a local Ollama server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        transport: Transport | None = None,
        sleep: Sleep = time.sleep,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        self.base_url = parse_base_url(base_url)
        self.model = model.strip()
        if not self.model:
            raise LLMConfigError("EMBEDDING_MODEL is not set.")
        if is_cloud_model(self.model):
            raise LLMConfigError(
                "EMBEDDING_MODEL looks like an Ollama cloud model. Cloud models run on remote "
                "servers, so resume text would leave this computer. Use a model downloaded to "
                "this machine."
            )
        if timeout_seconds <= 0:
            raise LLMConfigError("LLM_TIMEOUT_SECONDS must be greater than 0.")
        if not 0 <= max_retries <= MAX_RETRIES_LIMIT:
            raise LLMConfigError(f"LLM_MAX_RETRIES must be between 0 and {MAX_RETRIES_LIMIT}.")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        # Read by SemanticRetriever. This class does not add them itself, because embed()
        # cannot tell a query from a resume passage.
        self.query_prefix = normalize_prefix(query_prefix)
        self.document_prefix = normalize_prefix(document_prefix)
        self._url = self.base_url + EMBED_PATH
        self._transport = transport or ollama_client._urllib_transport
        self._sleep = sleep

    def __repr__(self) -> str:
        return f"OllamaEmbedder(base_url={self.base_url!r}, model={self.model!r})"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per text, in order. No request is made for an empty list."""
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingError("Cannot embed empty text.")
        body = json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8")
        raw = send_with_retries(
            self._transport,
            self._url,
            body,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            sleep=self._sleep,
        )
        return validate_vectors(_extract_embeddings(raw), len(texts))


def _extract_embeddings(raw: bytes) -> object:
    """The 'embeddings' value of the reply, still unchecked."""
    if len(raw) > MAX_BODY_BYTES:
        raise LLMResponseError("The embedding server reply was too large.")
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError):
        raise LLMResponseError("The embedding server reply was not valid JSON.") from None
    if not isinstance(data, dict):
        raise LLMResponseError("The embedding server reply was not a JSON object.")
    if "error" in data:
        raise LLMError("The embedding server reported an error.")
    if "embeddings" not in data:
        raise LLMResponseError("The embedding server reply was incomplete or malformed.")
    return data["embeddings"]