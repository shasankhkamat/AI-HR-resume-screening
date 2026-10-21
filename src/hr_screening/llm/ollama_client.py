"""Adapter for a local Ollama server, using only the standard library.

It sends one non streaming request to POST /api/chat and returns the assistant text
unchanged. Parsing, validation and evidence checks stay in llm/extractor.py.

Privacy design:
- Requests ignore proxy environment variables and never follow redirects, so resume text
  goes only to the configured URL.
- Only http and https URLs are accepted.
- Errors use fixed messages. Prompts, replies and response bodies are never logged or put in
  exception text, and provider exceptions are not chained.
"""

import http.client
import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from urllib.parse import urlsplit

from hr_screening.errors import (
    LLMConfigError,
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)

Transport = Callable[[str, bytes, float], tuple[int, bytes]]
Sleep = Callable[[float], None]

CHAT_PATH = "/api/chat"
MAX_BODY_BYTES = 1_048_576
MAX_RETRIES_LIMIT = 5
_ERROR_BODY_BYTES = 4096
_BACKOFF_SECONDS = 0.5
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_TRANSIENT_STATUS = frozenset({500, 502, 503, 504})
_NOT_REACHABLE = (
    "The model server could not be reached. Check that Ollama is running and that "
    "OLLAMA_BASE_URL is correct."
)
_CONNECTION_LOST = "The connection to the model server was lost."


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so a POST body is never forwarded to another address."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


# An empty ProxyHandler means "no proxy", whatever the environment says.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def _urllib_transport(url: str, body: bytes, timeout: float) -> tuple[int, bytes]:
    """POST ``body`` and return (status, body). Network failures raise OSError subclasses."""
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return response.status, response.read(MAX_BODY_BYTES + 1)
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, exc.read(MAX_BODY_BYTES + 1)
        finally:
            exc.close()
    except urllib.error.URLError as exc:
        reason = exc.reason
        raise (reason if isinstance(reason, OSError) else OSError("connection failed")) from None


def parse_base_url(base_url: str) -> str:
    message = "OLLAMA_BASE_URL must be an http or https address such as http://localhost:11434."
    try:
        parts = urlsplit(base_url.strip())
        host = parts.hostname
        _ = parts.port  # raises ValueError for an invalid port
    except ValueError:
        raise LLMConfigError(message) from None
    if parts.scheme not in ("http", "https") or not host:
        raise LLMConfigError(message)
    if parts.username or parts.password:
        raise LLMConfigError("OLLAMA_BASE_URL must not contain a user name or password.")
    if parts.query or parts.fragment:
        raise LLMConfigError(message)
    return base_url.strip().rstrip("/")


def base_url_is_local(base_url: str) -> bool:
    """True if the URL points at this computer (localhost, 127.0.0.1 or ::1)."""
    try:
        host = urlsplit(base_url.strip()).hostname
    except ValueError:
        return False
    return host is not None and host.lower() in _LOCAL_HOSTS


def is_cloud_model(model: str) -> bool:
    lowered = model.lower()
    return lowered.endswith(("-cloud", ":cloud"))


class OllamaClient:
    """Implements LLMClient for a local Ollama server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        max_output_tokens: int = 2048,
        transport: Transport | None = None,
        sleep: Sleep = time.sleep,
    ) -> None:
        self.base_url = parse_base_url(base_url)
        self.model = model.strip()
        if not self.model:
            raise LLMConfigError("LLM_MODEL is not set.")
        if is_cloud_model(self.model):
            raise LLMConfigError(
                "LLM_MODEL looks like an Ollama cloud model. Cloud models run on remote servers, "
                "so resume text would leave this computer. Use a model downloaded to this machine."
            )
        if timeout_seconds <= 0:
            raise LLMConfigError("LLM_TIMEOUT_SECONDS must be greater than 0.")
        if not 0 <= max_retries <= MAX_RETRIES_LIMIT:
            raise LLMConfigError(f"LLM_MAX_RETRIES must be between 0 and {MAX_RETRIES_LIMIT}.")
        if max_output_tokens < 1:
            raise LLMConfigError("LLM_MAX_OUTPUT_TOKENS must be at least 1.")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
        self._url = self.base_url + CHAT_PATH
        self._transport = transport or _urllib_transport
        self._sleep = sleep

    def __repr__(self) -> str:
        return f"OllamaClient(base_url={self.base_url!r}, model={self.model!r})"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": self.max_output_tokens},
        }
        raw = self._send(json.dumps(payload).encode("utf-8"))
        return _extract_content(raw)

    def _send(self, body: bytes) -> bytes:
        return send_with_retries(
            self._transport,
            self._url,
            body,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            sleep=self._sleep,
        )


def send_with_retries(
    transport: Transport,
    url: str,
    body: bytes,
    *,
    timeout_seconds: float,
    max_retries: int,
    sleep: Sleep,
) -> bytes:
    """POST ``body`` and return the reply body, retrying transient failures a bounded number
    of times. Errors use fixed messages, and provider exceptions are never chained.

    Shared by the chat client and the embedding client, so both behave the same way.
    """
    attempts = max_retries + 1
    for attempt in range(1, attempts + 1):
        error: LLMError
        try:
            status, raw = transport(url, body, timeout_seconds)
        except TimeoutError:
            error, transient = LLMTimeoutError("The model server did not answer in time."), True
        except ConnectionRefusedError:
            raise LLMConnectionError(_NOT_REACHABLE) from None
        except (ConnectionResetError, ConnectionAbortedError, http.client.HTTPException):
            error, transient = LLMConnectionError(_CONNECTION_LOST), True
        except OSError:
            raise LLMConnectionError(_NOT_REACHABLE) from None
        else:
            if status == 200:
                return raw
            error, transient = _status_error(status, raw)
        if not transient or attempt == attempts:
            raise error from None
        logger.warning(
            "Model server request failed (%s); retry %d of %d",
            type(error).__name__,
            attempt,
            max_retries,
        )
        sleep(_BACKOFF_SECONDS * attempt)
    raise LLMError("The model request failed.")  # pragma: no cover - loop always returns


def _status_error(status: int, raw: bytes) -> tuple[LLMError, bool]:
    """Map an HTTP error status to (error, is_transient)."""
    if status == 404:
        if _says_model_missing(raw):
            return (
                LLMModelNotFoundError(
                    "The configured model is not installed. Run 'ollama list' to see installed "
                    "models, or 'ollama pull <model>' to download one."
                ),
                False,
            )
        return LLMError("The model server returned HTTP 404. Check OLLAMA_BASE_URL."), False
    if status == 429:
        return LLMRateLimitError("The model server is rate limiting requests."), True
    if status in _TRANSIENT_STATUS:
        return LLMError(f"The model server failed (HTTP {status})."), True
    return LLMError(f"The model server rejected the request (HTTP {status})."), False


def _says_model_missing(raw: bytes) -> bool:
    """Only a yes or no is taken from the error body. The text itself is never kept."""
    try:
        data = json.loads(raw[:_ERROR_BODY_BYTES])
    except (ValueError, RecursionError):
        return False
    message = data.get("error") if isinstance(data, dict) else None
    return isinstance(message, str) and "not found" in message.lower()


def _extract_content(raw: bytes) -> str:
    """Return message.content unchanged, or raise LLMResponseError."""
    if len(raw) > MAX_BODY_BYTES:
        raise LLMResponseError("The model server reply was too large.")
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError):
        raise LLMResponseError("The model server reply was not valid JSON.") from None
    if not isinstance(data, dict):
        raise LLMResponseError("The model server reply was not a JSON object.")
    if "error" in data:
        raise LLMError("The model server reported an error.")
    message = data.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or data.get("done") is False:
        raise LLMResponseError("The model server reply was incomplete or malformed.")
    if data.get("done_reason") == "length":
        raise LLMResponseError(
            "The model stopped at the output token limit, so its reply is incomplete. "
            "Raise LLM_MAX_OUTPUT_TOKENS or use a different model."
        )
    if not content.strip():
        raise LLMResponseError("The model returned an empty reply.")
    return content