"""The Ollama embedding client and the embedder factory. No Ollama, model or network is used."""

import json
import logging

import pytest

from hr_screening.config import Settings
from hr_screening.errors import (
    EmbeddingError,
    LLMConfigError,
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from hr_screening.llm import ollama_client
from hr_screening.llm.ollama_client import MAX_BODY_BYTES
from hr_screening.llm.ollama_embedder import OllamaEmbedder
from hr_screening.rag.factory import build_embedder

SECRET = "sk-test-secret-123"
TEXT = "Jane Doe built dashboards at Example Corp"
BASE = "http://localhost:11434"
_ENV = (
    "EMBEDDING_MODEL",
    "OLLAMA_BASE_URL",
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES",
    "LLM_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


def ok(vectors) -> tuple[int, bytes]:
    return 200, json.dumps({"model": "m", "embeddings": vectors}).encode("utf-8")


class FakeTransport:
    def __init__(self, *results) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, dict, float]] = []

    def __call__(self, url: str, body: bytes, timeout: float):
        self.calls.append((url, json.loads(body), timeout))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def make(transport, **kwargs) -> OllamaEmbedder:
    kwargs.setdefault("sleep", lambda seconds: None)
    return OllamaEmbedder(BASE, "embed-model", transport=transport, **kwargs)


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture
def captured_logs():
    logger = logging.getLogger("hr_screening")
    handler = _Capture()
    saved = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler.messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(saved)


# --- the request ------------------------------------------------------------------------------


def test_request_url_and_payload():
    transport = FakeTransport(ok([[1, 0], [0, 1]]))
    embedder = OllamaEmbedder(
        BASE + "/", " embed-model ", timeout_seconds=12.5, transport=transport
    )
    embedder.embed(["first text", "second text"])
    url, payload, timeout = transport.calls[0]
    assert url == "http://localhost:11434/api/embed"
    assert payload == {"model": "embed-model", "input": ["first text", "second text"]}
    assert timeout == 12.5


def test_vectors_come_back_in_order_as_floats():
    vectors = make(FakeTransport(ok([[1, 2], [3, 4.5]]))).embed(["a", "b"])
    assert vectors == [[1.0, 2.0], [3.0, 4.5]]


def test_an_empty_list_makes_no_request():
    transport = FakeTransport()
    assert make(transport).embed([]) == []
    assert transport.calls == []


@pytest.mark.parametrize(
    "texts",
    [["ok", "  "], [""], ["ok", None], [5]],
    ids=["blank", "empty", "none", "int"],
)
def test_blank_or_non_text_input_is_rejected_without_a_request(texts):
    transport = FakeTransport()
    with pytest.raises(EmbeddingError, match="empty text"):
        make(transport).embed(texts)
    assert transport.calls == []


def test_the_default_transport_is_the_shared_one(monkeypatch):
    calls = []

    def recorder(url, body, timeout):
        calls.append(url)
        return ok([[1.0]])

    monkeypatch.setattr(ollama_client, "_urllib_transport", recorder)
    OllamaEmbedder(BASE, "embed-model").embed(["x"])
    assert calls == ["http://localhost:11434/api/embed"]


def test_construction_makes_no_network_call():
    transport = FakeTransport()
    make(transport)
    assert transport.calls == []


# --- replies that cannot be used --------------------------------------------------------------

BAD_REPLIES = [
    pytest.param(b"not json", id="not_json"),
    pytest.param(b"\xff\xfe", id="bad_utf8"),
    pytest.param(b"[]", id="json_array"),
    pytest.param(b"null", id="json_null"),
    pytest.param(b"{}", id="no_key"),
    pytest.param(b'{"embeddings": null}', id="null"),
    pytest.param(b'{"embeddings": "abc"}', id="string"),
    pytest.param(b'{"embeddings": [[1.0, 2.0]]}', id="wrong_count"),
    pytest.param(b'{"embeddings": [[1.0], []]}', id="empty_vector"),
    pytest.param(b'{"embeddings": [[1.0, 2.0], [1.0]]}', id="ragged"),
    pytest.param(b'{"embeddings": [["a"], ["b"]]}', id="text_values"),
    pytest.param(b'{"embeddings": [[NaN], [1.0]]}', id="nan"),
    pytest.param(b'{"embeddings": [[Infinity], [1.0]]}', id="infinity"),
    pytest.param(b'{"embeddings": [[1e30], [1.0]]}', id="huge"),
    pytest.param(b"[" * 100_000, id="nested_brackets"),
    pytest.param(b"x" * (MAX_BODY_BYTES + 1), id="too_large"),
]


@pytest.mark.parametrize("raw", BAD_REPLIES)
def test_unusable_replies_raise_a_clear_error(raw):
    with pytest.raises((LLMResponseError, EmbeddingError)):
        make(FakeTransport((200, raw))).embed(["a", "b"])


def test_an_error_object_in_a_200_reply_is_an_error():
    raw = json.dumps({"error": f"boom {SECRET}"}).encode()
    with pytest.raises(LLMError, match="reported an error") as caught:
        make(FakeTransport((200, raw))).embed(["a"])
    assert SECRET not in str(caught.value)


# --- failures the shared HTTP code handles ----------------------------------------------------


def test_a_timeout_is_retried_then_succeeds():
    transport = FakeTransport(TimeoutError(), ok([[1.0]]))
    assert make(transport).embed(["a"]) == [[1.0]]
    assert len(transport.calls) == 2


def test_retries_are_bounded():
    transport = FakeTransport(*[TimeoutError() for _ in range(3)])
    with pytest.raises(LLMTimeoutError):
        make(transport, max_retries=2).embed(["a"])
    assert len(transport.calls) == 3


def test_a_server_that_is_not_running_is_not_retried():
    transport = FakeTransport(ConnectionRefusedError())
    with pytest.raises(LLMConnectionError, match="Ollama is running"):
        make(transport).embed(["a"])
    assert len(transport.calls) == 1


def test_a_missing_embedding_model_is_reported_and_not_retried():
    body = json.dumps({"error": 'model "embed-model" not found, try pulling it first'}).encode()
    transport = FakeTransport((404, body))
    with pytest.raises(LLMModelNotFoundError, match="ollama pull"):
        make(transport).embed(["a"])
    assert len(transport.calls) == 1


def test_server_errors_and_rate_limits_are_retried_then_reported():
    with pytest.raises(LLMError, match="HTTP 503"):
        make(FakeTransport(*[(503, b"{}")] * 3)).embed(["a"])
    with pytest.raises(LLMRateLimitError):
        make(FakeTransport(*[(429, b"{}")] * 3)).embed(["a"])


# --- privacy ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "result",
    [
        TimeoutError(f"{SECRET} {TEXT}"),
        ConnectionRefusedError(f"{SECRET} {TEXT}"),
        OSError(f"{SECRET} {TEXT}"),
        (404, json.dumps({"error": f"model not found {SECRET} {TEXT}"}).encode()),
        (500, json.dumps({"error": f"{SECRET} {TEXT}"}).encode()),
        (200, f"{SECRET} {TEXT}".encode()),
        (200, json.dumps({"embeddings": [[f"{SECRET} {TEXT}"]]}).encode()),
        (200, json.dumps({"embeddings": [[1.0], [2.0]], "extra": SECRET}).encode()),
    ],
    ids=["timeout", "refused", "oserror", "missing", "server", "plain", "text_value", "count"],
)
def test_errors_and_logs_never_contain_texts_or_provider_output(result, captured_logs):
    with pytest.raises(LLMError) as caught:
        make(FakeTransport(*[result] * 3)).embed([TEXT])
    error = caught.value
    assert SECRET not in str(error)
    assert "Jane Doe" not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None or error.__suppress_context__
    joined = " ".join(captured_logs)
    assert SECRET not in joined
    assert "Jane Doe" not in joined


def test_successful_calls_log_no_texts_or_vectors(captured_logs):
    make(FakeTransport(TimeoutError(), ok([[0.123456, 0.654321]]))).embed([TEXT])
    joined = " ".join(captured_logs)
    assert "Jane Doe" not in joined
    assert "0.123456" not in joined
    assert "retry 1 of 2" in joined


def test_repr_shows_only_the_address_and_model():
    assert repr(make(FakeTransport())) == (
        "OllamaEmbedder(base_url='http://localhost:11434', model='embed-model')"
    )


# --- construction -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    ["", "localhost:11434", "ftp://localhost", "file:///etc/passwd", "http://u:pw@localhost"],
    ids=["empty", "no_scheme", "ftp", "file", "credentials"],
)
def test_bad_addresses_are_configuration_errors(url):
    with pytest.raises(LLMConfigError):
        OllamaEmbedder(url, "m")


@pytest.mark.parametrize("model", ["x:cloud", "big-model:120b-cloud", "Some-CLOUD"])
def test_cloud_models_are_refused(model):
    with pytest.raises(LLMConfigError, match="EMBEDDING_MODEL"):
        OllamaEmbedder(BASE, model)


@pytest.mark.parametrize("model", ["", "   "])
def test_a_blank_model_is_a_configuration_error(model):
    with pytest.raises(LLMConfigError, match="EMBEDDING_MODEL"):
        OllamaEmbedder(BASE, model)


@pytest.mark.parametrize(
    "kwargs", [{"timeout_seconds": 0}, {"max_retries": -1}, {"max_retries": 6}]
)
def test_bad_numbers_are_configuration_errors(kwargs):
    with pytest.raises(LLMConfigError):
        OllamaEmbedder(BASE, "m", **kwargs)


# --- the factory ------------------------------------------------------------------------------


def settings(**values) -> Settings:
    return Settings(_env_file=None, **values)


def test_the_factory_builds_an_embedder_from_the_settings(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("no network call may happen when building")

    monkeypatch.setattr(ollama_client, "_urllib_transport", refuse)
    embedder = build_embedder(
        settings(
            embedding_model="embed-model",
            ollama_base_url="http://127.0.0.1:9999/",
            llm_timeout_seconds=45,
            llm_max_retries=1,
        )
    )
    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.model == "embed-model"
    assert embedder.base_url == "http://127.0.0.1:9999"
    assert (embedder.timeout_seconds, embedder.max_retries) == (45, 1)


def test_no_llm_settings_or_api_key_are_needed():
    config = settings(embedding_model="embed-model", llm_api_key="sk-secret-value")
    embedder = build_embedder(config)
    assert "sk-secret-value" not in repr(embedder) + repr(vars(embedder))


@pytest.mark.parametrize(
    "model", ["your_embedding_model_here", "", "   ", "Your_Embedding_Model_Here"]
)
def test_a_missing_embedding_model_is_a_clear_error(model):
    with pytest.raises(LLMConfigError, match="EMBEDDING_MODEL is not set"):
        build_embedder(settings(embedding_model=model))


def test_the_default_settings_are_not_usable():
    with pytest.raises(LLMConfigError, match="EMBEDDING_MODEL"):
        build_embedder(settings())


def test_a_cloud_model_or_a_bad_address_is_refused_by_the_factory():
    with pytest.raises(LLMConfigError, match="cloud"):
        build_embedder(settings(embedding_model="big:cloud"))
    with pytest.raises(LLMConfigError, match="OLLAMA_BASE_URL"):
        build_embedder(settings(embedding_model="m", ollama_base_url="ftp://localhost"))