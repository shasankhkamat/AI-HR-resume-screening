"""Ollama adapter tests. None needs Ollama, a model, an API key or the internet.

Most tests use a fake transport. A few use the real urllib transport against a tiny HTTP
server bound to 127.0.0.1, which never leaves this computer.
"""

import http.client
import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from hr_screening.errors import (
    LLMConfigError,
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from hr_screening.llm.extractor import extract_profile_with_llm
from hr_screening.llm.ollama_client import (
    MAX_BODY_BYTES,
    OllamaClient,
    base_url_is_local,
)
from hr_screening.pipeline import LLM_SUCCESS_NOTE, screen_resume

SECRET = "sk-test-secret-123"
PROMPT = "PROMPT-WITH-RESUME-TEXT Jane Doe"
REPLY = "REPLY-WITH-RESUME-TEXT Jane Doe"
BASE = "http://localhost:11434"
VALID = {
    "skills": ["Python", "SQL"],
    "education": ["B.Tech in Computer Science, Example Institute, 2018"],
    "certifications": ["Example Cloud Practitioner"],
    "projects": [],
    "experience": [
        {"heading": "Data Analyst, Example Corp", "start": "Jan 2019", "end": "Dec 2021"},
        {"heading": "Senior Analyst, Sample Ltd", "start": "Jan 2022", "end": "Present"},
    ],
}


def ok(content: str = "{}", **extra) -> tuple[int, bytes]:
    body = {"model": "m", "message": {"role": "assistant", "content": content}, "done": True}
    body.update(extra)
    return 200, json.dumps(body).encode("utf-8")


class FakeTransport:
    """Plays back results in order. An Exception instance is raised instead of returned."""

    def __init__(self, *results) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, dict, float]] = []

    def __call__(self, url: str, body: bytes, timeout: float):
        self.calls.append((url, json.loads(body), timeout))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def make(transport, **kwargs) -> OllamaClient:
    kwargs.setdefault("sleep", lambda seconds: None)
    return OllamaClient(BASE, "my-model", transport=transport, **kwargs)


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


# --- request ---------------------------------------------------------------------------------


def test_request_url_and_payload():
    transport = FakeTransport(ok())
    client = OllamaClient(
        BASE + "/",
        " my-model ",
        timeout_seconds=12.5,
        max_output_tokens=777,
        transport=transport,
    )
    client.complete("SYS", "USER")
    url, payload, timeout = transport.calls[0]
    assert url == "http://localhost:11434/api/chat"
    assert timeout == 12.5
    assert payload == {
        "model": "my-model",
        "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "USER"},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_predict": 777},
    }


def test_reply_text_is_returned_unchanged():
    content = '  ```json\n{"skills": ["Python"]}\n```\n\n'
    assert make(FakeTransport(ok(content))).complete("s", "u") == content


def test_thinking_text_is_ignored():
    transport = FakeTransport(ok('{"a": 1}', message={"content": '{"a": 1}', "thinking": "hmm"}))
    assert make(transport).complete("s", "u") == '{"a": 1}'


def test_stop_reason_and_missing_reason_are_fine():
    assert make(FakeTransport(ok("x", done_reason="stop"))).complete("s", "u") == "x"
    assert make(FakeTransport(ok("x"))).complete("s", "u") == "x"


# --- retries ---------------------------------------------------------------------------------


def test_timeout_is_retried_then_succeeds():
    sleeps: list[float] = []
    transport = FakeTransport(TimeoutError(), ok("done"))
    client = make(transport, sleep=sleeps.append)
    assert client.complete("s", "u") == "done"
    assert len(transport.calls) == 2
    assert sleeps == [0.5]


def test_retries_are_bounded():
    sleeps: list[float] = []
    transport = FakeTransport(TimeoutError(), TimeoutError(), TimeoutError())
    client = make(transport, max_retries=2, sleep=sleeps.append)
    with pytest.raises(LLMTimeoutError, match="did not answer in time"):
        client.complete("s", "u")
    assert len(transport.calls) == 3
    assert sleeps == [0.5, 1.0]


def test_zero_retries_means_one_attempt():
    transport = FakeTransport(TimeoutError())
    with pytest.raises(LLMTimeoutError):
        make(transport, max_retries=0).complete("s", "u")
    assert len(transport.calls) == 1


@pytest.mark.parametrize(
    "error",
    [ConnectionResetError(), ConnectionAbortedError(), http.client.IncompleteRead(b"")],
    ids=["reset", "aborted", "incomplete"],
)
def test_lost_connection_is_retried(error):
    transport = FakeTransport(error, ok("done"))
    assert make(transport).complete("s", "u") == "done"
    assert len(transport.calls) == 2


def test_lost_connection_gives_up_after_the_retries():
    transport = FakeTransport(*[ConnectionResetError() for _ in range(3)])
    with pytest.raises(LLMConnectionError, match="connection"):
        make(transport).complete("s", "u")
    assert len(transport.calls) == 3


@pytest.mark.parametrize(
    "error",
    [ConnectionRefusedError(), socket.gaierror(), OSError("network down")],
    ids=["refused", "dns", "oserror"],
)
def test_unreachable_server_is_not_retried(error):
    transport = FakeTransport(error)
    with pytest.raises(LLMConnectionError, match="Ollama is running"):
        make(transport).complete("s", "u")
    assert len(transport.calls) == 1


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_server_errors_are_retried(status):
    transport = FakeTransport((status, b"{}"), ok("done"))
    assert make(transport).complete("s", "u") == "done"
    assert len(transport.calls) == 2


def test_server_error_gives_up_with_the_status_code():
    transport = FakeTransport(*[(503, b"{}")] * 3)
    with pytest.raises(LLMError, match="HTTP 503") as caught:
        make(transport).complete("s", "u")
    assert not isinstance(caught.value, LLMModelNotFoundError)
    assert len(transport.calls) == 3


def test_rate_limit_is_retried_then_reported():
    transport = FakeTransport(*[(429, b"{}")] * 3)
    with pytest.raises(LLMRateLimitError):
        make(transport).complete("s", "u")
    assert len(transport.calls) == 3


# --- model missing and other client errors ---------------------------------------------------


def test_missing_model_is_reported_clearly_and_not_retried():
    body = json.dumps({"error": 'model "my-model" not found, try pulling it first'}).encode()
    transport = FakeTransport((404, body))
    with pytest.raises(LLMModelNotFoundError, match="ollama pull"):
        make(transport).complete("s", "u")
    assert len(transport.calls) == 1


@pytest.mark.parametrize("body", [b"", b"not json", b'{"error": "something else"}', b"[]"])
def test_other_404_is_not_called_a_missing_model(body):
    with pytest.raises(LLMError, match="HTTP 404") as caught:
        make(FakeTransport((404, body))).complete("s", "u")
    assert not isinstance(caught.value, LLMModelNotFoundError)


@pytest.mark.parametrize("status", [400, 401, 403, 302, 418])
def test_client_errors_are_not_retried(status):
    transport = FakeTransport((status, b"{}"))
    with pytest.raises(LLMError, match=f"HTTP {status}"):
        make(transport).complete("s", "u")
    assert len(transport.calls) == 1


# --- malformed, empty and truncated replies --------------------------------------------------

MALFORMED = [
    pytest.param(b"not json", id="not_json"),
    pytest.param(b"\xff\xfe\x00", id="bad_utf8"),
    pytest.param(b"[]", id="json_array"),
    pytest.param(b"null", id="json_null"),
    pytest.param(b"{}", id="no_message"),
    pytest.param(b'{"message": "text"}', id="message_not_object"),
    pytest.param(b'{"message": {}}', id="no_content"),
    pytest.param(b'{"message": {"content": null}}', id="content_null"),
    pytest.param(b'{"message": {"content": 5}}', id="content_number"),
    pytest.param(b'{"message": {"content": "x"}, "done": false}', id="not_done"),
    pytest.param(b"[" * 100_000, id="nested_brackets"),
    pytest.param(b"x" * (MAX_BODY_BYTES + 1), id="too_large"),
]


@pytest.mark.parametrize("raw", MALFORMED)
def test_malformed_replies_raise_response_error(raw):
    with pytest.raises(LLMResponseError):
        make(FakeTransport((200, raw))).complete("s", "u")


def test_an_error_object_in_a_200_reply_is_an_error():
    raw = json.dumps({"error": f"boom {SECRET}"}).encode()
    with pytest.raises(LLMError, match="reported an error") as caught:
        make(FakeTransport((200, raw))).complete("s", "u")
    assert SECRET not in str(caught.value)


@pytest.mark.parametrize("content", ["", "   ", "\n\t"], ids=["empty", "spaces", "newline"])
def test_empty_output_is_an_error(content):
    with pytest.raises(LLMResponseError, match="empty"):
        make(FakeTransport(ok(content))).complete("s", "u")


@pytest.mark.parametrize("content", ['{"skills": ["Py', ""], ids=["partial", "nothing"])
def test_output_cut_off_by_the_token_limit_is_an_error(content):
    transport = FakeTransport(ok(content, done_reason="length"))
    with pytest.raises(LLMResponseError, match="token limit"):
        make(transport).complete("s", "u")


# --- privacy of errors and logs --------------------------------------------------------------


@pytest.mark.parametrize(
    "result",
    [
        TimeoutError(f"{SECRET} {PROMPT}"),
        ConnectionRefusedError(f"{SECRET} {PROMPT}"),
        ConnectionResetError(f"{SECRET} {PROMPT}"),
        OSError(f"{SECRET} {PROMPT}"),
        (404, json.dumps({"error": f"model not found {SECRET} {PROMPT}"}).encode()),
        (500, json.dumps({"error": f"{SECRET} {PROMPT}"}).encode()),
        (400, json.dumps({"error": f"{SECRET} {PROMPT}"}).encode()),
        (200, f"{SECRET} {PROMPT}".encode()),
        (200, json.dumps({"message": {"content": ""}, "extra": f"{SECRET} {PROMPT}"}).encode()),
    ],
    ids=[
        "timeout",
        "refused",
        "reset",
        "oserror",
        "missing_model",
        "server_error",
        "bad_request",
        "plain_text",
        "empty_reply",
    ],
)
def test_errors_and_logs_never_contain_provider_text(result, captured_logs):
    with pytest.raises(LLMError) as caught:
        make(FakeTransport(*[result] * 3)).complete(PROMPT, PROMPT)
    error = caught.value
    assert SECRET not in str(error)
    assert "Jane Doe" not in str(error)
    # The provider exception must not appear in a traceback: no cause, and any context hidden.
    assert error.__cause__ is None
    assert error.__context__ is None or error.__suppress_context__
    joined = " ".join(captured_logs)
    assert SECRET not in joined
    assert "Jane Doe" not in joined


def test_successful_calls_log_no_prompt_or_reply(captured_logs):
    make(FakeTransport(TimeoutError(), ok(REPLY))).complete(PROMPT, PROMPT)
    joined = " ".join(captured_logs)
    assert "Jane Doe" not in joined
    assert "retry 1 of 2" in joined


def test_repr_shows_neither_prompts_nor_secrets():
    text = repr(make(FakeTransport()))
    assert text == "OllamaClient(base_url='http://localhost:11434', model='my-model')"


# --- construction ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "",
        "localhost:11434",
        "ftp://localhost",
        "file:///etc/passwd",
        "http://",
        "not a url",
        "http://localhost:99999",
        "http://user:pw@localhost:11434",
        "http://localhost:11434?x=1",
    ],
    ids=["empty", "no_scheme", "ftp", "file", "no_host", "words", "bad_port", "creds", "query"],
)
def test_bad_base_urls_are_configuration_errors(url):
    with pytest.raises(LLMConfigError) as caught:
        OllamaClient(url, "m")
    assert "pw" not in str(caught.value).replace("password", "")


@pytest.mark.parametrize(
    "model",
    ["gpt-oss:120b-cloud", "x:cloud", "Some-Model-CLOUD"],
    ids=["suffix", "tag", "case"],
)
def test_cloud_models_are_refused(model):
    with pytest.raises(LLMConfigError, match="cloud model"):
        OllamaClient(BASE, model)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_seconds": 0},
        {"timeout_seconds": -1},
        {"max_retries": -1},
        {"max_retries": 6},
        {"max_output_tokens": 0},
    ],
    ids=["zero_timeout", "negative_timeout", "negative_retries", "many_retries", "no_tokens"],
)
def test_bad_numbers_are_configuration_errors(kwargs):
    with pytest.raises(LLMConfigError):
        OllamaClient(BASE, "m", **kwargs)


@pytest.mark.parametrize("model", ["", "   "])
def test_blank_model_is_a_configuration_error(model):
    with pytest.raises(LLMConfigError, match="LLM_MODEL"):
        OllamaClient(BASE, model)


def test_construction_makes_no_network_call():
    transport = FakeTransport()
    make(transport)
    assert transport.calls == []


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:11434", True),
        ("http://LOCALHOST", True),
        ("http://127.0.0.1:11434", True),
        ("http://[::1]:11434", True),
        ("http://192.168.1.20:11434", False),
        ("https://ollama.com", False),
        ("http://localhost.evil.example", False),
        ("garbage", False),
    ],
)
def test_base_url_is_local(url, expected):
    assert base_url_is_local(url) is expected


# --- the adapter together with the extractor and pipeline ------------------------------------


def test_adapter_feeds_the_extractor(sample_resume_text, today):
    client = make(FakeTransport(ok(json.dumps(VALID))))
    profile, dropped = extract_profile_with_llm(client, sample_resume_text, today=today)
    assert profile.skills == ["Python", "SQL"]
    assert profile.total_years_experience == 6.9
    assert profile.years_source == "date_ranges"
    assert dropped == 0


def test_invented_items_from_the_model_are_still_discarded(sample_resume_text, today):
    invented = {**VALID, "skills": ["Python", "Kubernetes"]}
    client = make(FakeTransport(ok(json.dumps(invented))))
    profile, dropped = extract_profile_with_llm(client, sample_resume_text, today=today)
    assert profile.skills == ["Python"]
    assert dropped == 1


def test_pipeline_with_adapter_keeps_the_score(
    tmp_path, make_docx, sample_resume_lines, sample_job, today
):
    path = make_docx(tmp_path / "cv.docx", sample_resume_lines)
    rules = screen_resume(path, sample_job, today=today)
    client = make(FakeTransport(ok(json.dumps(VALID))))
    report = screen_resume(path, sample_job, client=client, today=today)
    assert report.coverage_score == rules.coverage_score
    assert LLM_SUCCESS_NOTE in report.limitations
    assert not any("rule based parser was used" in item for item in report.limitations)


@pytest.mark.parametrize(
    ("result", "reason"),
    [
        (TimeoutError(), "did not answer in time"),
        (ConnectionRefusedError(), "could not be reached"),
        ((404, b'{"error": "model not found"}'), "not installed"),
        ((429, b"{}"), "rate limiting"),
        (ok("   "), "empty, cut off or invalid"),
        ((200, b"not json"), "empty, cut off or invalid"),
        (ok("not json at all"), "empty, cut off or invalid"),
        ((500, b"{}"), "rule based parser was used"),
    ],
    ids=["timeout", "refused", "no_model", "rate_limit", "empty", "bad_body", "bad_json", "http"],
)
def test_pipeline_falls_back_with_a_safe_reason(
    result, reason, tmp_path, make_docx, sample_resume_lines, sample_job, today
):
    path = make_docx(tmp_path / "cv.docx", sample_resume_lines)
    rules = screen_resume(path, sample_job, today=today)
    client = make(FakeTransport(*[result] * 3))
    report = screen_resume(path, sample_job, client=client, today=today)
    assert report.coverage_score == rules.coverage_score
    notes = " ".join(report.limitations)
    assert reason in notes
    assert "rule based parser was used" in notes
    assert LLM_SUCCESS_NOTE not in report.limitations


# --- the real urllib transport against a loopback server -------------------------------------


@pytest.fixture
def server():
    stop = threading.Event()
    state = SimpleNamespace(status=200, body=b"{}", delay=0.0, headers={}, requests=[], url="")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(length)
            state.requests.append((self.path, self.headers.get("Content-Type"), data))
            try:
                if state.delay:
                    stop.wait(state.delay)
                self.send_response(state.status)
                for name, value in state.headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(state.body)))
                self.end_headers()
                self.wfile.write(state.body)
            except OSError:
                pass

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    state.url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield state
    finally:
        stop.set()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def real(server, **kwargs) -> OllamaClient:
    kwargs.setdefault("sleep", lambda seconds: None)
    return OllamaClient(server.url, "my-model", **kwargs)


def test_real_transport_round_trip(server):
    server.body = ok("hello")[1]
    assert real(server).complete("SYS", "USER") == "hello"
    path, content_type, data = server.requests[0]
    assert path == "/api/chat"
    assert content_type == "application/json"
    assert json.loads(data)["options"] == {"temperature": 0, "num_predict": 2048}


def test_real_transport_ignores_proxy_settings(server, monkeypatch):
    for name in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    server.body = ok("direct")[1]
    assert real(server).complete("s", "u") == "direct"
    assert len(server.requests) == 1


def test_real_transport_reports_a_missing_model(server):
    server.status = 404
    server.body = b'{"error": "model \\"my-model\\" not found, try pulling it first"}'
    with pytest.raises(LLMModelNotFoundError):
        real(server).complete("s", "u")
    assert len(server.requests) == 1


def test_real_transport_retries_server_errors(server):
    server.status = 503
    with pytest.raises(LLMError, match="HTTP 503"):
        real(server, max_retries=1).complete("s", "u")
    assert len(server.requests) == 2


def test_real_transport_never_follows_redirects(server):
    server.status = 302
    server.headers = {"Location": server.url + "/elsewhere"}
    with pytest.raises(LLMError, match="HTTP 302"):
        real(server).complete("s", "u")
    assert len(server.requests) == 1


def test_real_transport_times_out(server):
    server.delay = 3.0
    with pytest.raises(LLMTimeoutError):
        real(server, timeout_seconds=0.3, max_retries=0).complete("s", "u")