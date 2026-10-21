"""Optional task prefixes for embedding models, and what they must never change."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from hr_screening.config import Settings
from hr_screening.llm.ollama_embedder import OllamaEmbedder
from hr_screening.pipeline import screen_resume
from hr_screening.rag.embeddings import FakeEmbedder, normalize_prefix
from hr_screening.rag.factory import build_embedder
from hr_screening.rag.requirements import verify_traceable
from hr_screening.rag.semantic import SemanticRetriever

RESUME = """SKILLS
Python, SQL

EXPERIENCE
- Built dashboards in Power BI.
- Wrote SQL against PostgreSQL.

EDUCATION
B.Sc. in Statistics
"""
JOB = """Data Analyst

Requirements
- Python, SQL
- Bachelor's degree in statistics
"""
_ENV = ("EMBEDDING_MODEL", "EMBEDDING_QUERY_PREFIX", "EMBEDDING_DOCUMENT_PREFIX")


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


class Recorder:
    """Remembers every text it is asked to embed. Optional prefix attributes are set per test."""

    model = "recorder"

    def __init__(self, **attributes) -> None:
        self.texts: list[str] = []
        for name, value in attributes.items():
            setattr(self, name, value)
        self._inner = FakeEmbedder(dimensions=4096)

    def embed(self, texts):
        self.texts += list(texts)
        return self._inner.embed(texts)


def chunk_texts(retriever) -> list[str]:
    return [chunk.text for chunk in retriever.chunks]


# --- the prefix itself -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("", ""),
        ("   ", ""),
        ("search_query:", "search_query: "),
        ("search_query: ", "search_query: "),
        ("  search_query:   ", "search_query: "),
    ],
)
def test_a_prefix_is_normalized_to_one_trailing_space_or_nothing(given, expected):
    assert normalize_prefix(given) == expected


# --- the retriever ---------------------------------------------------------------------------


def test_without_prefixes_the_texts_embedded_are_exactly_the_chunks_and_the_query():
    recorder = Recorder()
    retriever = SemanticRetriever(RESUME, recorder)
    recorder.texts.clear()
    retriever.for_skill("python")
    assert recorder.texts == ["python"]
    recorder = Recorder()
    retriever = SemanticRetriever(RESUME, recorder)
    assert recorder.texts == chunk_texts(retriever)


def test_the_document_prefix_goes_on_every_chunk_and_the_query_prefix_only_on_queries():
    recorder = Recorder()
    retriever = SemanticRetriever(
        RESUME, recorder, query_prefix="search_query:", document_prefix="search_document:"
    )
    assert recorder.texts == ["search_document: " + text for text in chunk_texts(retriever)]
    recorder.texts.clear()
    retriever.for_skill("python")
    retriever.for_text("Bachelor level degree")
    assert recorder.texts == ["search_query: python", "search_query: Bachelor level degree"]


def test_a_skill_query_keeps_its_aliases_after_the_prefix():
    recorder = Recorder()
    retriever = SemanticRetriever(RESUME, recorder, query_prefix="search_query:")
    recorder.texts.clear()
    retriever.for_skill("javascript")
    assert recorder.texts == ["search_query: javascript, js"]


def test_only_one_prefix_can_be_set():
    recorder = Recorder()
    retriever = SemanticRetriever(RESUME, recorder, document_prefix="search_document:")
    recorder.texts.clear()
    retriever.for_skill("sql")
    assert recorder.texts == ["sql"]


def test_the_embedder_s_own_prefixes_are_used_when_none_are_passed():
    recorder = Recorder(query_prefix="q:", document_prefix="d:")
    retriever = SemanticRetriever(RESUME, recorder)
    assert recorder.texts == ["d: " + text for text in chunk_texts(retriever)]
    recorder.texts.clear()
    retriever.for_skill("python")
    assert recorder.texts == ["q: python"]


def test_prefixes_passed_to_the_retriever_win_over_the_embedder_s():
    recorder = Recorder(query_prefix="q:", document_prefix="d:")
    retriever = SemanticRetriever(RESUME, recorder, query_prefix="", document_prefix="")
    assert recorder.texts == chunk_texts(retriever)


def test_returned_passages_never_contain_the_prefix_and_stay_traceable():
    recorder = Recorder(query_prefix="search_query:", document_prefix="search_document:")
    retriever = SemanticRetriever(RESUME, recorder)
    passages = retriever.for_skill("sql", top_k=9) + retriever.for_text("degree", top_k=9)
    assert passages
    for passage in passages:
        assert "search_document" not in passage.text
        assert passage.text == RESUME[passage.start : passage.end]


def test_a_prefix_does_not_change_which_chunks_exist():
    plain = SemanticRetriever(RESUME, Recorder())
    prefixed = SemanticRetriever(RESUME, Recorder(), document_prefix="search_document:")
    assert [(c.start, c.end, c.text) for c in plain.chunks] == [
        (c.start, c.end, c.text) for c in prefixed.chunks
    ]


# --- the Ollama embedder and the settings -----------------------------------------------------


class Transport:
    def __init__(self) -> None:
        self.bodies: list[dict] = []

    def __call__(self, url, body, timeout):
        self.bodies.append(json.loads(body))
        count = len(self.bodies[-1]["input"])
        vectors = [[1.0, 0.0]] * count
        return 200, json.dumps({"embeddings": vectors}).encode("utf-8")


def test_the_ollama_embedder_stores_normalized_prefixes_but_does_not_add_them_itself():
    transport = Transport()
    embedder = OllamaEmbedder(
        "http://localhost:11434",
        "embed-model",
        transport=transport,
        query_prefix="search_query:",
        document_prefix="search_document:",
    )
    assert (embedder.query_prefix, embedder.document_prefix) == (
        "search_query: ",
        "search_document: ",
    )
    embedder.embed(["plain text"])
    assert transport.bodies[0]["input"] == ["plain text"]


def test_the_ollama_embedder_has_no_prefixes_by_default():
    embedder = OllamaEmbedder("http://localhost:11434", "embed-model", transport=Transport())
    assert (embedder.query_prefix, embedder.document_prefix) == ("", "")


def test_the_retriever_sends_prefixed_texts_to_the_server():
    transport = Transport()
    embedder = OllamaEmbedder(
        "http://localhost:11434",
        "embed-model",
        transport=transport,
        query_prefix="search_query:",
        document_prefix="search_document:",
    )
    retriever = SemanticRetriever(RESUME, embedder)
    retriever.for_skill("python")
    sent = [text for body in transport.bodies for text in body["input"]]
    assert sent[:-1] == ["search_document: " + text for text in chunk_texts(retriever)]
    assert sent[-1] == "search_query: python"


def test_the_settings_default_to_no_prefixes():
    config = Settings(_env_file=None)
    assert (config.embedding_query_prefix, config.embedding_document_prefix) == ("", "")


def test_the_settings_read_the_prefixes_from_the_environment(monkeypatch):
    monkeypatch.setenv("EMBEDDING_QUERY_PREFIX", "search_query:")
    monkeypatch.setenv("EMBEDDING_DOCUMENT_PREFIX", "search_document:")
    config = Settings(_env_file=None)
    assert config.embedding_query_prefix == "search_query:"
    assert config.embedding_document_prefix == "search_document:"


@pytest.mark.parametrize("name", ["embedding_query_prefix", "embedding_document_prefix"])
def test_a_very_long_prefix_is_rejected(name):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{name: "x" * 101})


def test_the_factory_passes_the_prefixes_on():
    embedder = build_embedder(
        Settings(
            _env_file=None,
            embedding_model="embed-model",
            embedding_query_prefix="search_query:",
            embedding_document_prefix="search_document:",
        )
    )
    assert (embedder.query_prefix, embedder.document_prefix) == (
        "search_query: ",
        "search_document: ",
    )


# --- screening results never change ------------------------------------------------------------


@pytest.fixture
def resume_file(tmp_path, make_docx) -> Path:
    return make_docx(tmp_path / "resume.docx", RESUME.splitlines())


def test_prefixes_change_no_status_score_or_human_review_flag(resume_file, today):
    base = screen_resume(resume_file, JOB, today=today)
    plain = screen_resume(
        resume_file, JOB, today=today, rag_mode="semantic", embedder=Recorder()
    )
    prefixed = screen_resume(
        resume_file,
        JOB,
        today=today,
        rag_mode="semantic",
        embedder=Recorder(query_prefix="search_query:", document_prefix="search_document:"),
    )
    for report in (plain, prefixed):
        assert report.assessments == base.assessments
        assert report.coverage_score == base.coverage_score
        assert report.score_explanation == base.score_explanation
        assert report.human_review_required is True
        verify_traceable(report.retrieval, RESUME)