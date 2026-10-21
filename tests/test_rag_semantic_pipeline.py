"""Semantic retrieval in the pipeline: nothing changes in the result, failures fall back."""

import json
import logging
import socket
from pathlib import Path

import pytest
from pydantic import ValidationError

from hr_screening import pipeline
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
from hr_screening.evaluation.formatting import format_report_text
from hr_screening.ingestion.loader import load_resume_text
from hr_screening.models.report import RequirementKind, RequirementRetrieval, RetrievedPassage
from hr_screening.pipeline import (
    RETRIEVAL_FAILED_NOTE,
    RETRIEVAL_NOTE,
    RETRIEVAL_TRUNCATED_NOTE,
    screen_resume,
)
from hr_screening.rag.embeddings import FakeEmbedder
from hr_screening.rag.semantic import SemanticRetriever

FIXTURE = Path(__file__).parent / "fixtures" / "long_synthetic_resume.txt"
JOB = """Data Analyst

Requirements
- 3+ years of experience in analytics
- Python, SQL, Tableau and Kubernetes
- Bachelor's degree in computer science
- Example Cloud Practitioner certification

Preferred
- Spark
- Snowflake
"""
SECRET = "SECRET Jane Doe resume line sk-key"
SEMANTIC_NOTE_START = "Retrieved passages come from a semantic search using the embedding model"


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


@pytest.fixture
def resume_file(tmp_path, make_docx) -> Path:
    return make_docx(tmp_path / "long.docx", FIXTURE.read_text(encoding="utf-8").splitlines())


def embedder() -> FakeEmbedder:
    return FakeEmbedder(dimensions=4096)


def semantic(resume_file, today, **kwargs):
    kwargs.setdefault("embedder", embedder())
    return screen_resume(resume_file, JOB, today=today, rag_mode="semantic", **kwargs)


class FailsAfter:
    """Embeds the first calls normally, then fails."""

    model = "flaky"

    def __init__(self, good_calls: int, error: Exception) -> None:
        self.good_calls = good_calls
        self.error = error
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        if self.calls > self.good_calls:
            raise self.error
        return FakeEmbedder(dimensions=4096).embed(texts)


# --- the result is never changed --------------------------------------------------------------


def test_statuses_and_scores_are_identical_with_semantic_retrieval(resume_file, today):
    base = screen_resume(resume_file, JOB, today=today)
    report = semantic(resume_file, today)
    assert report.assessments == base.assessments
    assert report.coverage_score == base.coverage_score
    assert report.score_explanation == base.score_explanation
    assert report.human_review_required is True
    assert base.retrieval == []


def test_every_entry_is_marked_semantic_and_passages_are_exact(resume_file, today):
    text = load_resume_text(resume_file)
    report = semantic(resume_file, today)
    assert [(e.requirement, e.kind) for e in report.retrieval] == [
        (a.requirement, a.kind) for a in report.assessments
    ]
    assert {e.method for e in report.retrieval} == {"semantic"}
    count = 0
    for entry in report.retrieval:
        for passage in entry.passages:
            assert text[passage.start : passage.end] == passage.text
            assert 0 < passage.score <= 1
            count += 1
    assert count > 5


def test_the_report_names_the_embedding_model_and_says_a_passage_is_not_evidence(
    resume_file, today
):
    base = screen_resume(resume_file, JOB, today=today)
    report = semantic(resume_file, today)
    assert len(report.limitations) == len(base.limitations) + 1
    assert report.limitations[:-1] == base.limitations
    note = report.limitations[-1]
    assert note.startswith(SEMANTIC_NOTE_START + " fake-embedder.")
    assert "not evidence that a requirement is met" in note
    assert RETRIEVAL_NOTE not in report.limitations


def test_a_negated_mention_is_shown_but_does_not_change_the_status(resume_file, today):
    base = screen_resume(resume_file, JOB, today=today)
    report = semantic(resume_file, today)
    entry = next(e for e in report.retrieval if e.requirement == "kubernetes")
    assert entry.passages[0].mention_kinds == ["negated"]
    status = next(a.status for a in report.assessments if a.requirement == "kubernetes")
    assert status == next(a.status for a in base.assessments if a.requirement == "kubernetes")


def test_semantic_mode_needs_an_embedder_and_checks_before_reading_anything():
    with pytest.raises(ValueError, match="needs an embedder"):
        screen_resume("missing.docx", JOB, rag_mode="semantic")


def test_an_embedder_is_ignored_in_lexical_mode(resume_file, today):
    unused = embedder()
    report = screen_resume(resume_file, JOB, today=today, rag_mode="lexical", embedder=unused)
    assert unused.calls == []
    assert {e.method for e in report.retrieval} == {"lexical"}
    assert report.limitations[-1] == RETRIEVAL_NOTE


def test_semantic_retrieval_makes_no_network_call_with_a_fake_embedder(
    resume_file, today, monkeypatch
):
    def refuse(*args, **kwargs):
        raise AssertionError("the network must not be used")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    assert semantic(resume_file, today).retrieval


def test_a_long_resume_note_is_added_when_the_search_was_cut_short(
    resume_file, today, monkeypatch
):
    short = lambda text, emb: SemanticRetriever(text, emb, max_chunks=2)  # noqa: E731
    monkeypatch.setattr(pipeline, "SemanticRetriever", short)
    report = semantic(resume_file, today)
    assert report.limitations[-1] == RETRIEVAL_TRUNCATED_NOTE
    assert report.retrieval


# --- failures fall back to keyword search -----------------------------------------------------

REASONS = [
    pytest.param(LLMTimeoutError("x"), "the model did not answer in time", id="timeout"),
    pytest.param(LLMConnectionError("x"), "the model server could not be reached", id="down"),
    pytest.param(LLMModelNotFoundError("x"), "the configured model is not installed", id="model"),
    pytest.param(LLMRateLimitError("x"), "the model server was rate limiting requests", id="rate"),
    pytest.param(
        LLMResponseError("x"), "the model reply was empty, cut off or invalid", id="response"
    ),
    pytest.param(LLMError("HTTP 500"), "the model server returned an error", id="server_error"),
    pytest.param(
        EmbeddingError("x"), "the embedding model failed or returned unusable vectors", id="vectors"
    ),
    pytest.param(
        RuntimeError(SECRET),
        "the embedding model failed or returned unusable vectors",
        id="embedder_crash",
    ),
]


@pytest.mark.parametrize(("error", "reason"), REASONS)
def test_a_failing_embedder_falls_back_to_keyword_search_with_a_safe_note(
    error, reason, resume_file, today, captured_logs
):
    base = screen_resume(resume_file, JOB, today=today)
    report = semantic(resume_file, today, embedder=FakeEmbedder(error=error))
    note = f"Semantic search failed ({reason}), so keyword search was used instead."
    assert report.assessments == base.assessments
    assert report.coverage_score == base.coverage_score
    assert {e.method for e in report.retrieval} == {"lexical"}
    assert report.limitations == [*base.limitations, note, RETRIEVAL_NOTE]
    everything = report.model_dump_json() + " ".join(captured_logs)
    assert "SECRET" not in everything
    assert "sk-key" not in everything
    assert any("Semantic retrieval failed" in message for message in captured_logs)


def test_a_bug_outside_the_embedder_is_reported_as_an_unexpected_error(
    resume_file, today, monkeypatch, captured_logs
):
    def explode(text, emb):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(pipeline, "SemanticRetriever", explode)
    report = semantic(resume_file, today)
    note = "Semantic search failed (an unexpected error), so keyword search was used instead."
    assert note in report.limitations
    assert {e.method for e in report.retrieval} == {"lexical"}
    assert "SECRET" not in report.model_dump_json() + " ".join(captured_logs)


def test_a_failure_while_embedding_a_query_also_falls_back(resume_file, today):
    base = screen_resume(resume_file, JOB, today=today)
    flaky = FailsAfter(good_calls=2, error=LLMTimeoutError("slow"))
    report = semantic(resume_file, today, embedder=flaky)
    assert flaky.calls == 3
    assert {e.method for e in report.retrieval} == {"lexical"}
    assert report.limitations[-2].startswith("Semantic search failed (the model did not answer")
    assert report.assessments == base.assessments


def test_when_both_searches_fail_the_screening_result_is_still_returned(
    resume_file, today, monkeypatch
):
    def explode(text):
        raise RuntimeError("boom")

    base = screen_resume(resume_file, JOB, today=today)
    monkeypatch.setattr(pipeline, "LexicalRetriever", explode)
    report = semantic(resume_file, today, embedder=FakeEmbedder(error=LLMTimeoutError("x")))
    assert report.assessments == base.assessments
    assert report.retrieval == []
    assert report.limitations[-2].startswith("Semantic search failed")
    assert report.limitations[-1] == RETRIEVAL_FAILED_NOTE


def test_a_configuration_error_is_never_hidden_by_the_fallback(resume_file, today):
    broken = FakeEmbedder(error=LLMConfigError("EMBEDDING_MODEL is not set."))
    with pytest.raises(LLMConfigError):
        semantic(resume_file, today, embedder=broken)


# --- the report model -------------------------------------------------------------------------


def test_entries_default_to_the_keyword_method():
    entry = RequirementRetrieval(requirement="python", kind=RequirementKind.REQUIRED_SKILL)
    assert entry.method == "lexical"


def test_an_unknown_method_is_rejected():
    with pytest.raises(ValidationError):
        RequirementRetrieval(
            requirement="python", kind=RequirementKind.REQUIRED_SKILL, method="hybrid"
        )


def test_json_carries_the_method_for_every_entry(resume_file, today):
    lexical = screen_resume(resume_file, JOB, today=today, rag_mode="lexical")
    sem = semantic(resume_file, today)
    assert {e["method"] for e in json.loads(lexical.model_dump_json())["retrieval"]} == {"lexical"}
    assert {e["method"] for e in json.loads(sem.model_dump_json())["retrieval"]} == {"semantic"}


# --- the text report --------------------------------------------------------------------------


def test_the_semantic_section_uses_its_own_words(resume_file, today):
    text = format_report_text(semantic(resume_file, today))
    assert "RETRIEVED PASSAGES (for review)" in text
    assert "Found by semantic search" in text
    assert "Found by keyword search" not in text
    assert "is not evidence that a requirement is met" in text
    assert "does not change any status or the score" in text
    assert "similarity " in text
    assert "relevance " not in text


def test_the_keyword_section_still_uses_relevance(resume_file, today):
    text = format_report_text(screen_resume(resume_file, JOB, today=today, rag_mode="lexical"))
    assert "Found by keyword search" in text
    assert "relevance " in text
    assert "similarity " not in text


def test_a_mixed_report_explains_both_methods(resume_file, today):
    lexical = screen_resume(resume_file, JOB, today=today, rag_mode="lexical")
    extra = RequirementRetrieval(
        requirement="x",
        kind=RequirementKind.PREFERRED_SKILL,
        method="semantic",
        passages=[RetrievedPassage(start=0, end=3, text="abc", score=0.5)],
    )
    mixed = lexical.model_copy(update={"retrieval": [*lexical.retrieval, extra]})
    text = format_report_text(mixed)
    assert "Found by keyword search" in text
    assert "Found by semantic search" in text
    assert "Passage 1 (no section; chars 0-3; similarity 0.50)" in text