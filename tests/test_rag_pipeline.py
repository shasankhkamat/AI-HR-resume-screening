"""The pipeline with retrieval: statuses and scores must not change, failures must be safe."""

import json
import logging
import socket
from pathlib import Path

import pytest

from hr_screening import pipeline
from hr_screening.errors import NoExtractableTextError
from hr_screening.ingestion.loader import load_resume_text
from hr_screening.llm import ollama_client
from hr_screening.llm.fake import FakeLLMClient
from hr_screening.pipeline import (
    LLM_SUCCESS_NOTE,
    RETRIEVAL_FAILED_NOTE,
    RETRIEVAL_NOTE,
    RETRIEVAL_TRUNCATED_NOTE,
    screen_resume,
)
from hr_screening.rag.retrieval import LexicalRetriever

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
PAYLOAD = {
    "skills": ["Python", "SQL"],
    "education": ["B.Tech in Computer Science, Example Institute, 2018"],
    "certifications": ["Example Cloud Practitioner"],
    "projects": [],
    "experience": [
        {"heading": "Data Analyst, Example Corp", "start": "Jan 2019", "end": "Dec 2021"},
        {"heading": "Senior Analyst, Sample Ltd", "start": "Jan 2022", "end": "Present"},
    ],
}


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
def long_resume(tmp_path, make_docx) -> Path:
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    return make_docx(tmp_path / "long.docx", lines)


@pytest.fixture
def short_resume(tmp_path, make_docx, sample_resume_lines) -> Path:
    return make_docx(tmp_path / "short.docx", sample_resume_lines)


def status_of(report, requirement: str):
    return next(a.status for a in report.assessments if a.requirement == requirement)


# --- retrieval never changes the screening result ---------------------------------------------


def test_statuses_and_scores_are_identical_with_and_without_retrieval(long_resume, today):
    base = screen_resume(long_resume, JOB, today=today)
    rag = screen_resume(long_resume, JOB, today=today, rag_mode="lexical")
    assert rag.assessments == base.assessments
    assert rag.coverage_score == base.coverage_score
    assert rag.score_explanation == base.score_explanation
    assert rag.human_review_required is True
    assert base.retrieval == []
    assert rag.retrieval


def test_the_sample_resume_and_job_give_the_same_score_with_retrieval(
    short_resume, sample_job, today
):
    base = screen_resume(short_resume, sample_job, today=today)
    rag = screen_resume(short_resume, sample_job, today=today, rag_mode="lexical")
    assert base.coverage_score == rag.coverage_score
    assert rag.assessments == base.assessments


def test_retrieval_works_together_with_the_model_and_changes_nothing(long_resume, today):
    client = FakeLLMClient(json.dumps(PAYLOAD))
    llm_only = screen_resume(long_resume, JOB, client=client, today=today)
    both = screen_resume(
        long_resume, JOB, client=FakeLLMClient(json.dumps(PAYLOAD)), today=today, rag_mode="lexical"
    )
    assert both.assessments == llm_only.assessments
    assert both.coverage_score == llm_only.coverage_score
    assert LLM_SUCCESS_NOTE in both.limitations
    assert RETRIEVAL_NOTE in both.limitations


def test_a_retrieved_negated_line_does_not_change_the_status(long_resume, today):
    base = screen_resume(long_resume, JOB, today=today)
    rag = screen_resume(long_resume, JOB, today=today, rag_mode="lexical")
    kubernetes = next(e for e in rag.retrieval if e.requirement == "kubernetes")
    assert kubernetes.passages[0].mention_kinds == ["negated"]
    assert status_of(rag, "kubernetes") == status_of(base, "kubernetes")
    assert status_of(rag, "kubernetes").value != "supported"


def test_retrieved_passages_are_not_used_as_evidence(long_resume, today):
    base = screen_resume(long_resume, JOB, today=today)
    rag = screen_resume(long_resume, JOB, today=today, rag_mode="lexical")
    spark = next(a for a in rag.assessments if a.requirement == "spark")
    assert spark.evidence == next(a for a in base.assessments if a.requirement == "spark").evidence
    assert status_of(rag, "spark") == status_of(base, "spark")


# --- what retrieval adds --------------------------------------------------------------------


def test_retrieval_is_off_by_default(long_resume, today):
    report = screen_resume(long_resume, JOB, today=today)
    assert report.retrieval == []
    assert RETRIEVAL_NOTE not in report.limitations
    assert not any("retriev" in item.lower() for item in report.limitations)


def test_retrieval_adds_exactly_one_note_to_the_limitations(long_resume, today):
    base = screen_resume(long_resume, JOB, today=today)
    rag = screen_resume(long_resume, JOB, today=today, rag_mode="lexical")
    assert rag.limitations == [*base.limitations, RETRIEVAL_NOTE]
    assert "not evidence" in RETRIEVAL_NOTE


def test_every_passage_is_an_exact_slice_of_the_extracted_resume_text(long_resume, today):
    text = load_resume_text(long_resume)
    report = screen_resume(long_resume, JOB, today=today, rag_mode="lexical")
    count = 0
    for entry in report.retrieval:
        for passage in entry.passages:
            assert text[passage.start : passage.end] == passage.text
            count += 1
    assert count > 10


def test_there_is_one_entry_per_requirement(long_resume, today):
    report = screen_resume(long_resume, JOB, today=today, rag_mode="lexical")
    assert [(e.requirement, e.kind) for e in report.retrieval] == [
        (a.requirement, a.kind) for a in report.assessments
    ]


def test_a_short_resume_still_gets_a_valid_report(short_resume, sample_job, today):
    report = screen_resume(short_resume, sample_job, today=today, rag_mode="lexical")
    assert report.retrieval
    text = load_resume_text(short_resume)
    for entry in report.retrieval:
        for passage in entry.passages:
            assert text[passage.start : passage.end] == passage.text


def test_an_unknown_mode_is_rejected_before_anything_is_read():
    with pytest.raises(ValueError, match="rag_mode"):
        screen_resume("missing.docx", JOB, rag_mode="semantic")  # type: ignore[arg-type]


def test_a_blank_resume_still_fails_as_before_with_retrieval(tmp_path, make_docx):
    blank = make_docx(tmp_path / "blank.docx", ["   ", ""])
    with pytest.raises(NoExtractableTextError):
        screen_resume(blank, JOB, rag_mode="lexical")


# --- failures are safe -------------------------------------------------------------------------


def assert_unchanged_apart_from_the_failure(report, base) -> None:
    assert report.assessments == base.assessments
    assert report.coverage_score == base.coverage_score
    assert report.retrieval == []
    assert report.limitations == [*base.limitations, RETRIEVAL_FAILED_NOTE]
    assert "statuses and the score are not affected" in RETRIEVAL_FAILED_NOTE


def test_a_failing_retriever_leaves_the_screening_result_intact(
    long_resume, today, monkeypatch, captured_logs
):
    def explode(text):
        raise RuntimeError("SECRET Example Corp resume text")

    base = screen_resume(long_resume, JOB, today=today)
    monkeypatch.setattr(pipeline, "LexicalRetriever", explode)
    report = screen_resume(long_resume, JOB, today=today, rag_mode="lexical")
    assert_unchanged_apart_from_the_failure(report, base)
    everything = report.model_dump_json() + " ".join(captured_logs)
    assert "SECRET" not in everything
    assert "Example Corp resume text" not in " ".join(captured_logs)
    assert any("Retrieval failed (RuntimeError)" in message for message in captured_logs)


def test_an_error_while_mapping_requirements_is_handled_the_same_way(
    long_resume, today, monkeypatch
):
    def explode(retriever, assessments):
        raise ValueError("boom")

    base = screen_resume(long_resume, JOB, today=today)
    monkeypatch.setattr(pipeline, "retrieve_for_assessments", explode)
    report = screen_resume(long_resume, JOB, today=today, rag_mode="lexical")
    assert_unchanged_apart_from_the_failure(report, base)


def test_a_failed_traceability_check_drops_the_passages_instead_of_showing_them(
    long_resume, today, monkeypatch
):
    def reject(entries, text):
        raise ValueError("A retrieved passage does not match the resume text.")

    base = screen_resume(long_resume, JOB, today=today)
    monkeypatch.setattr(pipeline, "verify_traceable", reject)
    report = screen_resume(long_resume, JOB, today=today, rag_mode="lexical")
    assert_unchanged_apart_from_the_failure(report, base)


def test_a_retriever_with_wrong_offsets_is_never_shown(long_resume, today, monkeypatch):
    class Shifted(LexicalRetriever):
        def for_skill(self, skill, top_k=3):
            passages = super().for_skill(skill, top_k)
            return [
                type(p)(**{**p.__dict__, "start": p.start + 1, "end": p.end + 1}) for p in passages
            ]

    base = screen_resume(long_resume, JOB, today=today)
    monkeypatch.setattr(pipeline, "LexicalRetriever", Shifted)
    report = screen_resume(long_resume, JOB, today=today, rag_mode="lexical")
    assert_unchanged_apart_from_the_failure(report, base)


def test_a_long_resume_note_is_added_when_the_search_was_cut_short(
    long_resume, today, monkeypatch
):
    short = lambda text: LexicalRetriever(text, max_chunks=2)  # noqa: E731
    monkeypatch.setattr(pipeline, "LexicalRetriever", short)
    base = screen_resume(long_resume, JOB, today=today)
    report = screen_resume(long_resume, JOB, today=today, rag_mode="lexical")
    assert report.assessments == base.assessments
    assert report.retrieval
    assert report.limitations == [*base.limitations, RETRIEVAL_NOTE, RETRIEVAL_TRUNCATED_NOTE]


# --- no network ---------------------------------------------------------------------------------


def test_lexical_retrieval_makes_no_network_calls(long_resume, today, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the network must not be used")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(ollama_client, "_urllib_transport", refuse)
    report = screen_resume(long_resume, JOB, today=today, rag_mode="lexical")
    assert report.retrieval