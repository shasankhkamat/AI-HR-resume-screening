"""The --rag flag: opt in, local only, and the normal output stays exactly as before."""

import json
import socket
from pathlib import Path

import pytest

from hr_screening import main as main_module
from hr_screening import pipeline
from hr_screening.config import Settings
from hr_screening.evaluation.formatting import format_report_text
from hr_screening.ingestion.loader import load_resume_text
from hr_screening.llm import ollama_client
from hr_screening.llm.fake import FakeLLMClient
from hr_screening.pipeline import (
    LLM_SUCCESS_NOTE,
    RETRIEVAL_FAILED_NOTE,
    RETRIEVAL_NOTE,
    screen_resume,
)

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
HEADER = "RETRIEVED PASSAGES (for review)"
OLD_KEYS = {
    "assessments",
    "coverage_score",
    "score_explanation",
    "limitations",
    "human_review_required",
}


@pytest.fixture(autouse=True)
def quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "configure_logging", lambda: None)


@pytest.fixture
def files(tmp_path, make_docx):
    resume = make_docx(tmp_path / "cv.docx", FIXTURE.read_text(encoding="utf-8").splitlines())
    job = tmp_path / "job.txt"
    job.write_text(JOB, encoding="utf-8")
    return resume, ["--resume", str(resume), "--job-file", str(job)]


def run(argv, capsys):
    code = main_module.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def run_json(argv, capsys):
    code, out, err = run([*argv, "--json"], capsys)
    return code, json.loads(out), err


# --- the flag ---------------------------------------------------------------------------------


def test_retrieval_is_off_by_default():
    args = main_module._build_parser().parse_args(["--resume", "a.pdf", "--job-text", "x"])
    assert args.rag is None


def test_lexical_is_accepted():
    args = main_module._build_parser().parse_args(
        ["--resume", "a.pdf", "--job-text", "x", "--rag", "lexical"]
    )
    assert args.rag == "lexical"


@pytest.mark.parametrize("mode", ["semantic", "embeddings", "LEXICAL", ""])
def test_other_modes_are_rejected_for_now(mode, capsys):
    with pytest.raises(SystemExit) as caught:
        main_module._build_parser().parse_args(
            ["--resume", "a.pdf", "--job-text", "x", "--rag", mode]
        )
    assert caught.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


# --- the normal path is unchanged ------------------------------------------------------------


def test_default_text_output_is_exactly_the_old_report(files, today, capsys):
    resume, argv = files
    code, out, err = run(argv, capsys)
    expected = format_report_text(screen_resume(resume, JOB))
    assert code == 0
    assert out == expected + "\n"
    assert HEADER not in out
    assert err == ""


def test_default_json_has_exactly_the_old_keys(files, capsys):
    _, argv = files
    code, report, err = run_json(argv, capsys)
    assert code == 0
    assert set(report) == OLD_KEYS
    assert err == ""


def test_default_run_never_touches_the_model_or_the_network(files, monkeypatch, capsys):
    def refuse(*args, **kwargs):
        raise AssertionError("must not be called")

    monkeypatch.setattr(main_module, "build_llm_client", refuse)
    monkeypatch.setattr(main_module, "get_settings", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    _, argv = files
    code, _, _ = run(argv, capsys)
    assert code == 0


# --- with --rag lexical -------------------------------------------------------------------------


def test_text_output_gets_a_labelled_section_and_the_same_score(files, capsys):
    _, argv = files
    _, plain, _ = run(argv, capsys)
    code, rag, err = run([*argv, "--rag", "lexical"], capsys)
    assert code == 0
    assert HEADER in rag
    assert "is not evidence that a requirement is met" in rag
    assert "chars " in rag
    assert err == ""
    score_line = next(ln for ln in plain.splitlines() if ln.startswith("Requirement coverage"))
    assert score_line in rag.splitlines()
    assert plain.split("\n\nLIMITATIONS")[0] in rag


def test_json_output_lists_the_retrieval_and_keeps_the_assessments(files, capsys):
    resume, argv = files
    _, plain, _ = run_json(argv, capsys)
    code, rag, _ = run_json([*argv, "--rag", "lexical"], capsys)
    assert code == 0
    assert set(rag) == OLD_KEYS | {"retrieval"}
    for key in ("assessments", "coverage_score", "score_explanation", "human_review_required"):
        assert rag[key] == plain[key]
    assert len(rag["retrieval"]) == len(rag["assessments"])
    assert rag["limitations"] == [*plain["limitations"], RETRIEVAL_NOTE]


def test_json_passages_keep_text_offsets_and_scores_traceable(files, capsys):
    resume, argv = files
    text = load_resume_text(resume)
    _, rag, _ = run_json([*argv, "--rag", "lexical"], capsys)
    seen = 0
    for entry in rag["retrieval"]:
        for passage in entry["passages"]:
            assert text[passage["start"] : passage["end"]] == passage["text"]
            assert passage["score"] > 0
            assert passage["matched_terms"]
            seen += 1
    assert seen > 10


def test_the_kubernetes_passage_is_labelled_negated_in_json(files, capsys):
    _, argv = files
    _, rag, _ = run_json([*argv, "--rag", "lexical"], capsys)
    entry = next(e for e in rag["retrieval"] if e["requirement"] == "kubernetes")
    assert entry["passages"][0]["mention_kinds"] == ["negated"]


def test_retrieval_makes_no_network_call_and_builds_no_model_client(files, monkeypatch, capsys):
    def refuse(*args, **kwargs):
        raise AssertionError("the network or the model must not be used")

    monkeypatch.setattr(main_module, "build_llm_client", refuse)
    monkeypatch.setattr(main_module, "get_settings", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(ollama_client, "_urllib_transport", refuse)
    _, argv = files
    code, out, _ = run([*argv, "--rag", "lexical"], capsys)
    assert code == 0
    assert HEADER in out


def test_retrieval_and_the_model_can_be_combined(files, monkeypatch, capsys):
    client = FakeLLMClient(json.dumps(PAYLOAD))
    monkeypatch.setattr(main_module, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(main_module, "build_llm_client", lambda settings: client)
    _, argv = files
    _, report, err = run_json([*argv, "--llm", "--rag", "lexical"], capsys)
    assert len(client.calls) == 1
    assert LLM_SUCCESS_NOTE in report["limitations"]
    assert RETRIEVAL_NOTE in report["limitations"]
    assert report["retrieval"]
    assert err == ""


# --- failure ------------------------------------------------------------------------------------


def test_a_retrieval_failure_keeps_the_report_and_says_so(files, monkeypatch, capsys):
    def explode(text):
        raise RuntimeError("SECRET resume text")

    monkeypatch.setattr(pipeline, "LexicalRetriever", explode)
    _, argv = files
    _, plain, _ = run_json(argv, capsys)
    code, report, err = run_json([*argv, "--rag", "lexical"], capsys)
    assert code == 0
    assert report["assessments"] == plain["assessments"]
    assert report["coverage_score"] == plain["coverage_score"]
    assert "retrieval" not in report
    assert report["limitations"][-1] == RETRIEVAL_FAILED_NOTE
    assert "SECRET" not in json.dumps(report) + err


def test_a_retrieval_failure_in_text_output_has_the_note_and_no_section(
    files, monkeypatch, capsys
):
    def explode(text):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "LexicalRetriever", explode)
    _, argv = files
    code, out, _ = run([*argv, "--rag", "lexical"], capsys)
    assert code == 0
    assert HEADER not in out
    assert RETRIEVAL_FAILED_NOTE in out


def test_errors_before_retrieval_are_reported_as_before(tmp_path, capsys):
    code, out, err = run(
        ["--resume", str(tmp_path / "missing.docx"), "--job-text", JOB, "--rag", "lexical"], capsys
    )
    assert code == 2
    assert out == ""
    assert err.startswith("Error:")