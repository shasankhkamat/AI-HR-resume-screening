"""Error paths: fallback, malformed model output, bad input and privacy of logs."""

import json
import logging

import pytest

from hr_screening import main as main_module
from hr_screening.errors import (
    InvalidJobDescriptionError,
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
    UnreadableFileError,
)
from hr_screening.extraction.resume_parser import parse_resume
from hr_screening.ingestion.loader import load_resume_text
from hr_screening.llm.extractor import (
    MAX_RESPONSE_CHARS,
    extract_profile_with_llm,
    parse_payload,
)
from hr_screening.llm.fake import FakeLLMClient
from hr_screening.pipeline import screen_resume

FALLBACK_NOTE = "rule based parser was used"
SECRET = "sk-test-secret-123"
RESUME_ONLY_WORDS = ("Example Corp", "Sample Ltd", "Example Institute")


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def captured_logs():
    """Capture package logs directly, whatever the propagate setting is."""
    logger = logging.getLogger("hr_screening")
    handler = _Capture()
    saved_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(saved_level)


@pytest.fixture
def resume_path(tmp_path, make_docx, sample_resume_lines):
    return make_docx(tmp_path / "cv.docx", sample_resume_lines)


# --- fallback ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [LLMTimeoutError("slow"), LLMError("failed"), RuntimeError("boom"), ValueError("bad")],
)
def test_any_extraction_failure_falls_back_to_rules(error, resume_path, sample_job, today):
    report = screen_resume(resume_path, sample_job, client=FakeLLMClient(error=error), today=today)
    assert report.coverage_score == 73.5
    assert any(FALLBACK_NOTE in item for item in report.limitations)


def test_fallback_logs_only_the_error_class(resume_path, sample_job, today, captured_logs):
    secret_error = RuntimeError(f"{SECRET} Example Corp Jan 2019")
    screen_resume(resume_path, sample_job, client=FakeLLMClient(error=secret_error), today=today)
    messages = [record.getMessage() for record in captured_logs]
    assert any("LLM extraction failed (LLMError)" in message for message in messages)
    assert all(record.exc_info is None for record in captured_logs)
    joined = " ".join(messages)
    assert SECRET not in joined
    assert not any(word in joined for word in RESUME_ONLY_WORDS)


def test_full_run_logs_no_resume_text(resume_path, sample_job, today, captured_logs):
    screen_resume(resume_path, sample_job, today=today)
    joined = " ".join(record.getMessage() for record in captured_logs)
    assert "Python" not in joined
    assert not any(word in joined for word in RESUME_ONLY_WORDS)


def test_error_messages_do_not_expose_the_path_or_content(tmp_path):
    secret_name = tmp_path / "jane-doe-resume.pdf"
    with pytest.raises(UnreadableFileError) as caught:
        load_resume_text(secret_name)
    assert "jane-doe" not in str(caught.value)


# --- malformed model responses ----------------------------------------------------------------
# Each case has a short explicit id. Without ids, pytest builds the test name from the value, and
# the huge inputs below would make names of up to 500,000 characters. Windows rejects an
# environment variable over 32,767 characters, and pytest stores the test name in one.
BAD_RESPONSES = [
    pytest.param("", id="empty"),
    pytest.param("   ", id="whitespace"),
    pytest.param("not json", id="not_json"),
    pytest.param("Here is the JSON: {}", id="prose_before_json"),
    pytest.param("[]", id="json_array"),
    pytest.param("null", id="json_null"),
    pytest.param("42", id="json_number"),
    pytest.param('"text"', id="json_string"),
    pytest.param('{"skills": "Python"}', id="skills_not_list"),
    pytest.param('{"skills": [123]}', id="skill_is_number"),
    pytest.param('{"skills": [null]}', id="skill_is_null"),
    pytest.param('{"skills": [""]}', id="skill_is_empty"),
    pytest.param('{"skills": ["   "]}', id="skill_is_blank"),
    pytest.param('{"skills": null}', id="skills_null"),
    pytest.param('{"experience": [{"start": "Jan 2020"}]}', id="experience_missing_heading"),
    pytest.param('{"experience": ["Analyst"]}', id="experience_not_object"),
    pytest.param('{"education": [{"degree": "BSc"}]}', id="education_not_string"),
    pytest.param('{"skills": ["' + "x" * 301 + '"]}', id="oversized_item"),
    pytest.param('{"skills": ["Python"', id="truncated_json"),
    pytest.param("```json\n{bad}\n```", id="fenced_invalid_json"),
    pytest.param("[" * 100_000, id="nested_brackets"),
    pytest.param('{"skills":' * 50_000, id="deeply_nested_json"),
    pytest.param("x" * (MAX_RESPONSE_CHARS + 1), id="long_string"),
]



@pytest.mark.parametrize("bad", BAD_RESPONSES)
def test_malformed_responses_raise_a_clear_error(bad):
    with pytest.raises(LLMResponseError):
        parse_payload(bad)


@pytest.mark.parametrize("bad", [None, 123, b'{"skills": []}', ["skills"]])
def test_non_text_responses_are_rejected(bad):
    with pytest.raises(LLMResponseError, match="not text"):
        parse_payload(bad)


def test_error_messages_never_echo_the_model_response():
    with pytest.raises(LLMResponseError) as caught:
        parse_payload("Example Corp " + SECRET)
    assert SECRET not in str(caught.value)
    assert "Example Corp" not in str(caught.value)


@pytest.mark.parametrize("bad", BAD_RESPONSES)
def test_malformed_response_falls_back_and_adds_no_claims(bad, resume_path, sample_job, today):
    report = screen_resume(resume_path, sample_job, client=FakeLLMClient(bad), today=today)
    assert any(FALLBACK_NOTE in item for item in report.limitations)
    assert report.coverage_score == 73.5


def test_wellformed_but_invented_claims_become_nothing(sample_resume_text, today):
    payload = {
        "skills": ["Kubernetes", "Rust"],
        "education": ["PhD, MIT"],
        "certifications": ["CISSP"],
        "projects": ["Mars rover"],
        "experience": [{"heading": "CTO, Google", "start": "Jan 2000", "end": "Dec 2020"}],
        "total_years_experience": 40,
    }
    profile, dropped = extract_profile_with_llm(
        FakeLLMClient(json.dumps(payload)), sample_resume_text, today=today
    )
    assert dropped == 6
    assert profile.skills == []
    assert profile.education == []
    assert profile.certifications == []
    assert profile.experience == []
    assert profile.total_years_experience is None


def test_extra_keys_from_the_model_are_ignored(sample_resume_text, today):
    payload = {"skills": ["Python"], "salary": "1 million", "name": "Jane Doe"}
    profile, _ = extract_profile_with_llm(
        FakeLLMClient(json.dumps(payload)), sample_resume_text, today=today
    )
    assert profile.skills == ["Python"]
    assert "Jane" not in profile.model_dump_json()


def test_implausible_dates_from_the_model_do_not_crash(today):
    text = "EXPERIENCE\nEngineer   Jan 1900 - Present\n"
    payload = {"experience": [{"heading": "Engineer", "start": "Jan 1900", "end": "Present"}]}
    profile, _ = extract_profile_with_llm(FakeLLMClient(json.dumps(payload)), text, today=today)
    assert profile.total_years_experience is None
    assert profile.experience[0].duration_months is None
    assert any("could not be counted" in note for note in profile.extraction_notes)


# --- empty and malformed resume text -------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t"])
def test_blank_resume_text_does_not_call_the_model(text, today):
    client = FakeLLMClient("{}")
    profile, dropped = extract_profile_with_llm(client, text, today=today)
    assert client.calls == []
    assert dropped == 0
    assert profile.skills == []
    assert any("empty" in note for note in profile.extraction_notes)


@pytest.mark.parametrize("text", ["", "   ", "\x00\x01 ???", "SKILLS", "EXPERIENCE\n\n", "\ufeff"])
def test_rule_parser_handles_empty_or_junk_text(text, today):
    profile = parse_resume(text, today=today)
    assert profile.total_years_experience is None
    assert profile.skills == []


def test_resume_without_headings_still_screens(tmp_path, make_docx, today):
    path = make_docx(tmp_path / "cv.docx", ["I know Python and SQL"])
    report = screen_resume(path, "Requirements\n- Python\n- 3+ years of experience", today=today)
    statuses = {a.requirement: a.status.value for a in report.assessments}
    assert statuses == {"python": "supported", "3+ years of experience": "unverified"}


# --- invalid job requirements --------------------------------------------------------------------


@pytest.mark.parametrize("job", ["", "   \n", None])
def test_empty_job_description_is_a_clear_error(job, resume_path):
    with pytest.raises(InvalidJobDescriptionError, match="empty"):
        screen_resume(resume_path, job)  # type: ignore[arg-type]


def test_oversized_job_description_error_does_not_echo_the_text(resume_path):
    job = "SECRET-JOB-TEXT " * 5000
    with pytest.raises(InvalidJobDescriptionError) as caught:
        screen_resume(resume_path, job)
    assert "SECRET-JOB-TEXT" not in str(caught.value)


def test_job_without_recognizable_requirements_is_handled_safely(resume_path, today):
    report = screen_resume(resume_path, "Responsibilities\nYou will do things", today=today)
    assert report.assessments == []
    assert report.coverage_score is None
    assert "No requirements" in report.score_explanation
    assert report.human_review_required is True


def test_cli_reports_a_blank_job_without_traceback(tmp_path, monkeypatch, capsys, resume_path):
    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    code = main_module.main(["--resume", str(resume_path), "--job-text", "   "])
    captured = capsys.readouterr()
    assert code == 2
    assert "empty" in captured.err
    assert "Traceback" not in captured.err


def test_cli_survives_implausible_resume_dates(tmp_path, monkeypatch, capsys, make_docx):
    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    resume = make_docx(tmp_path / "cv.docx", ["EXPERIENCE", "Engineer   Jan 1900 - Present"])
    code = main_module.main(["--resume", str(resume), "--job-text", "3+ years of experience"])
    assert code == 0
    assert "UNVERIFIED" in capsys.readouterr().out