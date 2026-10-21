import json

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
from hr_screening.evaluation.formatting import format_report_text
from hr_screening.llm.fake import FakeLLMClient
from hr_screening.pipeline import LLM_SUCCESS_NOTE, screen_resume

FULL_PAYLOAD = {
    "skills": ["Python", "SQL"],
    "education": ["B.Tech in Computer Science, Example Institute, 2018"],
    "certifications": ["Example Cloud Practitioner"],
    "projects": [],
    "experience": [
        {"heading": "Data Analyst, Example Corp", "start": "Jan 2019", "end": "Dec 2021"},
        {"heading": "Senior Analyst, Sample Ltd", "start": "Jan 2022", "end": "Present"},
    ],
}


def _resume(tmp_path, make_docx, lines):
    return make_docx(tmp_path / "cv.docx", lines)


def test_pipeline_without_llm(tmp_path, make_docx, sample_resume_lines, sample_job, today):
    path = _resume(tmp_path, make_docx, sample_resume_lines)
    report = screen_resume(path, sample_job, today=today)
    assert report.coverage_score == 55.9
    text = format_report_text(report)
    assert "[PARTIAL] tableau" in text
    assert "human reviewer" in text


def test_pipeline_with_fake_llm(tmp_path, make_docx, sample_resume_lines, sample_job, today):
    path = _resume(tmp_path, make_docx, sample_resume_lines)
    client = FakeLLMClient(json.dumps(FULL_PAYLOAD))
    report = screen_resume(path, sample_job, client=client, today=today)
    assert len(client.calls) == 1
    assert report.coverage_score == 55.9


def test_provider_failure_falls_back_to_rules(
    tmp_path, make_docx, sample_resume_lines, sample_job, today
):
    path = _resume(tmp_path, make_docx, sample_resume_lines)
    client = FakeLLMClient(error=LLMRateLimitError("slow down"))
    report = screen_resume(path, sample_job, client=client, today=today)
    assert report.coverage_score == 55.9
    assert any("rule based parser was used" in item for item in report.limitations)


def test_malformed_model_output_falls_back_to_rules(
    tmp_path, make_docx, sample_resume_lines, sample_job, today
):
    path = _resume(tmp_path, make_docx, sample_resume_lines)
    report = screen_resume(path, sample_job, client=FakeLLMClient("not json"), today=today)
    assert any("rule based parser was used" in item for item in report.limitations)


def test_success_note_is_added_when_the_model_is_used(
    tmp_path, make_docx, sample_resume_lines, sample_job, today
):
    path = _resume(tmp_path, make_docx, sample_resume_lines)
    client = FakeLLMClient(json.dumps(FULL_PAYLOAD))
    report = screen_resume(path, sample_job, client=client, today=today)
    assert LLM_SUCCESS_NOTE in report.limitations
    assert "does not prove" in LLM_SUCCESS_NOTE
    assert not any("rule based parser was used" in item for item in report.limitations)


def test_no_model_note_without_a_client(
    tmp_path, make_docx, sample_resume_lines, sample_job, today
):
    path = _resume(tmp_path, make_docx, sample_resume_lines)
    report = screen_resume(path, sample_job, today=today)
    assert not any("language model" in item for item in report.limitations)
    assert not any("LLM extraction failed" in item for item in report.limitations)


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (LLMTimeoutError("x"), "the model did not answer in time"),
        (LLMConnectionError("x"), "the model server could not be reached"),
        (LLMModelNotFoundError("x"), "the configured model is not installed"),
        (LLMRateLimitError("x"), "the model server was rate limiting requests"),
        (LLMResponseError("x"), "the model reply was empty, cut off or invalid"),
    ],
    ids=["timeout", "connection", "model", "rate_limit", "response"],
)
def test_each_failure_gets_a_safe_reason(
    error, reason, tmp_path, make_docx, sample_resume_lines, sample_job, today
):
    path = _resume(tmp_path, make_docx, sample_resume_lines)
    report = screen_resume(path, sample_job, client=FakeLLMClient(error=error), today=today)
    assert f"LLM extraction failed ({reason}), so the rule based parser was used instead." in (
        report.limitations
    )
    assert report.coverage_score == 55.9
    assert LLM_SUCCESS_NOTE not in report.limitations


def test_unknown_llm_error_keeps_the_generic_note(
    tmp_path, make_docx, sample_resume_lines, sample_job, today
):
    path = _resume(tmp_path, make_docx, sample_resume_lines)
    report = screen_resume(path, sample_job, client=FakeLLMClient(error=LLMError("x")), today=today)
    assert "LLM extraction failed, so the rule based parser was used instead." in report.limitations


def test_fallback_note_never_contains_the_error_text(
    tmp_path, make_docx, sample_resume_lines, sample_job, today
):
    path = _resume(tmp_path, make_docx, sample_resume_lines)
    client = FakeLLMClient(error=LLMTimeoutError("sk-test-secret Jane Doe"))
    report = screen_resume(path, sample_job, client=client, today=today)
    assert "sk-test-secret" not in " ".join(report.limitations)
    assert "Jane Doe" not in " ".join(report.limitations)


def test_configuration_errors_are_not_hidden_by_the_fallback(
    tmp_path, make_docx, sample_resume_lines, sample_job, today
):
    path = _resume(tmp_path, make_docx, sample_resume_lines)
    client = FakeLLMClient(error=LLMConfigError("LLM_MODEL is not set."))
    with pytest.raises(LLMConfigError):
        screen_resume(path, sample_job, client=client, today=today)


@pytest.mark.parametrize(
    "client",
    [
        lambda: FakeLLMClient(json.dumps(FULL_PAYLOAD)),
        lambda: FakeLLMClient(error=LLMRateLimitError("slow down")),
        lambda: FakeLLMClient("not json"),
    ],
    ids=["model succeeds", "provider fails", "malformed output"],
)
def test_the_model_path_and_its_fallbacks_score_the_same_as_the_rules(
    tmp_path, make_docx, sample_resume_lines, sample_job, today, client
):
    path = _resume(tmp_path, make_docx, sample_resume_lines)
    rules = screen_resume(path, sample_job, today=today)
    report = screen_resume(path, sample_job, client=client(), today=today)
    assert report.coverage_score == rules.coverage_score
    assert [a.status for a in report.assessments] == [a.status for a in rules.assessments]
    assert report.human_review_required is True