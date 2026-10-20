import json

from hr_screening.errors import LLMRateLimitError
from hr_screening.evaluation.formatting import format_report_text
from hr_screening.llm.fake import FakeLLMClient
from hr_screening.pipeline import screen_resume

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
    assert report.coverage_score == 73.5
    text = format_report_text(report)
    assert "[PARTIAL] tableau" in text
    assert "human reviewer" in text


def test_pipeline_with_fake_llm(tmp_path, make_docx, sample_resume_lines, sample_job, today):
    path = _resume(tmp_path, make_docx, sample_resume_lines)
    client = FakeLLMClient(json.dumps(FULL_PAYLOAD))
    report = screen_resume(path, sample_job, client=client, today=today)
    assert len(client.calls) == 1
    assert report.coverage_score == 73.5


def test_provider_failure_falls_back_to_rules(
    tmp_path, make_docx, sample_resume_lines, sample_job, today
):
    path = _resume(tmp_path, make_docx, sample_resume_lines)
    client = FakeLLMClient(error=LLMRateLimitError("slow down"))
    report = screen_resume(path, sample_job, client=client, today=today)
    assert report.coverage_score == 73.5
    assert any("rule based parser was used" in item for item in report.limitations)


def test_malformed_model_output_falls_back_to_rules(
    tmp_path, make_docx, sample_resume_lines, sample_job, today
):
    path = _resume(tmp_path, make_docx, sample_resume_lines)
    report = screen_resume(path, sample_job, client=FakeLLMClient("not json"), today=today)
    assert any("rule based parser was used" in item for item in report.limitations)