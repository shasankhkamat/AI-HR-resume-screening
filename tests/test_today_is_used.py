"""The ``today`` value reaches every date calculation, so a run does not depend on the clock."""

from datetime import date

import pytest

from hr_screening.evaluation import evaluator
from hr_screening.extraction import resume_parser
from hr_screening.pipeline import screen_resume

ROLE = "Senior Data Analysis Lead, Acme   Jan 2019 - Present"
RESUME = ["SKILLS", "Python", "EXPERIENCE", ROLE, "Built reports"]
FIELD_JOB = "Requirements\n- 6+ years of experience in data analysis\n"
NO_FIELD_JOB = "Requirements\n- 6+ years of experience\n"


def experience(tmp_path, make_docx, job, today):
    report = screen_resume(make_docx(tmp_path / "cv.docx", RESUME), job, today=today)
    return next(a for a in report.assessments if a.kind.value == "experience")


@pytest.mark.parametrize("job", [FIELD_JOB, NO_FIELD_JOB], ids=["with field", "without field"])
@pytest.mark.parametrize(
    ("today", "years", "status"),
    [
        (date(2024, 1, 1), "5", "unverified"),  # 5 years: below the 6 required
        (date(2026, 1, 1), "7", "supported"),
        (date(2035, 1, 1), "16", "supported"),  # a future date is used as given
    ],
)
def test_the_same_resume_is_judged_at_the_date_that_was_supplied(
    tmp_path, make_docx, job, today, years, status
):
    item = experience(tmp_path, make_docx, job, today)
    assert f"About {years} years" in item.explanation
    if status == "supported":
        assert item.status.value == "supported"
    else:
        assert item.status.value != "supported"


def test_the_real_clock_is_never_read_when_today_is_supplied(tmp_path, make_docx, monkeypatch):
    class NoClock(date):
        @classmethod
        def today(cls):
            raise AssertionError("the real clock must not be read")

    monkeypatch.setattr(evaluator, "date", NoClock)
    monkeypatch.setattr(resume_parser, "date", NoClock)
    for job in (FIELD_JOB, NO_FIELD_JOB):
        item = experience(tmp_path, make_docx, job, date(2030, 1, 1))
        assert "About 11 years" in item.explanation
