import pytest
from pydantic import ValidationError

from hr_screening.models.candidate import CandidateProfile, WorkExperience
from hr_screening.models.report import (
    MatchStatus,
    RequirementAssessment,
    RequirementKind,
    ScreeningReport,
)


def _assessment(name: str, status: MatchStatus) -> RequirementAssessment:
    return RequirementAssessment(
        requirement=name,
        kind=RequirementKind.REQUIRED_SKILL,
        status=status,
        explanation="x",
    )


def test_too_many_skills_rejected():
    with pytest.raises(ValidationError):
        CandidateProfile(skills=["x"] * 101)


def test_empty_experience_heading_rejected():
    with pytest.raises(ValidationError):
        WorkExperience(heading="   ")


def test_negative_years_rejected():
    with pytest.raises(ValidationError):
        CandidateProfile(total_years_experience=-1)


def test_human_review_cannot_be_disabled():
    with pytest.raises(ValidationError):
        ScreeningReport(
            assessments=[], score_explanation="x", limitations=[], human_review_required=False
        )


def test_unresolved_lists_missing_and_unverified_only():
    report = ScreeningReport(
        assessments=[
            _assessment("a", MatchStatus.SUPPORTED),
            _assessment("b", MatchStatus.NOT_FOUND),
            _assessment("c", MatchStatus.UNVERIFIED),
            _assessment("d", MatchStatus.PARTIAL),
        ],
        score_explanation="x",
        limitations=[],
    )
    assert [a.requirement for a in report.unresolved] == ["b", "c"]