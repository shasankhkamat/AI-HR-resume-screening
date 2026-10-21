"""Scoring edge cases. The score is requirement coverage, not a prediction of anything."""

from itertools import product

import pytest

from hr_screening.evaluation.evaluator import evaluate
from hr_screening.evaluation.scoring import CREDIT, WEIGHTS, compute_coverage
from hr_screening.models.candidate import CandidateProfile
from hr_screening.models.job import JobRequirements
from hr_screening.models.report import (
    MatchStatus,
    RequirementAssessment,
    RequirementKind,
)

K = RequirementKind
S = MatchStatus


def _item(kind: RequirementKind, status: MatchStatus, name: str = "x") -> RequirementAssessment:
    return RequirementAssessment(requirement=name, kind=kind, status=status, explanation="e")


def _score(*items: RequirementAssessment) -> float | None:
    return compute_coverage(list(items))[0]


def test_weights_and_credits_are_unchanged():
    assert WEIGHTS == {
        K.REQUIRED_SKILL: 3.0,
        K.EXPERIENCE: 3.0,
        K.EDUCATION: 2.0,
        K.CERTIFICATION: 2.0,
        K.PREFERRED_SKILL: 1.0,
    }
    assert CREDIT == {S.SUPPORTED: 1.0, S.PARTIAL: 0.5, S.NOT_FOUND: 0.0, S.UNVERIFIED: 0.0}


def test_every_kind_has_a_positive_weight_and_every_status_a_credit():
    assert set(WEIGHTS) == set(RequirementKind)
    assert all(weight > 0 for weight in WEIGHTS.values())
    assert set(CREDIT) == set(MatchStatus)
    assert all(0 <= credit <= 1 for credit in CREDIT.values())


@pytest.mark.parametrize(
    ("status", "expected"),
    [(S.SUPPORTED, 100.0), (S.PARTIAL, 50.0), (S.NOT_FOUND, 0.0), (S.UNVERIFIED, 0.0)],
)
def test_all_requirements_in_one_status(status, expected):
    items = [_item(kind, status) for kind in RequirementKind]
    assert _score(*items) == expected


def test_mixed_statuses_and_weights():
    score = _score(
        _item(K.REQUIRED_SKILL, S.SUPPORTED),
        _item(K.EXPERIENCE, S.PARTIAL),
        _item(K.EDUCATION, S.NOT_FOUND),
        _item(K.PREFERRED_SKILL, S.UNVERIFIED),
    )
    assert score == 50.0  # (3 + 1.5) / 9


def test_requirement_weights_matter():
    heavy = _score(_item(K.REQUIRED_SKILL, S.SUPPORTED), _item(K.PREFERRED_SKILL, S.NOT_FOUND))
    light = _score(_item(K.REQUIRED_SKILL, S.NOT_FOUND), _item(K.PREFERRED_SKILL, S.SUPPORTED))
    assert (heavy, light) == (75.0, 25.0)


def test_empty_requirements_give_no_score():
    score, explanation = compute_coverage([])
    assert score is None
    assert "No requirements" in explanation


def test_rounding_is_one_decimal_and_ties_round_half_up():
    third = _score(
        _item(K.REQUIRED_SKILL, S.SUPPORTED),
        _item(K.REQUIRED_SKILL, S.NOT_FOUND),
        _item(K.REQUIRED_SKILL, S.NOT_FOUND),
    )
    assert third == 33.3
    tie = _score(
        _item(K.REQUIRED_SKILL, S.SUPPORTED),
        _item(K.PREFERRED_SKILL, S.PARTIAL),
        _item(K.CERTIFICATION, S.SUPPORTED),
        _item(K.EDUCATION, S.PARTIAL),
    )
    assert tie == 81.3  # exactly 81.25 before rounding


def test_score_is_deterministic_and_stays_in_bounds():
    kinds = list(RequirementKind)
    statuses = list(MatchStatus)
    for combo in product(product(kinds, statuses), repeat=2):
        items = [_item(kind, status) for kind, status in combo]
        first, second = _score(*items), _score(*items)
        assert first == second
        assert first is not None
        assert 0.0 <= first <= 100.0


def test_score_ignores_requirement_order():
    items = [_item(K.REQUIRED_SKILL, S.SUPPORTED), _item(K.EDUCATION, S.PARTIAL)]
    assert _score(*items) == _score(*reversed(items))


def test_sample_score_is_stable(sample_resume_text, sample_job, today):
    from hr_screening.extraction.job_parser import parse_job_description
    from hr_screening.extraction.resume_parser import parse_resume

    profile = parse_resume(sample_resume_text, today=today)
    report = evaluate(profile, parse_job_description(sample_job), sample_resume_text)
    assert report.coverage_score == 73.5


# --- duplicate requirements ----------------------------------------------------------------


def test_duplicate_skills_are_counted_once_ignoring_case_and_spacing():
    requirements = JobRequirements(required_skills=["python", "Python", " PYTHON "])
    report = evaluate(CandidateProfile(), requirements, "Python dev")
    assert [a.requirement for a in report.assessments] == ["python"]
    assert report.coverage_score == 100.0
    assert any("duplicate requirement(s) were counted once" in n for n in report.limitations)


def test_required_classification_wins_over_preferred():
    requirements = JobRequirements(required_skills=["sql"], preferred_skills=["SQL", "docker"])
    report = evaluate(CandidateProfile(), requirements, "SQL only")
    assert [(a.requirement, a.kind) for a in report.assessments] == [
        ("sql", K.REQUIRED_SKILL),
        ("docker", K.PREFERRED_SKILL),
    ]
    assert report.coverage_score == 75.0  # 3 earned of 3 + 1


def test_duplicate_certifications_are_counted_once():
    requirements = JobRequirements(certifications=["PMP certification", "pmp  Certification"])
    report = evaluate(CandidateProfile(), requirements, "CERTIFICATIONS\nPMP Certified")
    assert len(report.assessments) == 1


def test_no_duplicate_note_when_there_are_no_duplicates():
    report = evaluate(CandidateProfile(), JobRequirements(required_skills=["python"]), "Python")
    assert not any("duplicate" in n for n in report.limitations)


# --- the score stays a coverage summary -----------------------------------------------------


def test_unverified_counts_as_zero_and_is_explained():
    requirements = JobRequirements(min_years_experience=3)
    report = evaluate(CandidateProfile(), requirements, "No dates here")
    assert report.coverage_score == 0.0
    assert "not a measure" in report.score_explanation
    assert "understate" in report.score_explanation
    assert any("not a measure of overall suitability" in n for n in report.limitations)