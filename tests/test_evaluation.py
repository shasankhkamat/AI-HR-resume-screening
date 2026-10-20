from hr_screening.evaluation.evaluator import evaluate
from hr_screening.extraction.job_parser import parse_job_description
from hr_screening.extraction.resume_parser import parse_resume
from hr_screening.models.candidate import CandidateProfile
from hr_screening.models.job import JobRequirements
from hr_screening.models.report import MatchStatus


def _statuses(report):
    return {a.requirement: a.status for a in report.assessments}


def test_sample_resume_against_sample_job(sample_resume_text, sample_job, today):
    profile = parse_resume(sample_resume_text, today=today)
    report = evaluate(profile, parse_job_description(sample_job), sample_resume_text)
    statuses = _statuses(report)
    assert statuses["python"] == MatchStatus.SUPPORTED
    assert statuses["sql"] == MatchStatus.SUPPORTED
    assert statuses["tableau"] == MatchStatus.PARTIAL
    assert statuses["3+ years of experience"] == MatchStatus.SUPPORTED
    assert statuses["Bachelor level education or higher"] == MatchStatus.SUPPORTED
    assert statuses["Data Engineering certification"] == MatchStatus.UNVERIFIED
    assert statuses["spark"] == MatchStatus.NOT_FOUND
    assert report.coverage_score == 73.5
    assert report.human_review_required is True


def test_skill_evidence_is_copied_from_the_resume(sample_resume_text, sample_job, today):
    profile = parse_resume(sample_resume_text, today=today)
    report = evaluate(profile, parse_job_description(sample_job), sample_resume_text)
    for item in report.assessments:
        if item.status in (MatchStatus.SUPPORTED, MatchStatus.PARTIAL):
            assert item.kind.value.endswith("skill") is False or item.evidence
        if item.kind.value.endswith("skill"):
            for evidence in item.evidence:
                assert evidence.snippet in sample_resume_text


def test_missing_information_is_unverified_not_failed():
    text = "No details here."
    requirements = JobRequirements(
        required_skills=["python"],
        min_years_experience=3,
        min_degree_level=2,
    )
    report = evaluate(CandidateProfile(), requirements, text)
    statuses = _statuses(report)
    assert statuses["python"] == MatchStatus.NOT_FOUND
    assert statuses["3+ years of experience"] == MatchStatus.UNVERIFIED
    assert statuses["Bachelor level education or higher"] == MatchStatus.UNVERIFIED
    assert report.coverage_score == 0.0
    assert len(report.unresolved) == 3
    assert any("does not prove" in a.explanation for a in report.assessments)


def test_sql_does_not_match_inside_postgresql_exactly():
    report = evaluate(
        CandidateProfile(),
        JobRequirements(required_skills=["sql"]),
        "Worked with PostgreSQL daily",
    )
    assert report.assessments[0].status == MatchStatus.PARTIAL


def test_java_does_not_match_javascript():
    report = evaluate(
        CandidateProfile(),
        JobRequirements(required_skills=["java"]),
        "Built sites in JavaScript",
    )
    assert report.assessments[0].status == MatchStatus.NOT_FOUND


def test_symbols_in_skill_names():
    report = evaluate(
        CandidateProfile(),
        JobRequirements(required_skills=["c++"]),
        "Skills: C++, Rust",
    )
    assert report.assessments[0].status == MatchStatus.SUPPORTED


def test_no_requirements_means_no_score():
    report = evaluate(CandidateProfile(), JobRequirements(), "text")
    assert report.coverage_score is None