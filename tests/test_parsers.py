import pytest

from hr_screening.errors import InvalidJobDescriptionError
from hr_screening.extraction.degrees import detect_degree_level
from hr_screening.extraction.job_parser import MAX_JOB_CHARS, parse_job_description
from hr_screening.extraction.resume_parser import parse_resume
from hr_screening.models.candidate import DegreeLevel


def test_resume_sections_are_extracted(sample_resume_text, today):
    profile = parse_resume(sample_resume_text, today=today)
    assert profile.skills == ["Python", "SQL", "Pandas", "scikit-learn", "Power BI"]
    assert [e.heading for e in profile.experience] == [
        "Data Analyst, Example Corp",
        "Senior Analyst, Sample Ltd",
    ]
    assert profile.total_years_experience == 6.9
    assert profile.years_source == "date_ranges"
    assert profile.education[0].level == DegreeLevel.BACHELOR
    assert profile.certifications == ["Example Cloud Practitioner"]
    assert profile.projects == ["Churn prediction model using scikit-learn"]


def test_resume_without_headings_is_flagged(today):
    profile = parse_resume("Just some text about Python.", today=today)
    assert profile.skills == []
    assert profile.total_years_experience is None
    assert any("No standard section headings" in n for n in profile.extraction_notes)


def test_stated_years_are_used_when_no_dates(today):
    profile = parse_resume("Analyst with 5+ years of experience in reporting.", today=today)
    assert profile.total_years_experience == 5.0
    assert profile.years_source == "stated"
    assert "5+ years" in (profile.years_snippet or "")


def test_overlapping_roles_are_not_double_counted(today):
    text = (
        "EXPERIENCE\n"
        "Analyst, Alpha Co   Jan 2020 - Dec 2021\n"
        "Analyst, Beta Co   Jun 2021 - Dec 2022\n"
    )
    assert parse_resume(text, today=today).total_years_experience == 2.9


def test_numeric_dates(today):
    text = "EXPERIENCE\nEngineer, Gamma Co   03/2019 - 03/2021\n"
    assert parse_resume(text, today=today).total_years_experience == 2.0


def test_job_description_parsing(sample_job):
    req = parse_job_description(sample_job)
    assert req.required_skills == ["python", "sql", "tableau"]
    assert req.preferred_skills == ["spark"]
    assert req.min_years_experience == 3
    assert req.min_degree_level == DegreeLevel.BACHELOR
    assert req.certifications == ["Data Engineering certification"]


def test_job_without_headings_treats_skills_as_required():
    req = parse_job_description("We need Python and Docker.")
    assert req.required_skills == ["python", "docker"]
    assert any("heading" in n for n in req.notes)


def test_empty_job_description():
    with pytest.raises(InvalidJobDescriptionError):
        parse_job_description("   ")


def test_too_long_job_description():
    with pytest.raises(InvalidJobDescriptionError):
        parse_job_description("x" * (MAX_JOB_CHARS + 1))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Master of Science in Statistics", DegreeLevel.MASTER),
        ("B.E. in Mechanical Engineering", DegreeLevel.BACHELOR),
        ("PhD in Economics", DegreeLevel.DOCTORATE),
        ("Diploma in Business", DegreeLevel.DIPLOMA),
        ("MBA from Example Institute", DegreeLevel.MASTER),
        ("High school", None),
        ("Bengaluru University", None),
    ],
)
def test_degree_detection(text, expected):
    assert detect_degree_level(text) == expected