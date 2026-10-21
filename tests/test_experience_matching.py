"""'1 year of data analysis experience' against resumes that state a title or a duration."""

import pytest

from hr_screening.extraction.job_parser import parse_job_description
from hr_screening.extraction.resume_parser import parse_resume
from hr_screening.models.report import MatchStatus, RequirementKind
from hr_screening.pipeline import screen_resume

JOB = "Required Skills:\n- Python\n\nExperience:\n- 1 year of data analysis experience.\n"


@pytest.fixture
def status(tmp_path, make_docx, today):
    def _run(lines, job=JOB):
        report = screen_resume(make_docx(tmp_path / "cv.docx", lines), job, today=today)
        item = next(a for a in report.assessments if a.kind is RequirementKind.EXPERIENCE)
        return item.status, item, report

    return _run


# The experience line is not also a skill


def test_experience_field_is_not_listed_as_a_skill():
    job = parse_job_description(JOB)
    assert job.required_skills == ["python"]
    assert job.experience_field == "data analysis"
    assert job.min_years_experience == 1


def test_data_analysis_under_required_skills_is_still_a_skill():
    job = parse_job_description("Required Skills:\n- Data Analysis\n- Python\n")
    assert set(job.required_skills) == {"python", "data analysis"}


def test_data_analysis_listed_and_in_the_experience_line_is_one_skill():
    job = parse_job_description(
        "Required Skills:\n- Data Analysis\n\nExperience:\n- 1 year of data analysis experience.\n"
    )
    assert job.required_skills == ["data analysis"]
    assert job.min_years_experience == 1


def test_tools_named_in_an_experience_line_are_still_skills():
    job = parse_job_description("Requirements\n- 3+ years of experience with Python and SQL\n")
    assert {"python", "sql"} <= set(job.required_skills)


# Title rule: narrow


def test_stated_two_years_as_a_data_analyst_is_supported(status):
    state, item, _ = status(["Experience: 2 years as a Data Analyst."])
    assert state is MatchStatus.SUPPORTED
    assert item.evidence[0].snippet == "Experience: 2 years as a Data Analyst."


def test_dated_data_analyst_role_is_supported(status):
    lines = ["Experience", "Data Analyst, Acme Corp, Jan 2022 - Jan 2024"]
    assert status(lines)[0] is MatchStatus.SUPPORTED


@pytest.mark.parametrize(
    "role", ["Business Analyst", "Financial Analyst", "Analyst", "Reporting Intern"]
)
def test_other_analyst_roles_do_not_prove_data_analysis(status, role):
    state, item, _ = status([f"Experience: 2 years as a {role}."])
    assert state is MatchStatus.UNVERIFIED
    assert "not in data analysis" in item.explanation


def test_analyst_word_in_another_place_does_not_count(status):
    lines = ["Experience", "Analyst, Acme Corp, Jan 2022 - Jan 2024", "Worked with data."]
    assert status(lines)[0] is MatchStatus.UNVERIFIED


def test_data_and_analyst_apart_do_not_count(status):
    lines = ["Experience", "Reporting Analyst, Acme Corp, Jan 2022 - Jan 2024", "Managed data."]
    assert status(lines)[0] is MatchStatus.UNVERIFIED


# Duration


def test_one_year_of_data_analyst_meets_a_one_year_minimum(status):
    assert status(["Experience: 1 year as a Data Analyst."])[0] is MatchStatus.SUPPORTED


def test_six_months_does_not_meet_one_year(status):
    state, _, _ = status(["Experience: 6 months as a Data Analyst."])
    assert state is not MatchStatus.SUPPORTED


def test_six_month_internship_does_not_meet_one_year(status):
    lines = ["Experience", "Data Analyst Intern, Acme, Jan 2024 - Jun 2024"]
    assert status(lines)[0] is not MatchStatus.SUPPORTED


def test_short_stated_years_do_not_meet_a_longer_minimum(status):
    job = "Experience:\n- 3 years of data analysis experience.\n"
    state, _, _ = status(["Experience: 2 years as a Data Analyst."], job)
    assert state is not MatchStatus.SUPPORTED


# Ambiguous evidence


def test_two_statements_are_not_added_or_guessed():
    text = "Experience:\n2 years as a Data Analyst\n3 years as a Developer\n"
    assert parse_resume(text).total_years_experience is None


def test_two_statements_stay_unverified(status):
    lines = ["Experience", "2 years as a Data Analyst", "3 years as a Developer"]
    assert status(lines)[0] is MatchStatus.UNVERIFIED


def test_no_experience_information_stays_unverified(status):
    assert status(["Skills: Python"])[0] is MatchStatus.UNVERIFIED


def test_negated_experience_is_not_credited(status):
    lines = ["Skills: Python", "No experience as a Data Analyst."]
    assert status(lines)[0] is not MatchStatus.SUPPORTED


# Punctuation


def test_skills_lose_the_sentence_ending_full_stop():
    profile = parse_resume("Skills: Python, SQL, Tableau.\n")
    assert profile.skills == ["Python", "SQL", "Tableau"]


def test_dotted_skill_names_are_untouched():
    profile = parse_resume("Skills: Node.js, ASP.NET, Python\n")
    assert profile.skills == ["Node.js", "ASP.NET", "Python"]


def test_stated_months_are_reported_but_not_converted_to_years():
    profile = parse_resume("Experience:\n6 months as a Reporting Intern.\n")
    assert profile.total_years_experience is None
    assert any("not converted into years" in note for note in profile.extraction_notes)


def test_stated_months_note_reaches_the_report(status):
    _, _, report = status(["Experience: 6 months as a Reporting Intern."])
    assert any("duration in months" in line for line in report.limitations)
