"""Job headings, labelled resume sections, and what they mean for the assessments."""

import pytest

from hr_screening.batch import screen_batch
from hr_screening.extraction.job_parser import parse_job_description
from hr_screening.extraction.resume_parser import SectionIndex, parse_resume
from hr_screening.models.report import MatchStatus
from hr_screening.pipeline import screen_resume

JOB = """Job Title: Junior Data Analyst

Required Skills:
- Python
- SQL
- Tableau

Required Education:
- Bachelor's degree in Computer Science, Statistics, or a related field.

Experience:
- 1 year of data analysis experience.
"""
CV_A = [
    "Candidate A",
    "Skills: Python, SQL, Tableau.",
    "Experience: 2 years as a Data Analyst.",
    "Education: Bachelor's degree in Statistics.",
]
CV_B = [
    "Candidate B",
    "Skills: Python, SQL.",
    "Experience: 6 month internship as a Data Analyst.",
    "Education: Bachelor's degree in Computer Science.",
]


@pytest.fixture
def screen(tmp_path, make_docx, today):
    def _run(lines, job=JOB):
        return screen_resume(make_docx(tmp_path / "cv.docx", lines), job, today=today)

    return _run


def by_name(report):
    return {a.requirement: a for a in report.assessments}


# 1 to 5: the job description


def test_headings_are_not_unassessed_requirements():
    job = parse_job_description(JOB)
    assert job.unassessed_lines == []
    assert job.min_years_experience == 1
    assert job.min_degree_level is not None


def test_field_of_study_is_not_a_required_skill():
    job = parse_job_description(JOB)
    assert "statistics" not in job.required_skills
    assert {"python", "sql", "tableau"} <= set(job.required_skills)


def test_a_skill_listed_under_a_skills_heading_is_still_a_skill():
    job = parse_job_description("Required Skills:\n- Statistics\n- Python\n")
    assert set(job.required_skills) == {"statistics", "python"}


def test_unknown_skill_beside_a_known_one_is_surfaced():
    job = parse_job_description("Required Skills:\n- Python and Helm3\n")
    assert job.required_skills == ["python"]
    assert any("Helm3" in line for line in job.unassessed_lines)


def test_genuine_unknown_requirement_stays_visible():
    job = parse_job_description("Required Skills:\n- Python\n- Able to lift 20 kg boxes\n")
    assert job.unassessed_lines == ["Able to lift 20 kg boxes"]


@pytest.mark.parametrize(
    "heading",
    ["Required Skills", "Technical Skills", "Qualifications", "Requirements", "Skills"],
)
def test_common_headings_keep_their_content(heading):
    job = parse_job_description(f"{heading}:\n- Python\n- SQL\n")
    assert job.required_skills == ["python", "sql"]
    assert job.unassessed_lines == []


def test_inline_heading_keeps_its_text():
    job = parse_job_description("Required Education: Bachelor's degree in Statistics.\n")
    assert job.min_degree_level is not None
    assert job.unassessed_lines == []
    assert "statistics" not in job.required_skills


def test_arbitrary_colon_line_is_not_a_heading():
    job = parse_job_description("Requirements:\n- Python\n- Our stack includes: Rust\n")
    assert any("Our stack" in line for line in job.unassessed_lines)


# 6 to 10: the resume


def test_labelled_resume_sections_are_recognised(tmp_path, make_docx):
    from hr_screening.ingestion.loader import load_resume_text

    text = load_resume_text(make_docx(tmp_path / "a.docx", CV_A))
    profile = parse_resume(text)
    assert "Python" in profile.skills and "SQL" in profile.skills
    assert profile.education and profile.education[0].level is not None
    assert profile.total_years_experience == 2.0
    assert profile.extraction_notes == []
    index = SectionIndex(text)
    labels = (("Skills:", "skills"), ("Experience:", "experience"), ("Education:", "education"))
    for label, section in labels:
        assert index.at(text.index(label) + len(label) + 2) == section


def test_text_with_a_colon_is_not_taken_for_a_heading():
    profile = parse_resume("Notes: lots of things\nProject Name: Dashboard\n")
    assert not profile.education
    assert profile.total_years_experience is None


def test_bachelors_in_statistics_is_education_not_a_skill(screen):
    report = screen(CV_A)
    items = by_name(report)
    assert items["Bachelor level education or higher"].status is MatchStatus.SUPPORTED
    assert "statistics" not in items


def test_field_of_study_alone_is_not_skill_evidence(screen):
    job = "Required Skills:\n- Statistics\n"
    report = screen(["Education: Bachelor's degree in Statistics."], job)
    item = by_name(report)["statistics"]
    assert item.status is MatchStatus.UNVERIFIED
    assert "education section" in item.explanation


def test_field_of_study_does_not_give_related_skill_credit(screen):
    job = "Required Skills:\n- Data Analysis\n"
    report = screen(["Education: Bachelor's degree in Statistics."], job)
    assert by_name(report)["data analysis"].status is MatchStatus.NOT_FOUND


def test_skill_in_skills_section_still_counts_when_also_in_education(screen):
    job = "Required Skills:\n- Statistics\n"
    report = screen(["Education: BSc Statistics", "Skills: Statistics, R"], job)
    assert by_name(report)["statistics"].status is MatchStatus.SUPPORTED


def test_stated_two_years_in_the_required_field_is_supported(screen):
    job = "Experience:\n- 1 year of data analysis experience.\n"
    report = screen(["Experience: 2 years of data analysis."], job)
    item = by_name(report)["1+ years of experience"]
    assert item.status is MatchStatus.SUPPORTED
    assert item.evidence[0].snippet == "Experience: 2 years of data analysis."


def test_stated_years_as_a_data_analyst_meet_a_data_analysis_requirement(screen):
    report = screen(CV_A)
    item = by_name(report)["1+ years of experience"]
    assert item.status is MatchStatus.SUPPORTED
    assert item.evidence[0].snippet == "Experience: 2 years as a Data Analyst."


def test_stated_years_in_another_field_stay_unverified(screen):
    report = screen(["Experience: 2 years as a Business Analyst."] + CV_A[:2])
    item = by_name(report)["1+ years of experience"]
    assert item.status is MatchStatus.UNVERIFIED
    assert "not in data analysis" in item.explanation


def test_six_months_is_not_one_year(screen):
    report = screen(CV_B)
    item = by_name(report)["1+ years of experience"]
    assert item.status is MatchStatus.UNVERIFIED


def test_two_stated_year_figures_are_not_guessed():
    text = "Experience:\n2 years as a Data Analyst\n3 years as a Developer\n"
    assert parse_resume(text).total_years_experience is None


def test_missing_sections_still_say_so():
    profile = parse_resume("Just some words about me.\n")
    assert profile.extraction_notes
    assert profile.total_years_experience is None


def test_no_evidence_stays_unverified(screen):
    report = screen(["Candidate", "Enjoys hiking."])
    items = by_name(report)
    assert items["1+ years of experience"].status is MatchStatus.UNVERIFIED
    assert items["Bachelor level education or higher"].status is MatchStatus.UNVERIFIED


# 11: negation and aspiration


def test_negation_and_aspiration_are_unchanged(screen):
    job = "Required Skills:\n- Kubernetes\n- Docker\n"
    report = screen(["Skills: Python", "No experience with Kubernetes.", "Learning Docker."], job)
    items = by_name(report)
    assert items["kubernetes"].status is MatchStatus.NOT_FOUND
    assert items["docker"].status is MatchStatus.NOT_FOUND


# 12: batch workflow


def test_batch_still_orders_flags_and_isolates_errors(tmp_path, make_docx, today):
    folder = tmp_path / "cvs"
    folder.mkdir()
    make_docx(folder / "candidate_a.docx", CV_A)
    make_docx(folder / "candidate_b.docx", CV_B)
    make_docx(folder / "candidate_a_copy.docx", CV_A)
    (folder / "broken.docx").write_bytes(b"not a zip")
    summary = screen_batch(folder, JOB, today=today)
    assert summary.counts.failed == 1
    assert summary.counts.screened == 3
    assert [f.source_file for f in summary.failures] == ["broken.docx"]
    assert summary.results[-1].source_file == "candidate_b.docx"
    assert all(r.score_complete for r in summary.results)
    copies = [r for r in summary.results if r.duplicate_of]
    assert [r.source_file for r in copies] == ["candidate_a_copy.docx"]
