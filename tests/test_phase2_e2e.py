import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from hr_screening import main as main_module
from hr_screening.evaluation.formatting import format_report_text
from hr_screening.evaluation.scoring import CREDIT, WEIGHTS
from hr_screening.extraction.job_parser import parse_job_description
from hr_screening.ingestion.loader import load_resume_text
from hr_screening.llm.fake import FakeLLMClient
from hr_screening.models.report import MatchStatus
from hr_screening.pipeline import screen_resume
from hr_screening.rag.embeddings import FakeEmbedder

ROOT = Path(__file__).resolve().parents[1]
RANK = {"supported": 2, "partial": 1, "not_found": 0, "unverified": 0}


@pytest.fixture
def screen_file(tmp_path, make_docx, today):
    """Screen a generated DOCX through the real pipeline."""

    def _run(lines, job, *, name="cv", **kwargs):
        path = make_docx(tmp_path / f"{name}.docx", lines)
        return screen_resume(path, job, today=today, **kwargs)

    return _run


def item(report, requirement):
    return next(a for a in report.assessments if a.requirement == requirement)


def of_kind(report, kind):
    return [a for a in report.assessments if a.kind.value == kind]


def snippets(assessment):
    return [e.snippet for e in assessment.evidence]


def statuses(report):
    return {a.requirement: a.status.value for a in report.assessments}


def explained_years(assessment):
    return float(re.search(r"About ([\d.]+) years", assessment.explanation).group(1))


def recomputed_score(report):
    """The score from the published weights and credits, worked out here from the statuses."""
    possible = sum(WEIGHTS[a.kind] for a in report.assessments)
    earned = sum(WEIGHTS[a.kind] * CREDIT[a.status] for a in report.assessments)
    return round(100 * earned / possible + 1e-9, 1)


# --- A. strong match ------------------------------------------------------------------------

JOB_A = """Data Engineer

About the role
We build data platforms.

Requirements
- 3+ years of experience in data engineering
- Python and SQL
- Bachelor's degree in computer science
- AWS certification

Nice to have: Spark
"""
RESUME_A = [
    "SKILLS",
    "Python, SQL, Spark",
    "EXPERIENCE",
    "Data Engineer, Sample Ltd   Jan 2019 - Present",
    "Built data engineering pipelines in Python and SQL",
    "EDUCATION",
    "B.Sc. in Computer Science, Example University, 2018",
    "CERTIFICATIONS",
    "AWS Certified Solutions Architect",
]


def test_a_strong_match_is_parsed_grounded_and_scored_by_the_formula(
    screen_file, tmp_path, make_docx
):
    requirements = parse_job_description(JOB_A)
    assert set(requirements.required_skills) == {"python", "sql", "aws"}
    assert requirements.preferred_skills == ["spark"]
    assert requirements.min_years_experience == 3
    assert requirements.experience_field == "data engineering"
    assert requirements.certifications == ["AWS certification"]

    report = screen_file(RESUME_A, JOB_A)
    assert {a.status.value for a in report.assessments} == {"supported"}
    assert report.coverage_score == 100.0 == recomputed_score(report)

    resume_text = load_resume_text(make_docx(tmp_path / "again.docx", RESUME_A))
    for assessment in report.assessments:
        assert assessment.evidence, assessment.requirement
        assert all(snippet in resume_text for snippet in snippets(assessment))

    text = format_report_text(report)
    assert "Decision support only" in text
    assert "human reviewer" in text
    assert report.human_review_required is True


# --- B. partial match -----------------------------------------------------------------------

JOB_B = """Requirements
- 4+ years of experience in data engineering
- Python
- SQL
- Bachelor's degree
- PMP certification
"""
RESUME_B = [
    "SKILLS",
    "Python, MySQL",
    "EXPERIENCE",
    "Data Engineer, Sample Ltd   Jan 2024 - Present",
    "Data engineering for the finance team",
    "EDUCATION",
    "B.Sc. in Computer Science, expected 2027",
]


def test_b_partial_match_keeps_each_level_apart_and_scores_by_the_formula(screen_file):
    report = screen_file(RESUME_B, JOB_B)
    assert statuses(report) == {
        "python": "supported",
        "sql": "partial",  # MySQL is only a related technology
        "4+ years of experience": "partial",  # 2 years found, 4 asked
        "Bachelor level education or higher": "partial",  # in progress, not completed
        "PMP certification": "not_found",
    }
    assert "may be transferable" in item(report, "sql").explanation
    assert (
        "not counted as completed" in item(report, "Bachelor level education or higher").explanation
    )
    # weights 3 + 3 + 3 + 2 + 2 = 13, earned 3 + 1.5 + 1.5 + 1 + 0 = 7, so 53.8
    assert report.coverage_score == 53.8 == recomputed_score(report)
    assert item(report, "PMP certification").evidence == []


# --- C. missing required skills -------------------------------------------------------------

JOB_C = "Requirements\n- Python\n- SQL\n- Docker\n- Java\n"


def test_c_missing_skills_stay_not_found_and_vaguely_similar_skills_earn_nothing(screen_file):
    resume = [
        "SKILLS",
        "R, Excel, JavaScript, Podman",
        "EXPERIENCE",
        "Analyst, A   Jan 2020 - Dec 2021",
    ]
    report = screen_file(resume, JOB_C)
    assert set(statuses(report).values()) == {"not_found"}
    assert all(a.evidence == [] for a in report.assessments)
    assert report.coverage_score == 0.0


def test_c_a_model_cannot_invent_skills_the_resume_does_not_contain(screen_file):
    resume = ["SKILLS", "R, Excel"]
    invented = {"skills": ["Python", "SQL", "Docker", "Java"], "education": ["PhD in physics"]}
    report = screen_file(resume, JOB_C, client=FakeLLMClient(json.dumps(invented)))
    assert set(statuses(report).values()) == {"not_found"}
    assert any("could not be verified" in note for note in report.limitations)


# --- D. contradictory and negated evidence --------------------------------------------------

D_CASES = [
    pytest.param(
        ["SKILLS", "SQL", "SUMMARY", "I am currently learning Python."],
        "not_found",
        "being learned",
        ["I am currently learning Python."],
        id="learning",
    ),
    pytest.param(
        ["SKILLS", "SQL", "SUMMARY", "I no longer use Python."],
        "not_found",
        "negated",
        ["I no longer use Python."],
        id="no_longer",
    ),
    pytest.param(
        ["SKILLS", "Python, SQL", "SUMMARY", "I have no experience with Python in production."],
        "unverified",
        "denying it",
        ["Python, SQL", "I have no experience with Python in production."],
        id="listed_and_denied",
    ),
    pytest.param(
        ["SKILLS", "Python, SQL", "SUMMARY", "Currently learning advanced Python."],
        "supported",
        "learned or planned",
        ["Python, SQL", "Currently learning advanced Python."],
        id="listed_and_learning_both_shown",
    ),
    pytest.param(
        ["SKILLS", "SQL", "INTERESTS", "Python scripting, chess"],
        "unverified",
        "interests, hobbies or references",
        ["Python scripting, chess"],
        id="hobby_only",
    ),
    pytest.param(
        ["SKILLS", "SQL", "REFERENCES", "Jane Doe supervised my Python work"],
        "unverified",
        "interests, hobbies or references",
        ["Jane Doe supervised my Python work"],
        id="reference_only",
    ),
    pytest.param(
        ["SKILLS", "Python, SQL", "INTERESTS", "Python scripting, chess"],
        "supported",
        "mentioned in the resume",
        ["Python, SQL"],
        id="skills_section_beats_hobby",
    ),
]


@pytest.mark.parametrize(("resume", "status", "words", "evidence"), D_CASES)
def test_d_status_note_and_evidence_follow_the_source_text(
    screen_file, resume, status, words, evidence
):
    report = screen_file(resume, "Requirements\n- Python\n- SQL\n")
    python = item(report, "python")
    assert python.status.value == status
    assert words in python.explanation
    assert snippets(python) == evidence
    assert item(report, "sql").status.value == "supported"


@pytest.mark.parametrize(("resume", "status", "words", "evidence"), D_CASES)
def test_d_a_model_that_claims_the_skill_changes_nothing(
    screen_file, resume, status, words, evidence
):
    claim = FakeLLMClient(json.dumps({"skills": ["Python", "SQL"]}))
    report = screen_file(resume, "Requirements\n- Python\n- SQL\n", client=claim)
    rules = screen_file(resume, "Requirements\n- Python\n- SQL\n", name="rules")
    assert report.assessments == rules.assessments
    assert report.coverage_score == rules.coverage_score


# --- E. education edge cases ----------------------------------------------------------------

E_CASES = [
    # (resume line, expected status, the shorter text a model might return for it)
    ("B.Sc. in Statistics, State University, 2019", "supported", "B.Sc. in Statistics"),
    ("B.Sc. in Statistics, expected 2027", "partial", "B.Sc. in Statistics"),
    ("B.Sc. in Statistics, 2023 - Present", "partial", "B.Sc. in Statistics"),
    (
        "Pursuing a Bachelor of Science in Statistics",
        "partial",
        "Bachelor of Science in Statistics",
    ),
    (
        "Working towards a Bachelor of Science in Statistics",
        "partial",
        "Bachelor of Science in Statistics",
    ),
    ("B.Sc. in Statistics (dropped out)", "unverified", "B.Sc. in Statistics"),
    ("B.Sc. in Statistics (did not complete)", "unverified", "B.Sc. in Statistics"),
    (
        "Bachelor of Science in Statistics, degree not awarded",
        "unverified",
        "Bachelor of Science in Statistics",
    ),
    (
        "Completed coursework toward a Bachelor of Science in Statistics",
        "unverified",
        "Bachelor of Science in Statistics",
    ),
    (
        "Completed 90 credits towards a Bachelor of Science in Statistics",
        "unverified",
        "Bachelor of Science in Statistics",
    ),
    (
        "Bachelor of Science in Statistics, some college",
        "unverified",
        "Bachelor of Science in Statistics",
    ),
    ("Partial Bachelor degree in Statistics", "unverified", "Bachelor degree in Statistics"),
    ("Studied statistics at State University", "unverified", "Studied statistics"),
]


@pytest.mark.parametrize(("line", "status", "span"), E_CASES)
def test_e_an_unfinished_degree_is_never_a_completed_qualification(screen_file, line, status, span):
    report = screen_file(
        ["SKILLS", "SQL", "EDUCATION", line], "Requirements\n- Bachelor's degree\n"
    )
    education = of_kind(report, "education")[0]
    assert education.status.value == status
    if not line.startswith("Studied"):
        assert snippets(education) == [line]


@pytest.mark.parametrize(("line", "status", "span"), E_CASES)
def test_e_a_model_that_trims_the_qualifier_cannot_make_the_degree_complete(
    screen_file, line, status, span
):
    resume = ["SKILLS", "SQL", "EDUCATION", line]
    job = "Requirements\n- Bachelor's degree\n"
    client = FakeLLMClient(json.dumps({"education": [span]}))
    model = of_kind(screen_file(resume, job, client=client, name="model"), "education")[0]
    rules = of_kind(screen_file(resume, job, name="rules"), "education")[0]
    assert model.status == rules.status
    assert RANK[model.status.value] <= RANK[status]


@pytest.mark.parametrize(
    "line",
    [
        "Experience with Master Data Management",
        "Scrum Master experience",
        "Supporting associate engineers",
        "Associate Data Engineer role",
    ],
)
def test_e_ordinary_words_in_a_job_do_not_create_a_degree_requirement(screen_file, line):
    resume = ["SKILLS", "Python", "EDUCATION", "Diploma in Culinary Arts, 2013"]
    report = screen_file(resume, f"Requirements\n- Python\n- {line}\n")
    assert of_kind(report, "education") == []
    assert report.coverage_score == 100.0


# --- F. certification edge cases ------------------------------------------------------------

CERT_JOB = "Requirements\n- Python\n- AWS certification\n"
CERT_BASE = ["SKILLS", "Python, AWS"]
F_CASES = [
    (["CERTIFICATIONS", "AWS Certified Solutions Architect Associate"], "supported"),
    (["CERTIFICATIONS", "Certified Scrum Master", "PMP"], "unverified"),
    (["EXPERIENCE", "Deployed services on AWS and earned Certified Scrum Master"], "unverified"),
    (["EXPERIENCE", "Managed AWS migrations. Certified Scrum Master."], "unverified"),
    (["EXPERIENCE", "Managed AWS migrations; Scrum Master certified"], "unverified"),
    (
        ["EXPERIENCE", "Managed AWS migrations, then passed the Certified Scrum Master exam"],
        "unverified",
    ),
    (["CERTIFICATIONS", "AWS Solutions Architect (in progress, expected 2026)"], "unverified"),
    (["CERTIFICATIONS", "AWS Certified Cloud Practitioner (expired 2022)"], "unverified"),
    (["SUMMARY", "Planning to get an AWS certification next year"], "unverified"),
    (["EXPERIENCE", "Used AWS daily"], "unverified"),
]


@pytest.mark.parametrize(("extra", "status"), F_CASES)
def test_f_certification_evidence_must_belong_to_the_certification(screen_file, extra, status):
    report = screen_file([*CERT_BASE, *extra], CERT_JOB)
    certification = item(report, "AWS certification")
    assert certification.status.value == status
    assert report.coverage_score == recomputed_score(report)


DEGREE_OR_CERT = (
    "Requirements\n- Python\n- Bachelor's degree in computer science or relevant certification\n"
)
DEGREE_RESUME = ["SKILLS", "Python", "EDUCATION", "B.Sc. in Computer Science, 2019"]


def test_f_degree_or_certification_is_scored_as_the_degree_and_says_so(screen_file):
    report = screen_file(DEGREE_RESUME, DEGREE_OR_CERT)
    assert [a.kind.value for a in report.assessments] == ["required_skill", "education"]
    assert report.coverage_score == 100.0
    assert any("alternative to a degree" in note for note in report.limitations)
    only_cert = screen_file(
        ["SKILLS", "Python", "CERTIFICATIONS", "Cloud Practitioner"], DEGREE_OR_CERT
    )
    assert item(only_cert, "Bachelor level education or higher").status.value == "unverified"


def test_f_a_skill_named_on_a_degree_or_certification_line_is_still_scored_and_the_note_says_so(
    screen_file,
):
    job = "Requirements\n- SQL\n- Bachelor's degree or AWS certification\n"
    report = screen_file(["SKILLS", "SQL", "EDUCATION", "B.Sc. in Computer Science, 2019"], job)
    assert item(report, "aws").status.value == "not_found"
    note = next(n for n in report.limitations if "alternative to a degree" in n)
    assert "skill named on that line" in note


DEGREE_AND_CERT = "Requirements\n- SQL\n- Bachelor's degree and PMP certification\n"


def test_f_degree_and_certification_on_one_line_is_documented_not_silently_modelled(screen_file):
    both = [*DEGREE_RESUME, "CERTIFICATIONS", "PMP certification"]
    report = screen_file(both, DEGREE_AND_CERT)
    combined = of_kind(report, "certification")[0]
    assert combined.requirement == "Bachelor's degree and PMP certification"
    assert combined.status is not MatchStatus.SUPPORTED  # the whole line cannot be matched
    assert item(report, "Bachelor level education or higher").status.value == "supported"
    assert any("combines a degree and a certification" in note for note in report.limitations)


# --- G. experience relevance and role boundaries -------------------------------------------

EXPERIENCE_JOB = "Requirements\n- 4+ years of experience in data analysis\n"
SKILLS = ["SKILLS", "Python"]


def test_g_long_relevant_experience_in_the_title_is_supported(screen_file):
    resume = [*SKILLS, "EXPERIENCE", "Senior Data Analysis Lead, Acme   Jan 2019 - Present", "x"]
    experience = of_kind(screen_file(resume, EXPERIENCE_JOB), "experience")[0]
    assert experience.status.value == "supported"
    # Jan 2019 to the fixed test date (2026-01-01) is exactly 7 years. The clock is not used.
    assert explained_years(experience) == 7.0


def test_g_long_unrelated_experience_is_not_presented_as_verified(screen_file):
    resume = [*SKILLS, "EXPERIENCE", "Pastry Chef, Cafe Rose   Jan 2012 - Present", "Baked bread"]
    report = screen_file(resume, EXPERIENCE_JOB)
    experience = of_kind(report, "experience")[0]
    assert experience.status.value == "unverified"
    assert "no role mentions data analysis" in experience.explanation
    assert report.coverage_score == recomputed_score(report)


def test_g_only_the_relevant_roles_are_counted_among_several(screen_file):
    resume = [
        *SKILLS,
        "EXPERIENCE",
        "Data Analysis Lead, A   Jan 2018 - Dec 2019",
        "Reports",
        "Waiter, B   Jan 2020 - Dec 2021",
        "Served tables",
        "Data Analysis Lead, C   Jan 2022 - Dec 2024",
        "Dashboards",
    ]
    experience = of_kind(screen_file(resume, EXPERIENCE_JOB), "experience")[0]
    assert experience.status.value == "supported"
    assert 4.5 <= explained_years(experience) <= 5.0  # 2 + 3, the waiter's 2 are not counted
    assert "Waiter, B   Jan 2020 - Dec 2021" not in snippets(experience)


def test_g_overlapping_relevant_roles_are_merged_not_added(screen_file):
    resume = [
        *SKILLS,
        "EXPERIENCE",
        "Data Analysis Lead, A   Jan 2018 - Dec 2021",
        "x",
        "Data Analysis Lead, B   Jan 2020 - Dec 2022",
        "y",
    ]
    experience = of_kind(screen_file(resume, EXPERIENCE_JOB), "experience")[0]
    assert 4.5 <= explained_years(experience) <= 5.0  # Jan 2018 to Dec 2022, not 4 + 3


UNDATED_ROLES = [
    pytest.param("Data Analysis Intern, Foo (summer 2022)", id="undated_with_a_year"),
    pytest.param("Data Analysis Intern, Foo Corp", id="undated_without_any_year"),
    pytest.param("Data Analysis Intern at Foo Corp", id="undated_at_company"),
]


@pytest.mark.parametrize("intern", UNDATED_ROLES)
def test_g_an_undated_role_after_a_dated_one_does_not_borrow_its_years(screen_file, intern):
    resume = [
        *SKILLS,
        "EXPERIENCE",
        "Pastry Chef, Cafe Rose   Jan 2012 - Dec 2021",
        "Baked bread",
        intern,
        "Did data analysis",
    ]
    report = screen_file(resume, EXPERIENCE_JOB)
    experience = of_kind(report, "experience")[0]
    assert experience.status.value == "unverified"
    assert "9.9" not in experience.explanation
    assert intern in snippets(experience)
    assert report.coverage_score == recomputed_score(report)


@pytest.mark.parametrize("intern", UNDATED_ROLES)
def test_g_the_model_path_agrees_for_an_undated_role(screen_file, intern):
    resume = [
        *SKILLS,
        "EXPERIENCE",
        "Pastry Chef, Cafe Rose   Jan 2012 - Dec 2021",
        "Baked bread",
        intern,
        "Did data analysis",
    ]
    payload = {
        "experience": [
            {"heading": "Pastry Chef, Cafe Rose", "start": "Jan 2012", "end": "Dec 2021"},
            {"heading": intern, "start": "Jan 2012", "end": "Dec 2021"},  # borrowed dates
        ]
    }
    model = screen_file(resume, EXPERIENCE_JOB, client=FakeLLMClient(json.dumps(payload)))
    assert of_kind(model, "experience")[0].status.value == "unverified"


def test_g_a_bulleted_description_does_not_end_a_role(screen_file):
    resume = [
        *SKILLS,
        "EXPERIENCE",
        "Analyst, Acme   Jan 2018 - Present",
        "- Led the data analysis team",
        "- Built dashboards",
    ]
    assert of_kind(screen_file(resume, EXPERIENCE_JOB), "experience")[0].status.value == "supported"


def test_g_a_sentence_description_does_not_end_a_role(screen_file):
    resume = [
        *SKILLS,
        "EXPERIENCE",
        "Analyst, Acme   Jan 2018 - Present",
        "Led the data analysis team and built weekly dashboards for finance",
        "Reported results to senior management",
    ]
    assert of_kind(screen_file(resume, EXPERIENCE_JOB), "experience")[0].status.value == "supported"


ORDINARY_LAYOUTS = {
    "tools_line": ["Engineer, Acme Corp   Jan 2019 - Present", "Tools: Python, SQL, Airflow"],
    "location_line": ["Engineer, Acme Corp   Jan 2019 - Present", "Pune, India"],
    "title_below_company": ["Acme Corp   Jan 2019 - Present", "Data Engineer"],
    "title_above_company": ["Data Engineer", "Acme Corp | Pune   Jan 2019 - Present"],
    "capitalised_phrase": [
        "Engineer, Acme Corp   Jan 2019 - Present",
        "Worked With Teams, Finance",
    ],
}


@pytest.mark.parametrize("layout", list(ORDINARY_LAYOUTS))
def test_g_ordinary_resume_layouts_are_not_cut_into_false_undated_roles(screen_file, layout):
    resume = [*SKILLS, "EXPERIENCE", *ORDINARY_LAYOUTS[layout], "Built data engineering pipelines"]
    job = "Requirements\n- 3+ years of experience in data engineering\n"
    assert of_kind(screen_file(resume, job), "experience")[0].status.value == "supported"


def test_g_a_requirement_with_no_field_counts_total_years_and_says_so(screen_file):
    resume = [*SKILLS, "EXPERIENCE", "Pastry Chef, Cafe Rose   Jan 2012 - Present", "Baked bread"]
    report = screen_file(resume, "Requirements\n- 5+ years of experience\n")
    experience = of_kind(report, "experience")[0]
    assert experience.status.value == "supported"
    assert "without checking whether it is relevant" in experience.explanation


# --- H. job description parsing -------------------------------------------------------------


def parsed(job):
    requirements = parse_job_description(job)
    return requirements


def test_h_common_headings_and_inline_labels_give_the_right_kinds():
    job = "Requirements\n- Python\n- SQL\n- Nice to have: Spark\n\nPreferred\n- Tableau\n"
    requirements = parsed(job)
    assert requirements.required_skills == ["python", "sql"]
    assert set(requirements.preferred_skills) == {"spark", "tableau"}


def test_h_an_unfamiliar_heading_is_reported_not_silently_dropped(screen_file):
    job = "Who You Are\n- Python\n- 3+ years of experience\n\nNice to have\n- Spark\n"
    report = screen_file(["SKILLS", "Python"], job)
    assert [a.requirement for a in report.assessments] == ["spark"]
    assert any("does not recognise" in note and "python" in note for note in report.limitations)


def test_h_several_experience_year_requirements_are_reported():
    job = (
        "Requirements\n- 5+ years of experience in data analysis\n"
        "- 2+ years of experience with SQL\n"
    )
    requirements = parsed(job)
    assert requirements.min_years_experience == 5
    assert any("2 experience year requirements" in note for note in requirements.notes)


OPTIONAL_LINES = [
    "Python is not required",
    "Python (optional)",
    "No Python experience needed",
    "Python is a plus",
    "Python is nice to have",
]


@pytest.mark.parametrize("line", OPTIONAL_LINES)
def test_h_a_skill_the_job_says_is_optional_is_not_scored_as_required(line, screen_file):
    job = f"Requirements\n- SQL\n- {line}\n"
    requirements = parsed(job)
    assert requirements.required_skills == ["sql"]
    assert requirements.preferred_skills == ["python"]
    assert any("optional" in note and "python" in note for note in requirements.notes)
    report = screen_file(["SKILLS", "SQL"], job)
    assert item(report, "python").kind.value == "preferred_skill"
    assert report.coverage_score == 75.0  # SQL 3 of 3, Python 0 of 1, so 3 of 4


@pytest.mark.parametrize(
    "line",
    [
        "Python and SQL are required",
        "Strong Python skills",
        "Python, not Java",
        "Experience with Python (not required: R)",
    ],
)
def test_h_required_skills_stay_required_when_no_optional_wording_belongs_to_them(line):
    assert "python" in parsed(f"Requirements\n- {line}\n").required_skills


def test_h_a_required_skill_named_again_as_optional_stays_required():
    job = "Requirements\n- Python\n- Python is a plus for the data team\n"
    requirements = parsed(job)
    assert requirements.required_skills == ["python"]
    assert requirements.preferred_skills == []


def test_h_a_preferred_degree_or_experience_is_reported_as_not_scored():
    job = "Requirements\n- SQL\n\nPreferred\n- Master's degree\n- 5+ years of experience\n"
    requirements = parsed(job)
    assert requirements.min_degree_level is None
    assert requirements.min_years_experience is None
    assert any("preferred" in note.lower() and "not scored" in note for note in requirements.notes)


# --- I. ingestion and the command line ------------------------------------------------------


@pytest.fixture
def cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(main_module, "configure_logging", lambda: None)

    def _run(*argv):
        code = main_module.main([str(a) for a in argv])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return _run


@pytest.fixture
def job_file(tmp_path, sample_job):
    path = tmp_path / "job.txt"
    path.write_text(sample_job, encoding="utf-8")
    return path


def test_i_a_valid_docx_and_a_valid_pdf_give_the_same_report(
    cli, tmp_path, make_docx, make_pdf, sample_resume_lines, job_file
):
    docx = make_docx(tmp_path / "cv.docx", sample_resume_lines)
    pdf = make_pdf(tmp_path / "cv.pdf", sample_resume_lines)
    codes_and_reports = []
    for resume in (docx, pdf):
        code, out, err = cli("--resume", resume, "--job-file", job_file, "--json")
        assert (code, err) == (0, "")
        codes_and_reports.append(json.loads(out))
    assert codes_and_reports[0] == codes_and_reports[1]
    assert codes_and_reports[0]["coverage_score"] == 55.9


def test_i_json_output_has_the_documented_fields(
    cli, tmp_path, make_docx, sample_resume_lines, job_file
):
    resume = make_docx(tmp_path / "cv.docx", sample_resume_lines)
    code, out, _ = cli("--resume", resume, "--job-file", job_file, "--json")
    report = json.loads(out)
    assert code == 0
    assert set(report) == {
        "assessments",
        "coverage_score",
        "human_review_required",
        "limitations",
        "score_explanation",
        "score_complete",
        "unassessed_requirements",
    }
    assert report["human_review_required"] is True
    assert report["score_complete"] is True
    assert report["unassessed_requirements"] == []
    for assessment in report["assessments"]:
        assert set(assessment) == {"requirement", "kind", "status", "evidence", "explanation"}
        assert assessment["status"] in {"supported", "partial", "not_found", "unverified"}


def test_i_job_file_and_job_text_give_identical_output(
    cli, tmp_path, make_docx, sample_resume_lines, sample_job, job_file
):
    resume = make_docx(tmp_path / "cv.docx", sample_resume_lines)
    from_file = cli("--resume", resume, "--job-file", job_file, "--json")
    from_text = cli("--resume", resume, "--job-text", sample_job, "--json")
    assert from_file == from_text
    assert from_file[0] == 0


def test_i_the_text_report_states_that_a_human_must_decide(
    cli, tmp_path, make_docx, sample_resume_lines, job_file
):
    resume = make_docx(tmp_path / "cv.docx", sample_resume_lines)
    code, out, _ = cli("--resume", resume, "--job-file", job_file)
    assert code == 0
    assert "Decision support only" in out
    assert "REQUIREMENTS" in out


def test_i_resume_dir_allows_inside_and_blocks_outside_and_traversal(
    cli, tmp_path, make_docx, sample_resume_lines
):
    allowed = tmp_path / "incoming"
    allowed.mkdir()
    inside = make_docx(allowed / "a.docx", sample_resume_lines)
    outside = make_docx(tmp_path / "b.docx", sample_resume_lines)
    job = "Requirements\n- Python\n"
    assert cli("--resume", inside, "--resume-dir", allowed, "--job-text", job)[0] == 0
    for target in (outside, allowed / ".." / "b.docx"):
        code, out, err = cli("--resume", target, "--resume-dir", allowed, "--job-text", job)
        assert (code, out) == (2, "")
        assert "outside the allowed directory" in err


def _bad_files(tmp_path):
    corrupt_pdf = tmp_path / "bad.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.4 garbage")
    corrupt_docx = tmp_path / "bad.docx"
    corrupt_docx.write_bytes(b"PK\x03\x04junk")
    fake_docx = tmp_path / "fake.docx"
    fake_docx.write_text("plain text, not a zip", encoding="utf-8")
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    text = tmp_path / "cv.txt"
    text.write_text("hello", encoding="utf-8")
    legacy = tmp_path / "cv.doc"
    legacy.write_bytes(b"\xd0\xcf\x11\xe0")
    return {
        "corrupt_pdf": (corrupt_pdf, "could not be read"),
        "corrupt_docx": (corrupt_docx, "does not look like a valid .docx"),
        "fake_docx": (fake_docx, "does not look like a valid .docx"),
        "empty": (empty, "empty"),
        "txt": (text, "Unsupported file type '.txt'"),
        "legacy_doc": (legacy, "Legacy .doc"),
        "missing": (tmp_path / "nope.docx", "not found"),
        "directory": (tmp_path, "Unsupported"),
    }


@pytest.mark.parametrize(
    "name",
    [
        "corrupt_pdf",
        "corrupt_docx",
        "fake_docx",
        "empty",
        "txt",
        "legacy_doc",
        "missing",
        "directory",
    ],
)
def test_i_bad_resume_files_fail_with_a_clear_safe_error(cli, tmp_path, name):
    path, expected = _bad_files(tmp_path)[name]
    code, out, err = cli("--resume", path, "--job-text", "Requirements\n- Python\n")
    assert (code, out) == (2, "")
    assert err.startswith("Error: ")
    assert expected in err
    assert "Traceback" not in err
    assert str(tmp_path) not in err


def test_i_a_pdf_with_no_text_and_a_docx_with_no_text_ask_for_a_usable_file(
    cli, tmp_path, make_blank_pdf, make_docx
):
    code, _, err = cli("--resume", make_blank_pdf(tmp_path / "scan.pdf"), "--job-text", "Python")
    assert code == 2 and "OCR" in err
    code, _, err = cli("--resume", make_docx(tmp_path / "e.docx", [""]), "--job-text", "Python")
    assert code == 2 and "no text" in err


def test_i_bad_job_inputs_fail_cleanly(cli, tmp_path, make_docx, sample_resume_lines):
    resume = make_docx(tmp_path / "cv.docx", sample_resume_lines)
    huge = tmp_path / "huge.txt"
    huge.write_text("x" * 300_000, encoding="utf-8")
    cases = [
        (("--job-text", "   "), "empty"),
        (("--job-file", tmp_path / "missing.txt"), "could not be read"),
        (("--job-file", tmp_path), "could not be read"),
        (("--job-file", huge), "too large"),
    ]
    for args, expected in cases:
        code, out, err = cli("--resume", resume, *args)
        assert (code, out) == (2, "")
        assert expected in err and "Traceback" not in err
        assert str(tmp_path) not in err


def test_i_conflicting_or_missing_job_arguments_are_rejected_by_the_parser(
    tmp_path, make_docx, sample_resume_lines, job_file, monkeypatch
):
    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    resume = str(make_docx(tmp_path / "cv.docx", sample_resume_lines))
    for argv in (
        ["--resume", resume],
        ["--resume", resume, "--job-text", "x", "--job-file", str(job_file)],
    ):
        with pytest.raises(SystemExit) as stopped:
            main_module.main(argv)
        assert stopped.value.code == 2


def _subprocess(tmp_path, *argv):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    for name in ("LLM_API_KEY", "LLM_PROVIDER", "LLM_MODEL"):
        env.pop(name, None)
    return subprocess.run(
        [sys.executable, "-m", "hr_screening.main", *argv],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=tmp_path,
        env=env,
        timeout=60,
    )


def test_i_the_real_program_runs_and_shows_help(tmp_path):
    result = _subprocess(tmp_path, "--help")
    assert result.returncode == 0
    assert "Decision support only" in result.stdout


def test_i_the_real_program_reports_a_corrupt_pdf_without_a_traceback(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 garbage")
    result = _subprocess(tmp_path, "--resume", str(bad), "--job-text", "Requirements\n- Python")
    assert result.returncode == 2
    assert "Error: The PDF could not be read" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


def test_i_the_real_program_screens_a_pdf_end_to_end(
    tmp_path, make_pdf, sample_resume_lines, sample_job
):
    pdf = make_pdf(tmp_path / "cv.pdf", sample_resume_lines)
    result = _subprocess(tmp_path, "--resume", str(pdf), "--job-text", sample_job, "--json")
    assert result.returncode == 0
    assert json.loads(result.stdout)["coverage_score"] == 55.9


# --- J. model output and fallback -----------------------------------------------------------

VALID_PAYLOAD = {
    "skills": ["Python", "SQL"],
    "education": ["B.Sc. in Computer Science, Example University, 2018"],
    "certifications": ["AWS Certified Solutions Architect"],
    "projects": [],
    "experience": [{"heading": "Data Engineer, Sample Ltd", "start": "Jan 2019", "end": "Present"}],
}


def test_j_valid_model_output_scores_the_same_as_the_rules(screen_file):
    rules = screen_file(RESUME_A, JOB_A, name="rules")
    model = screen_file(RESUME_A, JOB_A, client=FakeLLMClient(json.dumps(VALID_PAYLOAD)))
    assert model.assessments == rules.assessments
    assert model.coverage_score == rules.coverage_score == 100.0
    assert any("language model performed the extraction" in n for n in model.limitations)


@pytest.mark.parametrize(
    "client",
    [
        FakeLLMClient("this is not json"),
        FakeLLMClient("[1, 2, 3]"),
        FakeLLMClient('{"skills": "Python"}'),
        FakeLLMClient(error=RuntimeError("provider down")),
    ],
    ids=["not_json", "not_object", "wrong_schema", "provider_failure"],
)
def test_j_malformed_output_and_provider_failure_fall_back_to_the_rules(screen_file, client):
    rules = screen_file(RESUME_A, JOB_A, name="rules")
    report = screen_file(RESUME_A, JOB_A, client=client)
    assert report.assessments == rules.assessments
    assert report.coverage_score == rules.coverage_score
    assert any("rule based parser was used instead" in n for n in report.limitations)
    assert not any("provider down" in n for n in report.limitations)


def test_j_missing_fields_can_only_make_the_result_less_favourable(screen_file):
    rules = screen_file(RESUME_A, JOB_A, name="rules")
    model = screen_file(RESUME_A, JOB_A, client=FakeLLMClient("{}"))
    for before, after in zip(rules.assessments, model.assessments, strict=True):
        assert RANK[after.status.value] <= RANK[before.status.value], before.requirement
    assert statuses(model)["Bachelor level education or higher"] == "unverified"
    assert model.coverage_score <= rules.coverage_score


def test_j_a_model_cannot_give_itself_a_degree_the_resume_does_not_state(screen_file):
    resume = ["SKILLS", "SQL", "EDUCATION", "Diploma in Culinary Arts, 2013"]
    payload = {"education": ["Master of Science in Data Science"]}
    report = screen_file(
        resume, "Requirements\n- Master's degree\n", client=FakeLLMClient(json.dumps(payload))
    )
    assert of_kind(report, "education")[0].status.value == "unverified"
    assert any("could not be verified" in n for n in report.limitations)


# --- K. retrieval invariance ----------------------------------------------------------------

K_SCENARIOS = {
    "strong": (RESUME_A, JOB_A),
    "partial": (RESUME_B, JOB_B),
    "missing": (["SKILLS", "R, Excel"], JOB_C),
    "learning": (D_CASES[0].values[0], "Requirements\n- Python\n- SQL\n"),
    "hobby": (D_CASES[4].values[0], "Requirements\n- Python\n- SQL\n"),
    "unfinished_degree": (
        ["SKILLS", "SQL", "EDUCATION", "Completed coursework toward a Bachelor of Science"],
        "Requirements\n- Bachelor's degree\n",
    ),
    "unrelated_cert": (
        [*CERT_BASE, "EXPERIENCE", "Deployed services on AWS and earned Certified Scrum Master"],
        CERT_JOB,
    ),
    "undated_role": (
        [
            *SKILLS,
            "EXPERIENCE",
            "Pastry Chef, Cafe Rose   Jan 2012 - Dec 2021",
            "Baked bread",
            "Data Analysis Intern, Foo Corp",
            "Did data analysis",
        ],
        EXPERIENCE_JOB,
    ),
    "degree_or_cert": (DEGREE_RESUME, DEGREE_OR_CERT),
}
K_MODES = ["lexical", "semantic", "model_lexical", "model_semantic"]


def _run_mode(path, job, today, mode):
    client = FakeLLMClient(json.dumps(VALID_PAYLOAD)) if mode.startswith("model") else None
    rag = mode.split("_")[-1]
    embedder = FakeEmbedder(dimensions=2048) if rag == "semantic" else None
    return screen_resume(path, job, today=today, client=client, rag_mode=rag, embedder=embedder)


@pytest.mark.parametrize("mode", K_MODES)
@pytest.mark.parametrize("name", list(K_SCENARIOS))
def test_k_retrieval_changes_only_retrieval_fields_and_documented_notes(
    name, mode, tmp_path, make_docx, today
):
    lines, job = K_SCENARIOS[name]
    path = make_docx(tmp_path / "cv.docx", lines)
    client = FakeLLMClient(json.dumps(VALID_PAYLOAD)) if mode.startswith("model") else None
    plain = screen_resume(path, job, today=today, client=client)
    report = _run_mode(path, job, today, mode)

    assert report.assessments == plain.assessments
    assert report.coverage_score == plain.coverage_score == recomputed_score(plain)
    assert report.score_explanation == plain.score_explanation
    assert report.human_review_required is True
    assert report.model_dump(exclude={"retrieval", "limitations"}) == plain.model_dump(
        exclude={"retrieval", "limitations"}
    )
    assert report.limitations[: len(plain.limitations)] == plain.limitations
    extra = report.limitations[len(plain.limitations) :]
    assert len(extra) == 1
    assert "not evidence that a requirement is met" in extra[0] or "is not evidence" in extra[0]


@pytest.mark.parametrize("mode", ["lexical", "semantic"])
@pytest.mark.parametrize("name", list(K_SCENARIOS))
def test_k_every_passage_is_an_exact_slice_of_the_resume_and_never_proof(
    name, mode, tmp_path, make_docx, today
):
    lines, job = K_SCENARIOS[name]
    path = make_docx(tmp_path / "cv.docx", lines)
    resume_text = load_resume_text(path)
    report = _run_mode(path, job, today, mode)
    assert [e.requirement for e in report.retrieval] == [a.requirement for a in report.assessments]
    for entry in report.retrieval:
        for passage in entry.passages:
            assert resume_text[passage.start : passage.end] == passage.text
    text = format_report_text(report)
    assert "not evidence that a requirement is met" in text
    assert "does not change any status or the score" in text