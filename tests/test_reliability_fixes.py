
import json
from datetime import date

import pytest

from hr_screening.evaluation.evaluator import evaluate
from hr_screening.extraction.degrees import detect_requested_degree_level
from hr_screening.extraction.job_parser import parse_job_description
from hr_screening.extraction.resume_parser import parse_resume
from hr_screening.llm.extractor import extract_profile_with_llm
from hr_screening.llm.fake import FakeLLMClient
from hr_screening.models.candidate import DegreeLevel
from hr_screening.pipeline import screen_resume

TODAY = date(2026, 1, 1)


def screen(resume_lines, job):
    text = "\n".join(resume_lines)
    profile = parse_resume(text, today=TODAY)
    return evaluate(profile, parse_job_description(job), text, today=TODAY)


def screen_with_model(resume_lines, job, payload):
    text = "\n".join(resume_lines)
    profile, _ = extract_profile_with_llm(FakeLLMClient(json.dumps(payload)), text, today=TODAY)
    return evaluate(profile, parse_job_description(job), text, today=TODAY)


def find(report, requirement):
    return next(a for a in report.assessments if a.requirement == requirement)


def snippets(assessment):
    return [e.snippet for e in assessment.evidence]


def year_notes(requirements):
    return [note for note in requirements.notes if "experience year" in note]


# --- F1: required skills must not vanish behind an unrecognised heading ----------------------


@pytest.mark.parametrize(
    "heading",
    ["Basic Requirements", "Minimum Requirements", "Basic Qualifications", "Key Requirements"],
)
def test_f1_common_requirement_headings_are_recognised(heading):
    job = f"{heading}\n- Python\n- SQL\n\nPreferred Qualifications\n- Spark\n"
    requirements = parse_job_description(job)
    assert requirements.required_skills == ["python", "sql"]
    assert requirements.preferred_skills == ["spark"]
    assert requirements.notes == []


def test_f1_an_unknown_heading_gets_a_note_and_nothing_is_invented():
    job = "Technical Must Haves\n- Python\n- SQL\n\nNice to have\n- Spark\n"
    requirements = parse_job_description(job)
    assert requirements.required_skills == []  # not guessed
    assert requirements.preferred_skills == ["spark"]
    (note,) = requirements.notes
    assert "does not recognise" in note
    assert "python" in note and "sql" in note
    report = screen(["SKILLS", "Spark"], job)
    assert any("does not recognise" in item for item in report.limitations)
    assert [a.requirement for a in report.assessments] == ["spark"]


def test_f1_a_normal_job_gets_no_such_note():
    job = "Requirements\n- Python\n\nResponsibilities\n- Use SQL daily\n\nPreferred\n- Spark\n"
    assert parse_job_description(job).notes == []


# --- F2: the model must not be able to remove "pursuing" or "did not complete" ---------------

F2_CASES = [
    pytest.param(
        "Pursuing Master's degree in Data Science, expected 2027",
        "Master's degree",
        "Master's degree in Data Science",
        "partial",
        id="pursuing",
    ),
    pytest.param(
        "B.Sc. in Computer Science (did not complete)",
        "Bachelor's degree",
        "B.Sc. in Computer Science",
        "unverified",
        id="did_not_complete",
    ),
    pytest.param(
        "Bachelor's degree in Statistics, expected 2027",
        "Bachelor's degree",
        "Bachelor's degree in Statistics",
        "partial",
        id="expected",
    ),
]


@pytest.mark.parametrize(("line", "requirement", "span", "expected"), F2_CASES)
def test_f2_a_trimmed_degree_is_judged_on_the_whole_resume_line(
    line, requirement, span, expected
):
    resume = ["EDUCATION", line]
    job = f"Requirements\n- {requirement}\n"
    rules = screen(resume, job).assessments[0]
    model = screen_with_model(resume, job, {"education": [span]}).assessments[0]
    assert rules.status.value == expected
    assert model.status == rules.status
    assert snippets(model) == [line]


def test_f2_the_model_path_and_rules_agree_on_a_finished_degree():
    line = "B.Tech in Computer Science, Example Institute, 2018"
    resume = ["EDUCATION", line]
    job = "Requirements\n- Bachelor's degree\n"
    model = screen_with_model(resume, job, {"education": ["B.Tech in Computer Science"]})
    assert model.assessments[0].status.value == "supported"
    assert snippets(model.assessments[0]) == [line]
    assert model.assessments[0] == screen(resume, job).assessments[0]


# --- F3: a certification marker elsewhere on the line is not evidence ---------------------


@pytest.mark.parametrize(
    "line",
    [
        "Managed AWS infrastructure. Certified Scrum Master.",
        "Managed AWS infrastructure, Certified Scrum Master",
        "Managed AWS infrastructure and certified Scrum Master",
        "Built AWS pipelines; PMP certified",
    ],
    ids=["sentences", "comma", "and", "semicolon"],
)
def test_f3_an_unrelated_certification_does_not_prove_aws(line):
    resume = ["EXPERIENCE", "Engineer, X   Jan 2020 - Dec 2022", line]
    item = find(screen(resume, "Requirements\n- AWS certification\n"), "AWS certification")
    assert item.status.value == "unverified"
    assert snippets(item) == [line]


@pytest.mark.parametrize(
    "resume",
    [
        ["EXPERIENCE", "Engineer, X   Jan 2020 - Dec 2022", "AWS Certified Solutions Architect"],
        ["EXPERIENCE", "Engineer, X   Jan 2020 - Dec 2022", "Certified AWS Solutions Architect"],
        ["EXPERIENCE", "Engineer, X   Jan 2020 - Dec 2022", "Certified in AWS (2023)"],
        ["CERTIFICATIONS", "AWS Solutions Architect Associate"],
        ["EXPERIENCE", "Managed AWS infrastructure. AWS Certified Cloud Practitioner."],
        ["EXPERIENCE", "Earned the AWS Solutions Architect Professional certificate"],
        ["EXPERIENCE", "AWS and Azure certified"],
        ["EXPERIENCE", "Certifications: Azure Administrator, AWS Solutions Architect"],
    ],
    ids=[
        "before",
        "after",
        "in",
        "section",
        "same_line_as_other_text",
        "name_then_certificate",
        "shared_marker",
        "label_list",
    ],
)
def test_f3_real_certifications_are_still_accepted(resume):
    item = find(screen(resume, "Requirements\n- AWS certification\n"), "AWS certification")
    assert item.status.value == "supported"


def test_f3_the_unrelated_certification_itself_is_still_found():
    resume = ["EXPERIENCE", "Managed AWS infrastructure. Certified Scrum Master."]
    report = screen(resume, "Requirements\n- Scrum Master certification\n")
    item = find(report, "Scrum Master certification")
    assert item.status.value == "supported"


# --- F4: an undated role must not inherit the years of the role above it -------------------

CHEF_THEN_UNDATED_INTERN = [
    "EXPERIENCE",
    "Pastry Chef, Cafe Rose   Jan 2012 - Dec 2021",
    "Baked bread",
    "Data Engineering Intern, Foo Corp (summer 2022)",
    "Built pipelines",
]
DATA_ENGINEERING_JOB = "Requirements\n- 3+ years of experience in data engineering\n"


def test_f4_a_dated_chef_followed_by_an_undated_intern_is_unverified():
    item = screen(CHEF_THEN_UNDATED_INTERN, DATA_ENGINEERING_JOB).assessments[0]
    assert item.status.value == "unverified"
    assert "without usable dates" in item.explanation
    assert "Data Engineering Intern, Foo Corp (summer 2022)" in snippets(item)


def test_f4_the_chef_years_are_not_counted_as_data_engineering():
    report = screen(CHEF_THEN_UNDATED_INTERN, DATA_ENGINEERING_JOB)
    assert report.coverage_score == 0.0
    assert "9.9" not in report.assessments[0].explanation


def test_f4_a_bulleted_line_with_a_year_does_not_end_the_role():
    roles = [
        "EXPERIENCE",
        "Data Engineer, Sample Ltd   Jan 2019 - Present",
        "- Migrated the 2019 warehouse with data engineering tools",
    ]
    item = screen(roles, DATA_ENGINEERING_JOB).assessments[0]
    assert item.status.value == "supported"


def test_f4_dated_relevant_roles_still_merge_and_count():
    roles = [
        "EXPERIENCE",
        "Pastry Chef, Cafe Rose   Jan 2012 - Dec 2021",
        "Baked bread",
        "Data Engineer, Sample Ltd   Jan 2019 - Present",
        "Built data engineering pipelines",
    ]
    item = screen(roles, DATA_ENGINEERING_JOB).assessments[0]
    assert item.status.value == "supported"
    assert "Pastry Chef, Cafe Rose   Jan 2012 - Dec 2021" not in snippets(item)


# --- F5: ordinary words are not degree requirements -----------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "Experience with Master Data Management",
        "Scrum Master experience",
        "Experience supporting associate engineers",
        "Associate Data Engineer role",
    ],
)
def test_f5_ordinary_words_create_no_degree_requirement(line):
    assert parse_job_description(f"Requirements\n- Python\n- {line}\n").min_degree_level is None


def test_f5_an_associate_line_does_not_lower_a_bachelors_requirement():
    job = (
        "Requirements\n- Bachelor's degree in computer science\n"
        "- Supporting associate engineers\n"
    )
    assert parse_job_description(job).min_degree_level is DegreeLevel.BACHELOR


@pytest.mark.parametrize(
    ("line", "level"),
    [
        ("Bachelor's degree in computer science", DegreeLevel.BACHELOR),
        ("B.Tech or B.Sc. in a quantitative field", DegreeLevel.BACHELOR),
        ("Master's degree in statistics", DegreeLevel.MASTER),
        ("Masters degree preferred", DegreeLevel.MASTER),
        ("Master of Science in data science", DegreeLevel.MASTER),
        ("MBA", DegreeLevel.MASTER),
        ("PhD in a related field", DegreeLevel.DOCTORATE),
        ("Associate degree", DegreeLevel.DIPLOMA),
        ("Diploma in engineering", DegreeLevel.DIPLOMA),
    ],
)
def test_f5_real_degree_requirements_are_still_found(line, level):
    assert detect_requested_degree_level(line) is level
    assert parse_job_description(f"Requirements\n- {line}\n").min_degree_level is level


# --- F6: "degree or certification" is one alternative, not two requirements ---------------

DEGREE_OR_CERT_JOB = (
    "Requirements\n- Bachelor's degree in computer science or relevant certification\n- Python\n"
)


def test_f6_the_line_becomes_an_education_requirement_only_and_says_so():
    requirements = parse_job_description(DEGREE_OR_CERT_JOB)
    assert requirements.min_degree_level is DegreeLevel.BACHELOR
    assert requirements.certifications == []
    (note,) = requirements.notes
    assert "alternative to a degree" in note
    assert "certification" in note


def test_f6_an_applicant_with_the_degree_is_not_penalised_for_a_missing_certification():
    resume = ["SKILLS", "Python", "EDUCATION", "B.Sc. in Computer Science, 2018"]
    report = screen(resume, DEGREE_OR_CERT_JOB)
    assert [a.kind.value for a in report.assessments] == ["required_skill", "education"]
    assert {a.status.value for a in report.assessments} == {"supported"}
    assert report.coverage_score == 100.0
    assert any("alternative to a degree" in item for item in report.limitations)


def test_f6_a_separate_certification_line_is_still_a_requirement():
    job = "Requirements\n- Bachelor's degree\n- AWS certification\n"
    assert parse_job_description(job).certifications == ["AWS certification"]


# --- F7: say when years were counted without checking the field ---------------------------

CHEF = ["EXPERIENCE", "Pastry Chef, Cafe Rose   Jan 2012 - Dec 2021", "Baked bread"]


@pytest.mark.parametrize(
    "job",
    [
        "Requirements\n- 5+ years of relevant experience\n",
        "Requirements\n- 5+ years of experience in a related field\n",
        "Requirements\n- 5+ years of experience\n",
    ],
)
def test_f7_a_requirement_with_no_field_says_relevance_was_not_checked(job):
    report = screen(CHEF, job)
    item = report.assessments[0]
    assert item.status.value == "supported"  # the score is not changed by the disclosure
    assert report.coverage_score == 100.0
    assert "names no field" in item.explanation
    assert "without checking whether it is relevant" in item.explanation


def test_f7_a_stated_total_with_no_field_says_it_too():
    resume = ["SUMMARY", "8 years of experience in sales"]
    item = screen(resume, "Requirements\n- 5+ years of experience\n").assessments[0]
    assert item.status.value == "supported"
    assert "names no field" in item.explanation


def test_f7_a_requirement_with_a_field_does_not_carry_the_note():
    roles = ["EXPERIENCE", "Data Engineer, Sample Ltd   Jan 2019 - Present", "Data engineering"]
    item = screen(roles, DATA_ENGINEERING_JOB).assessments[0]
    assert item.status.value == "supported"
    assert "names no field" not in item.explanation


def test_f7_nothing_is_added_when_no_years_could_be_counted():
    item = screen(["SKILLS", "Python"], "Requirements\n- 5+ years of experience\n").assessments[0]
    assert item.status.value == "unverified"
    assert "names no field" not in item.explanation


# --- F8: ranges and several year requirements -----------------------------------------------


@pytest.mark.parametrize(
    ("line", "years"),
    [
        ("3-5 years of experience in analytics", 3),
        ("3 - 5 years of experience in analytics", 3),
        ("3–5 years of experience in analytics", 3),
        ("3 to 5 years of experience in analytics", 3),
        ("3-5 yrs experience in analytics", 3),
        ("3+ years of experience in analytics", 3),
        ("5 years of experience in analytics", 5),
    ],
)
def test_f8_a_range_uses_its_lower_bound(line, years):
    requirements = parse_job_description(f"Requirements\n- {line}\n")
    assert requirements.min_years_experience == years
    assert year_notes(requirements) == []


def test_f8_a_range_keeps_its_field():
    requirements = parse_job_description("Requirements\n- 3-5 years of experience in analytics\n")
    assert requirements.experience_field == "analytics"


@pytest.mark.parametrize(
    "job",
    [
        "Requirements\n- 5+ years of experience in Python\n- 2+ years of experience in sales\n",
        "Requirements\n- 5+ years of Python experience and 2+ years of SQL experience\n",
    ],
    ids=["two_lines", "one_line"],
)
def test_f8_several_year_requirements_are_reported_not_dropped_silently(job):
    requirements = parse_job_description(job)
    assert requirements.min_years_experience == 5  # still only one is assessed
    (note,) = year_notes(requirements)
    assert "2 experience year requirements" in note
    assert "Only the first" in note
    report = screen(CHEF, job)
    assert any("2 experience year requirements" in item for item in report.limitations)


def test_f8_one_year_requirement_has_no_note():
    requirements = parse_job_description("Requirements\n- 5+ years of experience\n")
    assert year_notes(requirements) == []


# --- F9: skill mentions ---------------------------------------------------------------------


@pytest.mark.parametrize("heading", ["INTERESTS", "HOBBIES", "REFERENCES"])
def test_f9_a_skill_only_in_a_hobby_like_section_is_not_counted(heading):
    report = screen([heading, "I enjoy Python on weekends"], "Requirements\n- Python\n")
    item = report.assessments[0]
    assert item.status.value == "unverified"
    assert "interests, hobbies or references" in item.explanation
    assert snippets(item) == ["I enjoy Python on weekends"]
    assert report.coverage_score == 0.0


def test_f9_a_skill_in_the_skills_section_still_counts_beside_an_interest():
    resume = ["SKILLS", "Python", "INTERESTS", "Python on weekends"]
    item = screen(resume, "Requirements\n- Python\n").assessments[0]
    assert item.status.value == "supported"
    assert snippets(item) == ["Python"]


@pytest.mark.parametrize(
    "line",
    ["I no longer use Python", "Python (no longer used)", "Python - no longer"],
)
def test_f9_a_skill_no_longer_used_is_not_supported(line):
    item = screen(["SKILLS", line], "Requirements\n- Python\n").assessments[0]
    assert item.status.value == "not_found"
    assert snippets(item) == [line]


def test_f9_no_longer_does_not_hide_a_different_skill_in_the_same_line():
    line = "I no longer use Java but I use Python daily"
    item = screen(["SKILLS", line], "Requirements\n- Python\n").assessments[0]
    assert item.status.value == "supported"


def test_f9_a_learning_statement_is_shown_beside_a_listed_skill():
    resume = ["SKILLS", "Python", "SUMMARY", "Currently learning Python"]
    item = screen(resume, "Requirements\n- Python\n").assessments[0]
    assert item.status.value == "supported"  # kept: a skill list entry is real evidence
    assert snippets(item) == ["Python", "Currently learning Python"]
    assert "learned or planned" in item.explanation


def test_f9_a_plain_listed_skill_is_unchanged():
    item = screen(["SKILLS", "Python"], "Requirements\n- Python\n").assessments[0]
    assert item.status.value == "supported"
    assert item.explanation == (
        "The skill is mentioned in the resume. Depth and recency are not assessed."
    )
    assert snippets(item) == ["Python"]


# --- F10: inline preferred labels --------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "Nice to have: Spark",
        "Preferred: Spark",
        "Bonus: Spark",
        "Bonus points: Spark",
        "Good to have - Spark",
        "Preferred qualifications: Spark",
        "nice to have – Spark",
    ],
)
def test_f10_an_inline_preferred_label_makes_the_skill_preferred(line):
    requirements = parse_job_description(f"Requirements\n- Python\n- {line}\n")
    assert requirements.required_skills == ["python"]
    assert requirements.preferred_skills == ["spark"]


def test_f10_standalone_preferred_headings_still_work():
    for heading in ("Preferred", "Nice to have", "Bonus"):
        requirements = parse_job_description(f"Requirements\n- Python\n\n{heading}\n- Spark\n")
        assert requirements.required_skills == ["python"]
        assert requirements.preferred_skills == ["spark"]


def test_f10_the_preferred_skill_gets_the_existing_weight_of_one():
    job = "Requirements\n- Python\n- Nice to have: Spark\n"
    report = screen(["SKILLS", "Python"], job)
    assert [a.kind.value for a in report.assessments] == ["required_skill", "preferred_skill"]
    assert report.coverage_score == 75.0  # 3 of 4 points, not 3 of 6


# --- the retrieval invariant still holds for these cases -----------------------------------

RETRIEVAL_CASES = {
    "interests": (["INTERESTS", "I enjoy Python on weekends"], "Requirements\n- Python\n"),
    "cert": (
        ["EXPERIENCE", "Managed AWS infrastructure. Certified Scrum Master."],
        "Requirements\n- AWS certification\n",
    ),
    "undated_role": (CHEF_THEN_UNDATED_INTERN, DATA_ENGINEERING_JOB),
    "degree_or_cert": (
        ["SKILLS", "Python", "EDUCATION", "B.Sc. in Computer Science, 2018"],
        DEGREE_OR_CERT_JOB,
    ),
}


@pytest.mark.parametrize("name", list(RETRIEVAL_CASES))
def test_retrieval_still_changes_no_status_or_score(name, tmp_path, make_docx, today):
    lines, job = RETRIEVAL_CASES[name]
    path = make_docx(tmp_path / "cv.docx", lines)
    base = screen_resume(path, job, today=today)
    rag = screen_resume(path, job, today=today, rag_mode="lexical")
    assert rag.assessments == base.assessments
    assert rag.coverage_score == base.coverage_score
    assert rag.human_review_required is True