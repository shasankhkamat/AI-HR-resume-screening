"""Years of experience in a named field: only roles whose text names the field are counted.

Before this, '3+ years of experience in data engineering' was compared with the total of ALL dated
roles, so ten years as a pastry chef satisfied it. These tests pin the narrower behavior and what
it still cannot do. The check is a word check on the role's own text. It cannot prove a role was in
the field, and it misses roles that describe the field in other words.
"""

import pytest

from hr_screening.extraction.experience_field import field_terms, find_experience_field
from hr_screening.extraction.job_parser import parse_job_description
from hr_screening.models.job import JobRequirements
from hr_screening.pipeline import screen_resume
from hr_screening.rag.embeddings import FakeEmbedder

FIELD_JOB = "Data Engineer\n\nRequirements\n- 3+ years of experience in data engineering\n"
PLAIN_JOB = "Data Engineer\n\nRequirements\n- 3+ years of experience\n"
CHEF = [
    "Pastry Chef, Cafe Rose   Jan 2012 - Dec 2021",
    "Baked bread and desserts for 200 guests a day",
]
DATA_ROLE = [
    "Data Engineer, Sample Ltd   Jan 2019 - Present",
    "Built data engineering pipelines in Python",
]


def screen(tmp_path, make_docx, today, roles, job=FIELD_JOB, mode=None, extra=()):
    path = make_docx(tmp_path / "r.docx", ["EXPERIENCE", *roles, *extra])
    embedder = FakeEmbedder(dimensions=2048) if mode == "semantic" else None
    report = screen_resume(path, job, today=today, rag_mode=mode, embedder=embedder)
    return next(a for a in report.assessments if a.kind.value == "experience"), report


# --- the job line ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "field"),
    [
        ("3+ years of experience in data engineering", "data engineering"),
        ("5+ years of data engineering experience", "data engineering"),
        ("3+ years of experience in analytics, including dashboards", "analytics"),
        ("7 years of experience with Python", "Python"),
        ("3+ years of experience", None),
        ("3+ years of experience in a related field", None),
        ("2+ years of relevant experience", None),
        ("3+ years of experience in a fast paced environment", None),
        ("3 years of experience in a startup environment", None),
    ],
)
def test_the_field_is_read_from_the_years_line(line, field):
    assert find_experience_field(line) == field
    assert parse_job_description(f"Role\n\nRequirements\n- {line}\n").experience_field == field


def test_old_requirement_objects_still_work_without_a_field():
    assert JobRequirements(min_years_experience=3).experience_field is None
    assert field_terms(None) == [] and field_terms("a related field") == []


# --- relevant, unrelated, partial, insufficient, mixed ---------------------------------------


def test_documented_relevant_experience_is_supported_with_the_role_as_evidence(
    tmp_path, make_docx, today
):
    item, _ = screen(tmp_path, make_docx, today, DATA_ROLE)
    assert item.status.value == "supported"
    assert [e.snippet for e in item.evidence] == [DATA_ROLE[0]]
    assert "data engineering" in item.explanation


def test_many_years_in_an_unrelated_field_are_unverified_not_supported(tmp_path, make_docx, today):
    item, report = screen(tmp_path, make_docx, today, CHEF)
    assert item.status.value == "unverified"
    assert "no role mentions data engineering" in item.explanation
    assert [e.snippet for e in item.evidence] == [CHEF[0]]
    assert report.coverage_score == 0.0
    assert report.human_review_required is True


def test_only_the_relevant_role_is_counted_in_a_mixed_history(tmp_path, make_docx, today):
    role = "Data Engineer, Sample Ltd   Jan 2020 - Dec 2021"
    roles = [*CHEF, role, "Wrote data engineering code"]
    item, _ = screen(tmp_path, make_docx, today, roles)
    assert item.status.value == "partial"  # about 1.9 relevant years, not the ten from the chef
    assert "About 1.9 years" in item.explanation
    assert [e.snippet for e in item.evidence] == [role]


def test_enough_relevant_years_in_a_mixed_history_are_supported(tmp_path, make_docx, today):
    roles = [*CHEF, *DATA_ROLE]
    item, _ = screen(tmp_path, make_docx, today, roles)
    assert item.status.value == "supported"
    assert CHEF[0] not in [e.snippet for e in item.evidence]


def test_a_short_relevant_stretch_is_not_found_but_not_padded_with_other_years(
    tmp_path, make_docx, today
):
    role = "Data Engineer, Sample Ltd   Jan 2021 - Jun 2021"
    roles = [*CHEF, role, "Built data engineering jobs"]
    item, _ = screen(tmp_path, make_docx, today, roles)
    assert item.status.value == "not_found"
    assert "Only about 0.4 years" in item.explanation


def test_overlapping_relevant_roles_are_merged_not_added(tmp_path, make_docx, today):
    roles = [
        "Data Engineer, A Ltd   Jan 2020 - Dec 2020",
        "Data engineering work",
        "Data Engineer, B Ltd   Jan 2020 - Dec 2020",
        "More data engineering work",
    ]
    item, _ = screen(tmp_path, make_docx, today, roles)
    assert "about 0.9 years" in item.explanation  # two identical years merged, not 1.8


def test_a_role_with_no_usable_dates_is_not_counted(tmp_path, make_docx, today):
    roles = ["Data Engineer, Sample Ltd", "Built data engineering pipelines"]
    item, _ = screen(tmp_path, make_docx, today, roles)
    assert item.status.value == "unverified"
    assert "no dated roles" in item.explanation


def test_a_stated_total_counts_only_when_the_statement_names_the_field(
    tmp_path, make_docx, today
):
    path = make_docx(tmp_path / "a.docx", ["SUMMARY", "8 years of experience in data engineering"])
    other = make_docx(tmp_path / "b.docx", ["SUMMARY", "8 years of experience in catering"])
    named = screen_resume(path, FIELD_JOB, today=today).assessments[0]
    unnamed = screen_resume(other, FIELD_JOB, today=today).assessments[0]
    assert named.status.value == "supported"
    assert unnamed.status.value == "unverified"
    assert "not in data engineering" in unnamed.explanation


# --- the role's own text decides -------------------------------------------------------------


def test_the_field_in_the_education_section_does_not_make_a_role_relevant(
    tmp_path, make_docx, today
):
    extra = ["EDUCATION", "B.Sc. in Data Engineering, Example University, 2011"]
    item, _ = screen(tmp_path, make_docx, today, CHEF, extra=extra)
    assert item.status.value == "unverified"


def test_the_field_in_the_next_role_does_not_make_the_earlier_role_relevant(
    tmp_path, make_docx, today
):
    roles = [
        "Pastry Chef, Cafe Rose   Jan 2010 - Dec 2018",
        "Baked bread",
        "Data Engineer, Sample Ltd   Jan 2021 - Jun 2021",
        "Built data engineering jobs",
    ]
    item, _ = screen(tmp_path, make_docx, today, roles)
    assert item.status.value == "not_found"
    assert "Only about 0.4 years" in item.explanation


def test_a_role_that_describes_the_field_in_other_words_goes_to_a_reviewer(
    tmp_path, make_docx, today
):
    roles = ["Backend Developer, Sample Ltd   Jan 2019 - Present", "Built ETL pipelines"]
    item, _ = screen(tmp_path, make_docx, today, roles)
    assert item.status.value == "unverified"  # missing evidence, not proof of absence
    assert "may describe it in other words" in item.explanation


# --- what must not change --------------------------------------------------------------------


def test_a_requirement_with_no_field_still_uses_total_dated_years(tmp_path, make_docx, today):
    item, _ = screen(tmp_path, make_docx, today, CHEF, job=PLAIN_JOB)
    assert item.status.value == "supported"
    assert "dated roles with overlaps merged" in item.explanation


@pytest.mark.parametrize("mode", ["lexical", "semantic"])
@pytest.mark.parametrize("roles", [CHEF, [*CHEF, *DATA_ROLE]])
def test_retrieval_still_changes_nothing(tmp_path, make_docx, today, roles, mode):
    base_item, base = screen(tmp_path, make_docx, today, roles)
    item, rag = screen(tmp_path, make_docx, today, roles, mode=mode)
    assert item == base_item
    assert rag.assessments == base.assessments
    assert rag.coverage_score == base.coverage_score
    assert rag.human_review_required is True


def test_the_evaluator_still_works_without_a_date_argument(today):
    from hr_screening.evaluation.evaluator import evaluate
    from hr_screening.extraction.resume_parser import parse_resume

    text = "EXPERIENCE\n" + "\n".join(DATA_ROLE)
    profile = parse_resume(text, today=today)
    report = evaluate(profile, JobRequirements(min_years_experience=1), text)
    assert report.assessments[0].status.value == "supported"


# --- alternatives and combinations in the field ------------------------------------------------

OR_JOB = (
    "Data Engineer\n\nRequirements\n"
    "- 3+ years of experience in software development or data engineering\n"
)
AND_JOB = (
    "Data Engineer\n\nRequirements\n- 3+ years of experience in data engineering and finance\n"
)


def test_either_side_of_an_or_is_enough(tmp_path, make_docx, today):
    item, _ = screen(tmp_path, make_docx, today, DATA_ROLE, job=OR_JOB)
    assert item.status.value == "supported"
    unrelated, _ = screen(tmp_path, make_docx, today, CHEF, job=OR_JOB)
    assert unrelated.status.value == "unverified"


def test_every_part_of_an_and_is_required(tmp_path, make_docx, today):
    item, _ = screen(tmp_path, make_docx, today, DATA_ROLE, job=AND_JOB)
    assert item.status.value == "unverified"