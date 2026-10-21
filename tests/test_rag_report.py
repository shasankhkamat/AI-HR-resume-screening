"""Report models, requirement mapping and text formatting for retrieved passages."""

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from hr_screening.evaluation.evaluator import evaluate
from hr_screening.evaluation.formatting import format_report_text
from hr_screening.extraction.job_parser import parse_job_description
from hr_screening.extraction.resume_parser import parse_resume
from hr_screening.models.report import (
    MatchStatus,
    RequirementAssessment,
    RequirementKind,
    RequirementRetrieval,
    RetrievedPassage,
    ScreeningReport,
)
from hr_screening.rag.requirements import (
    EXPERIENCE_NOTE,
    retrieve_for_assessments,
    to_report_passage,
    verify_traceable,
)
from hr_screening.rag.retrieval import LexicalRetriever

FIXTURE = Path(__file__).parent / "fixtures" / "long_synthetic_resume.txt"
JOB = """Data Analyst

Requirements
- 3+ years of experience in analytics
- Python, SQL, Tableau and Kubernetes
- Bachelor's degree in computer science
- Example Cloud Practitioner certification

Preferred
- Spark
- Snowflake
"""
K = RequirementKind


@pytest.fixture(scope="module")
def resume() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def retriever(resume) -> LexicalRetriever:
    return LexicalRetriever(resume)


@pytest.fixture(scope="module")
def base_report(resume) -> ScreeningReport:
    profile = parse_resume(resume, today=date(2026, 1, 1))
    return evaluate(profile, parse_job_description(JOB), resume)


@pytest.fixture(scope="module")
def entries(retriever, base_report) -> list[RequirementRetrieval]:
    return retrieve_for_assessments(retriever, base_report.assessments)


def entry_for(entries, requirement: str) -> RequirementRetrieval:
    return next(e for e in entries if e.requirement == requirement)


def assessment(requirement: str, kind: RequirementKind) -> RequirementAssessment:
    return RequirementAssessment(
        requirement=requirement, kind=kind, status=MatchStatus.NOT_FOUND, explanation="e"
    )


# --- models -------------------------------------------------------------------------------


def test_a_report_has_an_empty_retrieval_list_by_default():
    report = ScreeningReport(assessments=[], score_explanation="x", limitations=[])
    assert report.retrieval == []


def test_a_retrieved_passage_keeps_its_fields():
    passage = RetrievedPassage(
        start=5,
        end=20,
        text="Python, SQL",
        section="skills",
        score=1.25,
        matched_terms=["python"],
        mention_kinds=["affirmative"],
    )
    assert (passage.start, passage.end, passage.score) == (5, 20, 1.25)
    assert passage.model_dump()["mention_kinds"] == ["affirmative"]


@pytest.mark.parametrize(
    "changes",
    [
        {"end": 5},
        {"end": 4},
        {"start": -1},
        {"text": ""},
        {"score": -0.1},
        {"mention_kinds": ["proficient"]},
    ],
    ids=["end_equals", "end_before", "negative_start", "empty_text", "negative_score", "bad_kind"],
)
def test_invalid_passages_are_rejected(changes):
    values = {"start": 5, "end": 20, "text": "x", "score": 1.0} | changes
    with pytest.raises(ValidationError):
        RetrievedPassage(**values)


def test_a_report_with_retrieval_survives_a_json_round_trip(entries, base_report):
    report = base_report.model_copy(update={"retrieval": entries})
    again = ScreeningReport.model_validate_json(report.model_dump_json())
    assert again == report


# --- mapping assessments to retrieval entries -------------------------------------------------


def test_there_is_one_entry_per_assessment_in_the_same_order(entries, base_report):
    assert [(e.requirement, e.kind) for e in entries] == [
        (a.requirement, a.kind) for a in base_report.assessments
    ]


def test_passages_are_exact_slices_and_keep_their_scores(entries, retriever, resume):
    verify_traceable(entries, resume)
    expected = retriever.for_skill("python")
    got = entry_for(entries, "python").passages
    assert [(p.start, p.end, p.score) for p in got] == [(p.start, p.end, p.score) for p in expected]
    for entry in entries:
        assert len(entry.passages) <= 3
        for passage in entry.passages:
            assert resume[passage.start : passage.end] == passage.text


def test_skill_passages_say_how_the_skill_is_mentioned(entries):
    assert entry_for(entries, "kubernetes").passages[0].mention_kinds == ["negated"]
    assert entry_for(entries, "spark").passages[0].mention_kinds == ["aspirational"]
    assert set(entry_for(entries, "python").passages[0].mention_kinds) == {"affirmative"}


def test_a_passage_found_through_a_related_skill_has_no_mention_kind(entries):
    tableau = entry_for(entries, "tableau")
    assert tableau.passages
    assert all(p.mention_kinds == [] for p in tableau.passages)
    assert any("power bi" in p.matched_terms for p in tableau.passages)


def test_education_is_searched_with_degree_words(entries):
    entry = entry_for(entries, "Bachelor level education or higher")
    assert entry.passages[0].section == "education"
    assert entry.passages[0].mention_kinds == []


def test_the_requested_level_does_not_stop_the_education_search(retriever):
    master = assessment("Master level education or higher", K.EDUCATION)
    result = retrieve_for_assessments(retriever, [master])[0]
    assert result.passages[0].section == "education"


def test_certifications_are_searched_by_their_words(entries):
    entry = entry_for(entries, "Example Cloud Practitioner certification")
    assert entry.passages[0].section == "certifications"


def test_experience_is_not_searched_and_says_why(entries):
    entry = entry_for(entries, "3+ years of experience")
    assert entry.passages == []
    assert entry.note == EXPERIENCE_NOTE
    assert "word experience" in entry.note


def test_a_skill_that_is_not_in_the_resume_has_no_passages(retriever):
    result = retrieve_for_assessments(retriever, [assessment("fastapi", K.REQUIRED_SKILL)])[0]
    assert result.passages == []
    assert result.note is None


def test_an_empty_resume_gives_empty_entries_without_errors():
    empty = LexicalRetriever("")
    items = [
        assessment("python", K.REQUIRED_SKILL),
        assessment("Bachelor level education or higher", K.EDUCATION),
        assessment("PMP certification", K.CERTIFICATION),
        assessment("3+ years of experience", K.EXPERIENCE),
    ]
    results = retrieve_for_assessments(empty, items)
    assert [r.passages for r in results] == [[], [], [], []]
    verify_traceable(results, "")


def test_top_k_is_passed_through(retriever, base_report):
    one = retrieve_for_assessments(retriever, base_report.assessments, top_k=1)
    assert all(len(e.passages) <= 1 for e in one)
    assert len(entry_for(one, "python").passages) == 1


def test_mapping_does_not_change_the_assessments(retriever, base_report):
    before = [a.model_copy(deep=True) for a in base_report.assessments]
    retrieve_for_assessments(retriever, base_report.assessments)
    assert base_report.assessments == before


def test_passages_convert_to_report_passages_without_losing_anything(retriever):
    passage = retriever.for_skill("kubernetes")[0]
    converted = to_report_passage(passage)
    assert (converted.start, converted.end, converted.text) == (
        passage.start,
        passage.end,
        passage.text,
    )
    assert converted.score == passage.score
    assert converted.section == passage.section
    assert converted.matched_terms == list(passage.matched_terms)


# --- the traceability check ----------------------------------------------------------------


def test_a_changed_passage_text_is_caught(entries, resume):
    entry = entry_for(entries, "python")
    bad = entry.passages[0].model_copy(update={"text": entry.passages[0].text + " extra"})
    tampered = entry.model_copy(update={"passages": [bad]})
    with pytest.raises(ValueError, match="does not match"):
        verify_traceable([tampered], resume)


def test_shifted_offsets_are_caught(entries, resume):
    entry = entry_for(entries, "python")
    passage = entry.passages[0]
    shifted = passage.model_copy(update={"start": passage.start + 1, "end": passage.end + 1})
    with pytest.raises(ValueError, match="does not match"):
        verify_traceable([entry.model_copy(update={"passages": [shifted]})], resume)


def test_a_passage_checked_against_a_different_resume_is_caught(entries):
    with pytest.raises(ValueError, match="does not match"):
        verify_traceable(entries, "a completely different text " * 100)


# --- text formatting ---------------------------------------------------------------------------


def test_a_report_without_retrieval_prints_no_retrieval_section(base_report):
    assert "RETRIEVED PASSAGES" not in format_report_text(base_report)


def test_the_section_is_added_without_changing_the_rest_of_the_report(base_report, entries):
    plain = format_report_text(base_report)
    full = format_report_text(base_report.model_copy(update={"retrieval": entries}))
    head, rest = full.split("\n\nRETRIEVED PASSAGES (for review)", 1)
    tail = rest[rest.index("\n\nLIMITATIONS") :]
    assert head + tail == plain


def test_the_section_is_labelled_and_comes_before_the_limitations(base_report, entries):
    text = format_report_text(base_report.model_copy(update={"retrieval": entries}))
    assert text.index("NOT FOUND OR UNVERIFIED") < text.index("RETRIEVED PASSAGES (for review)")
    assert text.index("RETRIEVED PASSAGES (for review)") < text.index("LIMITATIONS")
    assert "is not evidence that a requirement is met" in text
    assert "does not change any status or the score" in text


def test_each_passage_shows_section_offsets_score_and_mention_kind(base_report, entries):
    text = format_report_text(base_report.model_copy(update={"retrieval": entries}))
    top = entry_for(entries, "kubernetes").passages[0]
    header = (
        f"Passage 1 ({top.section}; chars {top.start}-{top.end}; "
        f"relevance {top.score:.2f}; skill mentioned as: negated)"
    )
    assert header in text
    assert "[required skill] kubernetes" in text
    assert "        | - Has no hands on experience with Kubernetes" in text


def test_multi_line_passages_are_indented_and_complete(base_report, entries):
    text = format_report_text(base_report.model_copy(update={"retrieval": entries}))
    passage = entry_for(entries, "Example Cloud Practitioner certification").passages[0]
    for line in passage.text.splitlines():
        assert f"        | {line}" in text


def test_entries_without_passages_and_notes_are_explained(base_report):
    items = [
        RequirementRetrieval(requirement="fastapi", kind=K.REQUIRED_SKILL),
        RequirementRetrieval(requirement="3+ years", kind=K.EXPERIENCE, note=EXPERIENCE_NOTE),
        RequirementRetrieval(
            requirement="x",
            kind=K.PREFERRED_SKILL,
            passages=[RetrievedPassage(start=0, end=3, text="abc", score=0.5)],
        ),
    ]
    text = format_report_text(base_report.model_copy(update={"retrieval": items}))
    assert "No matching passage found. This does not show that the candidate lacks it." in text
    assert f"    {EXPERIENCE_NOTE}" in text
    assert "Passage 1 (no section; chars 0-3; relevance 0.50)" in text


# --- related text versus a passage that names the skill -----------------------------------------


def skill_entry(*passages):
    return RequirementRetrieval(
        requirement="tableau", kind=K.REQUIRED_SKILL, passages=list(passages)
    )


def with_entries(base_report, *items):
    return format_report_text(base_report.model_copy(update={"retrieval": list(items)}))


def test_passages_that_never_name_the_skill_are_called_related_text(base_report):
    related = RetrievedPassage(start=0, end=3, text="abc", score=0.5)
    text = with_entries(base_report, skill_entry(related, related))
    assert text.count("None of these passages names the skill itself.") == 1
    assert "They are related text only, not evidence of the skill." in text
    assert "Related text only: this passage" not in text  # the entry level line is enough


def test_in_a_mixed_entry_only_the_passages_without_the_skill_are_marked(base_report):
    named = RetrievedPassage(start=0, end=3, text="abc", score=0.9, mention_kinds=["affirmative"])
    related = RetrievedPassage(start=4, end=7, text="def", score=0.4)
    text = with_entries(base_report, skill_entry(named, related))
    assert "None of these passages" not in text
    first, second = text.split("Passage 2", 1)
    assert "Related text only" not in first.split("Passage 1", 1)[1]
    assert "Related text only: this passage does not name the skill itself." in second


def test_a_passage_that_names_the_skill_keeps_its_label_and_no_related_note(base_report):
    named = RetrievedPassage(start=0, end=3, text="abc", score=0.9, mention_kinds=["negated"])
    text = with_entries(base_report, skill_entry(named))
    assert "skill mentioned as: negated" in text
    assert "related text" not in text.lower()


def test_text_queries_are_never_called_related_text(base_report):
    plain = RetrievedPassage(start=0, end=3, text="abc", score=0.5)
    degree = RequirementRetrieval(requirement="Bachelor level", kind=K.EDUCATION, passages=[plain])
    cert = RequirementRetrieval(requirement="Cert", kind=K.CERTIFICATION, passages=[plain])
    text = with_entries(base_report, degree, cert)
    assert "related text" not in text.lower()
    assert "names the skill" not in text


def test_the_related_text_wording_applies_to_semantic_entries_too(base_report):
    passage = RetrievedPassage(start=0, end=3, text="abc", score=0.41)
    entry = skill_entry(passage).model_copy(update={"method": "semantic"})
    text = with_entries(base_report, entry)
    assert "None of these passages names the skill itself." in text
    assert "similarity 0.41" in text


def test_the_wording_changes_no_status_score_or_assessment_text(base_report, entries):
    before = base_report.model_dump()
    plain = format_report_text(base_report)
    full = format_report_text(base_report.model_copy(update={"retrieval": entries}))
    assert base_report.model_dump() == before  # formatting does not touch the report
    requirements = plain.split("\n\nLIMITATIONS")[0]
    assert full.startswith(requirements)  # statuses, evidence and the score line are identical
    assert "Requirement coverage score:" in full
    assert "Decision support only. A human reviewer must make any hiring decision." in full
    assert "does not change any status or the score" in full
    assert "Read each one yourself." in full
    assert [a.status for a in base_report.assessments] == [
        a.status for a in base_report.model_copy(update={"retrieval": entries}).assessments
    ]