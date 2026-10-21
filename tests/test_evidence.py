"""Evidence must be exact resume text. These tests cover traceability and weak evidence."""

import json

import pytest
from pydantic import ValidationError

from hr_screening.evaluation.evaluator import evaluate
from hr_screening.extraction.degrees import DegreeState, degree_state
from hr_screening.extraction.job_parser import parse_job_description
from hr_screening.extraction.resume_parser import parse_resume
from hr_screening.llm.extractor import extract_profile_with_llm
from hr_screening.llm.fake import FakeLLMClient
from hr_screening.models.candidate import CandidateProfile, DegreeLevel, Education
from hr_screening.models.job import JobRequirements
from hr_screening.models.report import Evidence, MatchStatus
from hr_screening.textutil import FoldedText, line_at, normalize, original_span

SUPPORTED = MatchStatus.SUPPORTED
PARTIAL = MatchStatus.PARTIAL
NOT_FOUND = MatchStatus.NOT_FOUND
UNVERIFIED = MatchStatus.UNVERIFIED


def _cert(requirement: str, text: str, profile: CandidateProfile | None = None):
    requirements = JobRequirements(certifications=[requirement])
    return evaluate(profile or CandidateProfile(), requirements, text).assessments[0]


def _education(line: str, required: DegreeLevel = DegreeLevel.BACHELOR):
    text = "EDUCATION\n" + line
    profile = parse_resume(text)
    requirements = JobRequirements(min_degree_level=required)
    return evaluate(profile, requirements, text).assessments[0]


def _assert_traceable(report, text: str) -> None:
    haystack = normalize(text)
    for item in report.assessments:
        for evidence in item.evidence:
            assert normalize(evidence.snippet) in haystack, (item.requirement, evidence.snippet)


# --- text helpers: original span versus normalized matching text ---------------------------


def test_original_span_keeps_the_resume_wording():
    text = "Built   dashboards\nin  POWER BI for clients"
    assert original_span(text, "dashboards in power bi") == "dashboards\nin  POWER BI"
    assert original_span(text, "power bi") == "POWER BI"
    assert original_span(text, "tableau") is None


def test_normalize_is_only_for_comparison():
    assert normalize("  Power\n  BI ") == "power bi"
    assert "Power\n  BI" in "x Power\n  BI y"


def test_whole_word_matching_in_text_helper():
    text = "A presentation. Present"
    assert FoldedText(text).find_spans("present") == [(16, 23)]
    assert FoldedText(text).find_spans("present", whole_word=False)[0] == (2, 9)
    assert FoldedText(text).find_spans("   ") == []


def test_line_at_window_keeps_the_match_visible_on_long_lines():
    text = "x" * 400 + " Python " + "y" * 400
    snippet = line_at(text, text.index("Python"))
    assert "Python" in snippet
    assert snippet in text
    assert len(snippet) <= 200


# --- skills ----------------------------------------------------------------------------------


def test_skill_evidence_is_the_exact_line(sample_resume_text):
    requirements = JobRequirements(required_skills=["tableau", "python"])
    report = evaluate(CandidateProfile(), requirements, sample_resume_text)
    by_name = {a.requirement: a for a in report.assessments}
    assert by_name["python"].evidence[0].snippet == "Python, SQL, Pandas, scikit-learn, Power BI"
    assert by_name["tableau"].status == PARTIAL
    _assert_traceable(report, sample_resume_text)


def test_keyword_mention_in_a_negated_line_is_not_support():
    text = "No Python."
    item = evaluate(CandidateProfile(), JobRequirements(required_skills=["python"]), text)
    assert item.assessments[0].status == NOT_FOUND


# --- certifications ----------------------------------------------------------------------------


def test_genuine_certification_line_supports():
    text = "CERTIFICATIONS\nAWS Certified Solutions Architect, Associate"
    item = _cert("AWS Certified Solutions Architect", text)
    assert item.status == SUPPORTED
    assert item.evidence[0].snippet == "AWS Certified Solutions Architect, Associate"


def test_certification_terms_scattered_over_lines_do_not_support():
    item = _cert("Data Engineering certification", "Data Analyst\nSoftware Engineering Lead")
    assert item.status == NOT_FOUND


@pytest.mark.parametrize(
    "line",
    [
        "Working toward AWS Certified Solutions Architect (exam 2027)",
        "Not AWS Certified Solutions Architect",
        "AWS Certified Solutions Architect (expired 2022)",
        "In progress: AWS Certified Solutions Architect",
        "Preparing for AWS Certified Solutions Architect",
        "Planning to take AWS Certified Solutions Architect",
    ],
)
def test_negated_or_unfinished_certifications_are_not_supported(line):
    item = _cert("AWS Certified Solutions Architect", line)
    assert item.status == UNVERIFIED
    assert item.evidence[0].snippet == line
    assert "not counted as held" in item.explanation


def test_job_title_is_not_a_certification():
    text = "EXPERIENCE\nSolutions Architect at AWS"
    item = _cert("AWS Certified Solutions Architect", text)
    assert item.status == UNVERIFIED
    assert "not as a listed certification" in item.explanation


def test_certification_listed_under_the_heading_counts_without_marker_word():
    text = "CERTIFICATIONS\nAWS Solutions Architect Associate"
    assert _cert("AWS Certified Solutions Architect", text).status == SUPPORTED


def test_conflicting_certification_statements_are_unverified():
    text = "CERTIFICATIONS\nPMP Certified\nNot PMP Certified"
    item = _cert("PMP certification", text)
    assert item.status == UNVERIFIED
    assert len(item.evidence) == 2


def test_other_certifications_stay_unverified_with_their_text(sample_resume_text):
    profile = parse_resume(sample_resume_text)
    item = _cert("Data Engineering certification", sample_resume_text, profile)
    assert item.status == UNVERIFIED
    assert item.evidence[0].snippet == "Example Cloud Practitioner"


def test_certification_with_no_searchable_terms_is_unverified():
    assert _cert("certified", "anything").status == UNVERIFIED


# --- education -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "state"),
    [
        ("B.Tech in Computer Science, 2018", DegreeState.COMPLETE),
        ("Currently pursuing B.Tech in CS", DegreeState.IN_PROGRESS),
        ("BSc Economics (expected 2027)", DegreeState.IN_PROGRESS),
        ("MBA 2024 - Present", DegreeState.IN_PROGRESS),
        ("PhD Candidate in Economics", DegreeState.IN_PROGRESS),
        ("Dropped out of Master's program", DegreeState.INCOMPLETE),
        ("Did not complete bachelor's degree", DegreeState.INCOMPLETE),
        ("No bachelor's degree", DegreeState.AMBIGUOUS),
    ],
)
def test_degree_state(line, state):
    assert degree_state(line) == state


def test_completed_degree_supports():
    item = _education("B.Tech in Computer Science, Example Institute, 2018")
    assert item.status == SUPPORTED


@pytest.mark.parametrize(
    "line",
    ["Currently pursuing B.Tech in CS", "B.Tech in CS (expected 2027)", "B.Tech 2023 - Present"],
)
def test_in_progress_degree_is_partial_not_supported(line):
    item = _education(line)
    assert item.status == PARTIAL
    assert "not counted as completed" in item.explanation


@pytest.mark.parametrize(
    "line", ["Dropped out of Master's program", "No bachelor's degree", "Incomplete B.Tech"]
)
def test_incomplete_or_ambiguous_degree_is_unverified(line):
    item = _education(line)
    assert item.status == UNVERIFIED
    assert item.evidence[0].snippet == line


def test_in_progress_degree_below_the_request_is_unverified():
    assert _education("Pursuing Diploma in Business", DegreeLevel.MASTER).status == UNVERIFIED


def test_completed_lower_degree_still_gets_partial_credit():
    assert _education("B.Tech in CS, 2018", DegreeLevel.MASTER).status == PARTIAL


def test_completed_degree_wins_over_in_progress_one():
    text = "EDUCATION\nB.Tech in CS, 2018\nMBA (expected 2027)"
    report = evaluate(parse_resume(text), JobRequirements(min_degree_level=2), text)
    assert report.assessments[0].status == SUPPORTED
    assert report.assessments[0].evidence[0].snippet == "B.Tech in CS, 2018"


def test_education_claim_missing_from_the_resume_loses_its_status():
    profile = CandidateProfile(education=[Education(text="PhD in Physics, Oxford", level=4)])
    report = evaluate(profile, JobRequirements(min_degree_level=2), "Just some other text")
    item = report.assessments[0]
    assert item.status == UNVERIFIED
    assert item.evidence == []
    assert "could not be traced" in item.explanation


# --- experience evidence -----------------------------------------------------------------------


def test_experience_evidence_is_the_resume_line(sample_resume_text, today):
    profile = parse_resume(sample_resume_text, today=today)
    report = evaluate(profile, JobRequirements(min_years_experience=3), sample_resume_text)
    snippets = [e.snippet for e in report.assessments[0].evidence]
    assert snippets == [
        "Data Analyst, Example Corp   Jan 2019 - Dec 2021",
        "Senior Analyst, Sample Ltd   Jan 2022 - Present",
    ]
    assert all(s in sample_resume_text for s in snippets)


def test_stated_years_evidence_must_exist_in_the_resume():
    profile = CandidateProfile(
        total_years_experience=9, years_source="stated", years_snippet="9 years of experience"
    )
    report = evaluate(profile, JobRequirements(min_years_experience=5), "Unrelated resume text")
    item = report.assessments[0]
    assert item.status == UNVERIFIED
    assert item.evidence == []


def test_whole_report_is_traceable_for_rules_and_llm(sample_resume_text, sample_job, today):
    requirements = parse_job_description(sample_job)
    rule_profile = parse_resume(sample_resume_text, today=today)
    _assert_traceable(evaluate(rule_profile, requirements, sample_resume_text), sample_resume_text)
    payload = {
        "skills": ["python"],
        "education": ["b.tech   in computer science, example institute, 2018"],
        "certifications": ["example cloud practitioner"],
        "experience": [
            {"heading": "data analyst, example corp", "start": "jan 2019", "end": "dec 2021"}
        ],
    }
    llm_profile, _ = extract_profile_with_llm(
        FakeLLMClient(json.dumps(payload)), sample_resume_text, today=today
    )
    _assert_traceable(evaluate(llm_profile, requirements, sample_resume_text), sample_resume_text)


def test_evidence_model_rejects_empty_snippets():
    with pytest.raises(ValidationError):
        Evidence(snippet="")


# --- LLM extraction: grounding in the original text ------------------------------------------


def _extract(payload: dict, text: str, today):
    return extract_profile_with_llm(FakeLLMClient(json.dumps(payload)), text, today=today)


def test_llm_items_are_stored_as_the_original_resume_text(sample_resume_text, today):
    payload = {
        "education": ["b.tech   IN computer science, example institute, 2018"],
        "certifications": ["EXAMPLE   cloud practitioner"],
        "projects": ["churn prediction MODEL using scikit-learn"],
        "skills": ["power bi"],
    }
    profile, dropped = _extract(payload, sample_resume_text, today)
    assert dropped == 0
    assert profile.education[0].text == "B.Tech in Computer Science, Example Institute, 2018"
    assert profile.certifications == ["Example Cloud Practitioner"]
    assert profile.projects == ["Churn prediction model using scikit-learn"]
    assert profile.skills == ["Power BI"]
    for text in [profile.education[0].text, *profile.certifications, *profile.projects]:
        assert text in sample_resume_text


def test_unsupported_certification_and_education_claims_are_discarded(sample_resume_text, today):
    payload = {
        "education": ["PhD in Physics, Oxford"],
        "certifications": ["AWS Certified Solutions Architect"],
    }
    profile, dropped = _extract(payload, sample_resume_text, today)
    assert profile.education == []
    assert profile.certifications == []
    assert dropped == 2
    assert profile.extraction_notes


def test_llm_items_need_whole_word_matches(today):
    text = "SKILLS\nPython\nEDUCATION\nMasterclass in design"
    profile, dropped = _extract({"education": ["Master"]}, text, today)
    assert profile.education == []
    assert dropped == 1


def test_llm_negated_skill_is_discarded(today):
    text = "SKILLS\nJava\nNotes\nI have no experience with Python"
    profile, dropped = _extract({"skills": ["Python", "Java"]}, text, today)
    assert profile.skills == ["Java"]
    assert dropped == 1


def test_llm_future_learning_skill_is_discarded(today):
    profile, dropped = _extract({"skills": ["Python"]}, "Currently learning Python", today)
    assert profile.skills == []
    assert dropped == 1


def test_llm_conflicting_skill_is_discarded(today):
    text = "Python\nNever used Python in production"
    profile, dropped = _extract({"skills": ["Python"]}, text, today)
    assert profile.skills == []
    assert dropped == 1


def test_llm_duplicate_items_are_kept_once(sample_resume_text, today):
    payload = {"skills": ["Python", "python", "PYTHON"], "certifications": ["x", "x"]}
    profile, _ = _extract(payload, sample_resume_text, today)
    assert profile.skills == ["Python"]


def test_llm_dates_must_sit_next_to_the_role(today):
    text = (
        "SKILLS\nPython\n"
        "EXPERIENCE\nAnalyst, Alpha Co   Jan 2024 - Dec 2024\n"
        "EDUCATION\nIn my presentation I used Jan 2010 and 2012\n"
    )
    payload = {
        "experience": [{"heading": "Analyst, Alpha Co", "start": "Jan 2010", "end": "Present"}]
    }
    profile, _ = _extract(payload, text, today)
    assert profile.experience[0].start is None
    assert profile.experience[0].end is None
    assert profile.total_years_experience is None
    assert profile.years_source == "unknown"


def test_llm_dates_on_the_line_after_the_heading_are_accepted(today):
    text = "EXPERIENCE\nAnalyst, Alpha Co\n\nJan 2022 - Dec 2023\n"
    payload = {
        "experience": [{"heading": "Analyst, Alpha Co", "start": "Jan 2022", "end": "Dec 2023"}]
    }
    profile, _ = _extract(payload, text, today)
    assert profile.experience[0].start == "Jan 2022"
    assert profile.total_years_experience == 1.9


def test_llm_present_is_not_matched_inside_another_word(today):
    text = "EXPERIENCE\nAnalyst, Alpha Co   Jan 2024 - presentation skills course\n"
    payload = {
        "experience": [{"heading": "Analyst, Alpha Co", "start": "Jan 2024", "end": "Present"}]
    }
    profile, _ = _extract(payload, text, today)
    assert profile.experience[0].start == "Jan 2024"
    assert profile.experience[0].end is None
    assert profile.total_years_experience is None


def test_llm_dates_are_stored_as_written_in_the_resume(sample_resume_text, today):
    payload = {
        "experience": [
            {"heading": "senior analyst, sample ltd", "start": "jan 2022", "end": "present"}
        ]
    }
    profile, _ = _extract(payload, sample_resume_text, today)
    entry = profile.experience[0]
    assert (entry.heading, entry.start, entry.end) == (
        "Senior Analyst, Sample Ltd",
        "Jan 2022",
        "Present",
    )
    assert profile.total_years_experience == 4.0


def test_llm_experience_heading_missing_from_resume_is_discarded(sample_resume_text, today):
    payload = {
        "experience": [{"heading": "CEO, Imaginary Corp", "start": "Jan 2019", "end": "Dec 2021"}]
    }
    profile, dropped = _extract(payload, sample_resume_text, today)
    assert profile.experience == []
    assert dropped == 1


def test_llm_item_with_extra_spacing_is_collapsed_to_fit_the_model_limit(today):
    original = "   ".join(["word"] * 55)
    text = "Projects\n" + original
    profile, dropped = _extract({"projects": [" ".join(["word"] * 55)]}, text, today)
    assert dropped == 0
    assert len(original) > 300
    assert profile.projects == [" ".join(["word"] * 55)]
    assert normalize(profile.projects[0]) in normalize(text)