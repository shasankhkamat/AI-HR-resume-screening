"""Regression tests for skill matching: false positives, negation, aspiration, sections."""

import pytest

from hr_screening.evaluation.evaluator import evaluate
from hr_screening.extraction.mentions import MentionKind, find_skill_mentions
from hr_screening.extraction.resume_parser import SectionIndex
from hr_screening.extraction.skills_vocab import skill_pattern
from hr_screening.models.candidate import CandidateProfile
from hr_screening.models.job import JobRequirements
from hr_screening.models.report import MatchStatus

SUPPORTED = MatchStatus.SUPPORTED
PARTIAL = MatchStatus.PARTIAL
NOT_FOUND = MatchStatus.NOT_FOUND
UNVERIFIED = MatchStatus.UNVERIFIED


def _assess(skill: str, text: str):
    requirements = JobRequirements(required_skills=[skill])
    return evaluate(CandidateProfile(), requirements, text).assessments[0]


def _status(skill: str, text: str) -> MatchStatus:
    return _assess(skill, text).status


@pytest.mark.parametrize(
    ("skill", "text", "expected"),
    [
        ("java", "Built sites in JavaScript", NOT_FOUND),
        ("java", "Java developer", SUPPORTED),
        ("javascript", "Java developer", NOT_FOUND),
        ("javascript", "JavaScript and TypeScript", SUPPORTED),
        ("sql", "Worked with PostgreSQL daily", PARTIAL),
        ("sql", "Strong SQL skills", SUPPORTED),
        ("sql", "Used NoSQL stores", NOT_FOUND),
        ("sql", "Administered MySQL", PARTIAL),
        ("postgresql", "Strong SQL skills", NOT_FOUND),
        ("postgresql", "Tuned Postgres queries", SUPPORTED),
        ("python", "CPython internals and IPython notebooks", NOT_FOUND),
        ("python", "pythonic style", NOT_FOUND),
    ],
)
def test_similar_names_are_not_confused(skill, text, expected):
    assert _status(skill, text) == expected


@pytest.mark.parametrize("text", ["PYTHON", "python", "PyThOn", "Skills: PYTHON, sql"])
def test_matching_ignores_case(text):
    assert _status("python", text) == SUPPORTED


@pytest.mark.parametrize(
    "text",
    ["(Python)", "Python,", "Python.", "Python/R", "[Python]", "Python;SQL", "\u2022 Python"],
)
def test_punctuation_does_not_block_matches(text):
    assert _status("python", text) == SUPPORTED


@pytest.mark.parametrize(
    ("skill", "text"),
    [
        ("python", "Python3 scripts"),
        ("python", "Python 3.11 services"),
        ("python", "Python3.11"),
        ("c++", "C++17 engine"),
        ("c++", "Skills: C++, Rust"),
        ("c#", "C# services"),
        ("java", "Java8 backend"),
        ("scikit-learn", "Models built with sklearn"),
        ("javascript", "Wrote JS widgets"),
    ],
)
def test_versions_and_aliases_still_match(skill, text):
    assert _status(skill, text) == SUPPORTED


@pytest.mark.parametrize("text", ["Python3rd", "Python3x"])
def test_version_suffix_needs_a_clean_end(text):
    assert skill_pattern("python").search(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "I have no experience with Python",
        "No Python experience",
        "Never used Python in production",
        "Without any Python experience",
        "Not familiar with Python",
        "I don't know Python",
        "Haven't worked with Python",
        "Lacks experience with Python",
        "Python: none",
        "Python (no experience)",
        "No experience with SQL, Java or Python",
    ],
)
def test_negated_mentions_are_not_experience(text):
    item = _assess("python", text)
    assert item.status == NOT_FOUND
    assert "does not prove" not in item.explanation
    assert item.evidence
    assert item.evidence[0].snippet in text


@pytest.mark.parametrize(
    "text",
    [
        "Currently learning Python",
        "I plan to learn Python next year",
        "Want to learn Python",
        "Interested in learning Python",
        "Learning goals: Python",
        "Python (currently learning)",
        "Learning Python and SQL",
    ],
)
def test_future_learning_is_not_experience(text):
    item = _assess("python", text)
    assert item.status == NOT_FOUND
    assert "learned or planned" in item.explanation


@pytest.mark.parametrize(
    ("skill", "text"),
    [
        ("python", "No Java, strong Python skills"),
        ("python", "Not only Python but also SQL"),
        ("python", "Built dashboards without Excel using Python"),
        ("python", "Machine learning with Python"),
        ("python", "Learned Python at work"),
        ("python", "Taught myself Python"),
        ("sql", "Experienced in SQL; no experience with Python"),
        ("python", "Python - not only scripts"),
    ],
)
def test_nearby_cues_do_not_flip_unrelated_mentions(skill, text):
    assert _status(skill, text) == SUPPORTED


def test_negation_and_learning_only_affect_their_own_skill():
    text = "Experienced in SQL; no experience with Python"
    assert _status("sql", text) == SUPPORTED
    assert _status("python", text) == NOT_FOUND


def test_conflicting_mentions_are_unverified_with_both_lines():
    text = "SKILLS\nPython, SQL\nNotes\nNo experience with Python in production"
    item = _assess("python", text)
    assert item.status == UNVERIFIED
    assert [e.snippet for e in item.evidence] == [
        "Python, SQL",
        "No experience with Python in production",
    ]
    assert all(e.snippet in text for e in item.evidence)


def test_negated_related_skill_gives_no_partial_credit():
    assert _status("tableau", "Used Power BI") == PARTIAL
    assert _status("tableau", "No experience with Power BI") == NOT_FOUND


def test_project_only_skill_is_labelled_as_project_evidence():
    text = "EXPERIENCE\nAnalyst, Alpha Co   Jan 2020 - Dec 2021\nPROJECTS\nWeather app in Python"
    item = _assess("python", text)
    assert item.status == SUPPORTED
    assert item.evidence[0].snippet == "Weather app in Python"
    assert "project" in item.explanation
    assert "not confirmation of professional" in item.explanation


def test_experience_mention_is_preferred_over_project_mention():
    text = "PROJECTS\nWeather app in Python\nEXPERIENCE\nBuilt ETL jobs in Python"
    item = _assess("python", text)
    assert item.status == SUPPORTED
    assert item.evidence[0].snippet == "Built ETL jobs in Python"
    assert "project" not in item.explanation


def test_section_index_maps_positions_to_headings():
    text = "SKILLS\nPython\nPROJECTS\nApp\nSummary\nHello"
    sections = SectionIndex(text)
    assert sections.at(text.index("Python")) == "skills"
    assert sections.at(text.index("App")) == "projects"
    assert sections.at(text.index("Hello")) is None
    assert SectionIndex("no headings").at(3) is None


def test_duplicate_mentions_give_one_assessment_and_one_evidence_line():
    report = evaluate(
        CandidateProfile(),
        JobRequirements(required_skills=["python"]),
        "Python, python, PYTHON\nPython again",
    )
    assert len(report.assessments) == 1
    assert len(report.assessments[0].evidence) == 1
    mentions = find_skill_mentions("Python, python, PYTHON\nPython again", "python")
    assert len(mentions) == 4


def test_unknown_skills_are_matched_as_whole_words():
    assert _status("rust", "I write Rust daily") == SUPPORTED
    assert _status("rust", "Trusted advisor") == NOT_FOUND
    assert _status("rust", "No mention at all") == NOT_FOUND
    assert "does not prove" in _assess("rust", "nothing").explanation


@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
def test_empty_resume_text_finds_nothing(text):
    item = _assess("python", text)
    assert item.status == NOT_FOUND
    assert item.evidence == []


def test_blank_skill_requirements_are_ignored():
    requirements = JobRequirements.model_construct(
        required_skills=["", "   "], preferred_skills=[], certifications=[], notes=[]
    )
    report = evaluate(CandidateProfile(), requirements, "Python here")
    assert report.assessments == []
    assert report.coverage_score is None
    assert find_skill_mentions("Python here", "  ") == []


def test_mention_kinds_are_reported():
    kinds = [m.kind for m in find_skill_mentions("Python. Never used Python.", "python")]
    assert kinds == [MentionKind.AFFIRMATIVE, MentionKind.NEGATED]


def test_a_keyword_mention_does_not_claim_proficiency():
    item = _assess("python", "Python")
    assert item.status == SUPPORTED
    assert "Depth and recency are not assessed" in item.explanation