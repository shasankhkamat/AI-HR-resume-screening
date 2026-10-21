"""Explicit denial of a skill: a related skill must not earn credit for it.

Found in the release check by running the real CLI on a synthetic resume that says
"I have no experience with Kubernetes", with a Docker line. The evaluator used to give partial
credit through Docker and hide the denial line.
"""

import pytest

from hr_screening.pipeline import screen_resume

JOB = "Requirements\n- Kubernetes\n- Python\n"
DENIAL = "I have no experience with Kubernetes."


@pytest.fixture
def screen(tmp_path, make_docx, today):
    def _run(lines, job=JOB):
        return screen_resume(make_docx(tmp_path / "cv.docx", lines), job, today=today)

    return _run


def kubernetes(report):
    return next(a for a in report.assessments if a.requirement == "kubernetes")


def test_a_denied_skill_with_nothing_related_is_not_found_and_shows_the_denial(screen):
    report = screen(["SKILLS", "Python", "SUMMARY", DENIAL])
    item = kubernetes(report)
    assert item.status.value == "not_found"
    assert [e.snippet for e in item.evidence] == [DENIAL]
    assert "negated" in item.explanation
    assert report.coverage_score == 50.0  # python 3 of 6


def test_a_denial_blocks_partial_credit_from_a_related_skill(screen):
    report = screen(["SKILLS", "Docker, Python", "SUMMARY", DENIAL])
    item = kubernetes(report)
    assert item.status.value == "not_found"
    assert item.status.value != "partial"
    assert report.coverage_score == 50.0  # the Docker line adds nothing: python 3 of 6


def test_the_denial_line_is_the_evidence_and_the_related_line_is_not(screen):
    item = kubernetes(screen(["SKILLS", "Docker, Python", "SUMMARY", DENIAL]))
    assert [e.snippet for e in item.evidence] == [DENIAL]
    assert "docker" not in item.explanation.lower()
    assert "denies this one" in item.explanation


def test_the_denial_is_found_wherever_it_stands_in_the_resume(screen):
    lines = ["SKILLS", "Docker, Python", "EXPERIENCE", "Analyst, Foo  Jan 2020 - Jan 2021", DENIAL]
    assert kubernetes(screen(lines)).status.value == "not_found"


def test_without_a_denial_a_related_skill_still_earns_partial_credit(screen):
    """The rule is unchanged for a resume that does not deny the skill."""
    report = screen(["SKILLS", "Docker, Python"])
    item = kubernetes(report)
    assert item.status.value == "partial"
    assert "was not found, but 'docker' appears" in item.explanation
    assert report.coverage_score == 75.0  # kubernetes 1.5 and python 3, of 6


def test_an_affirmative_and_a_negated_mention_stay_unverified(screen):
    report = screen(["SKILLS", "Kubernetes, Docker, Python", "SUMMARY", DENIAL])
    item = kubernetes(report)
    assert item.status.value == "unverified"
    assert DENIAL in [e.snippet for e in item.evidence]
    assert "Kubernetes, Docker, Python" in [e.snippet for e in item.evidence]
    assert report.coverage_score == 50.0


def test_a_learning_mention_alone_still_earns_related_credit(screen):
    """Only a denial changes the rule. A skill being learned is not a denial."""
    report = screen(["SKILLS", "Docker, Python", "SUMMARY", "Currently learning Kubernetes."])
    assert kubernetes(report).status.value == "partial"


def test_a_denied_preferred_skill_is_blocked_the_same_way(screen):
    job = "Requirements\n- Python\nPreferred\n- Kubernetes\n"
    report = screen(["SKILLS", "Docker, Python", "SUMMARY", DENIAL], job)
    item = kubernetes(report)
    assert item.kind.value == "preferred_skill"
    assert item.status.value == "not_found"
    assert [e.snippet for e in item.evidence] == [DENIAL]
