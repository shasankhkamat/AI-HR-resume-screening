"""End to end screening scenarios: resume file in, assessed report out, with and without retrieval.

Each scenario goes through the real pipeline (ingestion, parsing, evaluation, optional
retrieval, text report). The semantic embedder is the deterministic fake, so these tests show the
plumbing and the safeguards. They say nothing about how a real embedding model behaves.
"""

import pytest

from hr_screening.evaluation.formatting import format_report_text
from hr_screening.pipeline import screen_resume
from hr_screening.rag.embeddings import FakeEmbedder

JOB = """Data Engineer

Requirements
- 3+ years of experience in data engineering
- Python and SQL
- Experience with Docker
- Kubernetes
- Bachelor's degree in computer science
- AWS certification

Preferred
- Knowledge of Spark
"""

DEGREE = ["EDUCATION", "B.Sc. in Computer Science, Example University, 2018"]
ROLE = ["EXPERIENCE", "Engineer, Sample Ltd   Jan 2020 - Present"]

RESUMES = {
    "strong": [
        "SKILLS",
        "Python, SQL, Docker, Kubernetes, Spark",
        "EXPERIENCE",
        "Data Engineer, Sample Ltd   Jan 2019 - Present",
        "Built pipelines in Python and SQL and deployed them on Docker and Kubernetes",
        *DEGREE,
        "CERTIFICATIONS",
        "AWS Certified Solutions Architect",
    ],
    "partial": [
        "SKILLS",
        "Python, SQL",
        "EXPERIENCE",
        "Analyst, Sample Ltd   Jan 2023 - Present",
        "Built weekly reports in Python",
        *DEGREE,
    ],
    "irrelevant": [
        "SUMMARY",
        "Pastry chef with ten years in busy restaurants.",
        "EXPERIENCE",
        "Pastry Chef, Cafe Rose   Jan 2014 - Present",
        "Baked bread and desserts for 200 guests a day",
        "EDUCATION",
        "Diploma in Culinary Arts, 2013",
    ],
    "negated_and_learning": [
        "SKILLS",
        "Python, SQL",
        "I have no experience with Docker.",
        "Planning to learn Kubernetes next year.",
        "Currently learning Spark.",
        *ROLE,
        "Wrote Python and SQL jobs",
        *DEGREE,
    ],
    "contradiction": [
        "SKILLS",
        "Python, SQL, Docker",
        "I have no experience with Docker.",
        *ROLE,
        "Wrote Python and SQL jobs",
        *DEGREE,
    ],
    # Podman is a container tool but Docker is never mentioned. The fake embedder is told that
    # the two mean the same, so Docker's closest passages are about containers.
    "similar_but_absent": [
        "SKILLS",
        "Python, SQL, Podman",
        *ROLE,
        "Ran containers on Podman and wrote SQL jobs in Python",
        *DEGREE,
    ],
}
SIMILAR = {"podman": "container", "docker": "container", "containers": "container"}
MODES = ("lexical", "semantic")


def screen(tmp_path, make_docx, today, name, mode=None):
    path = make_docx(tmp_path / f"{name}.docx", RESUMES[name])
    embedder = FakeEmbedder(dimensions=4096, concepts=SIMILAR) if mode == "semantic" else None
    return screen_resume(path, JOB, today=today, rag_mode=mode, embedder=embedder)


def statuses(report):
    return {a.requirement: a.status.value for a in report.assessments}


def assessment(report, requirement):
    return next(a for a in report.assessments if a.requirement == requirement)


def entry(report, requirement):
    return next(e for e in report.retrieval if e.requirement == requirement)


# --- the three candidate types ---------------------------------------------------------------


def test_a_strong_candidate_is_supported_on_every_requirement(tmp_path, make_docx, today):
    report = screen(tmp_path, make_docx, today, "strong")
    assert set(statuses(report).values()) == {"supported"}
    assert report.coverage_score == 100.0
    assert all(a.evidence for a in report.assessments)


def test_a_partial_candidate_keeps_gaps_as_not_found_with_no_evidence(tmp_path, make_docx, today):
    report = screen(tmp_path, make_docx, today, "partial")
    found = statuses(report)
    assert found["python"] == found["sql"] == "supported"
    for gap in ("docker", "kubernetes", "aws", "AWS certification", "spark"):
        assert found[gap] == "not_found"
        assert assessment(report, gap).evidence == []
    assert 0 < report.coverage_score < 100


def test_an_irrelevant_candidate_gets_no_skill_or_certification_credit(
    tmp_path, make_docx, today
):
    report = screen(tmp_path, make_docx, today, "irrelevant")
    for a in report.assessments:
        if a.kind.value in ("required_skill", "preferred_skill", "certification"):
            assert a.status.value == "not_found", a.requirement
    strong = screen(tmp_path, make_docx, today, "strong")
    partial = screen(tmp_path, make_docx, today, "partial")
    assert report.coverage_score < partial.coverage_score < strong.coverage_score


# --- missing, negated, learning and contradictory --------------------------------------


def test_an_unmentioned_skill_and_a_denied_skill_are_told_apart(tmp_path, make_docx, today):
    report = screen(tmp_path, make_docx, today, "negated_and_learning")
    unmentioned = assessment(report, "aws")
    denied = assessment(report, "docker")
    learning = assessment(report, "kubernetes")
    for item in (unmentioned, denied, learning):
        assert item.status.value == "not_found"
    assert unmentioned.evidence == []
    assert "No mention found" in unmentioned.explanation
    assert "negated" in denied.explanation
    assert [e.snippet for e in denied.evidence] == ["I have no experience with Docker."]
    assert "learned or planned" in learning.explanation or "learning" in learning.explanation
    assert [e.snippet for e in learning.evidence] == ["Planning to learn Kubernetes next year."]


def test_a_resume_that_contradicts_itself_is_unverified_and_listed_for_review(
    tmp_path, make_docx, today
):
    report = screen(tmp_path, make_docx, today, "contradiction")
    docker = assessment(report, "docker")
    assert docker.status.value == "unverified"
    assert {e.snippet for e in docker.evidence} >= {"I have no experience with Docker."}
    review_list = format_report_text(report).split("NOT FOUND OR UNVERIFIED")[1]
    assert "docker: unverified" in review_list.split("LIMITATIONS")[0]


# --- retrieval never changes the outcome -----------------------------------------------------


@pytest.mark.parametrize("name", list(RESUMES))
@pytest.mark.parametrize("mode", MODES)
def test_retrieval_changes_no_status_score_explanation_or_evidence(
    tmp_path, make_docx, today, name, mode
):
    base = screen(tmp_path, make_docx, today, name)
    rag = screen(tmp_path, make_docx, today, name, mode)
    assert rag.assessments == base.assessments
    assert rag.coverage_score == base.coverage_score
    assert rag.score_explanation == base.score_explanation
    assert rag.retrieval and not base.retrieval
    assert rag.limitations[: len(base.limitations)] == base.limitations
    assert len(rag.limitations) == len(base.limitations) + 1


@pytest.mark.parametrize("mode", MODES)
def test_negated_and_learning_lines_are_labelled_when_retrieved(
    tmp_path, make_docx, today, mode
):
    report = screen(tmp_path, make_docx, today, "negated_and_learning", mode)
    docker = entry(report, "docker").passages
    kubernetes = entry(report, "kubernetes").passages
    assert docker and docker[0].mention_kinds == ["negated"]
    assert kubernetes and kubernetes[0].mention_kinds == ["aspirational"]
    assert statuses(report)["docker"] == statuses(report)["kubernetes"] == "not_found"


# --- a similar passage is not a demonstrated skill ---------------------------------------


def test_a_semantically_close_passage_does_not_make_the_skill_supported(
    tmp_path, make_docx, today
):
    base = screen(tmp_path, make_docx, today, "similar_but_absent")
    report = screen(tmp_path, make_docx, today, "similar_but_absent", "semantic")
    docker = entry(report, "docker").passages
    assert docker, "the fake embedder links Docker to Podman, so passages are expected here"
    for passage in docker:
        assert "docker" not in passage.text.lower()
        assert passage.mention_kinds == []
    assert statuses(report)["docker"] == "not_found"
    assert assessment(report, "docker").evidence == []
    assert report.coverage_score == base.coverage_score


def test_the_report_calls_such_a_passage_related_text_only(tmp_path, make_docx, today):
    report = screen(tmp_path, make_docx, today, "similar_but_absent", "semantic")
    docker_section = format_report_text(report).split("[required skill] docker")[1]
    docker_section = docker_section.split("[required skill] kubernetes")[0]
    assert "None of these passages names the skill itself" in docker_section
    assert "related text only, not evidence of the skill" in docker_section
    assert "skill mentioned as" not in docker_section


# --- human review ----------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(RESUMES))
@pytest.mark.parametrize("mode", (None, *MODES))
def test_every_report_requires_a_human_reviewer(tmp_path, make_docx, today, name, mode):
    report = screen(tmp_path, make_docx, today, name, mode)
    assert report.human_review_required is True
    text = format_report_text(report)
    assert "A human reviewer must make any hiring decision" in text
    unresolved = [a for a in report.assessments if a.status.value in ("not_found", "unverified")]
    if unresolved:
        listed = text.split("NOT FOUND OR UNVERIFIED")[1].split("LIMITATIONS")[0]
        for a in unresolved:
            assert f"{a.requirement}: {a.status.value}" in listed