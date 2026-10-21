"""Retrieval on the long synthetic resume: exact passages, honest labels, no verdicts."""

import json
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from hr_screening.extraction.mentions import MentionKind
from hr_screening.rag.retrieval import DEFAULT_TOP_K, LexicalRetriever, Passage

FIXTURE = Path(__file__).parent / "fixtures" / "long_synthetic_resume.txt"
SRC = Path(__file__).resolve().parents[1] / "src"
SCRIPT = """
import json, sys
from pathlib import Path
from hr_screening.rag.retrieval import LexicalRetriever
retriever = LexicalRetriever(Path(sys.argv[1]).read_text(encoding="utf-8"))
out = {}
for skill in ["python", "sql", "tableau", "kubernetes", "power bi", "machine learning"]:
    found = retriever.for_skill(skill)
    out[skill] = [[p.start, p.end, p.score, list(p.matched_terms)] for p in found]
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def resume() -> str:
    return FIXTURE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def retriever(resume) -> LexicalRetriever:
    return LexicalRetriever(resume)


def assert_exact(resume: str, passages: list[Passage]) -> None:
    for passage in passages:
        assert passage.text == resume[passage.start : passage.end]


# --- the retriever -------------------------------------------------------------------------


def test_the_fixture_is_split_in_several_chunks(retriever):
    assert len(retriever.chunks) >= 6
    assert retriever.truncated is False


def test_passages_are_exact_slices_of_the_resume(retriever, resume):
    for skill in ["python", "sql", "tableau", "kubernetes", "spark", "snowflake", "power bi"]:
        assert_exact(resume, retriever.for_skill(skill))
    assert_exact(resume, retriever.for_text("bachelor degree in computer science"))


def test_the_default_is_three_passages(retriever):
    assert DEFAULT_TOP_K == 3
    passages = retriever.for_skill("python")
    assert len(passages) == 3


def test_top_k_can_be_changed(retriever):
    assert len(retriever.for_skill("python", top_k=1)) == 1
    assert len(retriever.for_skill("python", top_k=2)) == 2


def test_a_skill_mentioned_in_several_sections_is_found_in_each(retriever):
    sections = {p.section for p in retriever.for_skill("python")}
    assert sections == {"skills", "experience", "projects"}


def test_passages_are_ranked_best_first(retriever):
    scores = [p.score for p in retriever.for_skill("python")]
    assert scores == sorted(scores, reverse=True)
    assert all(score > 0 for score in scores)


def test_matched_passages_really_contain_the_skill(retriever):
    for passage in retriever.for_skill("python"):
        assert "python" in passage.text.lower()
        assert passage.matched_terms == ("python",)


def test_a_skill_with_aliases_and_relatives_returns_the_most_relevant_passage_first(retriever):
    top = retriever.for_skill("sql")[0]
    assert top.section == "skills"
    assert "SQL" in top.text


def test_a_skill_that_is_not_in_the_resume_returns_nothing(retriever, resume):
    assert "fastapi" not in resume.lower()
    assert retriever.for_skill("fastapi") == []


def test_blank_requests_return_nothing(retriever):
    assert retriever.for_skill("") == []
    assert retriever.for_skill("   ") == []
    assert retriever.for_text("") == []
    assert retriever.for_text("the of and") == []


def test_text_queries_find_the_right_section(retriever):
    top = retriever.for_text("bachelor degree in computer science")[0]
    assert top.section == "education"
    assert "B.Tech in Computer Science" in top.text
    assert top.mention_kinds == ()


def test_certification_text_finds_the_certification_section(retriever):
    top = retriever.for_text("Example Cloud Practitioner certification")[0]
    assert top.section == "certifications"


def test_results_are_the_same_every_time(resume):
    first = LexicalRetriever(resume)
    second = LexicalRetriever(resume)
    assert first.for_skill("python") == second.for_skill("python")
    assert first.for_text("forecasting") == second.for_text("forecasting")


def test_an_empty_resume_gives_empty_results():
    empty = LexicalRetriever("")
    assert empty.chunks == ()
    assert empty.truncated is False
    assert empty.for_skill("python") == []
    assert empty.for_text("anything") == []


def test_the_chunk_limit_is_passed_through(resume):
    limited = LexicalRetriever(resume, max_chunks=2)
    assert len(limited.chunks) == 2
    assert limited.truncated is True


# --- a retrieved passage is not proof ------------------------------------------------------


def test_a_negated_mention_is_retrieved_and_labelled_negated(retriever):
    passages = retriever.for_skill("kubernetes")
    top = passages[0]
    assert "no hands on experience with Kubernetes" in top.text
    assert top.mention_kinds == (MentionKind.NEGATED,)
    assert top.section == "experience"


def test_a_learning_mention_is_retrieved_and_labelled_aspirational(retriever):
    top = retriever.for_skill("spark")[0]
    assert "Currently learning Spark" in top.text
    assert top.mention_kinds == (MentionKind.ASPIRATIONAL,)


def test_an_affirmative_mention_is_still_only_a_mention(retriever):
    for passage in retriever.for_skill("python"):
        assert passage.mention_kinds
        assert set(passage.mention_kinds) == {MentionKind.AFFIRMATIVE}
    names = {f.name for f in fields(Passage)}
    assert names.isdisjoint({"status", "verdict", "proficiency", "supported", "level", "years"})


def test_a_related_skill_passage_has_no_mention_of_the_requested_skill(retriever, resume):
    assert "tableau" not in resume.lower()
    passages = retriever.for_skill("tableau")
    assert passages
    for passage in passages:
        assert passage.mention_kinds == ()
        assert "power bi" in passage.matched_terms
        assert "Tableau" not in passage.text


def test_text_queries_never_carry_mention_kinds(retriever):
    assert all(p.mention_kinds == () for p in retriever.for_text("python forecasting"))


def test_the_project_section_is_visible_so_it_can_be_labelled_as_project_evidence(retriever):
    sections = [p.section for p in retriever.for_skill("forecasting")]
    assert "projects" in sections


def test_results_do_not_depend_on_python_hash_randomization():
    outputs = []
    for seed in ("1", "2", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(SRC)}
        run = subprocess.run(
            [sys.executable, "-c", SCRIPT, str(FIXTURE)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(json.loads(run.stdout))
    assert outputs[0] == outputs[1] == outputs[2]
    assert outputs[0]["python"]