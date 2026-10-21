"""Turn requirement assessments into retrieval queries and report entries.

For each assessment the retriever finds resume passages for a human to read. Nothing here
changes a status or the score, and a retrieved passage never counts as evidence by itself.

How each kind of requirement is searched:
- Required and preferred skills: the skill, its aliases and (at lower weight) the related
  skills the evaluator already knows about.
- Certifications: the words of the requirement.
- Education: a fixed set of degree words plus the requested level.
- Experience: not searched. A keyword query for "years of experience" would match any
  sentence that contains the word experience, including negated ones. The evaluator already
  cites the dated roles as evidence, so the entry carries a note instead of passages.
"""

from collections.abc import Sequence

from hr_screening.models.report import (
    RequirementAssessment,
    RequirementKind,
    RequirementRetrieval,
    RetrievedPassage,
)
from hr_screening.rag.retrieval import DEFAULT_TOP_K, LexicalRetriever, Passage

EXPERIENCE_NOTE = (
    "Not searched. The dated roles above are the evidence for years of experience, and a "
    "keyword search would match any sentence that contains the word experience."
)
_DEGREE_WORDS = "degree education university college institute"
_LEVELS = frozenset({"diploma", "bachelor", "master", "doctorate"})
_SKILL_KINDS = (RequirementKind.REQUIRED_SKILL, RequirementKind.PREFERRED_SKILL)


def retrieve_for_assessments(
    retriever: LexicalRetriever,
    assessments: Sequence[RequirementAssessment],
    top_k: int = DEFAULT_TOP_K,
) -> list[RequirementRetrieval]:
    """One retrieval entry per assessment, in the same order."""
    return [_retrieve_one(retriever, assessment, top_k) for assessment in assessments]


def _retrieve_one(
    retriever: LexicalRetriever, assessment: RequirementAssessment, top_k: int
) -> RequirementRetrieval:
    kind = assessment.kind
    if kind is RequirementKind.EXPERIENCE:
        return RequirementRetrieval(
            requirement=assessment.requirement, kind=kind, note=EXPERIENCE_NOTE
        )
    if kind in _SKILL_KINDS:
        passages = retriever.for_skill(assessment.requirement, top_k)
    elif kind is RequirementKind.EDUCATION:
        passages = retriever.for_text(_education_query(assessment.requirement), top_k)
    else:
        passages = retriever.for_text(assessment.requirement, top_k)
    return RequirementRetrieval(
        requirement=assessment.requirement,
        kind=kind,
        passages=[to_report_passage(passage) for passage in passages],
    )


def _education_query(requirement: str) -> str:
    """The education requirement reads 'Bachelor level education or higher'."""
    words = requirement.split()
    level = words[0].lower() if words else ""
    return f"{level} {_DEGREE_WORDS}" if level in _LEVELS else _DEGREE_WORDS


def to_report_passage(passage: Passage) -> RetrievedPassage:
    return RetrievedPassage(
        start=passage.start,
        end=passage.end,
        text=passage.text,
        section=passage.section,
        score=passage.score,
        matched_terms=list(passage.matched_terms),
        mention_kinds=[kind.value for kind in passage.mention_kinds],
    )


def verify_traceable(entries: Sequence[RequirementRetrieval], resume_text: str) -> None:
    """Raise ValueError unless every passage is an exact slice of the resume text."""
    for entry in entries:
        for passage in entry.passages:
            if resume_text[passage.start : passage.end] != passage.text:
                raise ValueError("A retrieved passage does not match the resume text.")