"""Models for the screening report."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class MatchStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"
    UNVERIFIED = "unverified"


class RequirementKind(StrEnum):
    REQUIRED_SKILL = "required_skill"
    PREFERRED_SKILL = "preferred_skill"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    CERTIFICATION = "certification"


class Evidence(BaseModel):
    """A short excerpt copied from the resume."""

    snippet: str = Field(min_length=1, max_length=600)


class RequirementAssessment(BaseModel):
    requirement: str
    kind: RequirementKind
    status: MatchStatus
    evidence: list[Evidence] = Field(default_factory=list)
    explanation: str


class RetrievedPassage(BaseModel):
    """An exact slice of the resume that keyword retrieval found, for a human to read.

    ``text == resume_text[start:end]``. A passage is not evidence that a requirement is met:
    it can be a negated or aspirational mention, or it can be related text found through an
    alias or a related skill. ``score`` ranks passages for one requirement and is not a
    measure of skill.
    """

    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str = Field(min_length=1)
    section: str | None = None
    score: float = Field(ge=0)
    matched_terms: list[str] = Field(default_factory=list)
    mention_kinds: list[Literal["affirmative", "negated", "aspirational"]] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def _end_is_after_start(self) -> "RetrievedPassage":
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self


class RequirementRetrieval(BaseModel):
    """The passages retrieved for one requirement. No passages is not proof of absence."""

    requirement: str
    kind: RequirementKind
    passages: list[RetrievedPassage] = Field(default_factory=list)
    note: str | None = None


class ScreeningReport(BaseModel):
    assessments: list[RequirementAssessment]
    coverage_score: float | None = Field(default=None, ge=0, le=100)
    score_explanation: str
    limitations: list[str]
    human_review_required: Literal[True] = True
    # Empty unless retrieval was requested. It never changes statuses or the score.
    retrieval: list[RequirementRetrieval] = Field(default_factory=list)

    @property
    def unresolved(self) -> list[RequirementAssessment]:
        """Requirements with no supporting evidence (not found or unverified)."""
        bad = (MatchStatus.NOT_FOUND, MatchStatus.UNVERIFIED)
        return [a for a in self.assessments if a.status in bad]