"""Models for the screening report."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


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


class ScreeningReport(BaseModel):
    assessments: list[RequirementAssessment]
    coverage_score: float | None = Field(default=None, ge=0, le=100)
    score_explanation: str
    limitations: list[str]
    human_review_required: Literal[True] = True

    @property
    def unresolved(self) -> list[RequirementAssessment]:
        """Requirements with no supporting evidence (not found or unverified)."""
        bad = (MatchStatus.NOT_FOUND, MatchStatus.UNVERIFIED)
        return [a for a in self.assessments if a.status in bad]