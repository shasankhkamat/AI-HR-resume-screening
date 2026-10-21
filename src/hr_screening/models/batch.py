"""Models for a batch screening run.

Source metadata (file names, ranks, duplicate flags) is kept apart from the report, and the
report is the same ``ScreeningReport`` a single run produces. Nothing here holds resume text,
a candidate name or contact details. A file name is shown as the file system gives it, with
control characters replaced.
"""

from typing import Literal

from pydantic import BaseModel, Field

from hr_screening.models.report import RequirementKind, ScreeningReport


class RequirementCounts(BaseModel):
    """How many requirements ended in each state. ``unassessed`` has no status and no weight."""

    supported: int = Field(default=0, ge=0)
    partial: int = Field(default=0, ge=0)
    not_found: int = Field(default=0, ge=0)
    unverified: int = Field(default=0, ge=0)
    unassessed: int = Field(default=0, ge=0)


class ReviewItem(BaseModel):
    """A requirement a person has to look at, whatever the score is."""

    requirement: str
    kind: RequirementKind | None = None  # None for job text that was not assessed
    reason: Literal["unverified", "partial", "unassessed"]


class ScreenedResume(BaseModel):
    source_file: str
    rank: int = Field(ge=1)
    coverage_score: float | None = Field(default=None, ge=0, le=100)
    score_complete: bool
    required_supported: int = Field(ge=0)
    required_total: int = Field(ge=0)
    counts: RequirementCounts
    needs_review: list[ReviewItem] = Field(default_factory=list)
    duplicate_group: int | None = None
    duplicate_of: str | None = None  # the reference file, set only on the later copies
    report: ScreeningReport


class FailedResume(BaseModel):
    """A file that could not be screened. It has no score and no rank."""

    source_file: str
    error_type: str
    error: str


class DuplicateGroup(BaseModel):
    group: int = Field(ge=1)
    fingerprint: str
    reference_file: str
    duplicate_files: list[str]


class BatchCounts(BaseModel):
    files_found: int = Field(ge=0)
    screened: int = Field(ge=0)
    failed: int = Field(ge=0)
    duplicate_files: int = Field(ge=0)
    duplicate_groups: int = Field(ge=0)
    complete_scores: int = Field(ge=0)
    incomplete_scores: int = Field(ge=0)
    needs_review: int = Field(ge=0)  # screened resumes with at least one review item
    requirements: RequirementCounts


class BatchSummary(BaseModel):
    human_review_required: Literal[True] = True
    notice: str
    ordering_rule: str
    ordering_caveats: list[str]
    extra_skills: list[str]
    job_notes: list[str]
    counts: BatchCounts
    results: list[ScreenedResume]
    failures: list[FailedResume]
    duplicate_groups: list[DuplicateGroup]
    ignored_files: list[str]
    limitations: list[str]
