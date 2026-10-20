"""Structured candidate data, built from untrusted resume text."""

from enum import IntEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]


class DegreeLevel(IntEnum):
    DIPLOMA = 1
    BACHELOR = 2
    MASTER = 3
    DOCTORATE = 4


class WorkExperience(BaseModel):
    heading: ShortText
    start: str | None = Field(default=None, max_length=40)
    end: str | None = Field(default=None, max_length=40)
    duration_months: int | None = Field(default=None, ge=0, le=1200)


class Education(BaseModel):
    text: ShortText
    level: DegreeLevel | None = None


class CandidateProfile(BaseModel):
    """Only what the resume text supports. Empty means 'not found', not 'absent'."""

    skills: list[ShortText] = Field(default_factory=list, max_length=100)
    experience: list[WorkExperience] = Field(default_factory=list, max_length=30)
    education: list[Education] = Field(default_factory=list, max_length=10)
    certifications: list[ShortText] = Field(default_factory=list, max_length=20)
    projects: list[ShortText] = Field(default_factory=list, max_length=20)
    total_years_experience: float | None = Field(default=None, ge=0, le=80)
    years_source: Literal["date_ranges", "stated", "unknown"] = "unknown"
    years_snippet: str | None = Field(default=None, max_length=300)
    extraction_notes: list[str] = Field(default_factory=list)