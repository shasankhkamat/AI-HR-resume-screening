"""Explicit, inspectable job requirements."""

from pydantic import BaseModel, Field

from hr_screening.models.candidate import DegreeLevel, ShortText


class JobRequirements(BaseModel):
    required_skills: list[ShortText] = Field(default_factory=list, max_length=50)
    preferred_skills: list[ShortText] = Field(default_factory=list, max_length=50)
    min_years_experience: int | None = Field(default=None, ge=0, le=50)
    # The field the years must be in, as written in the job ('data engineering'), or None.
    experience_field: str | None = Field(default=None, max_length=100)
    min_degree_level: DegreeLevel | None = None
    certifications: list[ShortText] = Field(default_factory=list, max_length=10)
    notes: list[str] = Field(default_factory=list)
    # Lines under a required heading that gave no skill, years, degree or certification. Nothing
    # on them is scored. The report lists them so a reviewer knows they were not assessed.
    unassessed_lines: list[str] = Field(default_factory=list, max_length=50)