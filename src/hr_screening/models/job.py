"""Explicit, inspectable job requirements."""

from pydantic import BaseModel, Field

from hr_screening.models.candidate import DegreeLevel, ShortText


class JobRequirements(BaseModel):
    required_skills: list[ShortText] = Field(default_factory=list, max_length=50)
    preferred_skills: list[ShortText] = Field(default_factory=list, max_length=50)
    min_years_experience: int | None = Field(default=None, ge=0, le=50)
    min_degree_level: DegreeLevel | None = None
    certifications: list[ShortText] = Field(default_factory=list, max_length=10)
    notes: list[str] = Field(default_factory=list)