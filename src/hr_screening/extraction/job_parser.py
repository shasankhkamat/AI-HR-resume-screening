"""Rule based job description parser that produces explicit requirements."""

import re

from hr_screening.errors import InvalidJobDescriptionError
from hr_screening.extraction.degrees import detect_degree_level
from hr_screening.extraction.skills_vocab import SKILL_ALIASES, skill_pattern
from hr_screening.models.candidate import DegreeLevel
from hr_screening.models.job import JobRequirements

MAX_JOB_CHARS = 50_000

_REQUIRED = {
    "requirements",
    "required",
    "required skills",
    "required qualifications",
    "qualifications",
    "minimum qualifications",
    "must have",
    "what you need",
    "what we are looking for",
    "skills required",
}
_PREFERRED = {
    "preferred",
    "preferred skills",
    "preferred qualifications",
    "nice to have",
    "good to have",
    "bonus",
    "bonus points",
}
_OTHER = {
    "responsibilities",
    "key responsibilities",
    "about us",
    "about the role",
    "about the job",
    "benefits",
    "what you will do",
    "what we offer",
    "overview",
}
_YEARS = re.compile(r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)", re.IGNORECASE)


def parse_job_description(text: str) -> JobRequirements:
    stripped = text.strip() if isinstance(text, str) else ""
    if not stripped:
        raise InvalidJobDescriptionError("The job description is empty.")
    if len(stripped) > MAX_JOB_CHARS:
        raise InvalidJobDescriptionError(
            f"The job description is longer than {MAX_JOB_CHARS} characters."
        )

    required, preferred, other = _split(stripped)
    notes: list[str] = []
    if not required and not preferred:
        required = other
        notes.append(
            "No 'Requirements' or 'Preferred' heading was found, "
            "so all detected skills were treated as required."
        )

    required_skills = _find_skills(required)
    preferred_skills = [s for s in _find_skills(preferred) if s not in required_skills]
    if not required_skills and not preferred_skills:
        notes.append(
            "No skills from the built in vocabulary were detected. "
            "Requirements may be incomplete."
        )
    if any("certif" in line.lower() for line in preferred):
        notes.append("Preferred certifications are listed but are not scored.")

    return JobRequirements(
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        min_years_experience=_find_years(required),
        min_degree_level=_find_degree(required),
        certifications=_find_certifications(required),
        notes=notes,
    )


def _heading_kind(line: str) -> str | None:
    if len(line) > 50:
        return None
    key = " ".join(re.sub(r"[^a-z ]", "", line.lower()).split())
    if key in _REQUIRED:
        return "required"
    if key in _PREFERRED:
        return "preferred"
    if key in _OTHER:
        return "other"
    return None


def _split(text: str) -> tuple[list[str], list[str], list[str]]:
    buckets: dict[str, list[str]] = {"required": [], "preferred": [], "other": []}
    current = "other"
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*\u2022\u00b7 \t").strip()
        if not line:
            continue
        kind = _heading_kind(line)
        if kind is not None:
            current = kind
            continue
        buckets[current].append(line)
    return buckets["required"], buckets["preferred"], buckets["other"]


def _find_skills(lines: list[str]) -> list[str]:
    joined = "\n".join(lines)
    found = []
    for canonical in SKILL_ALIASES:
        match = skill_pattern(canonical).search(joined)
        if match:
            found.append((match.start(), canonical))
    return [name for _, name in sorted(found)]


def _find_years(lines: list[str]) -> int | None:
    for line in lines:
        if "experience" not in line.lower():
            continue
        match = _YEARS.search(line)
        if match and 0 < int(match.group(1)) <= 50:
            return int(match.group(1))
    return None


def _find_degree(lines: list[str]) -> DegreeLevel | None:
    levels = [level for line in lines if (level := detect_degree_level(line)) is not None]
    return min(levels) if levels else None


def _find_certifications(lines: list[str]) -> list[str]:
    found: list[str] = []
    for line in lines:
        if "certif" in line.lower() and line[:300] not in found:
            found.append(line[:300])
    return found[:10]