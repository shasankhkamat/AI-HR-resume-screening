"""Rule based resume parser: plain text to CandidateProfile. Makes no external calls.

It does not extract names or contact details, by design.
"""

import re
from bisect import bisect_right
from datetime import date

from hr_screening.extraction.dates import (
    DATE_PATTERN,
    PRESENT_PATTERN,
    month_span,
    total_years,
    unusable_date_ranges,
)
from hr_screening.extraction.degrees import detect_degree_level
from hr_screening.models.candidate import CandidateProfile, Education, WorkExperience
from hr_screening.textutil import line_at

_SECTIONS = {
    "skills": ("skills", "technical skills", "core skills", "key skills", "skills summary"),
    "experience": (
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "employment",
        "work history",
    ),
    "education": ("education", "academic background", "education and training"),
    "certifications": (
        "certifications",
        "certification",
        "certificates",
        "licenses and certifications",
    ),
    "projects": ("projects", "academic projects", "personal projects", "key projects"),
    # Not work. A skill mentioned only here is not counted as skill evidence.
    "interests": (
        "interests",
        "hobbies",
        "references",
        "hobbies and interests",
        "interests and hobbies",
    ),
}
_OTHER_HEADINGS = (
    "summary",
    "profile",
    "objective",
    "awards",
    "achievements",
    "publications",
    "volunteering",
)
_HEADINGS = {alias: key for key, aliases in _SECTIONS.items() for alias in aliases}
_HEADINGS.update({alias: "other" for alias in _OTHER_HEADINGS})

_RANGE = re.compile(
    rf"(?P<start>{DATE_PATTERN})\s*(?:-|\u2013|\u2014|\bto\b)\s*"
    rf"(?P<end>{DATE_PATTERN}|{PRESENT_PATTERN})",
    re.IGNORECASE,
)
_STATED_YEARS = re.compile(
    r"\b(\d{1,2})\s*\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:[\w-]+\s+){0,3}?experience",
    re.IGNORECASE,
)
# The first years figure of a line in the experience section: '2 years as a Data Analyst'.
_SECTION_YEARS = re.compile(
    r"^(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b(?!\s+(?:old|ago))", re.IGNORECASE
)
_SKILL_SPLIT = re.compile(r"[,;|\u2022\u00b7\u25cf\u25aa]|\s{2,}")


def parse_resume(text: str, *, today: date | None = None) -> CandidateProfile:
    today = today or date.today()
    sections = _split_sections(text)
    notes: list[str] = []
    if not sections:
        notes.append(
            "No standard section headings (SKILLS, EXPERIENCE, EDUCATION) were found, "
            "so structured extraction is limited."
        )

    experience = _parse_experience(sections.get("experience", []), today)
    skipped = unusable_date_ranges(experience, today)
    if skipped:
        notes.append(
            f"{skipped} dated role(s) had date ranges that could not be counted "
            "(in the future, reversed or implausible)."
        )
    years = total_years(experience, today)
    source = "date_ranges" if years is not None else "unknown"
    snippet = None
    if years is None:
        stated = find_stated_years(text)
        if stated is not None:
            years, snippet = stated
            source = "stated"
        else:
            stated = _stated_years_in_section(text, sections.get("experience", []))
            if stated is not None:
                years, snippet = stated
                source = "stated"
            elif "experience" not in sections:
                notes.append(
                    "No experience section was found, so years of experience are unknown."
                )
            else:
                notes += _month_notes(sections["experience"])

    return CandidateProfile(
        skills=_parse_skills(sections.get("skills", [])),
        experience=experience,
        education=_parse_education(sections.get("education", [])),
        certifications=_plain_items(sections.get("certifications", []), 20),
        projects=_plain_items(sections.get("projects", []), 20),
        total_years_experience=years,
        years_source=source,
        years_snippet=snippet,
        extraction_notes=notes,
    )


def find_stated_years(text: str) -> tuple[float, str] | None:
    """Find a statement like '5+ years of experience'. Returns (years, source line)."""
    match = _STATED_YEARS.search(text)
    if match is None:
        return None
    return float(match.group(1)), line_at(text, match.start())


_SECTION_MONTHS = re.compile(r"^(\d{1,2})\s*(?:months?|mos?)\b", re.IGNORECASE)


def _month_notes(lines: list[str]) -> list[str]:
    """Say so when the experience section states a duration in months. It is not converted."""
    return [
        f"The experience section states a duration in months ('{line[:80]}'). Months are not "
        "converted into years, so this is not counted as years of experience. Read it."
        for line in lines
        if _SECTION_MONTHS.match(_strip_bullet(line))
    ][:3]


def _stated_years_in_section(text: str, lines: list[str]) -> tuple[float, str] | None:
    """Years stated on a line of the experience section, such as '2 years as a Data Analyst'.

    Only a single statement is used. With two or more the total is unclear and nothing is
    guessed. Months are never turned into years. The snippet is the line as written.
    """
    found = []
    for line in lines:
        match = _SECTION_YEARS.match(_strip_bullet(line))
        if match is not None and 0 < int(match.group(1)) <= 50:
            found.append((float(match.group(1)), line))
    if len(found) != 1:
        return None
    years, line = found[0]
    position = text.find(line)
    return (years, line_at(text, position)) if position >= 0 else None


def _heading_key(line: str) -> str | None:
    """Return the section key if ``line`` is a section heading, else None."""
    return _HEADINGS.get(line.rstrip(":").strip().lower()) if len(line) <= 40 else None


def _labelled(line: str) -> tuple[str, str] | None:
    """A heading with its text on the same line: 'Skills: Python, SQL'. Returns (key, text).

    The label must start the line, with no bullet, and must be a known heading, so ordinary
    text with a colon is not taken for a heading. The text after the colon must not be empty.
    """
    label, colon, value = line.partition(":")
    if not colon or not value.strip():
        return None
    key = _heading_key(label.strip())
    return (key, value.strip()) if key is not None else None


class SectionIndex:
    """Maps a character position in the resume text to its section heading, if any."""

    def __init__(self, text: str) -> None:
        self._starts: list[int] = []
        self._names: list[str | None] = []
        offset = 0
        for raw in text.splitlines(keepends=True):
            line = raw.strip()
            labelled = _labelled(line) if line else None
            key = labelled[0] if labelled else (_heading_key(line) if line else None)
            if key is not None:
                self._starts.append(offset)
                self._names.append(None if key == "other" else key)
            offset += len(raw)

    def at(self, position: int) -> str | None:
        """Section key such as 'projects' or 'experience', or None if unknown."""
        found = bisect_right(self._starts, position) - 1
        return self._names[found] if found >= 0 else None


def _split_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        labelled = _labelled(line)
        if labelled is not None:
            current = labelled[0]
            sections.setdefault(current, []).append(labelled[1])
            continue
        key = _heading_key(line)
        if key is not None:
            current = key
            sections.setdefault(key, [])
            continue
        if current is not None:
            sections[current].append(line)
    sections.pop("other", None)
    sections.pop("interests", None)
    return sections


def _strip_bullet(line: str) -> str:
    return line.lstrip("-*\u2022\u00b7 \t").strip()


def _plain_items(lines: list[str], limit: int) -> list[str]:
    items = [_strip_bullet(line)[:300] for line in lines]
    return [item for item in items if item][:limit]


def _parse_skills(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    skills: list[str] = []
    for line in lines:
        if ":" in line:
            label, rest = line.split(":", 1)
            if len(label) <= 30:
                line = rest
        for part in _SKILL_SPLIT.split(line):
            item = part.strip(" -\u2013\u2014*\t")
            if len(item) > 1 and item.endswith(".") and not item.endswith(".."):
                item = item[:-1].rstrip()  # the full stop that ends a sentence: 'Tableau.'
            key = item.lower()
            if 1 <= len(item) <= 40 and key not in seen:
                seen.add(key)
                skills.append(item)
    return skills[:100]


def _clean_heading(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" ,|-\u2013\u2014()\t")


def _parse_experience(lines: list[str], today: date) -> list[WorkExperience]:
    entries: list[WorkExperience] = []
    previous = ""
    for line in lines:
        match = _RANGE.search(line)
        if match is None:
            previous = line
            continue
        heading = (
            _clean_heading(line[: match.start()] + " " + line[match.end() :])
            or _clean_heading(previous)
            or "Unlabeled role"
        )
        start, end = match.group("start"), match.group("end")
        entries.append(
            WorkExperience(
                heading=heading[:300],
                start=start,
                end=end,
                duration_months=month_span(start, end, today),
            )
        )
        previous = ""
    return entries[:30]


def _parse_education(lines: list[str]) -> list[Education]:
    cleaned = [_strip_bullet(line) for line in lines if _strip_bullet(line)]
    entries = []
    for line in cleaned:
        level = detect_degree_level(line)
        if level is not None:
            entries.append(Education(text=line[:300], level=level))
    if not entries and cleaned:
        entries.append(Education(text=cleaned[0][:300], level=None))
    return entries[:10]