"""Rule based job description parser that produces explicit requirements."""

import re

from hr_screening.errors import InvalidJobDescriptionError
from hr_screening.extraction.degrees import detect_requested_degree_level
from hr_screening.extraction.experience_field import find_experience_field, is_work_field
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
    "minimum requirements",
    "basic requirements",
    "basic qualifications",
    "key requirements",
    "essential requirements",
    "essential skills",
    "requirements and qualifications",
    "must have",
    "what you need",
    "what we are looking for",
    "skills required",
    "required education",
    "required experience",
    "required skills and experience",
    "education",
    "education requirements",
    "educational requirements",
    "experience",
    "experience requirements",
    "work experience",
    "education and experience",
    "experience and education",
    "qualifications and experience",
    "skills and qualifications",
    "skills and experience",
    "skills",
    "technical skills",
    "core skills",
    "key skills",
    "technical requirements",
}
_PREFERRED = {
    "preferred",
    "preferred skills",
    "preferred qualifications",
    "preferred requirements",
    "nice to have",
    "good to have",
    "bonus",
    "bonus points",
    "desirable",
    "desirable skills",
    "preferred education",
    "preferred experience",
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
    "about the company",
    "about the team",
    "company overview",
    "job summary",
    "position summary",
    "role overview",
    "job description",
    "how to apply",
}
# A label with a value on the same line that describes the job and is not a requirement:
# 'Job Title: Junior Data Analyst'. The line is not scored and is not listed as unassessed.
_METADATA_LABELS = frozenset(
    {"job title", "title", "position", "location", "company", "department", "salary", "job type"}
    | {"employment type", "reports to", "job id", "reference"}
)
# A years figure, or a range such as '3-5 years' or '3 to 5 years' (the lower bound is the
# requirement). The upper bound is consumed so that it is never read as a second requirement.
_YEARS = re.compile(
    r"(?P<low>\d{1,2})\s*(?:\+|(?:-|–|—|\bto\b)\s*\d{1,2})?\s*(?:years?|yrs?)\b",
    re.IGNORECASE,
)
# A label at the start of a line that makes the rest of the line preferred: 'Nice to have: Spark'.
_INLINE_PREFERRED = re.compile(
    r"^(?:nice\s+to\s+have|good\s+to\s+have|preferred(?:\s+(?:skills|qualifications|requirements))?"
    r"|bonus(?:\s+points)?|desirable(?:\s+skills)?)\s*[:\-–—]\s*(?P<rest>\S.*)$",
    re.IGNORECASE,
)
# A skill the job itself calls optional: 'Python is a plus', 'Python (optional)', 'Python is not
# required', 'No Python experience needed'. Only wording right next to the skill counts.
_OPTIONAL_WORDS = r"(?:optional|a\s+plus|a\s+bonus|nice\s+to\s+have|desirable)"
_OPTIONAL_AFTER = re.compile(
    rf"^\s*\(\s*{_OPTIONAL_WORDS}\b"
    rf"|^\s*\(\s*not\s+required\s*\)"
    rf"|^\s*(?:is|are)\s+(?:also\s+)?(?:{_OPTIONAL_WORDS}"
    r"|not\s+(?:strictly\s+)?(?:required|needed|necessary|essential))\b",
    re.IGNORECASE,
)
_NO_BEFORE = re.compile(r"\b(?:no|without)\s+(?:\w+\s+){0,2}$", re.IGNORECASE)
_REQUIRED_AFTER_NO = re.compile(
    r"^\s*(?:experience\s+)?(?:is\s+|are\s+)?(?:required|needed|necessary)\b", re.IGNORECASE
)
_OR = re.compile(r"\bor\b", re.IGNORECASE)
_DEGREE_WORD = re.compile(r"\bdegrees?\b", re.IGNORECASE)
_MAX_NAMED_SKILLS = 5
_MAX_UNASSESSED = 50
# Text left over on a line after the recognised parts are removed. It is split into pieces at
# commas, semicolons, 'and', 'or', 'plus' and '&'. A piece that looks like a name is reported.
_PIECE_SPLIT = re.compile(r"[,;()|]|&|\s+(?:and|or|plus)\s+", re.IGNORECASE)
# A degree, optionally 'degree', and the field list after 'in' or 'of', up to 'and', a full
# stop, a semicolon, a bracket or the end of the line: "Bachelor's degree in Computer Science,
# Statistics, or a related field". The fields are not skills. Text after 'and' is not masked.
_DEGREE_PHRASE = re.compile(
    r"\b(?i:ph\.?\s?d|doctorate|doctoral|master(?:['\u2019]?s)?|m\.?sc|m\.?tech|m\.?eng|mba"
    r"|bachelor(?:['\u2019]?s)?|b\.?sc|b\.?tech|b\.?eng|b\.?com|bba|diploma"
    r"|associate(?:['\u2019]?s)?)\b(?:\s+(?i:degree))?"
    r"(?:\s+(?i:in|of)\s+[^.;()]*?(?=\s+(?i:and)\b|[.;()]|$))?"
)
# A years figure and everything after it up to the next comma or semicolon, such as
# '3+ years of experience in Data Engineering'. The field is not a skill.
_YEARS_PHRASE = re.compile(
    r"\d{1,2}\s*(?:\+|(?:-|\u2013|\u2014|\bto\b)\s*\d{1,2})?\s*(?:years?|yrs?)\b[^,;]*",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#./'-]*")
# Ordinary words that start a requirement or are capitalised in a title. They are never read as
# the name of a skill. Anything else that looks like a name is reported for a person to check.
_GENERIC_WORDS = frozenset(
    {"experience", "knowledge", "familiarity", "proficiency", "strong", "good", "excellent"}
    | {"proven", "solid", "ability", "working", "understanding", "expertise", "skills", "skill"}
    | {"demonstrated", "deep", "basic", "advanced", "relevant", "minimum", "required", "tools"}
    | {"professional", "practical", "fluent", "fluency", "exposure", "command", "background"}
    | {"degree", "years", "year", "hands-on", "hands", "on", "the", "a", "an", "of", "in", "with"}
    | {"communication", "teamwork", "english", "senior", "junior", "technical", "technology"}
    | {"environment", "environments", "systems", "software", "data", "analytics", "analysis"}
)


def parse_job_description(text: str) -> JobRequirements:
    stripped = text.strip() if isinstance(text, str) else ""
    if not stripped:
        raise InvalidJobDescriptionError("The job description is empty.")
    if len(stripped) > MAX_JOB_CHARS:
        raise InvalidJobDescriptionError(
            f"The job description is longer than {MAX_JOB_CHARS} characters."
        )

    required, preferred, other, inline_preferred = _split(stripped)
    notes: list[str] = []
    if not required and not preferred:
        required = other
        other = []
        notes.append(
            "No 'Requirements' or 'Preferred' heading was found, "
            "so all detected skills were treated as required."
        )
    preferred = [*preferred, *inline_preferred]
    if not required and preferred and other:
        notes += _ignored_lines_notes(other)

    required_skills = _find_skills(required)
    optional = [s for s in _optional_skills(required) if s in required_skills]
    required_skills = [s for s in required_skills if s not in optional]
    preferred_skills = [s for s in _find_skills(preferred) if s not in required_skills]
    preferred_skills += [s for s in optional if s not in preferred_skills]
    if optional:
        notes.append(
            "A requirement line calls a skill optional or not required "
            f"({', '.join(optional[:_MAX_NAMED_SKILLS])}). It was scored as preferred, not "
            "required. Check the job description."
        )
    if not required_skills and not preferred_skills:
        notes.append(
            "No skills from the built in vocabulary were detected. "
            "Requirements may be incomplete."
        )
    if any("certif" in line.lower() for line in preferred):
        notes.append("Preferred certifications are listed but are not scored.")
    if any(_names_degree_or_years(line) for line in preferred):
        notes.append(
            "A preferred line names a degree or years of experience. Preferred items other than "
            "skills are not scored."
        )
    notes += _year_notes(required)
    certifications, alternative, combined = _find_certifications(required)
    if alternative:
        notes.append(
            "A requirement line offers a certification as an alternative to a degree. It was "
            "scored as an education requirement only, and the certification alternative was not "
            "scored. Any skill named on that line was still scored as a required skill. A "
            "candidate who holds the certification instead of the degree needs a manual check."
        )
    if combined:
        notes.append(
            "A requirement line combines a degree and a certification. It became one "
            "certification requirement made of the whole line, which a resume rarely matches "
            "word for word. Check the degree and the certification by hand."
        )

    return JobRequirements(
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        min_years_experience=_find_years(required),
        experience_field=_find_experience_field(required),
        min_degree_level=_find_degree(required),
        certifications=certifications,
        notes=notes,
        unassessed_lines=_unassessed_lines(required),
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


def _is_metadata_label(label: str) -> bool:
    return " ".join(re.sub(r"[^a-z ]", "", label.lower()).split()) in _METADATA_LABELS


def _split(text: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """Lines under a required, preferred or other heading, plus inline preferred lines.

    An inline label ('Nice to have: Spark') sends the rest of its line to the last list wherever
    the line stands, so it cannot be mistaken for a required skill.
    """
    buckets: dict[str, list[str]] = {"required": [], "preferred": [], "other": []}
    inline: list[str] = []
    current = "other"
    for raw in text.splitlines():
        line = raw.strip().lstrip("-*•· \t").strip()
        if not line:
            continue
        kind = _heading_kind(line)
        if kind is not None:
            current = kind
            continue
        labelled = _INLINE_PREFERRED.match(line)
        if labelled is not None:
            inline.append(labelled.group("rest").strip())
            continue
        label, colon, value = line.partition(":")
        if colon and value.strip():
            if _is_metadata_label(label):
                current = "other"
                continue
            label_kind = _heading_kind(label)
            if label_kind is not None:
                # 'Experience: 3+ years' or 'Skills: Python, SQL'. The whole line is kept, so the
                # words of the label stay with the value, and the label starts that section.
                current = label_kind
                buckets[current].append(line)
                continue
        buckets[current].append(line)
    return buckets["required"], buckets["preferred"], buckets["other"], inline


def _ignored_lines_notes(lines: list[str]) -> list[str]:
    """Say so when requirement like lines sit under no recognised heading and were not scored.

    This only fires when the job has no required lines at all, so required skills that were
    lost behind an unfamiliar heading cannot go unnoticed. Nothing is guessed or scored.
    """
    skills = _find_skills(lines)
    found = bool(
        skills
        or _find_years(lines) is not None
        or _find_degree(lines) is not None
        or _find_certifications(lines)[0]
    )
    if not found:
        return []
    shown = f" (skills found there: {', '.join(skills[:_MAX_NAMED_SKILLS])})" if skills else ""
    return [
        "Some lines sit under a heading this tool does not recognise, so they were not scored "
        f"as requirements{shown}. Check the job description, and put them under a "
        "'Requirements' heading if they are required."
    ]


def _unassessed_lines(lines: list[str]) -> list[str]:
    """Job text that was not assessed: whole required lines, and parts of recognised lines.

    A line that gave no skill, years, degree or certification is listed whole. In a line where
    something was recognised, the rest is checked for pieces that look like the name of a skill
    (a capitalised word such as Terraform, an upper case word such as CI, a name with a digit,
    plus or hash). Such a piece is listed, so 'Python and Terraform' cannot read as fully
    assessed. A skill written in plain lower case is not detected, and a capitalised word that
    is not a skill (a field of study) is listed too. Text is job text, cut to the ShortText
    limit, and repeated entries are listed once.
    """
    found: list[str] = []

    def add(text: str) -> None:
        if text[:300] not in found:
            found.append(text[:300])

    for line in lines:
        recognised = (
            _find_skills([line])
            or _year_matches([line])
            or detect_requested_degree_level(line) is not None
            or "certif" in line.lower()
        )
        if not recognised:
            add(line)
        elif "certif" not in line.lower():
            for piece in _unrecognised_pieces(line):
                add(piece)
    return found[:_MAX_UNASSESSED]


def _unrecognised_pieces(line: str) -> list[str]:
    """Pieces of a recognised line that look like a skill name the vocabulary does not know."""
    label, colon, value = line.partition(":")
    rest = value if colon and _heading_kind(label) is not None else line  # drop 'Skills:'
    for canonical in SKILL_ALIASES:
        rest = skill_pattern(canonical).sub(",", rest)
    rest = _DEGREE_PHRASE.sub(",", rest)
    rest = _YEARS_PHRASE.sub(",", rest)
    found = []
    for piece in _PIECE_SPLIT.split(rest):
        piece = piece.strip(" :-\u2013\u2014\u2022")
        tokens = [t for t in _TOKEN.findall(piece) if t.lower() not in _GENERIC_WORDS]
        if any(_looks_like_a_name(t) for t in tokens):
            found.append(piece)
    return found


def _looks_like_a_name(token: str) -> bool:
    if any(ch.isdigit() for ch in token) or any(ch in token for ch in "+#"):
        return True
    if "." in token.strip("."):
        return True
    if token.isupper():
        return True
    return token[0].isupper()


def _optional_skills(lines: list[str]) -> list[str]:
    """Skills whose every mention in these lines is marked optional by the job itself."""
    flags: dict[str, list[bool]] = {}
    for line in (_mask_degree_fields(raw) for raw in lines):
        for canonical in SKILL_ALIASES:
            for match in skill_pattern(canonical).finditer(line):
                after = line[match.end() : match.end() + 40]
                before = line[max(0, match.start() - 20) : match.start()]
                optional = _OPTIONAL_AFTER.match(after) is not None or (
                    _NO_BEFORE.search(before) is not None
                    and _REQUIRED_AFTER_NO.match(after) is not None
                )
                flags.setdefault(canonical, []).append(optional)
    return [name for name, marked in flags.items() if all(marked)]


def _names_degree_or_years(line: str) -> bool:
    return detect_requested_degree_level(line) is not None or bool(_year_matches([line]))


def _mask_degree_fields(line: str) -> str:
    """The line without its degree and the fields of study that follow it."""
    return _DEGREE_PHRASE.sub(" ", line)


def _mask_work_field(line: str) -> str:
    """Remove a work field ('data analysis') named by an experience requirement.

    '1 year of data analysis experience' asks for work history in a field. It is assessed as
    experience, so the field is not also listed as a required skill. Tools are not affected:
    'experience with Python' still lists Python. Under a skills heading the skill is kept.
    """
    if "experience" not in line.lower() or _YEARS.search(line) is None:
        return line
    field = find_experience_field(line)
    if field is None or not is_work_field(field):
        return line
    return re.sub(re.escape(field), " ", line, count=1, flags=re.IGNORECASE)


def _find_skills(lines: list[str]) -> list[str]:
    joined = "\n".join(_mask_work_field(_mask_degree_fields(line)) for line in lines)
    found = []
    for canonical in SKILL_ALIASES:
        match = skill_pattern(canonical).search(joined)
        if match:
            found.append((match.start(), canonical))
    return [name for _, name in sorted(found)]


def _year_matches(lines: list[str]) -> list[tuple[int, str]]:
    """Every years figure on a line that mentions experience, as (years, line), in order."""
    matches = []
    for line in lines:
        if "experience" not in line.lower():
            continue
        for match in _YEARS.finditer(line):
            years = int(match.group("low"))
            if 0 < years <= 50:
                matches.append((years, line))
    return matches


def _find_years(lines: list[str]) -> int | None:
    matches = _year_matches(lines)
    return matches[0][0] if matches else None


def _year_notes(lines: list[str]) -> list[str]:
    """Only one years requirement is assessed. Say so when the job lists more than one."""
    matches = _year_matches(lines)
    if len(matches) < 2:
        return []
    listed = ", ".join(str(years) for years, _ in matches)
    return [
        f"The job lists {len(matches)} experience year requirements ({listed} years). "
        f"Only the first, {matches[0][0]}+ years, was assessed. Check the others against the "
        "resume manually."
    ]


def _find_experience_field(lines: list[str]) -> str | None:
    """The field named on the line that gave the years, if that line names one."""
    matches = _year_matches(lines)
    return find_experience_field(matches[0][1]) if matches else None


def _find_degree(lines: list[str]) -> DegreeLevel | None:
    levels = [level for line in lines if (level := detect_requested_degree_level(line)) is not None]
    return min(levels) if levels else None


def _find_certifications(lines: list[str]) -> tuple[list[str], bool, bool]:
    """Certification requirements, then whether a line offered a certification as an alternative
    to a degree, then whether a line combined a degree and a certification.

    'Bachelor's degree or relevant certification' is one alternative. Treating the whole line as
    a certification requirement would make one that no resume can meet, so it is left out.
    'Bachelor's degree and PMP certification' cannot be split safely, so it stays one
    certification requirement and the report says so.
    """
    found: list[str] = []
    alternative = False
    combined = False
    for line in lines:
        if "certif" not in line.lower():
            continue
        names_degree = detect_requested_degree_level(line) is not None or _DEGREE_WORD.search(line)
        if names_degree and _OR.search(line):
            alternative = True
            continue
        combined = combined or bool(names_degree)
        if line[:300] not in found:
            found.append(line[:300])
    return found[:10], alternative, combined