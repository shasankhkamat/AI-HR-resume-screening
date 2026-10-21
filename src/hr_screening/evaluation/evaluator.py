"""Compare a candidate profile with job requirements, citing resume evidence.

Evidence snippets are exact text from the resume. Every snippet is checked to be
traceable to the resume text before the report is returned. That check proves the text
exists in the resume. It does not prove that the text means what the requirement needs.
"""

import re
from collections.abc import Sequence
from datetime import date

from hr_screening.evaluation.scoring import compute_coverage
from hr_screening.extraction.dates import DATE_PATTERN, PRESENT_PATTERN, total_years
from hr_screening.extraction.degrees import DegreeState, degree_state
from hr_screening.extraction.experience_field import FIELD_TITLES, field_alternatives
from hr_screening.extraction.mentions import MentionKind, SkillMention, find_skill_mentions
from hr_screening.extraction.resume_parser import SectionIndex
from hr_screening.extraction.skills_vocab import TRANSFERABLE
from hr_screening.models.candidate import CandidateProfile, DegreeLevel, WorkExperience
from hr_screening.models.job import JobRequirements
from hr_screening.models.report import (
    Evidence,
    MatchStatus,
    RequirementAssessment,
    RequirementKind,
    ScreeningReport,
    UnassessedRequirement,
)
from hr_screening.textutil import FoldedText, line_at, normalize

STANDARD_LIMITATIONS = (
    "Decision support only. A human reviewer must make any hiring decision.",
    "Not found does not mean absent: text extraction can miss content and resumes can omit "
    "relevant experience.",
    "Skills are matched by keyword with a small synonym map. Statements such as 'no "
    "experience with X' or 'learning X' are detected by simple rules that can miss unusual "
    "wording. Depth, recency and quality of skills are not assessed.",
    "Evidence is exact resume text, checked to exist in the resume. That check does not "
    "prove the text means what the requirement needs.",
    "The coverage score counts how many listed requirements have supporting text. It is not "
    "a measure of overall suitability.",
    "Job requirements are extracted with simple rules and can be missed or misread. "
    "Check them against the original job description.",
)
UNASSESSED_REASON = (
    "This job text was not recognised as a skill from the built in vocabulary, years of "
    "experience, a degree or a certification, so it was not assessed. It is not a requirement "
    "that was found missing. Give a skill by name with --extra-skill, or check it by hand."
)
INCOMPLETE_SCORE_NOTE = (
    "INCOMPLETE: {count} line(s) of the job description were not assessed and are not in this "
    "score. The score covers only the requirements listed as assessed. Review the unassessed "
    "lines before relying on it."
)
_CERT_STOPWORDS = frozenset(
    {
        "certified",
        "certification",
        "certifications",
        "certificate",
        "required",
        "preferred",
        "and",
        "the",
        "for",
        "with",
        "valid",
        "from",
        "any",
        "relevant",
    }
)
_CERT_MARKER = re.compile(r"\b(?:certified|certification|certificate|licen[sc]ed?|credential)s?\b")
_CERT_NOT_HELD = re.compile(
    r"(?<![\w'])(?:no|not|never|without|none)(?![\w'])|\w+n't\b|"
    r"\b(?:pursuing|preparing|studying|working\s+towards?|in\s+progress|expected|anticipated|"
    r"planned|planning|scheduled|upcoming|enrolled|ongoing|pending|expired|lapsed|revoked|"
    r"aspiring|aiming|will)\b"
)
_MAX_NAMED_DUPLICATES = 5
# Sections where a skill mention is not evidence of work: a hobby or a reference is not
# experience. A mention in a project is still counted, with a note.
_WEAK_SECTIONS = ("projects", "interests")
# A field of study ("Bachelor's degree in Statistics") names a subject, not a skill. A skill
# word inside the education section is not skill evidence by itself.
_NOT_SKILL_EVIDENCE = (*_WEAK_SECTIONS, "education")
_LEARNING_NOTE = (
    " A line elsewhere says the skill is being learned or planned. Read both lines."
)
_NO_FIELD_NOTE = (
    "This requirement names no field, so total experience was counted without checking "
    "whether it is relevant."
)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_DATE_RANGE = re.compile(
    rf"{DATE_PATTERN}\s*(?:-|\u2013|\u2014|\bto\b)\s*(?:{DATE_PATTERN}|{PRESENT_PATTERN})",
    re.IGNORECASE,
)
_MAX_ROLE_LINE_CHARS = 120
# A role heading names a title and a company: 'Data Analysis Intern, Foo Corp' or '... at Foo'.
_ROLE_SEPARATOR = re.compile(r",|\||@|\bat\b|\s[-\u2013\u2014]\s", re.IGNORECASE)
_ROLE_CONNECTORS = frozenset({"of", "and", "at", "in", "for", "the", "a", "an", "to", "on", "&"})
_MAX_ROLE_WORDS = 10
# A title word is required, so a tools line ('Tools: Python, SQL') or a place ('Pune, India')
# inside a role is not mistaken for the start of another role.
_TITLE_WORDS = frozenset(
    {"intern", "internship", "trainee", "apprentice", "engineer", "developer", "analyst"}
    | {"scientist", "manager", "consultant", "lead", "associate", "assistant", "specialist"}
    | {"officer", "director", "executive", "coordinator", "architect", "administrator"}
    | {"technician", "designer", "researcher", "programmer", "advisor", "supervisor"}
    | {"freelancer", "contractor", "head", "president", "representative", "volunteer"}
)
_CAPITALISED_SHARE = 0.75
# 'Certifications: AWS, Azure' names several certifications after one label.
_CERT_LABEL = re.compile(
    r"[a-z ]{0,30}(?:certifications?|certificates?|licen[sc]es?|credentials?)\s*:"
)
_CERT_TOKEN = re.compile(r"[a-z0-9+#]+")
_CERT_CLAUSE = re.compile(r"[,:|]")
_MARKER_BEFORE = 5  # 'AWS Solutions Architect Professional certificate'
_MARKER_AFTER = 3  # 'Certified in AWS'
# Words that start a new phrase, so a certification marker after them names what follows it:
# 'Deployed on AWS and earned Certified Scrum Master' says nothing about AWS.
_NEW_PHRASE = frozenset(
    {"and", "plus", "also", "then", "earned", "obtained", "received", "passed", "achieved"}
    | {"gained", "attained", "holds", "hold"}
)


def evaluate(
    profile: CandidateProfile,
    requirements: JobRequirements,
    resume_text: str,
    *,
    extra_limitations: Sequence[str] = (),
    today: date | None = None,
) -> ScreeningReport:
    duplicates: list[str] = []
    seen_skills: set[str] = set()
    required_skills = _dedupe(requirements.required_skills, seen_skills, duplicates)
    preferred_skills = _dedupe(requirements.preferred_skills, seen_skills, duplicates)
    certifications = _dedupe(requirements.certifications, set(), duplicates)

    sections = SectionIndex(resume_text)
    kind = RequirementKind.REQUIRED_SKILL
    assessments = [_assess_skill(s, kind, resume_text, sections) for s in required_skills]
    if requirements.min_years_experience is not None:
        years = requirements.min_years_experience
        field = requirements.experience_field
        assessments.append(
            _assess_experience(profile, years, resume_text, field, today or date.today())
        )
    if requirements.min_degree_level is not None:
        assessments.append(_assess_education(profile, requirements.min_degree_level))
    assessments.extend(
        _assess_certification(c, profile, resume_text, sections) for c in certifications
    )
    kind = RequirementKind.PREFERRED_SKILL
    assessments.extend(_assess_skill(s, kind, resume_text, sections) for s in preferred_skills)
    assessments = _verify_evidence(assessments, resume_text)

    score, explanation = compute_coverage(assessments)
    unassessed = [
        UnassessedRequirement(text=line, reason=UNASSESSED_REASON)
        for line in requirements.unassessed_lines
    ]
    if unassessed:
        explanation = f"{explanation} {INCOMPLETE_SCORE_NOTE.format(count=len(unassessed))}"
    notes = []
    if duplicates:
        shown = ", ".join(sorted(set(duplicates))[:_MAX_NAMED_DUPLICATES])
        notes.append(
            f"{len(duplicates)} duplicate requirement(s) were counted once ({shown}). "
            "A skill listed as both required and preferred keeps the required classification."
        )
    limitations = [
        *STANDARD_LIMITATIONS,
        *profile.extraction_notes,
        *requirements.notes,
        *notes,
        *extra_limitations,
    ]
    return ScreeningReport(
        assessments=assessments,
        coverage_score=score,
        score_explanation=explanation,
        limitations=limitations,
        unassessed_requirements=unassessed,
    )


def _dedupe(items: Sequence[str], seen: set[str], duplicates: list[str]) -> list[str]:
    """Drop blank and repeated requirements, ignoring case and spacing. Order is kept."""
    unique = []
    for item in items:
        key = normalize(item)
        if not key:
            continue
        if key in seen:
            duplicates.append(item.strip())
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _evidence(text: str, position: int) -> Evidence:
    return Evidence(snippet=line_at(text, position))


def _distinct_evidence(text: str, mentions: Sequence[SkillMention]) -> list[Evidence]:
    items: list[Evidence] = []
    for mention in mentions:
        item = _evidence(text, mention.start)
        if item not in items:
            items.append(item)
    return items


def _assess_skill(
    skill: str, kind: RequirementKind, text: str, sections: SectionIndex | None = None
) -> RequirementAssessment:
    sections = sections or SectionIndex(text)
    mentions = find_skill_mentions(text, skill)
    affirmative = [m for m in mentions if m.kind is MentionKind.AFFIRMATIVE]
    negated = [m for m in mentions if m.kind is MentionKind.NEGATED]
    aspirational = [m for m in mentions if m.kind is MentionKind.ASPIRATIONAL]

    if affirmative and negated:
        return RequirementAssessment(
            requirement=skill,
            kind=kind,
            status=MatchStatus.UNVERIFIED,
            evidence=_distinct_evidence(text, [affirmative[0], negated[0]]),
            explanation=(
                "The resume both mentions this skill and contains a statement denying it. "
                "A reviewer must read both lines."
            ),
        )
    if affirmative:
        return _assess_affirmative(skill, kind, text, sections, affirmative, aspirational)

    if negated:
        # A denial blocks credit for a related skill: "no experience with Kubernetes" is a
        # statement about Kubernetes, and a Docker line does not cancel it.
        return _not_found(
            skill,
            kind,
            "Only a negated mention was found (for example 'no experience with it'). "
            "It is not counted as experience, and a related skill does not earn credit "
            "while the resume denies this one.",
            _distinct_evidence(text, negated[:1]),
        )
    related = _transferable(skill, kind, text, sections)
    if related is not None:
        return related
    if aspirational:
        return _not_found(
            skill,
            kind,
            "The skill appears only as something being learned or planned. "
            "It is not counted as existing experience.",
            _distinct_evidence(text, aspirational[:1]),
        )
    return _not_found(
        skill,
        kind,
        "No mention found in the resume text. This does not prove a lack of the skill.",
        [],
    )


def _assess_affirmative(
    skill: str,
    kind: RequirementKind,
    text: str,
    sections: SectionIndex,
    affirmative: Sequence[SkillMention],
    aspirational: Sequence[SkillMention],
) -> RequirementAssessment:
    """A skill with at least one affirmative mention and no denial.

    A mention outside projects and interests is evidence. A mention only in a project counts with
    a note. A mention only in an interests, hobbies or references section is not counted: a
    reviewer must read it. If a line also says the skill is being learned, that line is shown
    beside the evidence and the explanation says so, so it cannot be missed.
    """
    learning = list(aspirational[:1])
    note = _LEARNING_NOTE if learning else ""
    primary = next(
        (m for m in affirmative if sections.at(m.start) not in _NOT_SKILL_EVIDENCE), None
    )
    if primary is not None:
        return RequirementAssessment(
            requirement=skill,
            kind=kind,
            status=MatchStatus.SUPPORTED,
            evidence=_distinct_evidence(text, [primary, *learning]),
            explanation=(
                "The skill is mentioned in the resume. Depth and recency are not assessed." + note
            ),
        )
    in_projects = [m for m in affirmative if sections.at(m.start) == "projects"]
    if in_projects:
        return RequirementAssessment(
            requirement=skill,
            kind=kind,
            status=MatchStatus.SUPPORTED,
            evidence=_distinct_evidence(text, [in_projects[0], *learning]),
            explanation=(
                "The skill is mentioned only in the projects section (project based "
                "evidence). This is a mention, not confirmation of professional "
                "experience or proficiency." + note
            ),
        )
    if all(sections.at(m.start) == "education" for m in affirmative):
        return RequirementAssessment(
            requirement=skill,
            kind=kind,
            status=MatchStatus.UNVERIFIED,
            evidence=_distinct_evidence(text, [affirmative[0], *learning]),
            explanation=(
                "The word appears only in the education section, where it may be a field of "
                "study rather than a skill. That is not counted as skill evidence. A reviewer "
                "must check it." + note
            ),
        )
    return RequirementAssessment(
        requirement=skill,
        kind=kind,
        status=MatchStatus.UNVERIFIED,
        evidence=_distinct_evidence(text, [affirmative[0], *learning]),
        explanation=(
            "The skill is mentioned only in an interests, hobbies or references section. That is "
            "not counted as skill evidence. A reviewer must check it." + note
        ),
    )


def _not_found(
    skill: str, kind: RequirementKind, explanation: str, evidence: list[Evidence]
) -> RequirementAssessment:
    return RequirementAssessment(
        requirement=skill,
        kind=kind,
        status=MatchStatus.NOT_FOUND,
        evidence=evidence,
        explanation=explanation,
    )


def _transferable(
    skill: str, kind: RequirementKind, text: str, sections: SectionIndex
) -> RequirementAssessment | None:
    for related in TRANSFERABLE.get(skill.strip().lower(), ()):
        hits = [
            m
            for m in find_skill_mentions(text, related)
            if m.kind is MentionKind.AFFIRMATIVE and sections.at(m.start) != "education"
        ]
        if hits:
            return RequirementAssessment(
                requirement=skill,
                kind=kind,
                status=MatchStatus.PARTIAL,
                evidence=[_evidence(text, hits[0].start)],
                explanation=(
                    f"'{skill}' was not found, but '{related}' appears and may be transferable."
                ),
            )
    return None


def _experience_line(folded: FoldedText, entry: WorkExperience) -> str | None:
    """The resume line for a role: the heading's line, else a line holding both dates."""
    spans = folded.find_spans(entry.heading)
    if spans:
        return line_at(folded.text, spans[0][0])
    if entry.start and entry.end:
        for start, _ in folded.find_spans(entry.start):
            line = line_at(folded.text, start)
            if FoldedText(line).find_spans(entry.end):
                return line
    return None


def _years_evidence(profile: CandidateProfile, text: str) -> list[Evidence]:
    if profile.years_source == "date_ranges":
        folded = FoldedText(text)
        items: list[Evidence] = []
        for entry in profile.experience[:5]:
            if entry.duration_months is None:
                continue
            line = _experience_line(folded, entry)
            if line and Evidence(snippet=line) not in items:
                items.append(Evidence(snippet=line))
        return items
    if profile.years_snippet:
        return [Evidence(snippet=profile.years_snippet)]
    return []


def _role_blocks(
    profile: CandidateProfile, text: str
) -> list[tuple[WorkExperience, str, str | None]]:
    """Each dated role with its text: the role's line up to the next role or section heading.

    Returns (role, block text, the role's own line). A role whose heading cannot be found in the
    resume gets an empty block, so it can never be judged relevant.
    """
    folded = FoldedText(text)
    sections = SectionIndex(text)
    starts = []
    for entry in profile.experience:
        spans = folded.find_spans(entry.heading)
        starts.append(text.rfind("\n", 0, spans[0][0]) + 1 if spans else None)
    known = sorted(s for s in starts if s is not None)
    blocks = []
    for entry, start in zip(profile.experience, starts, strict=True):
        if start is None or entry.duration_months is None:
            blocks.append((entry, "", None))
            continue
        end = len(text)
        section = sections.at(start)
        position = text.find("\n", start)
        while position != -1 and position + 1 < len(text):
            position += 1
            stop = text.find("\n", position)
            line = text[position : len(text) if stop == -1 else stop]
            if position in known or sections.at(position) != section or _undated_role_line(line):
                end = position
                break
            position = stop
        blocks.append((entry, text[start:end], _experience_line(folded, entry)))
    return blocks


def _undated_role_line(line: str) -> bool:
    """A short line that looks like a role heading but has no usable date range.

    Either it carries a year ('Intern, Foo (summer 2022)') or it reads as a title and a company
    ('Data Analysis Intern, Foo Corp'). The role above it must not take over the text below this
    line, because the years of the dated role would then be credited to work that has no dates.
    A bulleted line is a description, so a year inside it does not count. A bare title with no
    company ('Data Analyst') cannot be told from a title line under a dated company line, so it
    is not treated as a boundary.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_ROLE_LINE_CHARS or stripped[0] in "-*\u2022\u00b7":
        return False
    if _DATE_RANGE.search(stripped) is not None:
        return False
    return _YEAR.search(stripped) is not None or _reads_as_role_heading(stripped)


def _reads_as_role_heading(line: str) -> bool:
    """A short capitalised line with a title word and a company separator, and no sentence end."""
    if line[-1] in ".:;!?" or _ROLE_SEPARATOR.search(line) is None:
        return False
    words = [w.strip(",|@-\u2013\u2014") for w in line.split()]
    words = [w for w in words if any(c.isalpha() for c in w) and w.lower() not in _ROLE_CONNECTORS]
    if not 2 <= len(words) <= _MAX_ROLE_WORDS:
        return False
    if not any(w.lower().rstrip("s") in _TITLE_WORDS for w in words):
        return False
    return sum(w[0].isupper() for w in words) / len(words) >= _CAPITALISED_SHARE


def _undated_lines_naming_the_field(text: str, alternatives: list[list[str]]) -> list[Evidence]:
    """Undated role like lines in the experience section that name the field."""
    sections = SectionIndex(text)
    found: list[Evidence] = []
    offset = 0
    for raw in text.splitlines(keepends=True):
        line = raw.strip()
        if (
            sections.at(offset) == "experience"
            and _undated_role_line(line)
            and _names_the_field(line, alternatives)
        ):
            found.append(Evidence(snippet=line_at(text, offset + raw.find(line))))
        offset += len(raw)
    return found


def _stem(word: str) -> str:
    """Drop a plural s or an ing ending, so 'engineering' and 'engineers' meet at 'engineer'."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word[:-3] if len(word) > 5 and word.endswith("ing") else word


def _has_phrase(words: list[str], phrase: tuple[str, ...]) -> bool:
    size = len(phrase)
    return any(tuple(words[i : i + size]) == phrase for i in range(len(words) - size + 1))


def _names_the_field(block: str, alternatives: list[list[str]]) -> bool:
    """True if the block holds every word of at least one alternative of the field, or one of
    the job titles that the field is documented to accept."""
    plain = re.findall(r"[a-z0-9+#]+", block.lower())
    words = {_stem(word) for word in plain}
    if any(all(_stem(term) in words for term in terms) for terms in alternatives):
        return True
    for terms in alternatives:
        titles = FIELD_TITLES.get(tuple(terms), ())
        if any(_has_phrase(plain, title) for title in titles):
            return True
    return False


def _assess_field_experience(
    profile: CandidateProfile,
    required: int,
    text: str,
    field: str,
    alternatives: list[list[str]],
    today: date,
) -> RequirementAssessment:
    """Years in a named field. Only roles whose own text names the field are counted.

    Role text is checked for the field's words, nothing more. That cannot prove a role was in the
    field, and it misses roles that describe the field in other words, so when no role names it
    the result is unverified for a reviewer, not not found.
    """
    label = f"{required}+ years of experience"
    kind = RequirementKind.EXPERIENCE
    if profile.years_source == "stated":
        snippet = profile.years_snippet or ""
        if snippet and _names_the_field(snippet, alternatives):
            return _assess_experience_years(profile, required, text)
        return _unverified_field(
            label,
            kind,
            f"The resume states about {profile.total_years_experience:g} years of experience "
            f"but not in {field}, so relevant experience could not be confirmed.",
            [Evidence(snippet=snippet)] if snippet else [],
        )
    blocks = _role_blocks(profile, text)
    relevant = [
        entry for entry, block, _ in blocks if block and _names_the_field(block, alternatives)
    ]
    lines = [Evidence(snippet=line) for _, block, line in blocks if line and block]
    relevant_lines = [
        Evidence(snippet=line)
        for entry, block, line in blocks
        if line and entry in relevant
    ]
    years = total_years(relevant, today) if relevant else None
    if years is None:
        undated = _undated_lines_naming_the_field(text, alternatives)
        if undated:
            return _unverified_field(
                label,
                kind,
                f"{field[:1].upper()}{field[1:]} appears in a role without usable dates, so the "
                "years could not be tied to that field. Read the roles.",
                _distinct([*undated, *lines]),
            )
        total = profile.total_years_experience
        return _unverified_field(
            label,
            kind,
            f"About {total:g} years of dated roles were found, but no role mentions {field}, "
            "so relevant experience could not be confirmed. The resume may describe it in "
            "other words. Read the roles.",
            _distinct(lines),
        )
    basis = f"dated roles that mention {field}, with overlaps merged"
    return _years_result(label, kind, years, required, basis, _distinct(relevant_lines))


def _distinct(items: list[Evidence]) -> list[Evidence]:
    unique: list[Evidence] = []
    for item in items:
        if item not in unique:
            unique.append(item)
    return unique


def _unverified_field(
    label: str, kind: RequirementKind, explanation: str, evidence: list[Evidence]
) -> RequirementAssessment:
    return RequirementAssessment(
        requirement=label,
        kind=kind,
        status=MatchStatus.UNVERIFIED,
        evidence=evidence,
        explanation=explanation,
    )


def _years_result(
    label: str,
    kind: RequirementKind,
    years: float,
    required: int,
    basis: str,
    evidence: list[Evidence],
) -> RequirementAssessment:
    if years >= required:
        status = MatchStatus.SUPPORTED
        explanation = f"About {years:g} years found from {basis}."
    elif years >= required / 2:
        status = MatchStatus.PARTIAL
        explanation = (
            f"About {years:g} years found from {basis}, below the {required} requested. "
            "Dates may be incomplete."
        )
    else:
        status = MatchStatus.NOT_FOUND
        explanation = (
            f"Only about {years:g} years could be found from {basis}. "
            "The resume may omit roles."
        )
    return RequirementAssessment(
        requirement=label, kind=kind, status=status, evidence=evidence, explanation=explanation
    )


def _assess_experience(
    profile: CandidateProfile,
    required: int,
    text: str,
    field: str | None = None,
    today: date | None = None,
) -> RequirementAssessment:
    alternatives = field_alternatives(field)
    if field and alternatives and profile.total_years_experience is not None:
        return _assess_field_experience(
            profile, required, text, field, alternatives, today or date.today()
        )
    result = _assess_experience_years(profile, required, text)
    if profile.total_years_experience is None:
        return result  # nothing was counted, so there is nothing to disclose
    return result.model_copy(update={"explanation": f"{result.explanation} {_NO_FIELD_NOTE}"})


def _assess_experience_years(
    profile: CandidateProfile, required: int, text: str
) -> RequirementAssessment:
    """Total dated years, for a requirement that names no field (or a stated total that does)."""
    label = f"{required}+ years of experience"
    kind = RequirementKind.EXPERIENCE
    years = profile.total_years_experience
    if years is None:
        return RequirementAssessment(
            requirement=label,
            kind=kind,
            status=MatchStatus.UNVERIFIED,
            explanation=(
                "The resume has no dated roles or stated total years, "
                "so experience could not be calculated."
            ),
        )
    basis = (
        "dated roles with overlaps merged"
        if profile.years_source == "date_ranges"
        else "a statement in the resume"
    )
    return _years_result(label, kind, years, required, basis, _years_evidence(profile, text))


def _assess_education(profile: CandidateProfile, required: DegreeLevel) -> RequirementAssessment:
    label = f"{required.name.capitalize()} level education or higher"
    kind = RequirementKind.EDUCATION
    found = [e for e in profile.education if e.level is not None]
    if not found:
        return RequirementAssessment(
            requirement=label,
            kind=kind,
            status=MatchStatus.UNVERIFIED,
            explanation="No degree could be identified in the education section.",
        )
    complete = [e for e in found if degree_state(e.text) is DegreeState.COMPLETE]
    in_progress = [e for e in found if degree_state(e.text) is DegreeState.IN_PROGRESS]
    if complete:
        best = max(complete, key=lambda e: e.level)
        evidence = [Evidence(snippet=best.text)]
        if best.level >= required:
            return RequirementAssessment(
                requirement=label,
                kind=kind,
                status=MatchStatus.SUPPORTED,
                evidence=evidence,
                explanation="A degree at or above the requested level is listed.",
            )
        return RequirementAssessment(
            requirement=label,
            kind=kind,
            status=MatchStatus.PARTIAL,
            evidence=evidence,
            explanation=(
                f"The highest degree found is {best.level.name.lower()}, below the request. "
                "Other or equivalent credentials may be omitted."
            ),
        )
    if in_progress:
        best = max(in_progress, key=lambda e: e.level)
        if best.level >= required:
            return RequirementAssessment(
                requirement=label,
                kind=kind,
                status=MatchStatus.PARTIAL,
                evidence=[Evidence(snippet=best.text)],
                explanation=(
                    "A degree at the requested level is listed as in progress or expected. "
                    "It is not counted as completed."
                ),
            )
    shown = [Evidence(snippet=e.text) for e in found[:3]]
    return RequirementAssessment(
        requirement=label,
        kind=kind,
        status=MatchStatus.UNVERIFIED,
        evidence=shown,
        explanation=(
            "The degrees listed are unfinished, below the requested level while in progress, "
            "or worded in a way that cannot be judged automatically. Check them manually."
        ),
    )


def _word_in(word: str, lowered_text: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", lowered_text) is not None


def _assess_certification(
    requirement: str,
    profile: CandidateProfile,
    text: str,
    sections: SectionIndex | None = None,
) -> RequirementAssessment:
    kind = RequirementKind.CERTIFICATION
    words = [
        w
        for w in re.findall(r"[a-z0-9+#]+", requirement.lower())
        if len(w) > 2 and w not in _CERT_STOPWORDS
    ]
    if not words:
        return RequirementAssessment(
            requirement=requirement,
            kind=kind,
            status=MatchStatus.UNVERIFIED,
            explanation="No searchable terms could be derived from this requirement.",
        )
    sections = sections or SectionIndex(text)
    held, not_held, indirect = _certification_lines(text, words, sections)
    if held and not_held:
        return RequirementAssessment(
            requirement=requirement,
            kind=kind,
            status=MatchStatus.UNVERIFIED,
            evidence=[held[0], not_held[0]],
            explanation="The resume contains conflicting statements about this certification.",
        )
    if held:
        return RequirementAssessment(
            requirement=requirement,
            kind=kind,
            status=MatchStatus.SUPPORTED,
            evidence=[held[0]],
            explanation="All key terms of the certification appear together on one resume line.",
        )
    if not_held:
        return RequirementAssessment(
            requirement=requirement,
            kind=kind,
            status=MatchStatus.UNVERIFIED,
            evidence=[not_held[0]],
            explanation=(
                "The certification is mentioned as negated, in progress, planned or expired. "
                "It is not counted as held."
            ),
        )
    if indirect:
        return RequirementAssessment(
            requirement=requirement,
            kind=kind,
            status=MatchStatus.UNVERIFIED,
            evidence=[indirect[0]],
            explanation=(
                "The key terms appear, but not as a listed certification (for example in a job "
                "title or description). They are not counted as a certification."
            ),
        )
    if profile.certifications:
        return RequirementAssessment(
            requirement=requirement,
            kind=kind,
            status=MatchStatus.UNVERIFIED,
            evidence=[Evidence(snippet=c) for c in profile.certifications[:3]],
            explanation=(
                "Other certifications are listed. Whether they are equivalent "
                "cannot be judged automatically."
            ),
        )
    return RequirementAssessment(
        requirement=requirement,
        kind=kind,
        status=MatchStatus.NOT_FOUND,
        explanation="No matching certification found. It may exist but not be listed.",
    )


def _marker_names_terms(lowered: str, words: list[str]) -> bool:
    """True if a certification marker sits next to the key terms, in one sentence and clause.

    'AWS Certified Solutions Architect' qualifies. 'Managed AWS infrastructure. Certified Scrum
    Master.' does not: the marker belongs to another certification, so it says nothing about AWS.
    A key term must be within a few words of the marker, without a comma or a sentence end in
    between. A marker that opens a new phrase ('... and certified Scrum Master') looks forward
    only. A line that starts with a label such as 'Certifications:' names everything after it.
    """
    for sentence in re.split(r"(?<=[.;!?])\s+", lowered):
        if not all(word in _CERT_TOKEN.findall(sentence) for word in words):
            continue
        if _CERT_LABEL.match(sentence):
            return True
        for clause in _CERT_CLAUSE.split(sentence):
            tokens = _CERT_TOKEN.findall(clause)
            for index, token in enumerate(tokens):
                if not _CERT_MARKER.fullmatch(token):
                    continue
                opens_phrase = index > 0 and tokens[index - 1] in _NEW_PHRASE
                before = [] if opens_phrase else tokens[max(index - _MARKER_BEFORE, 0) : index]
                nearby = before + tokens[index + 1 : index + 1 + _MARKER_AFTER]
                if any(word in nearby for word in words):
                    return True
    return False


def _certification_lines(
    text: str, words: list[str], sections: SectionIndex
) -> tuple[list[Evidence], list[Evidence], list[Evidence]]:
    """Split lines holding every key term into held, not held and indirect mentions."""
    held: list[Evidence] = []
    not_held: list[Evidence] = []
    indirect: list[Evidence] = []
    offset = 0
    for raw in text.splitlines(keepends=True):
        line = raw.strip()
        lowered = line.lower()
        if line and all(_word_in(w, lowered) for w in words):
            position = offset + raw.find(line) + max(lowered.find(words[0]), 0)
            item = Evidence(snippet=line_at(text, position))
            credible = sections.at(offset) == "certifications" or _marker_names_terms(
                lowered, words
            )
            if not credible:
                indirect.append(item)
            elif _CERT_NOT_HELD.search(lowered):
                not_held.append(item)
            else:
                held.append(item)
        offset += len(raw)
    return held, not_held, indirect


def _verify_evidence(
    assessments: list[RequirementAssessment], text: str
) -> list[RequirementAssessment]:
    """Remove evidence that is not traceable to the resume text.

    A supported or partial result with no traceable evidence left becomes unverified.
    """
    haystack = normalize(text)
    verified = []
    for item in assessments:
        traceable = [e for e in item.evidence if normalize(e.snippet) in haystack]
        needs_evidence = item.status in (MatchStatus.SUPPORTED, MatchStatus.PARTIAL)
        if len(traceable) == len(item.evidence) and (traceable or not needs_evidence):
            verified.append(item)
            continue
        note = " Evidence that could not be traced to the resume text was removed."
        status = item.status
        if needs_evidence and not traceable:
            status = MatchStatus.UNVERIFIED
            note += " The status was changed to unverified because no traceable evidence remains."
        update = {"evidence": traceable, "status": status, "explanation": item.explanation + note}
        verified.append(item.model_copy(update=update))
    return verified