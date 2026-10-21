"""Compare a candidate profile with job requirements, citing resume evidence.

Evidence snippets are exact text from the resume. Every snippet is checked to be
traceable to the resume text before the report is returned. That check proves the text
exists in the resume. It does not prove that the text means what the requirement needs.
"""

import re
from collections.abc import Sequence

from hr_screening.evaluation.scoring import compute_coverage
from hr_screening.extraction.degrees import DegreeState, degree_state
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


def evaluate(
    profile: CandidateProfile,
    requirements: JobRequirements,
    resume_text: str,
    *,
    extra_limitations: Sequence[str] = (),
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
        assessments.append(_assess_experience(profile, years, resume_text))
    if requirements.min_degree_level is not None:
        assessments.append(_assess_education(profile, requirements.min_degree_level))
    assessments.extend(
        _assess_certification(c, profile, resume_text, sections) for c in certifications
    )
    kind = RequirementKind.PREFERRED_SKILL
    assessments.extend(_assess_skill(s, kind, resume_text, sections) for s in preferred_skills)
    assessments = _verify_evidence(assessments, resume_text)

    score, explanation = compute_coverage(assessments)
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
        primary = next((m for m in affirmative if sections.at(m.start) != "projects"), None)
        if primary is None:
            return RequirementAssessment(
                requirement=skill,
                kind=kind,
                status=MatchStatus.SUPPORTED,
                evidence=_distinct_evidence(text, affirmative[:1]),
                explanation=(
                    "The skill is mentioned only in the projects section (project based "
                    "evidence). This is a mention, not confirmation of professional "
                    "experience or proficiency."
                ),
            )
        return RequirementAssessment(
            requirement=skill,
            kind=kind,
            status=MatchStatus.SUPPORTED,
            evidence=_distinct_evidence(text, [primary]),
            explanation="The skill is mentioned in the resume. Depth and recency are not assessed.",
        )

    related = _transferable(skill, kind, text)
    if related is not None:
        return related
    if negated:
        return _not_found(
            skill,
            kind,
            "Only a negated mention was found (for example 'no experience with it'). "
            "It is not counted as experience.",
            _distinct_evidence(text, negated[:1]),
        )
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


def _transferable(skill: str, kind: RequirementKind, text: str) -> RequirementAssessment | None:
    for related in TRANSFERABLE.get(skill.strip().lower(), ()):
        hits = [
            m for m in find_skill_mentions(text, related) if m.kind is MentionKind.AFFIRMATIVE
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


def _assess_experience(
    profile: CandidateProfile, required: int, text: str
) -> RequirementAssessment:
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
        requirement=label,
        kind=kind,
        status=status,
        evidence=_years_evidence(profile, text),
        explanation=explanation,
    )


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
            credible = sections.at(offset) == "certifications" or _CERT_MARKER.search(lowered)
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