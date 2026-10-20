"""Compare a candidate profile with job requirements, citing resume evidence."""

import re
from collections.abc import Sequence

from hr_screening.evaluation.scoring import compute_coverage
from hr_screening.extraction.skills_vocab import TRANSFERABLE, skill_pattern
from hr_screening.models.candidate import CandidateProfile, DegreeLevel
from hr_screening.models.job import JobRequirements
from hr_screening.models.report import (
    Evidence,
    MatchStatus,
    RequirementAssessment,
    RequirementKind,
    ScreeningReport,
)
from hr_screening.textutil import line_at

STANDARD_LIMITATIONS = (
    "Decision support only. A human reviewer must make any hiring decision.",
    "Not found does not mean absent: text extraction can miss content and resumes can omit "
    "relevant experience.",
    "Skills are matched by keyword with a small synonym map. Depth, recency and quality of "
    "skills are not assessed.",
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


def evaluate(
    profile: CandidateProfile,
    requirements: JobRequirements,
    resume_text: str,
    *,
    extra_limitations: Sequence[str] = (),
) -> ScreeningReport:
    kind = RequirementKind.REQUIRED_SKILL
    assessments = [_assess_skill(s, kind, resume_text) for s in requirements.required_skills]
    if requirements.min_years_experience is not None:
        assessments.append(_assess_experience(profile, requirements.min_years_experience))
    if requirements.min_degree_level is not None:
        assessments.append(_assess_education(profile, requirements.min_degree_level))
    assessments.extend(
        _assess_certification(c, profile, resume_text) for c in requirements.certifications
    )
    kind = RequirementKind.PREFERRED_SKILL
    assessments.extend(_assess_skill(s, kind, resume_text) for s in requirements.preferred_skills)

    score, explanation = compute_coverage(assessments)
    limitations = [
        *STANDARD_LIMITATIONS,
        *profile.extraction_notes,
        *requirements.notes,
        *extra_limitations,
    ]
    return ScreeningReport(
        assessments=assessments,
        coverage_score=score,
        score_explanation=explanation,
        limitations=limitations,
    )


def _assess_skill(skill: str, kind: RequirementKind, text: str) -> RequirementAssessment:
    match = skill_pattern(skill).search(text)
    if match:
        return RequirementAssessment(
            requirement=skill,
            kind=kind,
            status=MatchStatus.SUPPORTED,
            evidence=[Evidence(snippet=line_at(text, match.start()))],
            explanation="The skill is mentioned in the resume. Depth and recency are not assessed.",
        )
    for related in TRANSFERABLE.get(skill.strip().lower(), ()):
        related_match = skill_pattern(related).search(text)
        if related_match:
            return RequirementAssessment(
                requirement=skill,
                kind=kind,
                status=MatchStatus.PARTIAL,
                evidence=[Evidence(snippet=line_at(text, related_match.start()))],
                explanation=(
                    f"'{skill}' was not found, but '{related}' appears and may be transferable."
                ),
            )
    return RequirementAssessment(
        requirement=skill,
        kind=kind,
        status=MatchStatus.NOT_FOUND,
        explanation="No mention found in the resume text. This does not prove a lack of the skill.",
    )


def _years_evidence(profile: CandidateProfile) -> list[Evidence]:
    if profile.years_source == "date_ranges":
        return [
            Evidence(snippet=f"{e.heading} ({e.start} to {e.end})")
            for e in profile.experience[:5]
            if e.duration_months is not None
        ]
    if profile.years_snippet:
        return [Evidence(snippet=profile.years_snippet)]
    return []


def _assess_experience(profile: CandidateProfile, required: int) -> RequirementAssessment:
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
        evidence=_years_evidence(profile),
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
    best = max(found, key=lambda e: e.level)
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


def _word_in(word: str, lowered_text: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", lowered_text) is not None


def _assess_certification(
    requirement: str, profile: CandidateProfile, text: str
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
    lowered = text.lower()
    if all(_word_in(w, lowered) for w in words):
        position = lowered.find(words[0])
        return RequirementAssessment(
            requirement=requirement,
            kind=kind,
            status=MatchStatus.SUPPORTED,
            evidence=[Evidence(snippet=line_at(text, max(position, 0)))],
            explanation="All key terms of the certification appear in the resume.",
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