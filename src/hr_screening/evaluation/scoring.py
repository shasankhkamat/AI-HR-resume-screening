"""Requirement coverage score.

score = 100 * sum(weight * credit) / sum(weight), over the requirements in the job description.
Credit: supported 1.0, partial 0.5, not found 0, unverified 0.
Weights: required skill 3, experience 3, education 2, certification 2, preferred skill 1.
Weights and credits are judgment calls, not statistics. Unverified items count as zero,
so the score can understate a candidate.
"""

from collections.abc import Sequence

from hr_screening.models.report import MatchStatus, RequirementAssessment, RequirementKind

CREDIT = {
    MatchStatus.SUPPORTED: 1.0,
    MatchStatus.PARTIAL: 0.5,
    MatchStatus.NOT_FOUND: 0.0,
    MatchStatus.UNVERIFIED: 0.0,
}
WEIGHTS = {
    RequirementKind.REQUIRED_SKILL: 3.0,
    RequirementKind.EXPERIENCE: 3.0,
    RequirementKind.EDUCATION: 2.0,
    RequirementKind.CERTIFICATION: 2.0,
    RequirementKind.PREFERRED_SKILL: 1.0,
}
SCORE_EXPLANATION = (
    "Coverage score = 100 x sum(weight x credit) / sum(weight) over the listed requirements. "
    "Credit: supported 1.0, partial 0.5, not found 0, unverified 0. Weights: required skill 3, "
    "experience 3, education 2, certification 2, preferred skill 1. Unverified items count as "
    "zero, so the score can understate a candidate. It is not a measure of overall worth."
)


def compute_coverage(assessments: Sequence[RequirementAssessment]) -> tuple[float | None, str]:
    if not assessments:
        return None, "No requirements were detected, so no score was calculated."
    possible = sum(WEIGHTS[a.kind] for a in assessments)
    earned = sum(WEIGHTS[a.kind] * CREDIT[a.status] for a in assessments)
    return round(100 * earned / possible, 1), SCORE_EXPLANATION