"""The screening workflow: resume file + job description -> report. Nothing is stored."""

import logging
from datetime import date
from pathlib import Path

from hr_screening.errors import (
    LLMConfigError,
    LLMConnectionError,
    LLMError,
    LLMModelNotFoundError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from hr_screening.evaluation.evaluator import evaluate
from hr_screening.extraction.job_parser import parse_job_description
from hr_screening.extraction.resume_parser import parse_resume
from hr_screening.ingestion.loader import load_resume_text
from hr_screening.llm.base import LLMClient
from hr_screening.llm.extractor import extract_profile_with_llm
from hr_screening.models.candidate import CandidateProfile
from hr_screening.models.report import ScreeningReport

logger = logging.getLogger(__name__)

LLM_SUCCESS_NOTE = (
    "A language model performed the extraction. Extracted items were checked to exist in the "
    "resume text. That check does not prove they are correct or mean what the model says."
)
_FAILURE_REASONS = (
    (LLMTimeoutError, "the model did not answer in time"),
    (LLMConnectionError, "the model server could not be reached"),
    (LLMModelNotFoundError, "the configured model is not installed"),
    (LLMRateLimitError, "the model server was rate limiting requests"),
    (LLMResponseError, "the model reply was empty, cut off or invalid"),
)


def _fallback_note(error: LLMError) -> str:
    """A fixed, safe explanation. It never includes the error text, only a known reason."""
    for kind, reason in _FAILURE_REASONS:
        if isinstance(error, kind):
            return f"LLM extraction failed ({reason}), so the rule based parser was used instead."
    return "LLM extraction failed, so the rule based parser was used instead."


def screen_resume(
    resume_path: str | Path,
    job_description: str,
    *,
    client: LLMClient | None = None,
    base_dir: str | Path | None = None,
    today: date | None = None,
) -> ScreeningReport:
    """Screen one resume against one job description.

    If a client is given and it fails or returns something invalid, the rule based
    parser is used instead and the report says so. A configuration error is not an
    operational failure, so it is raised instead of hidden.
    """
    resume_text = load_resume_text(resume_path, base_dir=base_dir)
    requirements = parse_job_description(job_description)

    notes: list[str] = []
    profile: CandidateProfile | None = None
    if client is not None:
        try:
            profile, _ = extract_profile_with_llm(client, resume_text, today=today)
            notes.append(LLM_SUCCESS_NOTE)
        except LLMConfigError:
            raise
        except LLMError as exc:
            logger.warning(
                "LLM extraction failed (%s); using rule based parser", type(exc).__name__
            )
            notes.append(_fallback_note(exc))
    if profile is None:
        profile = parse_resume(resume_text, today=today)
    return evaluate(profile, requirements, resume_text, extra_limitations=notes)