"""The screening workflow: resume file + job description -> report. Nothing is stored."""

import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Literal

from hr_screening.errors import (
    EmbeddingError,
    InvalidExtraSkillError,
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
from hr_screening.extraction.skills_vocab import skill_pattern
from hr_screening.ingestion.loader import load_resume_text
from hr_screening.llm.base import LLMClient
from hr_screening.llm.extractor import extract_profile_with_llm
from hr_screening.models.candidate import CandidateProfile
from hr_screening.models.job import JobRequirements
from hr_screening.models.report import RequirementRetrieval, ScreeningReport
from hr_screening.rag.embeddings import Embedder
from hr_screening.rag.requirements import retrieve_for_assessments, verify_traceable
from hr_screening.rag.retrieval import LexicalRetriever, Retriever
from hr_screening.rag.semantic import SemanticRetriever
from hr_screening.textutil import normalize

logger = logging.getLogger(__name__)

LLM_SUCCESS_NOTE = (
    "A language model performed the extraction. Extracted items were checked to exist in the "
    "resume text. That check does not prove they are correct or mean what the model says."
)
RAG_MODES = ("lexical", "semantic")
RETRIEVAL_NOTE = (
    "Retrieved passages come from a keyword search over the resume. A passage is not evidence "
    "that a requirement is met, and the search can miss text that uses different words."
)
SEMANTIC_NOTE = (
    "Retrieved passages come from a semantic search{model}. Passages are ranked by embedding "
    "similarity, which can match related wording and can also match unrelated text. A passage "
    "is not evidence that a requirement is met."
)
RETRIEVAL_TRUNCATED_NOTE = (
    "The resume is very long, so only its first part was searched for passages."
)
RETRIEVAL_FAILED_NOTE = (
    "Passage retrieval failed, so this report has no retrieved passages. "
    "The statuses and the score are not affected."
)
MAX_EXTRA_SKILLS = 20
MAX_EXTRA_SKILL_CHARS = 60
_SHORT_SKILL_CHARS = 2
_MAX_REQUIRED_SKILLS = 50  # the limit of JobRequirements.required_skills
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


def _clean_extra_skills(extra: Sequence[str]) -> list[str]:
    """Validated, de-duplicated extra skill names. A bad value is an error, never ignored."""
    if isinstance(extra, str):
        raise InvalidExtraSkillError("Extra skills must be given as a list of names.")
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in extra:
        name = " ".join(str(raw).split())
        if not name:
            raise InvalidExtraSkillError("An extra skill is empty.")
        if len(name) > MAX_EXTRA_SKILL_CHARS:
            raise InvalidExtraSkillError(
                f"An extra skill is longer than {MAX_EXTRA_SKILL_CHARS} characters. "
                "Give a skill name, not a sentence."
            )
        key = normalize(name)
        if key not in seen:
            seen.add(key)
            cleaned.append(name)
    if len(cleaned) > MAX_EXTRA_SKILLS:
        raise InvalidExtraSkillError(f"More than {MAX_EXTRA_SKILLS} extra skills were given.")
    return cleaned


def _with_extra_skills(requirements: JobRequirements, extra: Sequence[str]) -> JobRequirements:
    """Add skills named by the person running the screen as required skills.

    They are assessed like any required skill: a whole word match in the resume, with the same
    negation and learning rules. They are never added to the built in vocabulary. A skill the
    job already lists as preferred becomes required, because the person asked for it. A line of
    the job description that names an extra skill is no longer listed as unassessed.
    """
    names = _clean_extra_skills(extra)
    if not names:
        return requirements
    keys = {normalize(name) for name in names}
    already = {normalize(s) for s in requirements.required_skills}
    added = [name for name in names if normalize(name) not in already]
    required = [*requirements.required_skills, *added]
    if len(required) > _MAX_REQUIRED_SKILLS:
        raise InvalidExtraSkillError("Too many required skills with the extra skills added.")
    preferred = [s for s in requirements.preferred_skills if normalize(s) not in keys]
    patterns = [skill_pattern(name) for name in names]
    unassessed = [
        line
        for line in requirements.unassessed_lines
        if not any(pattern.search(line) for pattern in patterns)
    ]
    notes = [
        *requirements.notes,
        "Required skills added by the person running the screen (--extra-skill): "
        f"{', '.join(names)}. They are matched as whole words in the resume text.",
    ]
    short = [name for name in names if len(name) <= _SHORT_SKILL_CHARS]
    if short:
        notes.append(
            f"The extra skill(s) {', '.join(short)} are very short, so they can match unrelated "
            "text. Read the evidence line for each."
        )
    return requirements.model_copy(
        update={
            "required_skills": required,
            "preferred_skills": preferred,
            "unassessed_lines": unassessed,
            "notes": notes,
        }
    )


def prepare_requirements(
    job_description: str, extra_skills: Sequence[str] = ()
) -> JobRequirements:
    """Parse a job description and add the extra skills. A batch does this once, not per resume."""
    return _with_extra_skills(parse_job_description(job_description), extra_skills)


def check_rag_options(rag_mode: str | None, embedder: Embedder | None) -> None:
    if rag_mode is not None and rag_mode not in RAG_MODES:
        raise ValueError(f"rag_mode must be one of {RAG_MODES} or None.")
    if rag_mode == "semantic" and embedder is None:
        raise ValueError("rag_mode 'semantic' needs an embedder.")


def screen_resume(
    resume_path: str | Path,
    job_description: str,
    *,
    client: LLMClient | None = None,
    base_dir: str | Path | None = None,
    today: date | None = None,
    rag_mode: Literal["lexical", "semantic"] | None = None,
    embedder: Embedder | None = None,
    extra_skills: Sequence[str] = (),
) -> ScreeningReport:
    """Screen one resume against one job description.

    If a client is given and it fails or returns something invalid, the rule based
    parser is used instead and the report says so. A configuration error is not an
    operational failure, so it is raised instead of hidden.

    With ``rag_mode="lexical"`` (keyword search) or ``"semantic"`` (embeddings, which needs an
    ``embedder``) the report also lists resume passages for each requirement. They are added
    after evaluation, so statuses and the score are exactly the same as without them. If
    semantic search fails, keyword search is used and the report says so.

    ``extra_skills`` are additional required skills named by the person running the screen. They
    are assessed for this run only. Lines of the job description that no rule could assess are
    listed in ``report.unassessed_requirements``, and the score is then flagged as incomplete.
    """
    check_rag_options(rag_mode, embedder)
    resume_text = load_resume_text(resume_path, base_dir=base_dir)
    requirements = prepare_requirements(job_description, extra_skills)
    return screen_text(
        resume_text,
        requirements,
        client=client,
        today=today,
        rag_mode=rag_mode,
        embedder=embedder,
    )


def screen_text(
    resume_text: str,
    requirements: JobRequirements,
    *,
    client: LLMClient | None = None,
    today: date | None = None,
    rag_mode: Literal["lexical", "semantic"] | None = None,
    embedder: Embedder | None = None,
) -> ScreeningReport:
    """Screen resume text that is already loaded against requirements that are already parsed.

    This is the one place where a report is built, for a single resume and for a batch, so both
    give the same result for the same resume.
    """
    check_rag_options(rag_mode, embedder)
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
    report = evaluate(profile, requirements, resume_text, extra_limitations=notes, today=today)
    if rag_mode is not None:
        report = _add_retrieval(report, resume_text, rag_mode, embedder)
    return report


def _retrieve(
    report: ScreeningReport, resume_text: str, retriever: Retriever
) -> list[RequirementRetrieval]:
    entries = retrieve_for_assessments(retriever, report.assessments)
    verify_traceable(entries, resume_text)
    return entries


def _semantic_reason(error: Exception) -> str:
    if isinstance(error, EmbeddingError):
        return "the embedding model failed or returned unusable vectors"
    for kind, reason in _FAILURE_REASONS:
        if isinstance(error, kind):
            return reason
    if isinstance(error, LLMError):
        return "the model server returned an error"
    return "an unexpected error"


def _with_retrieval(
    report: ScreeningReport,
    entries: list[RequirementRetrieval],
    retriever: Retriever,
    notes: list[str],
) -> ScreeningReport:
    if retriever.truncated:
        notes = [*notes, RETRIEVAL_TRUNCATED_NOTE]
    limitations = [*report.limitations, *notes]
    return report.model_copy(update={"retrieval": entries, "limitations": limitations})


def _add_retrieval(
    report: ScreeningReport,
    resume_text: str,
    rag_mode: str,
    embedder: Embedder | None,
) -> ScreeningReport:
    """Attach retrieved passages. Only ``retrieval`` and ``limitations`` are changed.

    Retrieval is an optional extra. Semantic search is tried first when requested. If it fails
    for any reason except a configuration error, keyword search is used instead and the report
    says so. If keyword search also fails, screening continues without passages and the report
    says so. Only error class names are logged, never resume text.
    """
    notes: list[str] = []
    if rag_mode == "semantic":
        try:
            semantic = SemanticRetriever(resume_text, embedder)  # type: ignore[arg-type]
            entries = _retrieve(report, resume_text, semantic)
            model = getattr(embedder, "model", None)
            named = f" using the embedding model {model}" if model else ""
            note = SEMANTIC_NOTE.format(model=named)
            return _with_retrieval(report, entries, semantic, [note])
        except LLMConfigError:
            raise
        except Exception as exc:
            logger.warning(
                "Semantic retrieval failed (%s); using keyword search", type(exc).__name__
            )
            notes.append(
                f"Semantic search failed ({_semantic_reason(exc)}), "
                "so keyword search was used instead."
            )
    try:
        retriever = LexicalRetriever(resume_text)
        entries = _retrieve(report, resume_text, retriever)
    except Exception as exc:
        logger.warning(
            "Retrieval failed (%s); the report has no retrieved passages", type(exc).__name__
        )
        limitations = [*report.limitations, *notes, RETRIEVAL_FAILED_NOTE]
        return report.model_copy(update={"limitations": limitations})
    return _with_retrieval(report, entries, retriever, [*notes, RETRIEVAL_NOTE])