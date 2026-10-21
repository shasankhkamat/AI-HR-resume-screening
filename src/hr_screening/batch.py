"""Screen every resume in one folder against one job description.

The single resume pipeline does the screening (``pipeline.screen_text``), so a resume gets the
same report in a batch as on its own. This module adds only what a batch needs: finding the
files, a fingerprint to flag duplicates, an order for human review, and the summary.

Rules that are easy to miss:

* Files are read in file name order, case insensitive, and only from the folder itself. Sub
  folders are not searched.
* The job description is parsed once. A model client or an embedder, when one is given, is
  shared by every resume. This module never builds one.
* One bad file never stops the batch. It is listed as failed and gets no score and no rank.
  A model configuration error is the exception, because every file would fail the same way.
* A duplicate is flagged and still screened and listed. Nothing is skipped or removed.
* The order depends only on the reports, never on the order files were found or on retrieval.
"""

import hashlib
import logging
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import Literal

from hr_screening.errors import (
    BatchInputError,
    LLMConfigError,
    OutputWriteError,
    ScreeningError,
)
from hr_screening.ingestion.loader import load_resume_text
from hr_screening.llm.base import LLMClient
from hr_screening.models.batch import (
    BatchCounts,
    BatchSummary,
    DuplicateGroup,
    FailedResume,
    RequirementCounts,
    ReviewItem,
    ScreenedResume,
)
from hr_screening.models.report import MatchStatus, RequirementKind, ScreeningReport
from hr_screening.pipeline import check_rag_options, prepare_requirements, screen_text
from hr_screening.rag.embeddings import Embedder

logger = logging.getLogger(__name__)

NOTICE = (
    "Decision support only. This list is ordered for human review. It does not accept, reject "
    "or recommend any candidate, and a person must make every hiring decision."
)
ORDERING_RULE = (
    "Ordered for human review by (1) higher coverage score, (2) more supported required items "
    "(required skills, experience, education and certifications; preferred skills are not "
    "counted), (3) file name in case insensitive order, then exact file name. A resume with no "
    "score comes last. The order does not depend on the order files were found."
)
INCOMPLETE_CAVEAT = (
    "Some scores are marked INCOMPLETE. They leave out parts of the job description that the "
    "tool could not assess, so they are not directly comparable with complete scores or with "
    "each other. They are ordered by the same rule and must be reviewed by hand."
)
UNVERIFIED_CAVEAT = (
    "Unverified requirements count as zero in the score, so a score can understate a "
    "candidate. The unverified count is shown beside every score."
)
LIMITATIONS = (
    "The score counts how many listed requirements have supporting text. It is not a measure "
    "of suitability, and the order is not a ranking of people.",
    "Duplicates are found by comparing resume text after whitespace is normalised. A resume "
    "with one changed word or date is not detected as a duplicate.",
    "Only the files directly inside the folder are screened. Sub folders are not searched.",
    "A failed file is not screened. Check each failure, because a resume that could not be "
    "read has not been assessed at all.",
    "Retrieved passages (--rag) never change a status, a score or the order.",
)
_IGNORED_NAMES = frozenset({"thumbs.db", "desktop.ini"})
_MAX_TEXT = 300
_FINGERPRINT_CHARS = 16
_REQUIRED_KINDS = (
    RequirementKind.REQUIRED_SKILL,
    RequirementKind.EXPERIENCE,
    RequirementKind.EDUCATION,
    RequirementKind.CERTIFICATION,
)


def safe_text(text: str, limit: int = _MAX_TEXT) -> str:
    """Text for a summary or a terminal: control characters are replaced and it is cut short."""
    cleaned = "".join("?" if ord(ch) < 32 or ord(ch) == 127 else ch for ch in text)
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def name_key(name: str) -> tuple[str, str]:
    """File name order: case insensitive first, then the exact name so the order is total."""
    return (name.casefold(), name)


def fingerprint(text: str) -> str | None:
    """A stable hash of resume text after conservative normalisation, or None for no text.

    Normalisation: Unicode NFC, then every run of whitespace (spaces, tabs, line breaks,
    non breaking spaces) becomes one space and the ends are trimmed. Letter case, punctuation
    and numbers are kept, so two different resumes are not merged by accident.
    """
    normalised = " ".join(unicodedata.normalize("NFC", text).split())
    if not normalised:
        return None
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:_FINGERPRINT_CHARS]


def discover_resume_files(
    directory: Path, exclude: Iterable[Path] = ()
) -> tuple[list[Path], list[str]]:
    """The files to screen, in file name order, and the names that were left out and why.

    Only the folder itself is read. Sub folders are listed as left out, not entered. Hidden
    files, Word lock files (``~$...``), ``Thumbs.db`` and ``desktop.ini`` are left out, and so
    is any file in ``exclude`` (the job description). Every other file is returned, including
    unsupported types, so that it is reported as a failure instead of vanishing.
    """
    skip = set()
    for path in exclude:
        try:
            skip.add(Path(path).resolve())
        except OSError:
            continue
    files: list[Path] = []
    ignored: list[str] = []
    try:
        entries = sorted(directory.iterdir(), key=lambda p: name_key(p.name))
    except OSError as exc:
        raise BatchInputError("The resume folder could not be read.") from exc
    for entry in entries:
        label = safe_text(entry.name)
        lowered = entry.name.lower()
        try:
            resolved = entry.resolve()
            is_directory = entry.is_dir()
        except OSError:
            ignored.append(f"{label} (not accessible)")
            continue
        if is_directory:
            ignored.append(f"{label}/ (sub folder, not searched)")
        elif resolved in skip:
            ignored.append(f"{label} (the job description file)")
        elif lowered.startswith((".", "~$")) or lowered in _IGNORED_NAMES:
            ignored.append(f"{label} (hidden or temporary file)")
        else:
            files.append(entry)
    return files, ignored


def _counts(report: ScreeningReport) -> RequirementCounts:
    by_status = {status: 0 for status in MatchStatus}
    for item in report.assessments:
        by_status[item.status] += 1
    return RequirementCounts(
        supported=by_status[MatchStatus.SUPPORTED],
        partial=by_status[MatchStatus.PARTIAL],
        not_found=by_status[MatchStatus.NOT_FOUND],
        unverified=by_status[MatchStatus.UNVERIFIED],
        unassessed=len(report.unassessed_requirements),
    )


def _review_items(report: ScreeningReport) -> list[ReviewItem]:
    items = [
        ReviewItem(
            requirement=a.requirement,
            kind=a.kind,
            reason="unverified" if a.status is MatchStatus.UNVERIFIED else "partial",
        )
        for a in report.assessments
        if a.status in (MatchStatus.UNVERIFIED, MatchStatus.PARTIAL)
    ]
    items += [
        ReviewItem(requirement=u.text, kind=None, reason="unassessed")
        for u in report.unassessed_requirements
    ]
    return items


def _order_key(entry: tuple[str, ScreeningReport]) -> tuple[float, int, str, str]:
    name, report = entry
    score = -1.0 if report.coverage_score is None else report.coverage_score
    supported = sum(
        1
        for a in report.assessments
        if a.kind in _REQUIRED_KINDS and a.status is MatchStatus.SUPPORTED
    )
    return (-score, -supported, *name_key(name))


def screen_batch(
    directory: str | Path,
    job_description: str,
    *,
    client: LLMClient | None = None,
    today: date | None = None,
    rag_mode: Literal["lexical", "semantic"] | None = None,
    embedder: Embedder | None = None,
    extra_skills: Sequence[str] = (),
    exclude: Iterable[str | Path] = (),
) -> BatchSummary:
    """Screen every file in ``directory`` and return the summary. Nothing is written to disk.

    Raises ``BatchInputError`` when the folder is missing or has nothing to screen, and the usual
    job description errors for a bad job description. Per file problems are returned in
    ``failures``, never raised.
    """
    root = Path(directory)
    if not root.exists():
        raise BatchInputError("The resume folder was not found.")
    if not root.is_dir():
        raise BatchInputError("The resume path is not a folder.")
    files, ignored = discover_resume_files(root, [Path(p) for p in exclude])
    if not files:
        raise BatchInputError("The resume folder has no files to screen.")
    check_rag_options(rag_mode, embedder)
    requirements = prepare_requirements(job_description, extra_skills)  # once for the batch

    screened: list[tuple[str, ScreeningReport, str | None]] = []
    failures: list[FailedResume] = []
    for path in files:  # file name order
        name = safe_text(path.name)
        try:
            text = load_resume_text(path, base_dir=root)
            report = screen_text(
                text,
                requirements,
                client=client,
                today=today,
                rag_mode=rag_mode,
                embedder=embedder,
            )
        except LLMConfigError:
            raise
        except ScreeningError as exc:
            failures.append(
                FailedResume(
                    source_file=name, error_type=type(exc).__name__, error=safe_text(str(exc))
                )
            )
        except Exception as exc:  # one file must not stop the batch; only the class is logged
            logger.warning("Screening a file failed unexpectedly (%s)", type(exc).__name__)
            failures.append(
                FailedResume(
                    source_file=name,
                    error_type="UnexpectedError",
                    error="An unexpected error stopped this file from being screened.",
                )
            )
        else:
            screened.append((name, report, fingerprint(text)))

    results = _ordered_results(screened)
    groups = _duplicate_groups(results, {name: fp for name, _, fp in screened})
    totals = _totals(results)
    counts = BatchCounts(
        files_found=len(files),
        screened=len(results),
        failed=len(failures),
        duplicate_files=sum(len(g.duplicate_files) for g in groups),
        duplicate_groups=len(groups),
        complete_scores=sum(1 for r in results if r.score_complete),
        incomplete_scores=sum(1 for r in results if not r.score_complete),
        needs_review=sum(1 for r in results if r.needs_review),
        requirements=totals,
    )
    caveats = [UNVERIFIED_CAVEAT]
    if counts.incomplete_scores:
        caveats.insert(0, INCOMPLETE_CAVEAT)
    logger.info(
        "Batch finished: %d files, %d screened, %d failed",
        counts.files_found,
        counts.screened,
        counts.failed,
    )
    return BatchSummary(
        notice=NOTICE,
        ordering_rule=ORDERING_RULE,
        ordering_caveats=caveats,
        extra_skills=[safe_text(s) for s in extra_skills],
        job_notes=list(requirements.notes),
        counts=counts,
        results=results,
        failures=failures,
        duplicate_groups=groups,
        ignored_files=ignored,
        limitations=list(LIMITATIONS),
    )


def _ordered_results(
    screened: list[tuple[str, ScreeningReport, str | None]],
) -> list[ScreenedResume]:
    ordered = sorted(((n, r) for n, r, _ in screened), key=_order_key)
    results = []
    for rank, (name, report) in enumerate(ordered, start=1):
        required = [a for a in report.assessments if a.kind in _REQUIRED_KINDS]
        results.append(
            ScreenedResume(
                source_file=name,
                rank=rank,
                coverage_score=report.coverage_score,
                score_complete=report.score_complete,
                required_supported=sum(1 for a in required if a.status is MatchStatus.SUPPORTED),
                required_total=len(required),
                counts=_counts(report),
                needs_review=_review_items(report),
                report=report,
            )
        )
    return results


def _duplicate_groups(
    results: list[ScreenedResume], prints: dict[str, str | None]
) -> list[DuplicateGroup]:
    """Flag resumes with the same fingerprint. The reference is the first by file name."""
    members: dict[str, list[str]] = defaultdict(list)
    for name in sorted(prints, key=name_key):
        if prints[name] is not None:
            members[str(prints[name])].append(name)
    repeated = sorted(
        (names for names in members.values() if len(names) > 1), key=lambda n: name_key(n[0])
    )
    by_name = {r.source_file: r for r in results}
    groups = []
    for number, names in enumerate(repeated, start=1):
        reference, *others = names
        groups.append(
            DuplicateGroup(
                group=number,
                fingerprint=prints[reference] or "",
                reference_file=reference,
                duplicate_files=others,
            )
        )
        by_name[reference].duplicate_group = number
        for other in others:
            by_name[other].duplicate_group = number
            by_name[other].duplicate_of = reference
    return groups


def _totals(results: list[ScreenedResume]) -> RequirementCounts:
    total = RequirementCounts()
    for result in results:
        for field in RequirementCounts.model_fields:
            setattr(total, field, getattr(total, field) + getattr(result.counts, field))
    return total


def write_batch_outputs(output_dir: str | Path, rendered: dict[str, str]) -> list[Path]:
    """Write text files into ``output_dir`` and return their paths.

    ``rendered`` maps a file name to its text, and is built by the caller, so this function does
    not choose formats. Existing files of the same name from an earlier run are replaced.
    Names always end in .json or .txt, so a resume (.pdf or .docx) cannot be overwritten.
    """
    folder = Path(output_dir)
    written: list[Path] = []
    try:
        folder.mkdir(parents=True, exist_ok=True)
        for name, content in rendered.items():
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.suffix.lower() not in (".json", ".txt"):
                raise OutputWriteError("An output file name was not allowed.")
            target.write_text(content, encoding="utf-8", newline="\n")
            written.append(target)
    except OSError as exc:
        raise OutputWriteError("The output folder or a file in it could not be written.") from exc
    return written

