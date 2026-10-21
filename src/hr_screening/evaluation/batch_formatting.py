"""Text and JSON rendering of a batch summary, and of the per resume files."""

import json
from typing import Any

from hr_screening.evaluation.formatting import format_report_text
from hr_screening.models.batch import BatchSummary, ScreenedResume

_KIND_LABELS = {
    "required_skill": "required skill",
    "preferred_skill": "preferred skill",
    "experience": "experience",
    "education": "education",
    "certification": "certification",
}


def _trim(data: dict[str, Any]) -> dict[str, Any]:
    """Drop an empty ``retrieval`` list from a report, as the single resume JSON does."""
    report = data.get("report")
    if isinstance(report, dict) and not report.get("retrieval"):
        report.pop("retrieval", None)
    return data


def resume_json(result: ScreenedResume) -> str:
    """One resume: the source metadata and, under ``report``, the usual report."""
    return json.dumps(_trim(result.model_dump(mode="json")), indent=2, ensure_ascii=False)


def batch_json(summary: BatchSummary) -> str:
    """The whole summary. Keys are fixed and the order follows the models."""
    data = summary.model_dump(mode="json")
    data["results"] = [_trim(item) for item in data["results"]]
    return json.dumps(data, indent=2, ensure_ascii=False)


def resume_text(result: ScreenedResume) -> str:
    """One resume as text: a source header, then the usual report."""
    header = [
        f"SOURCE FILE: {result.source_file}",
        f"Position in the review order: {result.rank} (an order for human review, not a decision)",
    ]
    if result.duplicate_of:
        header.append(f"DUPLICATE of {result.duplicate_of} (group {result.duplicate_group})")
    return "\n".join([*header, "", format_report_text(result.report)])


def _score(result: ScreenedResume) -> str:
    return "no score" if result.coverage_score is None else f"{result.coverage_score:.1f}"


def _bullets(items: list[str], empty: str = "none") -> list[str]:
    return [f"- {item}" for item in items] or [f"- {empty}"]


def batch_text(summary: BatchSummary) -> str:
    c = summary.counts
    r = c.requirements
    lines = ["BATCH SCREENING SUMMARY", summary.notice, "", "FILES"]
    lines += [
        f"Found: {c.files_found}   Screened: {c.screened}   Failed: {c.failed}   "
        f"Left out: {len(summary.ignored_files)}",
        f"Complete scores: {c.complete_scores}   INCOMPLETE scores: {c.incomplete_scores}",
        f"Duplicate files: {c.duplicate_files} in {c.duplicate_groups} group(s)",
        f"Resumes with something to review: {c.needs_review}",
        "Requirements over all screened resumes: "
        f"supported {r.supported}, partial {r.partial}, not found {r.not_found}, "
        f"unverified {r.unverified}, not assessed {r.unassessed}",
        "",
        "ORDERED FOR HUMAN REVIEW",
        summary.ordering_rule,
    ]
    lines += [f"NOTE: {text}" for text in summary.ordering_caveats]
    lines.append("")
    if not summary.results:
        lines.append("No resume was screened.")
    for result in summary.results:
        counts = result.counts
        flag = "complete" if result.score_complete else "INCOMPLETE SCORE"
        lines.append(f"{result.rank}. {result.source_file}")
        lines.append(
            f"    score {_score(result)} ({flag}); required items supported "
            f"{result.required_supported} of {result.required_total}"
        )
        lines.append(
            f"    supported {counts.supported}, partial {counts.partial}, "
            f"not found {counts.not_found}, UNVERIFIED {counts.unverified}, "
            f"NOT ASSESSED {counts.unassessed}"
        )
        if result.duplicate_of:
            lines.append(f"    DUPLICATE of {result.duplicate_of} (group {result.duplicate_group})")
        elif result.duplicate_group:
            lines.append(f"    reference copy for duplicate group {result.duplicate_group}")
        if result.needs_review:
            lines.append(f"    needs review: {len(result.needs_review)} item(s), listed below")
    lines += ["", "NEEDS HUMAN REVIEW (whatever the score is)"]
    reviewed = [res for res in summary.results if res.needs_review]
    for result in reviewed:
        lines.append(f"{result.source_file}")
        for item in result.needs_review:
            label = _KIND_LABELS.get(item.kind.value, "") if item.kind else "job text"
            shown = "not assessed" if item.reason == "unassessed" else item.reason
            lines.append(f"    [{shown}] {item.requirement} ({label})")
    if not reviewed:
        lines.append("- none. This does not remove the need to read each report.")
    lines += ["", "DUPLICATES (flagged, never removed)"]
    lines += _bullets(
        [
            f"group {g.group}: reference {g.reference_file}; same text: "
            + ", ".join(g.duplicate_files)
            for g in summary.duplicate_groups
        ]
    )
    lines += ["", "FAILED FILES (not screened, no score)"]
    lines += _bullets([f"{f.source_file}: {f.error}" for f in summary.failures])
    if summary.ignored_files:
        lines += ["", "LEFT OUT"]
        lines += _bullets(summary.ignored_files)
    if summary.extra_skills:
        lines += ["", "EXTRA SKILLS ADDED FOR THIS RUN"]
        lines += _bullets(summary.extra_skills)
    if summary.job_notes:
        lines += ["", "JOB DESCRIPTION NOTES"]
        lines += _bullets(summary.job_notes)
    lines += ["", "LIMITATIONS"]
    lines += _bullets(summary.limitations)
    return "\n".join(lines)
