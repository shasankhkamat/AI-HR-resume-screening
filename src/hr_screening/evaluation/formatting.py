"""Plain text rendering of a screening report."""

from hr_screening.models.report import RequirementKind, ScreeningReport

_KIND_LABELS = {
    RequirementKind.REQUIRED_SKILL: "required skill",
    RequirementKind.PREFERRED_SKILL: "preferred skill",
    RequirementKind.EXPERIENCE: "experience",
    RequirementKind.EDUCATION: "education",
    RequirementKind.CERTIFICATION: "certification",
}


_RETRIEVAL_INTRO = (
    "Found by keyword search. A passage is not evidence that a requirement is met, and it does "
    "not change any status or the score. Read each one yourself."
)
_SEMANTIC_INTRO = (
    "Found by semantic search: passages are ranked by embedding similarity (higher is closer). "
    "This can match related wording and can also match unrelated text. A passage is not "
    "evidence that a requirement is met, and it does not change any status or the score. "
    "Read each one yourself."
)
_NO_PASSAGE = "No matching passage found. This does not show that the candidate lacks it."


def _retrieval_lines(report: ScreeningReport) -> list[str]:
    methods = {entry.method for entry in report.retrieval}
    lines = ["", "RETRIEVED PASSAGES (for review)"]
    if "lexical" in methods:
        lines.append(_RETRIEVAL_INTRO)
    if "semantic" in methods:
        lines.append(_SEMANTIC_INTRO)
    for entry in report.retrieval:
        score_label = "similarity" if entry.method == "semantic" else "relevance"
        lines.append(f"[{_KIND_LABELS[entry.kind]}] {entry.requirement}")
        if entry.note:
            lines.append(f"    {entry.note}")
        elif not entry.passages:
            lines.append(f"    {_NO_PASSAGE}")
        for number, passage in enumerate(entry.passages, start=1):
            facts = [
                passage.section or "no section",
                f"chars {passage.start}-{passage.end}",
                f"{score_label} {passage.score:.2f}",
            ]
            if passage.mention_kinds:
                kinds = ", ".join(dict.fromkeys(passage.mention_kinds))
                facts.append(f"skill mentioned as: {kinds}")
            lines.append(f"    Passage {number} ({'; '.join(facts)})")
            lines += [f"        | {text}" for text in passage.text.splitlines()]
    return lines


def format_report_text(report: ScreeningReport) -> str:
    lines = [
        "SCREENING REPORT",
        "Decision support only. A human reviewer must make any hiring decision.",
        "",
    ]
    if report.coverage_score is None:
        lines.append("Requirement coverage score: not available")
    else:
        lines.append(f"Requirement coverage score: {report.coverage_score:.1f} / 100")
    lines += [report.score_explanation, "", "REQUIREMENTS"]
    for item in report.assessments:
        line = f"[{item.status.value.upper()}] {item.requirement} ({_KIND_LABELS[item.kind]})"
        lines.append(line)
        lines.append(f"    {item.explanation}")
        lines += [f'    Evidence: "{e.snippet}"' for e in item.evidence]
    lines += ["", "NOT FOUND OR UNVERIFIED (check manually)"]
    lines += [f"- {a.requirement}: {a.status.value}" for a in report.unresolved] or ["- none"]
    if report.retrieval:
        lines += _retrieval_lines(report)
    lines += ["", "LIMITATIONS"]
    lines += [f"- {text}" for text in report.limitations]
    return "\n".join(lines)