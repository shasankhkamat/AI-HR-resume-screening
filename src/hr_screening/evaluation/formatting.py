"""Plain text rendering of a screening report."""

from hr_screening.models.report import RequirementKind, ScreeningReport

_KIND_LABELS = {
    RequirementKind.REQUIRED_SKILL: "required skill",
    RequirementKind.PREFERRED_SKILL: "preferred skill",
    RequirementKind.EXPERIENCE: "experience",
    RequirementKind.EDUCATION: "education",
    RequirementKind.CERTIFICATION: "certification",
}


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
    lines += ["", "LIMITATIONS"]
    lines += [f"- {text}" for text in report.limitations]
    return "\n".join(lines)