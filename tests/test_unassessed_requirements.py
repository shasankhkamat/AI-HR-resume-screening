"""Job lines the tool cannot assess, and skills the person running the screen adds."""

import json
import re
from pathlib import Path

import pytest

from hr_screening import main as main_module
from hr_screening.errors import InvalidExtraSkillError
from hr_screening.evaluation.formatting import format_report_text
from hr_screening.extraction.job_parser import parse_job_description
from hr_screening.pipeline import MAX_EXTRA_SKILLS, screen_resume

ROOT = Path(__file__).resolve().parents[1]
JOB = "Requirements\n- Python\n- Experience with Terraform and Rust\n- Bachelor's degree\n"
RESUME = [
    "EDUCATION",
    "BSc Computer Science",
    "SKILLS",
    "Python, Terraform",
    "No experience with Rust.",
]


@pytest.fixture
def screen(tmp_path, make_docx, today):
    def _run(job=JOB, lines=RESUME, **kwargs):
        return screen_resume(make_docx(tmp_path / "cv.docx", lines), job, today=today, **kwargs)

    return _run


@pytest.fixture
def cli(monkeypatch, capsys):
    monkeypatch.setattr(main_module, "configure_logging", lambda: None)

    def _run(*argv):
        code = main_module.main([str(a) for a in argv])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return _run


def item(report, name):
    return next(a for a in report.assessments if a.requirement == name)


# --- the parser -------------------------------------------------------------------------


def test_a_required_line_with_nothing_recognised_is_listed_as_unassessed():
    found = parse_job_description(JOB)
    assert found.required_skills == ["python"]
    assert found.unassessed_lines == ["Experience with Terraform and Rust"]


def test_lines_with_a_skill_years_degree_or_certification_are_not_unassessed():
    job = (
        "Requirements\n- Python\n- 3+ years of experience\n- Master's degree\n"
        "- Data Engineering certification\n"
    )
    assert parse_job_description(job).unassessed_lines == []


def test_preferred_and_responsibility_lines_are_not_listed():
    job = "Requirements\n- Python\nPreferred\n- Terraform\nResponsibilities\n- Build things\n"
    assert parse_job_description(job).unassessed_lines == []


def test_the_parser_does_not_treat_arbitrary_words_as_skills():
    """The vocabulary is closed. Words such as Terraform, Excel, R, Rust and Go are not skills."""
    found = parse_job_description("Requirements\n- Terraform, Excel, R, Rust and Go\n")
    assert found.required_skills == []
    assert found.unassessed_lines == ["Terraform, Excel, R, Rust and Go"]


@pytest.mark.parametrize(
    ("line", "listed"),
    [
        ("Python and Terraform", ["Terraform"]),
        ("Rust and Go, Bachelor's degree", ["Rust", "Go"]),
        ("Hands-on experience with Python, R, or Go", ["R", "Go"]),
        ("3+ years of experience, Rust", ["Rust"]),
        ("Python, SQL and CI", ["CI"]),
        ("Python and Helm3", ["Helm3"]),
    ],
)
def test_an_unknown_skill_on_a_recognised_line_is_listed_not_hidden(line, listed):
    """'Python and Terraform' must not read as fully assessed because Python was recognised."""
    found = parse_job_description(f"Requirements\n- {line}\n")
    assert found.unassessed_lines == listed


@pytest.mark.parametrize(
    "line",
    [
        "Strong Python and SQL skills",
        "3+ years of experience in Data Engineering",
        "Bachelor's degree in Computer Science",
        "Master's degree in Data Science or related field",
        "Experience with Tableau",
        "Data Engineering certification",
        "Python, SQL, Tableau and Kubernetes",
    ],
)
def test_fully_recognised_lines_are_not_listed(line):
    assert parse_job_description(f"Requirements\n- {line}\n").unassessed_lines == []


def test_known_limitation_an_unknown_skill_in_plain_lower_case_is_not_detected():
    """A name in lower case cannot be told from an ordinary word. This pins the limit."""
    found = parse_job_description("Requirements\n- Python and terraform\n")
    assert found.unassessed_lines == []


def test_a_listed_piece_is_covered_by_an_extra_skill(tmp_path, make_docx, today):
    lines = ["EDUCATION", "BSc CS", "SKILLS", "Python, Terraform"]
    path = make_docx(tmp_path / "cv.docx", lines)
    job = "Requirements\n- Python and Terraform\n- Bachelor's degree\n"
    before = screen_resume(path, job, today=today)
    assert [u.text for u in before.unassessed_requirements] == ["Terraform"]
    assert before.score_complete is False
    after = screen_resume(path, job, today=today, extra_skills=["Terraform"])
    assert after.unassessed_requirements == []
    assert after.score_complete is True


# --- the report -------------------------------------------------------------------------


def test_an_unassessed_line_is_not_an_assessment_and_the_score_ignores_it(screen):
    report = screen()
    kinds = [a.kind.value for a in report.assessments]
    assert kinds == ["required_skill", "education"]
    assert [u.text for u in report.unassessed_requirements] == [
        "Experience with Terraform and Rust"
    ]
    assert report.coverage_score == 100.0  # python and the degree. Nothing else was assessed.
    assert report.score_complete is False


def test_the_score_explanation_says_it_is_incomplete(screen):
    report = screen()
    assert "INCOMPLETE" in report.score_explanation
    assert "1 line(s)" in report.score_explanation


def test_unassessed_is_different_from_assessed_and_not_found(screen):
    """Assessed and not found has a status and costs score. Unassessed has neither."""
    report = screen(extra_skills=["Rust"], lines=["EDUCATION", "BSc CS", "SKILLS", "Python"])
    rust = item(report, "Rust")
    assert rust.status.value == "not_found"
    assert report.coverage_score < 100.0
    assert report.unassessed_requirements == []
    assert report.score_complete is True


def test_a_complete_job_is_reported_as_complete(screen):
    report = screen(job="Requirements\n- Python\n- Bachelor's degree\n")
    assert report.score_complete is True
    assert report.unassessed_requirements == []
    assert "INCOMPLETE" not in report.score_explanation


def test_the_json_report_carries_the_flag_and_the_lines(cli, tmp_path, make_docx):
    resume = make_docx(tmp_path / "cv.docx", RESUME)
    code, out, _ = cli("--resume", resume, "--job-text", JOB, "--json")
    data = json.loads(out)
    assert code == 0
    assert data["score_complete"] is False
    assert [u["text"] for u in data["unassessed_requirements"]] == [
        "Experience with Terraform and Rust"
    ]
    assert "not assessed" in data["unassessed_requirements"][0]["reason"]
    assert "INCOMPLETE" in data["score_explanation"]


def test_the_text_report_flags_the_score_and_has_its_own_section(screen):
    text = format_report_text(screen())
    lines = text.splitlines()
    score = lines.index(next(ln for ln in lines if ln.startswith("Requirement coverage score")))
    assert lines[score + 1].startswith("SCORE INCOMPLETE")
    assert "NOT ASSESSED (not in the score; this is not the same as not found)" in text
    assert "- Experience with Terraform and Rust" in text
    section = text.split("\nNOT ASSESSED (")[1]
    assert "Experience with Terraform and Rust" in section.split("LIMITATIONS")[0]
    # The "not found" list stays about assessed requirements.
    not_found = text.split("NOT FOUND OR UNVERIFIED")[1].split("\nNOT ASSESSED (")[0]
    assert "Terraform" not in not_found


def test_a_complete_text_report_has_no_incomplete_wording(screen):
    text = format_report_text(screen(job="Requirements\n- Python\n"))
    assert "INCOMPLETE" not in text
    assert "NOT ASSESSED" not in text


# --- explicit extra skills --------------------------------------------------------------


def test_extra_skills_are_assessed_as_required_skills(screen):
    report = screen(extra_skills=["Terraform", "Rust"])
    terraform = item(report, "Terraform")
    assert terraform.kind.value == "required_skill"
    assert terraform.status.value == "supported"
    assert [e.snippet for e in terraform.evidence] == ["Python, Terraform"]
    rust = item(report, "Rust")
    assert rust.status.value == "not_found"  # "No experience with Rust." is a denial
    assert [e.snippet for e in rust.evidence] == ["No experience with Rust."]


def test_a_line_that_names_an_extra_skill_is_no_longer_unassessed(screen):
    report = screen(extra_skills=["Terraform"])
    assert report.unassessed_requirements == []
    assert report.score_complete is True


def test_a_line_naming_a_different_skill_stays_unassessed(screen):
    report = screen(extra_skills=["Kubernetes"])
    assert [u.text for u in report.unassessed_requirements] == [
        "Experience with Terraform and Rust"
    ]
    assert report.score_complete is False


def test_the_score_formula_includes_extra_skills_with_the_required_weight(screen):
    report = screen(extra_skills=["Terraform", "Rust"])
    # python 3 + terraform 3 + degree 2 earned, rust 0, of 3 + 3 + 3 + 2 = 11
    assert report.coverage_score == round(100 * 8 / 11, 1)


def test_extra_skills_are_noted_in_the_limitations_for_the_audit_trail(screen):
    report = screen(extra_skills=["Terraform"])
    note = next(ln for ln in report.limitations if "--extra-skill" in ln)
    assert "Terraform" in note


def test_a_very_short_extra_skill_gets_a_warning(screen):
    report = screen(extra_skills=["Go"])
    assert any("very short" in ln and "Go" in ln for ln in report.limitations)


def test_extra_skills_do_not_change_the_vocabulary(screen):
    screen(extra_skills=["Terraform"])
    assert parse_job_description("Requirements\n- Terraform\n").required_skills == []


def test_an_extra_skill_already_in_the_job_is_counted_once(screen):
    report = screen(job="Requirements\n- Python\n- Bachelor's degree\n", extra_skills=["python"])
    assert [a.requirement for a in report.assessments].count("python") == 1


def test_a_preferred_skill_named_as_extra_becomes_required(screen):
    job = "Requirements\n- Python\nPreferred\n- Terraform\n"
    report = screen(job=job, extra_skills=["Terraform"])
    assert item(report, "Terraform").kind.value == "required_skill"
    assert [a.requirement for a in report.assessments].count("Terraform") == 1
    assert not any(a.requirement == "terraform" for a in report.assessments)


def test_the_same_extra_skill_twice_is_counted_once(screen):
    report = screen(extra_skills=["Terraform", "terraform", " TERRAFORM "])
    names = [a.requirement.lower() for a in report.assessments]
    assert names.count("terraform") == 1


@pytest.mark.parametrize(
    "bad",
    [[""], ["   "], ["x" * 61], [f"skill{n}" for n in range(MAX_EXTRA_SKILLS + 1)], "Terraform"],
    ids=["empty", "blank", "too long", "too many", "bare string"],
)
def test_invalid_extra_skills_are_refused(screen, bad):
    with pytest.raises(InvalidExtraSkillError):
        screen(extra_skills=bad)


def test_extra_skills_are_matched_as_whole_words(screen):
    lines = ["EDUCATION", "BSc CS", "SKILLS", "Python, Terraforming"]
    report = screen(extra_skills=["Terraform"], lines=lines)
    assert item(report, "Terraform").status.value == "not_found"


def test_without_extra_skills_the_report_is_the_same_as_before(screen):
    assert screen().model_dump() == screen(extra_skills=()).model_dump()


def test_extra_skills_never_change_with_retrieval(screen):
    plain = screen(extra_skills=["Terraform"])
    with_rag = screen(extra_skills=["Terraform"], rag_mode="lexical")
    assert with_rag.coverage_score == plain.coverage_score
    assert [a.model_dump() for a in with_rag.assessments] == [
        a.model_dump() for a in plain.assessments
    ]
    assert with_rag.unassessed_requirements == plain.unassessed_requirements


# --- the command line -------------------------------------------------------------------


def test_the_cli_accepts_extra_skill_more_than_once(cli, tmp_path, make_docx):
    resume = make_docx(tmp_path / "cv.docx", RESUME)
    argv = ["--resume", resume, "--job-text", JOB, "--json"]
    code, out, _ = cli(*argv, "--extra-skill", "Terraform", "--extra-skill", "Rust")
    data = json.loads(out)
    assert code == 0
    assert data["score_complete"] is True
    names = [a["requirement"] for a in data["assessments"]]
    assert "Terraform" in names
    assert "Rust" in names


def test_the_cli_refuses_an_invalid_extra_skill_without_a_traceback(cli, tmp_path, make_docx):
    resume = make_docx(tmp_path / "cv.docx", RESUME)
    code, out, err = cli("--resume", resume, "--job-text", JOB, "--extra-skill", "  ")
    assert code == 2
    assert out == ""
    assert "extra skill is empty" in err
    assert "Traceback" not in err


def test_the_cli_without_the_flag_still_works_as_before(cli, tmp_path, make_docx):
    resume = make_docx(tmp_path / "cv.docx", RESUME)
    code, out, _ = cli("--resume", resume, "--job-text", JOB)
    assert code == 0
    assert out.startswith("SCREENING REPORT")


# --- the README -------------------------------------------------------------------------


def test_the_readme_no_longer_claims_unknown_skills_are_matched_automatically():
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    assert not re.search(r"outside the built in list are matched only when", readme)
    assert "--extra-skill" in readme
    assert "NOT ASSESSED" in readme
