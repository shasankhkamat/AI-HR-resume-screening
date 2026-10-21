"""Batch screening: one job description, a folder of resumes, a summary for human review.

Everything runs on synthetic documents and fakes. No model server and no network are used.
"""

import hashlib
import json
import logging
import re
from pathlib import Path

import pytest

from hr_screening import batch as batch_module
from hr_screening import main as main_module
from hr_screening import pipeline as pipeline_module
from hr_screening.batch import fingerprint, name_key, safe_text, screen_batch
from hr_screening.config import Settings
from hr_screening.errors import BatchInputError, LLMConfigError
from hr_screening.evaluation.batch_formatting import batch_json, batch_text
from hr_screening.llm.fake import FakeLLMClient
from hr_screening.pipeline import screen_resume
from hr_screening.rag.embeddings import FakeEmbedder

JOB = "Requirements\n- Python\n- SQL\n- Kubernetes\n- Bachelor's degree\n"
GOOD = ["EDUCATION", "BSc Computer Science", "SKILLS", "Python, SQL, Kubernetes"]
PARTIAL = ["EDUCATION", "BSc Physics", "SKILLS", "Python, SQL"]
WEAK = ["EDUCATION", "BSc Physics", "SKILLS", "Python"]
PAYLOAD = {
    "skills": ["Python", "SQL"],
    "education": ["BSc Computer Science"],
    "certifications": [],
    "projects": [],
    "experience": [],
}
_ENV = (
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "LLM_MODEL",
    "OLLAMA_BASE_URL",
    "EMBEDDING_MODEL",
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(main_module, "configure_logging", lambda: None)


@pytest.fixture
def folder(tmp_path, make_docx):
    """A resume folder with a space in its name. ``put`` adds one synthetic DOCX."""
    root = tmp_path / "my resumes"
    root.mkdir()

    def put(name, lines):
        return make_docx(root / name, lines)

    put.root = root
    return put


@pytest.fixture
def job_file(tmp_path):
    path = tmp_path / "job files" / "job desc.txt"
    path.parent.mkdir()
    path.write_text(JOB, encoding="utf-8")
    return path


@pytest.fixture
def cli(capsys):
    def _run(*argv):
        try:
            code = main_module.main([str(a) for a in argv])
        except SystemExit as stop:
            code = stop.code
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return _run


def names(summary):
    return [r.source_file for r in summary.results]


# --- discovery and failures -----------------------------------------------------------------


def test_every_supported_valid_file_is_screened(folder, make_pdf, today):
    folder("a.docx", GOOD)
    make_pdf(folder.root / "b.pdf", GOOD)
    summary = screen_batch(folder.root, JOB, today=today)
    assert sorted(names(summary)) == ["a.docx", "b.pdf"]
    assert summary.counts.files_found == 2
    assert summary.counts.screened == 2
    assert summary.counts.failed == 0
    assert all(r.report.human_review_required for r in summary.results)


def test_a_corrupt_file_does_not_stop_the_valid_files(folder, today):
    folder("a.docx", GOOD)
    (folder.root / "broken.docx").write_bytes(b"this is not a zip file")
    folder("c.docx", PARTIAL)
    summary = screen_batch(folder.root, JOB, today=today)
    assert sorted(names(summary)) == ["a.docx", "c.docx"]
    assert [f.source_file for f in summary.failures] == ["broken.docx"]
    assert "valid .docx" in summary.failures[0].error


def test_unsupported_files_are_failures_with_a_clear_reason(folder, today):
    folder("a.docx", GOOD)
    (folder.root / "notes.txt").write_text("hello", encoding="utf-8")
    (folder.root / "old.doc").write_bytes(b"x")
    summary = screen_batch(folder.root, JOB, today=today)
    reasons = {f.source_file: f.error for f in summary.failures}
    assert "Unsupported file type '.txt'" in reasons["notes.txt"]
    assert "Legacy .doc" in reasons["old.doc"]
    assert names(summary) == ["a.docx"]


def test_an_empty_or_unreadable_pdf_and_an_empty_file_are_failures(folder, make_blank_pdf, today):
    folder("a.docx", GOOD)
    make_blank_pdf(folder.root / "scan.pdf")
    (folder.root / "zero.docx").write_bytes(b"")
    summary = screen_batch(folder.root, JOB, today=today)
    reasons = {f.source_file: f.error for f in summary.failures}
    assert "no extractable text" in reasons["scan.pdf"]
    assert "empty" in reasons["zero.docx"]


def test_a_missing_folder_and_a_file_instead_of_a_folder_are_clear_errors(tmp_path, folder):
    with pytest.raises(BatchInputError, match="folder was not found"):
        screen_batch(tmp_path / "nope", JOB)
    one = folder("a.docx", GOOD)
    with pytest.raises(BatchInputError, match="not a folder"):
        screen_batch(one, JOB)


def test_an_empty_folder_is_a_clear_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(BatchInputError, match="no files to screen"):
        screen_batch(empty, JOB)


def test_a_folder_with_only_temporary_files_has_nothing_to_screen(tmp_path):
    root = tmp_path / "junk"
    root.mkdir()
    for name in ("Thumbs.db", "desktop.ini", ".DS_Store", "~$cv.docx"):
        (root / name).write_bytes(b"x")
    with pytest.raises(BatchInputError, match="no files to screen"):
        screen_batch(root, JOB)


def test_temporary_files_and_sub_folders_are_listed_as_left_out_not_failed(folder, today):
    folder("a.docx", GOOD)
    (folder.root / "Thumbs.db").write_bytes(b"x")
    (folder.root / "~$a.docx").write_bytes(b"x")
    nested = folder.root / "archive"
    nested.mkdir()
    folder.root.joinpath("archive", "inner.docx").write_bytes(b"x")
    summary = screen_batch(folder.root, JOB, today=today)
    assert names(summary) == ["a.docx"]
    assert summary.counts.failed == 0
    left_out = " | ".join(summary.ignored_files)
    assert "Thumbs.db" in left_out
    assert "~$a.docx" in left_out
    assert "archive/ (sub folder, not searched)" in left_out
    assert "inner.docx" not in left_out


def test_the_job_description_inside_the_folder_is_not_screened(folder, today):
    folder("a.docx", GOOD)
    job = folder.root / "job.txt"
    job.write_text(JOB, encoding="utf-8")
    summary = screen_batch(folder.root, JOB, today=today, exclude=[job])
    assert summary.counts.failed == 0
    assert "job.txt (the job description file)" in summary.ignored_files


def test_a_surprise_error_on_one_file_is_contained_and_never_leaks_its_message(
    folder, monkeypatch, today
):
    folder("a.docx", GOOD)
    folder("b.docx", PARTIAL)
    real = batch_module.screen_text

    def explode_on_a_resume_with_python_and_sql_only(text, *args, **kwargs):
        if "Physics" in text:
            raise RuntimeError("secret resume text SHOULD-NOT-APPEAR")
        return real(text, *args, **kwargs)

    monkeypatch.setattr(batch_module, "screen_text", explode_on_a_resume_with_python_and_sql_only)
    summary = screen_batch(folder.root, JOB, today=today)
    assert names(summary) == ["a.docx"]
    assert summary.failures[0].source_file == "b.docx"
    assert summary.failures[0].error_type == "UnexpectedError"
    assert "SHOULD-NOT-APPEAR" not in batch_json(summary)


def test_a_model_configuration_error_stops_the_batch(folder, monkeypatch, today):
    folder("a.docx", GOOD)

    def refuse(*args, **kwargs):
        raise LLMConfigError("bad model setting")

    monkeypatch.setattr(batch_module, "screen_text", refuse)
    with pytest.raises(LLMConfigError):
        screen_batch(folder.root, JOB, today=today)


# --- equivalence, parsing once, shared components -------------------------------------------


def test_a_batch_result_equals_the_single_resume_result(folder, today):
    for name, lines in (("a.docx", GOOD), ("b.docx", PARTIAL), ("c.docx", WEAK)):
        folder(name, lines)
    summary = screen_batch(folder.root, JOB, today=today, extra_skills=["Terraform"])
    assert len(summary.results) == 3
    for result in summary.results:
        single = screen_resume(
            folder.root / result.source_file, JOB, today=today, extra_skills=["Terraform"]
        )
        assert result.report.model_dump() == single.model_dump()


def test_the_job_description_is_parsed_once_per_batch(folder, monkeypatch, today):
    for name in ("a.docx", "b.docx", "c.docx"):
        folder(name, GOOD + [name])
    calls = []
    real = pipeline_module.parse_job_description
    monkeypatch.setattr(
        pipeline_module, "parse_job_description", lambda text: calls.append(1) or real(text)
    )
    screen_batch(folder.root, JOB, today=today)
    assert len(calls) == 1


def test_extra_skills_apply_to_every_resume_and_not_to_the_vocabulary(folder, today):
    folder("a.docx", [*GOOD, "Terraform"])
    folder("b.docx", GOOD)
    summary = screen_batch(folder.root, JOB, today=today, extra_skills=["Terraform"])
    by_name = {r.source_file: r.report for r in summary.results}
    for report in by_name.values():
        assert "Terraform" in [a.requirement for a in report.assessments]
    status = {n: next(a for a in r.assessments if a.requirement == "Terraform").status.value
              for n, r in by_name.items()}
    assert status == {"a.docx": "supported", "b.docx": "not_found"}
    assert summary.extra_skills == ["Terraform"]
    from hr_screening.extraction.job_parser import parse_job_description

    assert parse_job_description("Requirements\n- Terraform\n").required_skills == []


def test_a_model_client_and_embedder_are_built_once_for_the_whole_batch(
    folder, job_file, monkeypatch, cli
):
    for name in ("a.docx", "b.docx", "c.docx"):
        folder(name, GOOD + [name])
    built = {"llm": 0, "embedder": 0}
    client = FakeLLMClient(json.dumps(PAYLOAD))
    embedder = FakeEmbedder()

    def make_client(settings):
        built["llm"] += 1
        return client

    def make_embedder(settings):
        built["embedder"] += 1
        return embedder

    config = Settings(_env_file=None, llm_provider="ollama", llm_model="m", embedding_model="e")
    monkeypatch.setattr(main_module, "get_settings", lambda: config)
    monkeypatch.setattr(main_module, "build_llm_client", make_client)
    monkeypatch.setattr(main_module, "build_embedder", make_embedder)
    code, out, _ = cli(
        "--batch-dir", folder.root, "--job-file", job_file, "--llm", "--rag", "semantic", "--json"
    )
    assert code == 0
    assert built == {"llm": 1, "embedder": 1}
    assert len(client.calls) == 3  # one model call per resume, one shared client
    assert len(json.loads(out)["results"]) == 3


def test_nothing_is_initialised_when_models_are_not_requested(folder, job_file, monkeypatch, cli):
    folder("a.docx", GOOD)

    def forbid(name):
        def fail(*args, **kwargs):
            raise AssertionError(f"{name} must not be called")

        return fail

    monkeypatch.setattr(main_module, "build_llm_client", forbid("build_llm_client"))
    monkeypatch.setattr(main_module, "build_embedder", forbid("build_embedder"))
    monkeypatch.setattr(main_module, "get_settings", forbid("get_settings"))
    code, out, err = cli("--batch-dir", folder.root, "--job-file", job_file, "--json")
    assert code == 0
    assert err == ""
    code, _, _ = cli("--batch-dir", folder.root, "--job-file", job_file, "--rag", "lexical")
    assert code == 0  # keyword search builds no embedder either


def test_a_model_failure_falls_back_for_every_resume_and_the_batch_continues(
    folder, job_file, monkeypatch, cli
):
    from hr_screening.errors import LLMConnectionError

    folder("a.docx", GOOD)
    folder("b.docx", PARTIAL)
    client = FakeLLMClient(error=LLMConnectionError("down"))
    config = Settings(_env_file=None, llm_provider="ollama", llm_model="m")
    monkeypatch.setattr(main_module, "get_settings", lambda: config)
    monkeypatch.setattr(main_module, "build_llm_client", lambda settings: client)
    code, out, _ = cli("--batch-dir", folder.root, "--job-file", job_file, "--llm", "--json")
    data = json.loads(out)
    assert code == 0
    assert len(data["results"]) == 2
    for result in data["results"]:
        assert "the model server could not be reached" in " ".join(result["report"]["limitations"])


# --- order ----------------------------------------------------------------------------------


def test_files_are_processed_in_case_insensitive_file_name_order(folder, monkeypatch, today):
    for name in ("Zed.docx", "alex.docx", "Bob.docx", "carl.docx"):
        folder(name, GOOD + [name])
    seen = []
    real = batch_module.load_resume_text
    monkeypatch.setattr(
        batch_module, "load_resume_text", lambda p, **kw: seen.append(Path(p).name) or real(p, **kw)
    )
    monkeypatch.setattr(Path, "iterdir", _scrambled(Path.iterdir))
    screen_batch(folder.root, JOB, today=today)
    assert seen == ["alex.docx", "Bob.docx", "carl.docx", "Zed.docx"]


def _scrambled(real_iterdir):
    """An iterdir that returns entries in reverse, to show that the result does not depend on it."""

    def iterdir(self):
        return iter(sorted(real_iterdir(self), key=lambda p: p.name, reverse=True))

    return iterdir


def test_the_review_order_does_not_depend_on_how_files_are_enumerated(folder, monkeypatch, today):
    for name, lines in (
        ("d.docx", PARTIAL), ("a.docx", GOOD), ("c.docx", WEAK), ("b.docx", PARTIAL)
    ):
        folder(name, lines + [name])
    normal = batch_json(screen_batch(folder.root, JOB, today=today))
    monkeypatch.setattr(Path, "iterdir", _scrambled(Path.iterdir))
    scrambled = batch_json(screen_batch(folder.root, JOB, today=today))
    assert normal == scrambled


def test_a_higher_score_comes_first(folder, today):
    folder("a.docx", WEAK)
    folder("b.docx", GOOD)
    folder("c.docx", PARTIAL)
    summary = screen_batch(folder.root, JOB, today=today)
    assert names(summary) == ["b.docx", "c.docx", "a.docx"]
    assert [r.rank for r in summary.results] == [1, 2, 3]
    scores = [r.coverage_score for r in summary.results]
    assert scores == sorted(scores, reverse=True)


def test_equal_scores_are_ordered_by_supported_required_items_then_file_name(folder, today):
    """Both resumes score 66.7, but one has two supported required items and one has one."""
    job = "Requirements\n- Python\n- SQL\n- Kubernetes\n"
    folder("a first by name.docx", ["SKILLS", "MySQL, Docker, Python"])  # 3 + 1.5 + 1.5 of 9
    folder("z second by name.docx", ["SKILLS", "Python, SQL"])  # 3 + 3 of 9
    summary = screen_batch(folder.root, job, today=today)
    first, second = summary.results
    assert first.coverage_score == second.coverage_score == 66.7
    assert names(summary) == ["z second by name.docx", "a first by name.docx"]
    assert summary.results[0].required_supported == 2
    assert summary.results[1].required_supported == 1


def test_a_full_tie_is_broken_by_case_insensitive_file_name(folder, today):
    folder("Zed.docx", GOOD + ["Zed"])
    folder("alex.docx", GOOD + ["alex"])
    folder("Bob.docx", GOOD + ["Bob"])
    summary = screen_batch(folder.root, JOB, today=today)
    assert len({r.coverage_score for r in summary.results}) == 1
    assert names(summary) == ["alex.docx", "Bob.docx", "Zed.docx"]  # not "Bob, Zed, alex"


def test_the_exact_name_breaks_a_tie_that_case_folding_leaves():
    assert sorted(["a.docx", "A.docx"], key=name_key) == ["A.docx", "a.docx"]
    assert sorted(["b.docx", "A.docx", "a.docx"], key=name_key) == ["A.docx", "a.docx", "b.docx"]


def test_a_resume_without_a_score_is_listed_last(folder, today):
    folder("a.docx", GOOD)
    folder("b.docx", GOOD + ["b"])
    summary = screen_batch(folder.root, "Responsibilities\n- Build things\n", today=today)
    assert all(r.coverage_score is None for r in summary.results)
    assert [r.rank for r in summary.results] == [1, 2]


# --- duplicates -----------------------------------------------------------------------------


def test_identical_content_under_different_names_is_flagged_and_never_dropped(folder, today):
    folder("one.docx", GOOD)
    folder("two.docx", GOOD)
    folder("three.docx", PARTIAL)
    summary = screen_batch(folder.root, JOB, today=today)
    assert sorted(names(summary)) == ["one.docx", "three.docx", "two.docx"]  # all kept
    assert summary.counts.duplicate_files == 1
    assert summary.counts.duplicate_groups == 1
    group = summary.duplicate_groups[0]
    assert group.reference_file == "one.docx"
    assert group.duplicate_files == ["two.docx"]
    by_name = {r.source_file: r for r in summary.results}
    assert by_name["two.docx"].duplicate_of == "one.docx"
    assert by_name["one.docx"].duplicate_of is None
    assert by_name["one.docx"].duplicate_group == by_name["two.docx"].duplicate_group == 1
    assert by_name["three.docx"].duplicate_group is None


def test_the_reference_of_a_duplicate_group_is_the_first_name_in_any_enumeration(
    folder, monkeypatch, today
):
    for name in ("z copy.docx", "A original.docx", "m copy.docx"):
        folder(name, GOOD)
    folder("other.docx", PARTIAL)
    first = screen_batch(folder.root, JOB, today=today)
    monkeypatch.setattr(Path, "iterdir", _scrambled(Path.iterdir))
    second = screen_batch(folder.root, JOB, today=today)
    for summary in (first, second):
        group = summary.duplicate_groups[0]
        assert group.reference_file == "A original.docx"
        assert group.duplicate_files == ["m copy.docx", "z copy.docx"]
    assert batch_json(first) == batch_json(second)


def test_duplicate_groups_are_numbered_by_reference_name(folder, today):
    for name in ("b1.docx", "b2.docx"):
        folder(name, GOOD)
    for name in ("a1.docx", "a2.docx"):
        folder(name, PARTIAL)
    summary = screen_batch(folder.root, JOB, today=today)
    assert [(g.group, g.reference_file) for g in summary.duplicate_groups] == [
        (1, "a1.docx"),
        (2, "b1.docx"),
    ]


def test_a_difference_in_spacing_only_still_counts_as_the_same_resume(folder, today):
    folder("a.docx", ["SKILLS", "Python, SQL"])
    folder("b.docx", ["SKILLS", "Python,   SQL  "])
    summary = screen_batch(folder.root, JOB, today=today)
    assert summary.counts.duplicate_files == 1


def test_different_resumes_are_not_merged_by_the_normalisation(folder, today):
    folder("a.docx", ["SKILLS", "Python, SQL"])
    folder("b.docx", ["SKILLS", "python, sql"])  # only the case differs
    folder("c.docx", ["SKILLS", "Python, SQL, R"])
    summary = screen_batch(folder.root, JOB, today=today)
    assert summary.counts.duplicate_files == 0
    assert summary.duplicate_groups == []


def test_the_fingerprint_rules():
    assert fingerprint("a  b\n\tc") == fingerprint("a b c")
    assert fingerprint("a b") == fingerprint("a b")  # a non breaking space
    assert fingerprint("é") == fingerprint("é")  # the same letter, composed or not
    assert fingerprint("Python") != fingerprint("python")
    assert fingerprint("Python 3") != fingerprint("Python 4")
    assert fingerprint("") is None
    assert fingerprint(" \n\t ") is None
    assert re.fullmatch(r"[0-9a-f]{16}", fingerprint("x"))


def test_a_failed_file_is_never_part_of_a_duplicate_group(folder, today):
    folder("a.docx", GOOD)
    (folder.root / "b.docx").write_bytes(b"broken")
    (folder.root / "c.docx").write_bytes(b"broken")
    summary = screen_batch(folder.root, JOB, today=today)
    assert summary.duplicate_groups == []
    assert summary.counts.duplicate_files == 0


# --- failed files, counts, wording ----------------------------------------------------------


def test_failed_files_have_no_score_and_no_rank(folder, today):
    folder("a.docx", GOOD)
    (folder.root / "bad.docx").write_bytes(b"nope")
    summary = screen_batch(folder.root, JOB, today=today)
    data = json.loads(batch_json(summary))
    assert set(data["failures"][0]) == {"source_file", "error_type", "error"}
    assert "bad.docx" not in [r["source_file"] for r in data["results"]]
    assert [r["rank"] for r in data["results"]] == [1]


def test_the_json_summary_has_stable_keys_and_every_required_count(folder, today):
    folder("a.docx", GOOD)
    folder("b.docx", GOOD)
    (folder.root / "bad.docx").write_bytes(b"nope")
    data = json.loads(batch_json(screen_batch(folder.root, JOB, today=today)))
    assert list(data) == [
        "human_review_required", "notice", "ordering_rule", "ordering_caveats", "extra_skills",
        "job_notes", "counts", "results", "failures", "duplicate_groups", "ignored_files",
        "limitations",
    ]
    assert list(data["counts"]) == [
        "files_found", "screened", "failed", "duplicate_files", "duplicate_groups",
        "complete_scores", "incomplete_scores", "needs_review", "requirements",
    ]
    assert list(data["counts"]["requirements"]) == [
        "supported", "partial", "not_found", "unverified", "unassessed",
    ]
    assert data["counts"]["files_found"] == 3
    assert data["counts"]["screened"] == 2
    assert data["counts"]["failed"] == 1
    assert data["human_review_required"] is True
    result = data["results"][0]
    assert set(result) == {
        "source_file", "rank", "coverage_score", "score_complete", "required_supported",
        "required_total", "counts", "needs_review", "duplicate_group", "duplicate_of", "report",
    }
    assert "retrieval" not in result["report"]  # as in the single resume JSON


def test_the_text_summary_has_every_required_section_and_count(folder, today):
    folder("a.docx", GOOD)
    folder("b.docx", GOOD)
    (folder.root / "bad.docx").write_bytes(b"nope")
    text = batch_text(screen_batch(folder.root, JOB, today=today))
    for part in (
        "BATCH SCREENING SUMMARY", "Found: 3", "Screened: 2", "Failed: 1",
        "Duplicate files: 1 in 1 group(s)", "Complete scores: 2", "INCOMPLETE scores: 0",
        "ORDERED FOR HUMAN REVIEW", "NEEDS HUMAN REVIEW", "DUPLICATES", "FAILED FILES",
        "bad.docx:", "LIMITATIONS", "supported", "partial", "not found", "UNVERIFIED",
        "NOT ASSESSED",
    ):
        assert part in text


def test_incomplete_scores_and_unassessed_requirements_stay_visible_even_at_100(folder, today):
    job = "Requirements\n- Python\n- Experience with Terraform and Rust\n"
    folder("a.docx", ["SKILLS", "Python"])
    summary = screen_batch(folder.root, job, today=today)
    result = summary.results[0]
    assert result.coverage_score == 100.0
    assert result.score_complete is False
    assert result.counts.unassessed == 1
    assert [i.reason for i in result.needs_review] == ["unassessed"]
    text = batch_text(summary)
    assert "score 100.0 (INCOMPLETE SCORE)" in text
    assert "NOT ASSESSED 1" in text
    assert "[not assessed] Experience with Terraform and Rust" in text
    assert "INCOMPLETE scores: 1" in text
    assert any("INCOMPLETE" in c for c in summary.ordering_caveats)


def test_unverified_counts_are_shown_beside_the_score(folder, today):
    job = "Requirements\n- Python\n- SQL\n"
    folder("a.docx", ["SKILLS", "SQL", "INTERESTS", "Python"])  # python only in a hobby line
    summary = screen_batch(folder.root, job, today=today)
    result = summary.results[0]
    assert result.counts.unverified == 1
    assert [(i.requirement, i.reason) for i in result.needs_review] == [("python", "unverified")]
    text = batch_text(summary)
    assert "UNVERIFIED 1" in text
    assert "[unverified] python" in text
    assert "Unverified requirements count as zero" in text


def test_the_wording_is_about_human_review_and_never_a_decision(folder, today):
    folder("a.docx", GOOD)
    folder("b.docx", WEAK)
    summary = screen_batch(folder.root, JOB, today=today)
    text = batch_text(summary)
    assert "ordered for human review" in text.lower()
    assert "a person must make every hiring decision" in text
    lowered = text.lower() + batch_json(summary).lower()
    for phrase in (
        "recommended for", "recommend hiring", "should be hired", "rejected", "shortlisted",
        "top candidate", "best candidate", "selected", "hire this", "auto reject",
    ):
        assert phrase not in lowered


def test_the_text_summary_holds_no_resume_text(folder, today):
    marker = "UNIQUE-MARKER-LINE-4821"
    folder("a.docx", [*GOOD, f"Worked on {marker}"])
    summary = screen_batch(folder.root, JOB, today=today)
    assert marker not in batch_text(summary)


# --- retrieval boundaries -------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["lexical", "semantic"])
def test_retrieval_never_changes_scores_counts_or_order(folder, today, mode):
    folder("a.docx", GOOD)
    folder("b.docx", [*PARTIAL, "Docker"])
    folder("c.docx", WEAK)
    plain = screen_batch(folder.root, JOB, today=today)
    with_rag = screen_batch(folder.root, JOB, today=today, rag_mode=mode, embedder=FakeEmbedder())

    def essentials(summary):
        return [
            (r.source_file, r.rank, r.coverage_score, r.counts.model_dump(),
             [a.status.value for a in r.report.assessments])
            for r in summary.results
        ]

    assert essentials(plain) == essentials(with_rag)
    assert all(r.report.retrieval for r in with_rag.results)
    assert not any(r.report.retrieval for r in plain.results)


def test_retrieval_cannot_break_a_tie_between_equal_scores(folder, today):
    """Zed has far more matching passages than alex, yet the file name still decides."""
    rich = [*GOOD]
    for section in ("PROJECTS", "SUMMARY", "EXPERIENCE", "TOOLS", "COURSES"):
        rich += [section, "Python and SQL and Kubernetes work", "More Python, SQL, Kubernetes"]
    folder("Zed.docx", rich)
    folder("alex.docx", GOOD)
    plain = screen_batch(folder.root, JOB, today=today)
    with_rag = screen_batch(folder.root, JOB, today=today, rag_mode="lexical")
    assert plain.results[0].coverage_score == plain.results[1].coverage_score
    assert names(plain) == names(with_rag) == ["alex.docx", "Zed.docx"]
    zed = next(r for r in with_rag.results if r.source_file == "Zed.docx")
    alex = next(r for r in with_rag.results if r.source_file == "alex.docx")
    passages = lambda r: sum(len(e.passages) for e in r.report.retrieval)  # noqa: E731
    assert passages(zed) > passages(alex)


# --- safe names and errors ------------------------------------------------------------------


def test_control_characters_in_a_name_or_message_are_replaced():
    assert safe_text("a\nb\x1b[31mred\tc") == "a?b?[31mred?c"
    assert safe_text("x" * 400).endswith("...")
    assert len(safe_text("x" * 400)) == 300


def test_failure_messages_never_contain_resume_text(folder, today):
    (folder.root / "bad.docx").write_bytes(b"PRIVATE-CONTENT-IN-A-BROKEN-FILE")
    folder("a.docx", GOOD)
    summary = screen_batch(folder.root, JOB, today=today)
    assert "PRIVATE-CONTENT" not in batch_json(summary)


def test_nothing_is_logged_about_names_or_text(folder, today, caplog):
    marker = "PRIVATE-MARKER-7788"
    folder(f"{marker} candidate.docx", [*GOOD, f"Led {marker} project"])
    (folder.root / f"{marker} broken.docx").write_bytes(b"x")
    with caplog.at_level(logging.DEBUG, logger="hr_screening"):
        screen_batch(folder.root, JOB, today=today)
    assert caplog.records
    assert marker not in caplog.text


# --- command line ---------------------------------------------------------------------------


def test_the_cli_screens_a_folder_with_paths_that_contain_spaces(folder, job_file, cli):
    folder("cv one.docx", GOOD)
    folder("cv two.docx", PARTIAL)
    code, out, err = cli("--batch-dir", folder.root, "--job-file", job_file, "--json")
    data = json.loads(out)
    assert code == 0
    assert names_from(data) == ["cv one.docx", "cv two.docx"]
    assert err == ""


def names_from(data):
    return [r["source_file"] for r in data["results"]]


def test_exit_codes_for_success_partial_failure_and_total_failure(folder, job_file, cli, tmp_path):
    folder("a.docx", GOOD)
    assert cli("--batch-dir", folder.root, "--job-file", job_file)[0] == 0
    (folder.root / "bad.docx").write_bytes(b"x")
    code, out, _ = cli("--batch-dir", folder.root, "--job-file", job_file)
    assert code == 1  # some screened, some failed
    assert "FAILED FILES" in out
    total = tmp_path / "only bad"
    total.mkdir()
    (total / "bad.docx").write_bytes(b"x")
    code, out, _ = cli("--batch-dir", total, "--job-file", job_file)
    assert code == 2  # nothing could be screened
    assert "No resume was screened." in out


def test_the_cli_reports_a_missing_folder_an_empty_folder_and_a_bad_job(
    folder, job_file, cli, tmp_path
):
    code, out, err = cli("--batch-dir", tmp_path / "missing", "--job-file", job_file)
    assert (code, out) == (2, "")
    assert "folder was not found" in err
    assert "Traceback" not in err
    empty = tmp_path / "empty"
    empty.mkdir()
    code, out, err = cli("--batch-dir", empty, "--job-file", job_file)
    assert (code, out) == (2, "")
    assert "no files to screen" in err
    folder("a.docx", GOOD)
    code, out, err = cli("--batch-dir", folder.root, "--job-text", "   ")
    assert (code, out) == (2, "")
    assert "job description is empty" in err


def test_the_cli_keeps_the_job_description_out_of_the_screening(folder, cli):
    folder("a.docx", GOOD)
    job = folder.root / "job.txt"
    job.write_text(JOB, encoding="utf-8")
    code, out, _ = cli("--batch-dir", folder.root, "--job-file", job)
    assert code == 0
    assert "job.txt (the job description file)" in out


def test_the_cli_passes_extra_skills_to_every_resume(folder, job_file, cli):
    folder("a.docx", [*GOOD, "Terraform"])
    folder("b.docx", GOOD)
    code, out, _ = cli(
        "--batch-dir", folder.root, "--job-file", job_file, "--extra-skill", "Terraform",
        "--extra-skill", "Rust", "--json",
    )
    data = json.loads(out)
    assert code == 0
    assert data["extra_skills"] == ["Terraform", "Rust"]
    for result in data["results"]:
        assert "Terraform" in [a["requirement"] for a in result["report"]["assessments"]]


def test_batch_options_are_checked_against_each_other(folder, job_file, cli):
    one = folder("a.docx", GOOD)
    base = ["--job-file", str(job_file)]
    for argv in (
        ["--resume", str(one), "--batch-dir", str(folder.root), *base],
        ["--resume", str(one), "--output-dir", str(folder.root.parent / "o"), *base],
        ["--batch-dir", str(folder.root), "--resume-dir", str(folder.root), *base],
        ["--batch-dir", str(folder.root), "--batch-dir", str(folder.root), *base],
        base,
    ):
        code, out, err = cli(*argv)
        assert code == 2, argv
        assert out == ""
        assert "Traceback" not in err


def test_the_single_resume_command_is_unchanged(folder, job_file, cli):
    one = folder("a.docx", GOOD)
    code, out, err = cli("--resume", one, "--job-file", job_file)
    assert code == 0
    assert out.startswith("SCREENING REPORT")
    assert err == ""
    code, out, _ = cli("--resume", one, "--job-file", job_file, "--json")
    assert list(json.loads(out)) == [
        "assessments", "coverage_score", "score_explanation", "limitations",
        "human_review_required", "unassessed_requirements", "score_complete",
    ]
    assert cli("--resume", folder.root / "missing.docx", "--job-file", job_file)[0] == 2


# --- output files ---------------------------------------------------------------------------


def hashes(root):
    files = [p for p in root.iterdir() if p.is_file()]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def test_output_files_are_written_to_the_chosen_folder_and_inputs_are_untouched(
    folder, job_file, cli, tmp_path
):
    folder("cv one.docx", GOOD)
    folder("cv two.docx", PARTIAL)
    before = hashes(folder.root)
    out_dir = tmp_path / "out put"
    code, out, err = cli(
        "--batch-dir", folder.root, "--job-file", job_file, "--output-dir", out_dir
    )
    assert code == 0
    assert "Saved 6 file(s)" in err
    assert hashes(folder.root) == before
    found = [p.relative_to(out_dir) for p in out_dir.rglob("*") if p.is_file()]
    files = sorted(p.as_posix() for p in found)
    assert files == [
        "batch_summary.json",
        "batch_summary.txt",
        "reports/cv one.docx.report.json",
        "reports/cv one.docx.report.txt",
        "reports/cv two.docx.report.json",
        "reports/cv two.docx.report.txt",
    ]
    summary = json.loads((out_dir / "batch_summary.json").read_text(encoding="utf-8"))
    assert [r["source_file"] for r in summary["results"]] == ["cv one.docx", "cv two.docx"]
    one = json.loads((out_dir / "reports" / "cv one.docx.report.json").read_text(encoding="utf-8"))
    assert one["source_file"] == "cv one.docx"
    assert one["report"]["human_review_required"] is True
    text = (out_dir / "reports" / "cv one.docx.report.txt").read_text(encoding="utf-8")
    assert text.startswith("SOURCE FILE: cv one.docx")
    assert "SCREENING REPORT" in text
    assert (out_dir / "batch_summary.txt").read_text(encoding="utf-8") == out.rstrip("\n")


def test_the_output_folder_may_not_be_the_resume_folder(folder, job_file, cli):
    folder("a.docx", GOOD)
    before = hashes(folder.root)
    code, out, err = cli(
        "--batch-dir", folder.root, "--job-file", job_file, "--output-dir", folder.root
    )
    assert (code, out) == (2, "")
    assert "different folder" in err
    assert hashes(folder.root) == before


def test_an_output_folder_inside_the_resume_folder_is_not_screened_on_the_next_run(
    folder, job_file, cli
):
    folder("a.docx", GOOD)
    inside = folder.root / "results"
    args = ("--batch-dir", folder.root, "--job-file", job_file, "--output-dir", inside)
    assert cli(*args)[0] == 0
    code, out, _ = cli(*args)
    assert code == 0  # the first run's output is not mistaken for resumes
    assert "results/ (sub folder, not searched)" in out


def test_an_output_path_that_is_a_file_is_refused_before_screening(folder, job_file, cli, tmp_path):
    folder("a.docx", GOOD)
    target = tmp_path / "taken"
    target.write_text("x", encoding="utf-8")
    code, out, err = cli(
        "--batch-dir", folder.root, "--job-file", job_file, "--output-dir", target
    )
    assert (code, out) == (2, "")
    assert "is a file" in err
    assert target.read_text(encoding="utf-8") == "x"


def test_stderr_never_carries_resume_text_or_names(folder, job_file, cli, tmp_path):
    marker = "PRIVATE-MARKER-9911"
    folder(f"{marker}.docx", [*GOOD, f"Built {marker} systems"])
    _, _, err = cli(
        "--batch-dir", folder.root, "--job-file", job_file, "--output-dir", tmp_path / "o"
    )
    assert marker not in err


# --- the README -----------------------------------------------------------------------------


def test_the_readme_describes_the_batch_workflow_and_its_limits():
    readme = " ".join((Path(__file__).resolve().parents[1] / "README.md")
                      .read_text(encoding="utf-8").split())
    for part in (
        "--batch-dir", "--output-dir", "Exit codes", "Order for human review",
        "INCOMPLETE", "Duplicates", "batch_summary.json", "does not accept, reject or recommend",
        "**Not run on Windows**", "A skill written in plain lower case",
    ):
        assert part in readme
    assert "Python and Terraform\" counts as assessed" not in readme
