
import pytest

from hr_screening import main as main_module


@pytest.fixture
def cli(monkeypatch, capsys):
    monkeypatch.setattr(main_module, "configure_logging", lambda: None)

    def _run(*argv):
        try:
            code = main_module.main([str(a) for a in argv])
        except SystemExit as stop:
            code = stop.code
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return _run


@pytest.fixture
def two_resumes(tmp_path, make_docx):
    return (
        make_docx(tmp_path / "a.docx", ["SKILLS", "Python"]),
        make_docx(tmp_path / "b.docx", ["SKILLS", "Java"]),
    )


JOB = "Requirements\n- Python\n"


def test_two_resumes_in_one_run_are_refused_not_silently_reduced_to_one(cli, two_resumes):
    first, second = two_resumes
    code, out, err = cli("--resume", first, "--resume", second, "--job-text", JOB, "--json")
    assert code == 2
    assert out == ""
    assert "--resume was given more than once" in err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    "extra",
    [
        ["--job-text", JOB],
        ["--rag", "lexical", "--rag", "lexical"],
        ["--resume-dir", ".", "--resume-dir", "."],
    ],
    ids=["job_text_twice", "rag_twice", "resume_dir_twice"],
)
def test_other_repeated_options_are_refused_too(cli, two_resumes, extra):
    code, out, err = cli("--resume", two_resumes[0], "--job-text", JOB, *extra)
    assert code == 2
    assert out == ""
    assert "more than once" in err


def test_a_repeated_job_file_is_refused(cli, two_resumes, tmp_path):
    job = tmp_path / "job.txt"
    job.write_text(JOB, encoding="utf-8")
    code, out, err = cli("--resume", two_resumes[0], "--job-file", job, "--job-file", job)
    assert (code, out) == (2, "")
    assert "--job-file was given more than once" in err


def test_each_option_given_once_still_works(cli, two_resumes, tmp_path):
    code, out, _ = cli(
        "--resume", two_resumes[0], "--resume-dir", tmp_path, "--job-text", JOB, "--rag", "lexical"
    )
    assert code == 0
    assert "Decision support only" in out