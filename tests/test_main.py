import json

from hr_screening import main as main_module


def _setup(tmp_path, monkeypatch, make_docx, lines, job):
    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    resume = make_docx(tmp_path / "cv.docx", lines)
    job_file = tmp_path / "job.txt"
    job_file.write_text(job, encoding="utf-8")
    return resume, job_file


def test_cli_text_output(tmp_path, monkeypatch, capsys, make_docx, sample_resume_lines, sample_job):
    resume, job_file = _setup(tmp_path, monkeypatch, make_docx, sample_resume_lines, sample_job)
    code = main_module.main(["--resume", str(resume), "--job-file", str(job_file)])
    assert code == 0
    assert "Requirement coverage score" in capsys.readouterr().out


def test_cli_json_output(tmp_path, monkeypatch, capsys, make_docx, sample_resume_lines, sample_job):
    resume, job_file = _setup(tmp_path, monkeypatch, make_docx, sample_resume_lines, sample_job)
    code = main_module.main(["--resume", str(resume), "--job-file", str(job_file), "--json"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["human_review_required"] is True


def test_cli_reports_errors_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(main_module, "configure_logging", lambda: None)
    bad = tmp_path / "cv.txt"
    bad.write_text("hello", encoding="utf-8")
    code = main_module.main(["--resume", str(bad), "--job-text", "Python"])
    assert code == 2
    assert "Unsupported" in capsys.readouterr().err