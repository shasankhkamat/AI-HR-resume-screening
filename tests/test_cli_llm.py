"""The --llm flag. The model is always faked, so no test needs Ollama or a network."""

import json

import pytest

from hr_screening import main as main_module
from hr_screening.config import Settings
from hr_screening.llm.fake import FakeLLMClient
from hr_screening.llm.ollama_client import OllamaClient
from hr_screening.pipeline import LLM_SUCCESS_NOTE

_ENV_NAMES = (
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "LLM_MODEL",
    "OLLAMA_BASE_URL",
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES",
    "LLM_MAX_OUTPUT_TOKENS",
)
PAYLOAD = {
    "skills": ["Python", "SQL"],
    "education": ["B.Tech in Computer Science, Example Institute, 2018"],
    "certifications": ["Example Cloud Practitioner"],
    "projects": [],
    "experience": [
        {"heading": "Data Analyst, Example Corp", "start": "Jan 2019", "end": "Dec 2021"},
        {"heading": "Senior Analyst, Sample Ltd", "start": "Jan 2022", "end": "Present"},
    ],
}


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(main_module, "configure_logging", lambda: None)


@pytest.fixture
def files(tmp_path, make_docx, sample_resume_lines, sample_job):
    resume = make_docx(tmp_path / "cv.docx", sample_resume_lines)
    job = tmp_path / "job.txt"
    job.write_text(sample_job, encoding="utf-8")
    return ["--resume", str(resume), "--job-file", str(job)]


def ollama_settings(**values) -> Settings:
    values.setdefault("llm_provider", "ollama")
    values.setdefault("llm_model", "my-model")
    return Settings(_env_file=None, **values)


def use_settings(monkeypatch, config: Settings) -> None:
    monkeypatch.setattr(main_module, "get_settings", lambda: config)


def forbid(name: str):
    def fail(*args, **kwargs):
        raise AssertionError(f"{name} must not be called")

    return fail


def run_json(argv, capsys):
    code = main_module.main([*argv, "--json"])
    captured = capsys.readouterr()
    return code, (json.loads(captured.out) if captured.out else None), captured.err


# --- without --llm -----------------------------------------------------------------------


def test_llm_is_off_by_default():
    args = main_module._build_parser().parse_args(["--resume", "a.pdf", "--job-text", "x"])
    assert args.llm is False


def test_without_llm_nothing_is_initialized(monkeypatch, capsys, files):
    monkeypatch.setattr(main_module, "build_llm_client", forbid("build_llm_client"))
    monkeypatch.setattr(main_module, "get_settings", forbid("get_settings"))
    code, report, err = run_json(files, capsys)
    assert code == 0
    assert report["coverage_score"] == 73.5
    assert LLM_SUCCESS_NOTE not in report["limitations"]
    assert not any("LLM" in item for item in report["limitations"])
    assert err == ""


# --- with --llm ----------------------------------------------------------------------------


def test_llm_success_adds_the_note_and_keeps_the_score(monkeypatch, capsys, files):
    client = FakeLLMClient(json.dumps(PAYLOAD))
    use_settings(monkeypatch, ollama_settings())
    monkeypatch.setattr(main_module, "build_llm_client", lambda settings: client)
    code, report, err = run_json([*files, "--llm"], capsys)
    assert code == 0
    assert len(client.calls) == 1
    assert report["coverage_score"] == 73.5
    assert LLM_SUCCESS_NOTE in report["limitations"]
    assert not any("rule based parser was used" in item for item in report["limitations"])
    assert report["human_review_required"] is True
    assert err == ""


def test_text_output_shows_the_note(monkeypatch, capsys, files):
    use_settings(monkeypatch, ollama_settings())
    client = FakeLLMClient(json.dumps(PAYLOAD))
    monkeypatch.setattr(main_module, "build_llm_client", lambda settings: client)
    assert main_module.main([*files, "--llm"]) == 0
    assert "A language model performed the extraction" in capsys.readouterr().out


def test_unreachable_server_falls_back_with_a_note(monkeypatch, capsys, files):
    def refuse(url, body, timeout):
        raise ConnectionRefusedError("secret detail")

    client = OllamaClient("http://localhost:11434", "my-model", transport=refuse)
    use_settings(monkeypatch, ollama_settings())
    monkeypatch.setattr(main_module, "build_llm_client", lambda settings: client)
    code, report, err = run_json([*files, "--llm"], capsys)
    assert code == 0
    assert report["coverage_score"] == 73.5
    notes = " ".join(report["limitations"])
    assert "the model server could not be reached" in notes
    assert "rule based parser was used" in notes
    assert "secret detail" not in notes + err


def test_malformed_model_output_falls_back(monkeypatch, capsys, files):
    use_settings(monkeypatch, ollama_settings())
    client = FakeLLMClient("not json")
    monkeypatch.setattr(main_module, "build_llm_client", lambda settings: client)
    code, report, _ = run_json([*files, "--llm"], capsys)
    assert code == 0
    assert report["coverage_score"] == 73.5
    assert any("rule based parser was used" in item for item in report["limitations"])


# --- configuration errors exit with code 2 and never fall back ---------------------------


def test_default_settings_are_a_clear_error(monkeypatch, capsys, files):
    use_settings(monkeypatch, Settings(_env_file=None))
    code = main_module.main([*files, "--llm"])
    captured = capsys.readouterr()
    assert code == 2
    assert "LLM_PROVIDER" in captured.err
    assert captured.out == ""
    assert "Traceback" not in captured.err


def test_missing_model_is_a_clear_error(monkeypatch, capsys, files):
    use_settings(monkeypatch, Settings(_env_file=None, llm_provider="ollama"))
    code = main_module.main([*files, "--llm"])
    captured = capsys.readouterr()
    assert code == 2
    assert "LLM_MODEL" in captured.err
    assert captured.out == ""


def test_unsupported_provider_is_a_clear_error(monkeypatch, capsys, files):
    use_settings(monkeypatch, ollama_settings(llm_provider="openai"))
    code = main_module.main([*files, "--llm"])
    captured = capsys.readouterr()
    assert code == 2
    assert "ollama" in captured.err
    assert captured.out == ""


def test_cloud_model_is_a_clear_error(monkeypatch, capsys, files):
    use_settings(monkeypatch, ollama_settings(llm_model="big:120b-cloud"))
    code = main_module.main([*files, "--llm"])
    captured = capsys.readouterr()
    assert code == 2
    assert "cloud" in captured.err
    assert captured.out == ""


def test_invalid_setting_value_names_the_setting_but_not_the_value(monkeypatch, capsys, files):
    def broken():
        return Settings(_env_file=None, llm_timeout_seconds=-5)

    monkeypatch.setattr(main_module, "get_settings", broken)
    code = main_module.main([*files, "--llm"])
    captured = capsys.readouterr()
    assert code == 2
    assert "LLM_TIMEOUT_SECONDS" in captured.err
    assert "-5" not in captured.err
    assert captured.out == ""
    assert "Traceback" not in captured.err


def test_invalid_job_stops_before_any_model_call(monkeypatch, capsys, files):
    use_settings(monkeypatch, ollama_settings())
    client = FakeLLMClient(json.dumps(PAYLOAD))
    monkeypatch.setattr(main_module, "build_llm_client", lambda settings: client)
    resume = files[1]
    code = main_module.main(["--resume", resume, "--job-text", "   ", "--llm"])
    captured = capsys.readouterr()
    assert code == 2
    assert "empty" in captured.err
    assert captured.out == ""
    assert client.calls == []


# --- privacy warning -----------------------------------------------------------------------


def test_remote_server_gets_a_warning(monkeypatch, capsys, files):
    use_settings(monkeypatch, ollama_settings(ollama_base_url="http://ollama.internal:11434"))
    client = FakeLLMClient(json.dumps(PAYLOAD))
    monkeypatch.setattr(main_module, "build_llm_client", lambda settings: client)
    code, _, err = run_json([*files, "--llm"], capsys)
    assert code == 0
    assert "not a local address" in err


@pytest.mark.parametrize("url", ["http://localhost:11434", "http://127.0.0.1:11434"])
def test_local_server_gets_no_warning(url, monkeypatch, capsys, files):
    use_settings(monkeypatch, ollama_settings(ollama_base_url=url))
    client = FakeLLMClient(json.dumps(PAYLOAD))
    monkeypatch.setattr(main_module, "build_llm_client", lambda settings: client)
    _, _, err = run_json([*files, "--llm"], capsys)
    assert err == ""


@pytest.mark.parametrize("flags", [[], ["--llm"]], ids=["rules", "llm"])
def test_invalid_setting_is_a_clean_error_before_logging_starts(monkeypatch, capsys, files, flags):
    def broken():
        return Settings(_env_file=None, llm_timeout_seconds=-5)

    monkeypatch.setattr(main_module, "configure_logging", broken)
    code = main_module.main([*files, *flags])
    captured = capsys.readouterr()
    assert code == 2
    assert "LLM_TIMEOUT_SECONDS" in captured.err
    assert "-5" not in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""