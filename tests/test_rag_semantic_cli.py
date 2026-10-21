"""The --rag semantic flag: opt in, config errors exit 2, failures fall back to keywords."""

import json
from pathlib import Path

import pytest

from hr_screening import main as main_module
from hr_screening.config import Settings
from hr_screening.ingestion.loader import load_resume_text
from hr_screening.llm.fake import FakeLLMClient
from hr_screening.llm.ollama_embedder import OllamaEmbedder
from hr_screening.pipeline import LLM_SUCCESS_NOTE, RETRIEVAL_NOTE
from hr_screening.rag.embeddings import FakeEmbedder

FIXTURE = Path(__file__).parent / "fixtures" / "long_synthetic_resume.txt"
JOB = """Data Analyst

Requirements
- 3+ years of experience in analytics
- Python, SQL, Tableau and Kubernetes
- Bachelor's degree in computer science
- Example Cloud Practitioner certification

Preferred
- Spark
- Snowflake
"""
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
_ENV = (
    "EMBEDDING_MODEL",
    "OLLAMA_BASE_URL",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_API_KEY",
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES",
)


@pytest.fixture(autouse=True)
def quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(main_module, "configure_logging", lambda: None)


@pytest.fixture
def files(tmp_path, make_docx):
    resume = make_docx(tmp_path / "cv.docx", FIXTURE.read_text(encoding="utf-8").splitlines())
    job = tmp_path / "job.txt"
    job.write_text(JOB, encoding="utf-8")
    return resume, ["--resume", str(resume), "--job-file", str(job)]


def use_settings(monkeypatch, **values) -> Settings:
    config = Settings(_env_file=None, **values)
    monkeypatch.setattr(main_module, "get_settings", lambda: config)
    return config


def use_embedder(monkeypatch, embedder) -> None:
    monkeypatch.setattr(main_module, "build_embedder", lambda settings: embedder)


def run(argv, capsys):
    code = main_module.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def run_json(argv, capsys):
    code, out, err = run([*argv, "--json"], capsys)
    return code, (json.loads(out) if out else None), err


# --- a working run ----------------------------------------------------------------------------


def test_semantic_output_has_its_own_labelled_section_and_the_same_score(
    files, monkeypatch, capsys
):
    _, argv = files
    _, plain, _ = run(argv, capsys)
    use_settings(monkeypatch)
    use_embedder(monkeypatch, FakeEmbedder(dimensions=4096))
    code, out, err = run([*argv, "--rag", "semantic"], capsys)
    assert code == 0
    assert "RETRIEVED PASSAGES (for review)" in out
    assert "Found by semantic search" in out
    assert "similarity " in out
    assert err == ""
    score_line = next(ln for ln in plain.splitlines() if ln.startswith("Requirement coverage"))
    assert score_line in out.splitlines()
    assert plain.split("\n\nLIMITATIONS")[0] in out


def test_semantic_json_keeps_assessments_and_traceable_passages(files, monkeypatch, capsys):
    resume, argv = files
    text = load_resume_text(resume)
    _, plain, _ = run_json(argv, capsys)
    use_settings(monkeypatch)
    use_embedder(monkeypatch, FakeEmbedder(dimensions=4096))
    code, report, _ = run_json([*argv, "--rag", "semantic"], capsys)
    assert code == 0
    for key in ("assessments", "coverage_score", "score_explanation"):
        assert report[key] == plain[key]
    assert {entry["method"] for entry in report["retrieval"]} == {"semantic"}
    seen = 0
    for entry in report["retrieval"]:
        for passage in entry["passages"]:
            assert text[passage["start"] : passage["end"]] == passage["text"]
            assert 0 < passage["score"] <= 1
            seen += 1
    assert seen > 5


def test_semantic_search_builds_no_model_client(files, monkeypatch, capsys):
    def refuse(*args, **kwargs):
        raise AssertionError("no model client may be built without --llm")

    monkeypatch.setattr(main_module, "build_llm_client", refuse)
    use_settings(monkeypatch)
    use_embedder(monkeypatch, FakeEmbedder(dimensions=4096))
    _, argv = files
    code, _, _ = run([*argv, "--rag", "semantic"], capsys)
    assert code == 0


def test_semantic_search_and_the_model_can_be_combined(files, monkeypatch, capsys):
    client = FakeLLMClient(json.dumps(PAYLOAD))
    use_settings(monkeypatch)
    monkeypatch.setattr(main_module, "build_llm_client", lambda settings: client)
    use_embedder(monkeypatch, FakeEmbedder(dimensions=4096))
    _, argv = files
    _, report, err = run_json([*argv, "--llm", "--rag", "semantic"], capsys)
    assert len(client.calls) == 1
    assert LLM_SUCCESS_NOTE in report["limitations"]
    assert any("semantic search using the embedding model" in n for n in report["limitations"])
    assert err == ""


# --- configuration errors exit with code 2 ----------------------------------------------------


def test_a_missing_embedding_model_is_a_clear_error(files, monkeypatch, capsys):
    use_settings(monkeypatch)
    _, argv = files
    code, out, err = run([*argv, "--rag", "semantic"], capsys)
    assert code == 2
    assert out == ""
    assert "EMBEDDING_MODEL is not set" in err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    ("values", "word"),
    [
        ({"embedding_model": "big:cloud"}, "cloud"),
        ({"embedding_model": "m", "ollama_base_url": "ftp://localhost"}, "OLLAMA_BASE_URL"),
    ],
    ids=["cloud_model", "bad_address"],
)
def test_other_configuration_problems_are_clear_errors(files, monkeypatch, capsys, values, word):
    use_settings(monkeypatch, **values)
    _, argv = files
    code, out, err = run([*argv, "--rag", "semantic"], capsys)
    assert code == 2
    assert out == ""
    assert word in err


def test_an_invalid_setting_names_the_setting_but_not_the_value(files, monkeypatch, capsys):
    def broken():
        return Settings(_env_file=None, llm_timeout_seconds=-5)

    monkeypatch.setattr(main_module, "get_settings", broken)
    _, argv = files
    code, out, err = run([*argv, "--rag", "semantic"], capsys)
    assert code == 2
    assert out == ""
    assert "LLM_TIMEOUT_SECONDS" in err
    assert "-5" not in err


def test_keyword_retrieval_still_reads_no_settings(files, monkeypatch, capsys):
    def refuse(*args, **kwargs):
        raise AssertionError("keyword retrieval must not read settings")

    monkeypatch.setattr(main_module, "get_settings", refuse)
    monkeypatch.setattr(main_module, "build_embedder", refuse)
    _, argv = files
    code, out, _ = run([*argv, "--rag", "lexical"], capsys)
    assert code == 0
    assert "Found by keyword search" in out


# --- privacy warning --------------------------------------------------------------------------


def test_a_remote_server_gets_a_warning(files, monkeypatch, capsys):
    use_settings(monkeypatch, ollama_base_url="http://ollama.internal:11434")
    use_embedder(monkeypatch, FakeEmbedder(dimensions=4096))
    _, argv = files
    code, _, err = run([*argv, "--rag", "semantic"], capsys)
    assert code == 0
    assert "not a local address" in err


@pytest.mark.parametrize("url", ["http://localhost:11434", "http://127.0.0.1:11434"])
def test_a_local_server_gets_no_warning(files, monkeypatch, capsys, url):
    use_settings(monkeypatch, ollama_base_url=url)
    use_embedder(monkeypatch, FakeEmbedder(dimensions=4096))
    _, argv = files
    _, _, err = run([*argv, "--rag", "semantic"], capsys)
    assert err == ""


# --- failure falls back to keyword search -----------------------------------------------------


def test_an_unreachable_server_falls_back_to_keyword_search(files, monkeypatch, capsys):
    def refuse(url, body, timeout):
        raise ConnectionRefusedError("SECRET detail")

    use_settings(monkeypatch)
    embedder = OllamaEmbedder("http://localhost:11434", "embed-model", transport=refuse)
    use_embedder(monkeypatch, embedder)
    _, argv = files
    _, plain, _ = run_json(argv, capsys)
    code, report, err = run_json([*argv, "--rag", "semantic"], capsys)
    assert code == 0
    assert report["assessments"] == plain["assessments"]
    assert report["coverage_score"] == plain["coverage_score"]
    assert {entry["method"] for entry in report["retrieval"]} == {"lexical"}
    assert report["limitations"][-2] == (
        "Semantic search failed (the model server could not be reached), "
        "so keyword search was used instead."
    )
    assert report["limitations"][-1] == RETRIEVAL_NOTE
    assert "SECRET" not in json.dumps(report) + err


def test_the_text_report_after_a_fallback_describes_keyword_search(files, monkeypatch, capsys):
    use_settings(monkeypatch)
    use_embedder(monkeypatch, FakeEmbedder(error=RuntimeError("boom")))
    _, argv = files
    code, out, _ = run([*argv, "--rag", "semantic"], capsys)
    assert code == 0
    assert "Found by keyword search" in out
    assert "Found by semantic search" not in out
    assert "Semantic search failed" in out