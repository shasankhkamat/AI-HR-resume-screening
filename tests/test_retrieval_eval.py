"""The synthetic retrieval check in evals/: valid cases, correct scoring, safe failures."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from hr_screening.config import Settings
from hr_screening.llm.ollama_embedder import OllamaEmbedder
from hr_screening.rag.embeddings import FakeEmbedder
from hr_screening.rag.retrieval import LexicalRetriever
from hr_screening.rag.semantic import SemanticRetriever

EVALS = Path(__file__).resolve().parents[1] / "evals"
_ENV = ("EMBEDDING_MODEL", "OLLAMA_BASE_URL", "LLM_TIMEOUT_SECONDS", "LLM_MAX_RETRIES")


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def evaluation(monkeypatch):
    monkeypatch.syspath_prepend(str(EVALS))
    path = EVALS / "run_retrieval_eval.py"
    spec = importlib.util.spec_from_file_location("run_retrieval_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def passage(text, mentions=()):
    return SimpleNamespace(text=text, mention_kinds=list(mentions))


def query(gold, mention=None):
    return {"id": "q", "type": "exact", "gold": gold, "mention": mention}


# --- the case file ---------------------------------------------------------------------------


def test_the_case_file_is_valid(evaluation):
    evaluation.check_cases()
    ids = [q["id"] for resume in evaluation.RESUMES for q in resume["queries"]]
    assert len(ids) == len(set(ids)) > 10


def test_a_quote_that_is_not_in_the_resume_is_caught(evaluation, monkeypatch):
    resumes = [{"id": "r", "text": "SKILLS\nPython\n", "queries": [
        {"id": "a", "type": "exact", "kind": "required_skill", "requirement": "python",
         "gold": ["Rust"], "mention": None}]}]
    monkeypatch.setattr(evaluation, "RESUMES", resumes)
    with pytest.raises(ValueError, match="exactly once"):
        evaluation.check_cases()


def test_other_case_mistakes_are_caught(evaluation, monkeypatch):
    base = {"id": "a", "type": "exact", "kind": "required_skill", "requirement": "python",
            "gold": ["Python"], "mention": None}
    text = "SKILLS\nPython\n"
    for changes, message in [
        ({"type": "mystery"}, "Unknown type"),
        ({"gold": None, "mention": "negated"}, "no gold"),
    ]:
        resumes = [{"id": "r", "text": text, "queries": [{**base, **changes}]}]
        monkeypatch.setattr(evaluation, "RESUMES", resumes)
        with pytest.raises(ValueError, match=message):
            evaluation.check_cases()
    resumes = [{"id": "r", "text": text, "queries": [base, base]}]
    monkeypatch.setattr(evaluation, "RESUMES", resumes)
    with pytest.raises(ValueError, match="Duplicate"):
        evaluation.check_cases()


# --- scoring ---------------------------------------------------------------------------------


def test_scoring_a_hit_at_each_rank_and_a_miss(evaluation):
    gold = query(["needle"])
    first = evaluation.score_query(gold, [passage("a needle here"), passage("other")])
    second = evaluation.score_query(gold, [passage("other"), passage("a needle here")])
    miss = evaluation.score_query(gold, [passage("other"), passage("more")])
    assert (first["rank"], second["rank"], miss["rank"]) == (1, 2, None)
    assert miss["returned"] == 2


def test_any_of_the_listed_quotes_counts(evaluation):
    row = evaluation.score_query(query(["one", "two"]), [passage("only two here")])
    assert row["rank"] == 1


def test_an_absent_skill_is_correct_only_when_nothing_is_returned(evaluation):
    empty = evaluation.score_query(query(None), [])
    wrong = evaluation.score_query(query(None), [passage("something")])
    assert (empty["returned"], wrong["returned"]) == (0, 1)
    assert empty["answerable"] is False
    assert evaluation.describe(empty) == "correct, nothing returned"
    assert evaluation.describe(wrong) == "wrong, 1 passages returned"


def test_the_mention_label_is_checked_on_the_correct_passage(evaluation):
    gold = query(["needle"], mention="negated")
    right = evaluation.score_query(gold, [passage("needle", ["negated"])])
    wrong = evaluation.score_query(gold, [passage("needle", ["affirmative"])])
    missing = evaluation.score_query(gold, [passage("other", ["negated"])])
    assert (right["mention_ok"], wrong["mention_ok"], missing["mention_ok"]) == (True, False, False)


# --- running ---------------------------------------------------------------------------------


def test_the_documented_keyword_results_still_hold(evaluation):
    rows = evaluation.run(LexicalRetriever)
    answerable = [r for r in rows if r["answerable"]]
    misses = {r["id"] for r in answerable if r["rank"] is None}
    assert len(rows) == 22
    assert len(answerable) == 19
    assert sum(r["rank"] == 1 for r in answerable) == 17
    assert misses == {"analyst-forecasting", "marketing-abtesting"}
    assert all(r["returned"] == 0 for r in rows if not r["answerable"])
    assert all(r["mention_ok"] for r in rows if r["mention_ok"] is not None)


def test_the_semantic_path_runs_with_a_fake_embedder(evaluation):
    embedder = FakeEmbedder(dimensions=4096)
    rows = evaluation.run(lambda text: SemanticRetriever(text, embedder))
    assert len(rows) == 22
    assert all(row["returned"] <= 3 for row in rows)


def test_the_report_prints_every_query_and_a_summary(evaluation, capsys):
    evaluation.print_report(evaluation.run(LexicalRetriever), "keyword search (lexical)")
    out = capsys.readouterr().out
    assert "22 queries" in out
    assert "analyst-python" in out
    assert "Absent skills: nothing returned for 3 of 3." in out
    assert "Mention labels correct: 2 of 2." in out


# --- the command -------------------------------------------------------------------------------


def test_main_runs_keyword_search_by_default(evaluation, capsys):
    assert evaluation.main([]) == 0
    assert "keyword search (lexical)" in capsys.readouterr().out


def test_main_runs_semantic_search_with_whatever_embedder_the_settings_give(
    evaluation, monkeypatch, capsys
):
    monkeypatch.setattr(evaluation, "get_settings", lambda: Settings(_env_file=None))
    fake = FakeEmbedder(dimensions=4096)
    monkeypatch.setattr(evaluation, "build_embedder", lambda settings: fake)
    assert evaluation.main(["--semantic"]) == 0
    assert "semantic search, model fake-embedder" in capsys.readouterr().out


def test_a_missing_embedding_model_stops_with_a_clear_message(evaluation, monkeypatch, capsys):
    monkeypatch.setattr(evaluation, "get_settings", lambda: Settings(_env_file=None))
    assert evaluation.main(["--semantic"]) == 2
    out = capsys.readouterr().out
    assert "EMBEDDING_MODEL is not set" in out
    assert "Answerable" not in out


def test_an_unreachable_server_stops_instead_of_falling_back(evaluation, monkeypatch, capsys):
    def refuse(url, body, timeout):
        raise ConnectionRefusedError("SECRET detail")

    embedder = OllamaEmbedder("http://localhost:11434", "embed-model", transport=refuse)
    monkeypatch.setattr(evaluation, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(evaluation, "build_embedder", lambda settings: embedder)
    assert evaluation.main(["--semantic"]) == 2
    out = capsys.readouterr().out
    assert "could not be reached" in out
    assert "SECRET" not in out
    assert "Answerable" not in out


def test_invalid_settings_are_reported_without_the_value(evaluation, monkeypatch, capsys):
    def broken():
        return Settings(_env_file=None, llm_timeout_seconds=-5)

    monkeypatch.setattr(evaluation, "get_settings", broken)
    assert evaluation.main(["--semantic"]) == 2
    out = capsys.readouterr().out
    assert "invalid settings" in out
    assert "-5" not in out