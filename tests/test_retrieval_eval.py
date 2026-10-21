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


def passage(text, mentions=(), score=0.5, section="experience", start=0):
    return SimpleNamespace(
        text=text,
        mention_kinds=list(mentions),
        score=score,
        section=section,
        start=start,
        end=start + len(text),
    )


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


def test_scoring_records_the_top_score_and_the_gold_passage_score(evaluation):
    gold = query(["needle"])
    row = evaluation.score_query(
        gold, [passage("other", score=0.8), passage("a needle", score=0.6), passage("x", score=0.1)]
    )
    assert (row["top_score"], row["gold_score"], row["rank"]) == (0.8, 0.6, 2)
    miss = evaluation.score_query(gold, [passage("other", score=0.8)])
    assert (miss["top_score"], miss["gold_score"]) == (0.8, None)
    empty = evaluation.score_query(query(None), [])
    assert (empty["top_score"], empty["gold_score"]) == (None, None)


# --- running ---------------------------------------------------------------------------------


def test_the_documented_keyword_results_still_hold(evaluation):
    rows = evaluation.run(LexicalRetriever)
    answerable = [r for r in rows if r["answerable"]]
    misses = {r["id"] for r in answerable if r["rank"] is None}
    assert len(rows) == 40
    assert len(answerable) == 31
    assert sum(r["rank"] == 1 for r in answerable) == 28
    assert misses == {"marketing-abtesting", "cloud-cicd", "cloud-monitoring"}
    assert all(r["returned"] == 0 for r in rows if not r["answerable"])
    assert all(r["mention_ok"] for r in rows if r["mention_ok"] is not None)


def test_the_semantic_path_runs_with_a_fake_embedder(evaluation):
    embedder = FakeEmbedder(dimensions=4096)
    rows = evaluation.run(lambda text: SemanticRetriever(text, embedder))
    assert len(rows) == 40
    assert all(row["returned"] <= 3 for row in rows)


def test_the_report_prints_every_query_and_a_summary(evaluation, capsys):
    evaluation.print_report(evaluation.run(LexicalRetriever), "keyword search (lexical)")
    out = capsys.readouterr().out
    assert "40 queries" in out
    assert "analyst-python" in out
    assert "Absent skills: nothing returned for 9 of 9." in out
    assert "Mention labels correct: 4 of 4." in out


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


# --- score diagnostics (--scores) ----------------------------------------------------------------


def row(id, answerable, top, gold=None):
    return {"id": id, "answerable": answerable, "top_score": top, "gold_score": gold,
            "type": "absent" if not answerable else "exact", "returned": 0 if top is None else 1}


SWEEP_ROWS = [
    row("absent-a", False, 0.50),
    row("absent-b", False, 0.30),
    row("absent-c", False, None),
    row("hit-high", True, 0.80, 0.80),
    row("hit-low", True, 0.40, 0.40),
    row("never-found", True, 0.60, None),
]


def sweep_at(evaluation, cutoff):
    return evaluation.threshold_sweep(SWEEP_ROWS, (cutoff,))[0]


def test_the_sweep_counts_absent_queries_that_become_empty(evaluation):
    assert sweep_at(evaluation, 0.0)["absent_empty"] == 1  # only the query with no passages
    assert sweep_at(evaluation, 0.30)["absent_empty"] == 2  # a score equal to the cutoff is dropped
    assert sweep_at(evaluation, 0.50)["absent_empty"] == 3
    assert sweep_at(evaluation, 0.50)["absent_total"] == 3


def test_the_sweep_counts_gold_passages_that_survive(evaluation):
    assert sweep_at(evaluation, 0.0)["gold_kept"] == 2
    assert sweep_at(evaluation, 0.40)["gold_kept"] == 1  # equal to the cutoff is dropped
    assert sweep_at(evaluation, 0.80)["gold_kept"] == 0
    assert sweep_at(evaluation, 0.0)["already_missed"] == 1
    assert sweep_at(evaluation, 0.0)["answerable_total"] == 3


def test_the_default_sweep_runs_from_zero_to_just_under_one(evaluation):
    cutoffs = [line["threshold"] for line in evaluation.threshold_sweep(SWEEP_ROWS)]
    assert cutoffs[0] == 0.0 and cutoffs[-1] == 0.95 and len(cutoffs) == 20


def test_the_score_printout_names_the_extreme_queries_and_the_sweep(evaluation, capsys):
    evaluation.print_scores(SWEEP_ROWS)
    out = capsys.readouterr().out
    assert "Highest top score, absent skill: 0.500 (absent-a)" in out
    assert "Lowest gold passage score:       0.400 (hit-low)" in out
    assert "The two ranges overlap" in out
    assert "Threshold sweep" in out
    assert "hit-high" in out
    assert "do not prove a safe cutoff" in out


def test_the_printout_says_when_a_cutoff_would_separate_the_run(evaluation, capsys):
    rows = [row("absent-a", False, 0.30), row("hit", True, 0.60, 0.60)]
    evaluation.print_scores(rows)
    assert "A cutoff between those two would separate this run." in capsys.readouterr().out


def test_scores_are_printed_only_when_asked_for(evaluation, monkeypatch, capsys):
    monkeypatch.setattr(evaluation, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(evaluation, "build_embedder", lambda settings: FakeEmbedder(4096))
    assert evaluation.main(["--semantic"]) == 0
    plain = capsys.readouterr().out
    assert "Score diagnostics" not in plain
    assert "Threshold sweep" not in plain
    assert evaluation.main(["--semantic", "--scores"]) == 0
    with_scores = capsys.readouterr().out
    assert with_scores.startswith(plain)  # the existing report is printed first and unchanged
    assert "Score diagnostics" in with_scores
    assert "Threshold sweep" in with_scores


def test_the_default_report_is_unchanged_by_the_score_fields(evaluation, capsys):
    assert evaluation.main([]) == 0
    out = capsys.readouterr().out
    assert "Score diagnostics" not in out
    assert "Absent skills: nothing returned for 9 of 9." in out


def test_scores_need_semantic_search(evaluation, capsys):
    with pytest.raises(SystemExit) as stop:
        evaluation.main(["--scores"])
    assert stop.value.code == 2
    assert "--scores needs --semantic" in capsys.readouterr().err


# --- where the scores came from (the trace) ------------------------------------------------------


def traced_row(id, type, rank, passages):
    """A row as score_query would make it, with the passages that were returned."""
    return {
        "id": id, "type": type, "answerable": type != "absent", "rank": rank,
        "top_score": passages[0]["score"] if passages else None,
        "gold_score": None, "top_scores": [p["score"] for p in passages],
        "trace": passages, "returned": len(passages),
    }


LONG_PASSAGE = {
    "score": 0.42, "section": "experience", "start": 10, "end": 400,
    "text": "Wrote SQL against PostgreSQL to reconcile billing data " * 6,
}


def test_scoring_keeps_the_returned_passages_for_the_trace(evaluation):
    row = evaluation.score_query(
        query(None), [passage("first", score=0.4, section="skills", start=5), passage("second")]
    )
    assert row["top_scores"] == [0.4, 0.5]
    assert row["trace"][0] == {
        "score": 0.4, "section": "skills", "start": 5, "end": 10, "text": "first"
    }
    assert len(row["trace"]) == 2


def test_the_trace_shows_the_passages_behind_absent_skills(evaluation, capsys):
    absent = traced_row("absent-a", "absent", None, [LONG_PASSAGE])
    evaluation.print_traces([absent])
    out = capsys.readouterr().out
    assert "Passages returned for absent skills" in out
    assert "absent-a (absent)" in out
    assert "1. 0.420  [experience, chars 10-400]" in out
    assert "Wrote SQL against PostgreSQL" in out
    assert out.count("\n") < 8  # a long passage is shortened, not printed in full
    assert "..." in out


def test_the_trace_includes_answerable_queries_that_were_not_first_and_skips_the_rest(
    evaluation, capsys
):
    one = {"score": 0.6, "section": None, "start": 0, "end": 3, "text": "abc"}
    rows = [
        traced_row("first", "exact", 1, [one]),
        traced_row("second", "exact", 2, [one]),
        traced_row("missed", "paraphrase", None, [one]),
    ]
    evaluation.print_traces(rows)
    out = capsys.readouterr().out
    assert "second (exact)" in out and "missed (paraphrase)" in out
    assert "first (exact)" not in out
    assert "no section" in out


def test_the_trace_prints_nothing_when_there_is_nothing_to_explain(evaluation, capsys):
    one = {"score": 0.6, "section": None, "start": 0, "end": 3, "text": "abc"}
    evaluation.print_traces([traced_row("first", "exact", 1, [one])])
    assert capsys.readouterr().out == ""


def test_the_score_table_lists_the_top_three_scores(evaluation, capsys):
    passages = [{"score": s, "section": None, "start": 0, "end": 1, "text": "x"}
                for s in (0.42, 0.401, 0.39)]
    evaluation.print_scores([traced_row("absent-a", "absent", None, passages)])
    out = capsys.readouterr().out
    assert "top 3 scores" in out
    assert "0.420 0.401 0.390" in out


def test_the_run_label_names_the_prefixes_when_the_embedder_has_them(
    evaluation, monkeypatch, capsys
):
    fake = FakeEmbedder(dimensions=4096)
    fake.query_prefix = "search_query: "
    fake.document_prefix = "search_document: "
    monkeypatch.setattr(evaluation, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(evaluation, "build_embedder", lambda settings: fake)
    assert evaluation.main(["--semantic"]) == 0
    out = capsys.readouterr().out
    assert "query prefix 'search_query:'" in out
    assert "document prefix 'search_document:'" in out


def test_the_run_label_has_no_prefix_text_without_prefixes(evaluation, monkeypatch, capsys):
    monkeypatch.setattr(evaluation, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(evaluation, "build_embedder", lambda settings: FakeEmbedder(4096))
    assert evaluation.main(["--semantic"]) == 0
    assert "prefix" not in capsys.readouterr().out


def test_the_expanded_cases_cover_absent_skills_next_to_skills_the_resume_has(evaluation):
    rows = evaluation.run(LexicalRetriever)
    by_type = {}
    for row in rows:
        by_type.setdefault(row["type"], []).append(row["id"])
    assert len(by_type["absent"]) == 9
    for needed in ("cloud-terraform", "analyst-tensorflow", "engineer-pytorch"):
        assert needed in by_type["absent"]
    assert {"cloud-gcp", "marketing-tableau"} <= set(by_type["related"])
    assert {"cloud-azure"} <= set(by_type["negated"])
    assert {"cloud-kubernetes"} <= set(by_type["aspirational"])