"""Tokenizing, query building and BM25 ranking on small hand made corpora."""

import pytest

from hr_screening.rag.chunking import Chunk
from hr_screening.rag.lexical import (
    LexicalIndex,
    Query,
    make_query,
    skill_query,
    terms,
    text_query,
    tokenize,
)


def chunks_of(*texts: str) -> list[Chunk]:
    """Chunks for a tiny corpus. Offsets are made up because the index never reads them."""
    return [Chunk(i, 0, len(t), t, None) for i, t in enumerate(texts)]


def ranked(index: LexicalIndex, query: Query, top_k: int = 3) -> list[int]:
    return [hit.chunk.index for hit in index.search(query, top_k)]


# --- tokenizing ---------------------------------------------------------------------------


def test_tokens_are_lower_case_and_split_on_punctuation():
    expected = ["python", "sql", "power", "bi", "tableau"]
    assert tokenize("Python, SQL; (Power BI)/Tableau.") == expected


def test_stop_words_and_single_letters_are_dropped():
    assert tokenize("The use of a SQL and R in an A/B test") == ["use", "sql", "test"]


def test_symbols_in_c_languages_are_kept():
    assert tokenize("C++ and C# plus C++17") == ["c++", "c#", "plus", "c++", "17"]


def test_plural_s_is_dropped_but_not_from_words_that_end_in_ss_us_or_sis():
    assert tokenize("dashboards APIs databases") == ["dashboard", "api", "database"]
    words = ["analysis", "business", "status", "aws", "sas"]
    assert tokenize("analysis business status aws sas") == words


def test_accents_and_case_are_folded():
    assert tokenize("ÉLÈVE Naïve") == ["élève", "naïve"]


@pytest.mark.parametrize("text", ["", "   ", "...", "a", "the of and"])
def test_text_without_words_gives_no_tokens(text):
    assert tokenize(text) == []
    assert terms(text) == []


def test_terms_add_adjacent_word_pairs():
    assert terms("Power BI dashboards") == ["power", "bi", "dashboard", "power bi", "bi dashboard"]


# --- queries ------------------------------------------------------------------------------


def test_skill_query_includes_aliases():
    weights = dict(skill_query("scikit-learn").terms)
    assert weights["sklearn"] == 1.0
    assert weights["scikit learn"] == 1.0


def test_skill_query_adds_related_skills_at_lower_weight():
    weights = dict(skill_query("tableau").terms)
    assert weights["tableau"] == 1.0
    assert weights["power bi"] == 0.5
    assert weights["looker"] == 0.5


def test_a_requirements_own_words_beat_related_words():
    # powerbi is an alias of power bi, and power bi is related to tableau but also its own name
    weights = dict(skill_query("power bi").terms)
    assert weights["power bi"] == 1.0
    assert weights["tableau"] == 0.5


def test_unknown_skill_is_searched_by_its_own_words():
    assert skill_query("Rust").terms == (("rust", 1.0),)


@pytest.mark.parametrize("skill", ["", "   ", "the"])
def test_blank_skill_gives_an_empty_query(skill):
    assert skill_query(skill).terms == ()


def test_query_terms_are_sorted_and_unique():
    query = make_query(["python python SQL"], ["sql tableau"])
    names = [term for term, _ in query.terms]
    assert names == sorted(set(names))
    assert dict(query.terms)["sql"] == 1.0


def test_text_query_uses_every_word_at_full_weight():
    query = text_query("Bachelor degree")
    assert dict(query.terms) == {"bachelor": 1.0, "degree": 1.0, "bachelor degree": 1.0}


# --- ranking ------------------------------------------------------------------------------


def test_the_best_matching_chunk_comes_first():
    index = LexicalIndex(
        chunks_of(
            "Cooked dinner for friends",
            "Python Python Python",
            "Used Python once in a long sentence about many other unrelated topics",
        )
    )
    assert ranked(index, text_query("python")) == [1, 2]


def test_chunks_without_any_query_term_are_not_returned():
    index = LexicalIndex(chunks_of("alpha beta", "gamma delta"))
    assert ranked(index, text_query("epsilon")) == []
    assert ranked(index, text_query("alpha")) == [0]


def test_matching_ignores_case():
    index = LexicalIndex(chunks_of("Wrote python scripts"))
    assert ranked(index, text_query("PYTHON")) == [0]


def test_a_phrase_match_beats_scattered_words():
    index = LexicalIndex(
        chunks_of("The power plant met bi weekly", "Built Power BI dashboards", "Unrelated text")
    )
    assert ranked(index, skill_query("power bi")) == [1, 0]


def test_a_rare_term_counts_for_more_than_a_common_one():
    corpus = ["python tools"] * 4 + ["python and snowflake loads"]
    index = LexicalIndex(chunks_of(*corpus))
    top = index.search(text_query("python snowflake"), 1)[0]
    assert top.chunk.index == 4


def test_equal_scores_keep_the_original_order():
    index = LexicalIndex(chunks_of("python sql", "python sql", "python sql", "python sql"))
    assert ranked(index, text_query("python"), top_k=4) == [0, 1, 2, 3]


def test_top_k_limits_the_number_of_hits():
    index = LexicalIndex(chunks_of(*["python tools"] * 6))
    assert len(index.search(text_query("python"), 2)) == 2
    assert len(index.search(text_query("python"), 10)) == 6


def test_hits_are_sorted_by_descending_score_and_scores_are_positive():
    index = LexicalIndex(chunks_of("python", "python python tools", "tools only", "python tools"))
    scores = [h.score for h in index.search(text_query("python tools"), 4)]
    assert scores == sorted(scores, reverse=True)
    assert all(score > 0 for score in scores)


def test_hits_list_the_query_terms_they_matched_in_sorted_order():
    index = LexicalIndex(chunks_of("sql and python daily"))
    hit = index.search(text_query("python sql tableau"), 1)[0]
    assert hit.matched_terms == ("python", "sql")


def test_scores_are_stable_between_runs():
    corpus = chunks_of("python sql dashboards", "forecasting with python", "power bi reports")
    first = LexicalIndex(corpus).search(skill_query("python"), 3)
    second = LexicalIndex(corpus).search(skill_query("python"), 3)
    assert first == second
    assert all(h.score == round(h.score, 9) for h in first)


# --- edges ---------------------------------------------------------------------------------


def test_an_empty_index_returns_nothing():
    assert LexicalIndex([]).search(text_query("python")) == []


def test_an_empty_query_returns_nothing():
    index = LexicalIndex(chunks_of("python"))
    assert index.search(Query(())) == []
    assert index.search(text_query("")) == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_top_k_below_one_is_rejected(top_k):
    with pytest.raises(ValueError):
        LexicalIndex(chunks_of("python")).search(text_query("python"), top_k)