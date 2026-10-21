"""Semantic retrieval with deterministic fake embedders: ranking, safety and exactness."""

from pathlib import Path

import pytest

from hr_screening.errors import (
    EmbeddingError,
    LLMConfigError,
    LLMTimeoutError,
)
from hr_screening.extraction.mentions import MentionKind
from hr_screening.models.report import MatchStatus, RequirementAssessment, RequirementKind
from hr_screening.rag.embeddings import FakeEmbedder, cosine_similarity
from hr_screening.rag.requirements import retrieve_for_assessments, verify_traceable
from hr_screening.rag.retrieval import LexicalRetriever
from hr_screening.rag.semantic import (
    DEFAULT_BATCH_SIZE,
    SemanticRetriever,
    skill_query_text,
)

FIXTURE = Path(__file__).parent / "fixtures" / "long_synthetic_resume.txt"
SECRET = "SECRET Jane Doe resume line"


@pytest.fixture(scope="module")
def resume() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def fake(**kwargs) -> FakeEmbedder:
    kwargs.setdefault("dimensions", 4096)
    return FakeEmbedder(**kwargs)


class VectorEmbedder:
    """Returns the vector registered for each text, or a default one."""

    def __init__(self, default, by_text=None) -> None:
        self.default = default
        self.by_text = by_text or {}
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [list(self.by_text.get(text, self.default)) for text in texts]


class ScriptedEmbedder:
    """Answers each call with the next function in the list."""

    def __init__(self, *answers) -> None:
        self.answers = list(answers)

    def embed(self, texts):
        return self.answers.pop(0)(texts)


def good(texts):
    return [[1.0, 0.0] for _ in texts]


# --- what is sent and how often ---------------------------------------------------------------


def test_chunks_are_embedded_in_batches_and_each_query_adds_one_call(resume):
    embedder = fake()
    retriever = SemanticRetriever(resume, embedder, batch_size=4)
    sizes = [len(call) for call in embedder.calls]
    assert sizes == [4, 4, 1]
    assert len(retriever.chunks) == 9
    retriever.for_skill("python")
    retriever.for_text("bachelor degree")
    assert [len(call) for call in embedder.calls[3:]] == [1, 1]


def test_the_default_batch_size_is_eight(resume):
    embedder = fake()
    SemanticRetriever(resume, embedder)
    assert DEFAULT_BATCH_SIZE == 8
    assert [len(call) for call in embedder.calls] == [8, 1]


def test_only_the_chunk_texts_and_the_query_are_sent(resume):
    embedder = fake()
    retriever = SemanticRetriever(resume, embedder)
    sent = [text for call in embedder.calls for text in call]
    assert sent == [chunk.text for chunk in retriever.chunks]
    retriever.for_skill("python")
    assert embedder.calls[-1] == ["python"]


def test_an_empty_resume_sends_nothing(resume):
    embedder = fake()
    retriever = SemanticRetriever("", embedder)
    assert retriever.chunks == ()
    assert retriever.for_skill("python") == []
    assert retriever.for_text("anything") == []
    assert embedder.calls == []


@pytest.mark.parametrize("skill", ["", "   "])
def test_blank_requests_send_nothing(resume, skill):
    embedder = fake()
    retriever = SemanticRetriever(resume, embedder)
    calls_before = len(embedder.calls)
    assert retriever.for_skill(skill) == []
    assert retriever.for_text(skill) == []
    assert len(embedder.calls) == calls_before


# --- passages ---------------------------------------------------------------------------------


def test_passages_are_exact_slices_with_sections_and_bounded_scores(resume):
    retriever = SemanticRetriever(resume, fake())
    for skill in ["python", "sql", "snowflake", "kubernetes"]:
        passages = retriever.for_skill(skill)
        assert 1 <= len(passages) <= 3
        for passage in passages:
            assert resume[passage.start : passage.end] == passage.text
            assert 0 < passage.score <= 1
            assert passage.score == round(passage.score, 9)
            assert passage.matched_terms == ()
    assert retriever.for_skill("python")[0].section in {"projects", "skills", "experience"}


def test_the_default_is_three_passages_and_top_k_can_change(resume):
    retriever = SemanticRetriever(resume, fake())
    assert len(retriever.for_skill("python")) == 3
    assert len(retriever.for_skill("python", top_k=1)) == 1


def test_passages_are_ranked_best_first_and_scores_are_the_cosine(resume):
    embedder = fake()
    retriever = SemanticRetriever(resume, embedder)
    passages = retriever.for_skill("python")
    scores = [p.score for p in passages]
    assert scores == sorted(scores, reverse=True)
    query, *_ = embedder.embed(["python"])
    chunk_vector = embedder.embed([passages[0].text])[0]
    assert passages[0].score == pytest.approx(cosine_similarity(query, chunk_vector), abs=1e-9)


def test_equal_scores_keep_the_original_order(resume):
    retriever = SemanticRetriever(resume, VectorEmbedder([1.0, 0.0]))
    passages = retriever.for_skill("python", top_k=9)
    assert [p.chunk_index for p in passages] == list(range(9))
    assert len({p.score for p in passages}) == 1


def test_passages_found_without_a_shared_word_when_the_embedder_sees_a_link(resume):
    concepts = {"intelligence": "bi", "bi": "bi", "power": "bi"}
    semantic = SemanticRetriever(resume, fake(concepts=concepts))
    top = semantic.for_text("business intelligence")[0]
    assert "Power BI" in top.text
    lexical = LexicalRetriever(resume).for_text("business intelligence")
    assert all("Power BI" not in p.text for p in lexical[:1])


def test_a_skill_that_is_not_there_returns_nothing(resume):
    assert "fastapi" not in resume.lower()
    assert SemanticRetriever(resume, fake()).for_skill("fastapi") == []


def test_skill_passages_say_how_the_skill_is_mentioned(resume):
    retriever = SemanticRetriever(resume, fake())
    assert retriever.for_skill("kubernetes")[0].mention_kinds == (MentionKind.NEGATED,)
    assert retriever.for_skill("spark")[0].mention_kinds == (MentionKind.ASPIRATIONAL,)
    assert set(retriever.for_skill("python")[0].mention_kinds) == {MentionKind.AFFIRMATIVE}


def test_text_queries_carry_no_mention_kinds(resume):
    retriever = SemanticRetriever(resume, fake())
    assert all(p.mention_kinds == () for p in retriever.for_text("python forecasting"))


def test_negative_similarity_is_never_returned(resume):
    embedder = VectorEmbedder([1.0, 0.0], by_text={"python": [-1.0, 0.0]})
    assert SemanticRetriever(resume, embedder).for_skill("python") == []


def test_a_minimum_similarity_removes_weak_passages(resume):
    loose = SemanticRetriever(resume, fake()).for_skill("python", top_k=9)
    strict = SemanticRetriever(resume, fake(), min_similarity=0.3).for_skill("python", top_k=9)
    assert 0 < len(strict) < len(loose)
    assert all(p.score > 0.3 for p in strict)


def test_results_are_the_same_every_time(resume):
    first = SemanticRetriever(resume, fake())
    second = SemanticRetriever(resume, fake())
    assert first.for_skill("python") == second.for_skill("python")
    assert first.for_text("forecasting") == second.for_text("forecasting")


def test_the_chunk_limit_is_passed_through(resume):
    limited = SemanticRetriever(resume, fake(), max_chunks=2)
    assert len(limited.chunks) == 2
    assert limited.truncated is True


def test_it_works_with_the_requirement_mapping(resume):
    retriever = SemanticRetriever(resume, fake())
    items = [
        RequirementAssessment(
            requirement=name,
            kind=kind,
            status=MatchStatus.NOT_FOUND,
            explanation="e",
        )
        for name, kind in [
            ("python", RequirementKind.REQUIRED_SKILL),
            ("Bachelor level education or higher", RequirementKind.EDUCATION),
            ("Example Cloud Practitioner certification", RequirementKind.CERTIFICATION),
            ("3+ years of experience", RequirementKind.EXPERIENCE),
        ]
    ]
    entries = retrieve_for_assessments(retriever, items)
    assert {entry.method for entry in entries} == {"semantic"}
    verify_traceable(entries, resume)
    assert entries[3].passages == []
    assert entries[2].passages[0].section == "certifications"


# --- settings ---------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [0, -1])
def test_unusable_settings_are_rejected(resume, value):
    with pytest.raises(ValueError):
        SemanticRetriever(resume, fake(), batch_size=value)
    with pytest.raises(ValueError):
        SemanticRetriever(resume, fake()).for_skill("python", top_k=value)


def test_skill_query_text_lists_the_skill_and_its_aliases():
    assert skill_query_text("scikit-learn") == "scikit-learn, sklearn, scikit learn"
    assert skill_query_text("  Rust ") == "rust"
    assert skill_query_text("   ") == ""


# --- unusable output and failures -------------------------------------------------------------


def test_the_wrong_number_of_vectors_for_the_chunks_is_an_error(resume):
    def drop_one(texts):
        return [[1.0, 0.0] for _ in texts][:-1]

    with pytest.raises(EmbeddingError, match="wrong number"):
        SemanticRetriever(resume, ScriptedEmbedder(drop_one))


def test_the_wrong_number_of_vectors_for_a_query_is_an_error(resume):
    retriever = SemanticRetriever(resume, ScriptedEmbedder(good, good, lambda texts: []))
    with pytest.raises(EmbeddingError, match="wrong number"):
        retriever.for_skill("python")


def test_a_query_vector_of_another_size_is_an_error(resume):
    embedder = ScriptedEmbedder(good, good, lambda texts: [[1.0, 0.0, 0.0]])
    with pytest.raises(EmbeddingError, match="different sizes"):
        SemanticRetriever(resume, embedder).for_skill("python")


def test_chunk_vectors_of_different_sizes_across_batches_are_an_error(resume):
    def other_size(texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    with pytest.raises(EmbeddingError, match="different sizes"):
        SemanticRetriever(resume, ScriptedEmbedder(good, other_size, good), batch_size=4)


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), "x", None],
    ids=["nan", "inf", "text", "none"],
)
def test_invalid_values_in_the_vectors_are_an_error(resume, bad):
    def broken(texts):
        return [[1.0, bad] for _ in texts]

    with pytest.raises(EmbeddingError):
        SemanticRetriever(resume, ScriptedEmbedder(broken))


def test_an_unexpected_embedder_failure_becomes_a_safe_error(resume):
    embedder = fake(error=RuntimeError(f"{SECRET} sk-key"))
    with pytest.raises(EmbeddingError, match="embedding model failed") as caught:
        SemanticRetriever(resume, embedder)
    assert "SECRET" not in str(caught.value)
    assert "sk-key" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None or caught.value.__suppress_context__


def test_known_errors_from_the_embedder_pass_through_unchanged(resume):
    with pytest.raises(LLMTimeoutError):
        SemanticRetriever(resume, fake(error=LLMTimeoutError("slow")))
    with pytest.raises(LLMConfigError):
        SemanticRetriever(resume, fake(error=LLMConfigError("EMBEDDING_MODEL is not set.")))


def test_error_messages_never_contain_resume_text(resume):
    cases = [
        fake(error=RuntimeError(resume[:200])),
        ScriptedEmbedder(lambda texts: [[1.0, float("nan")] for _ in texts]),
        ScriptedEmbedder(lambda texts: [[1.0, 0.0]]),
    ]
    for embedder in cases:
        with pytest.raises(EmbeddingError) as caught:
            SemanticRetriever(resume, embedder)
        assert "Analytics" not in str(caught.value)
        assert "Example Corp" not in str(caught.value)