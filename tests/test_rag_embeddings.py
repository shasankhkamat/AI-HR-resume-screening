"""The embedding interface: vector checks, cosine similarity and the deterministic fake."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hr_screening.errors import EmbeddingError
from hr_screening.rag.embeddings import (
    FakeEmbedder,
    cosine_similarity,
    dot,
    unit_vector,
    validate_vectors,
)

SRC = Path(__file__).resolve().parents[1] / "src"


# --- validate_vectors -------------------------------------------------------------------------


def test_valid_vectors_come_back_as_floats():
    assert validate_vectors([[1, 2], (3, 4.5)], 2) == [[1.0, 2.0], [3.0, 4.5]]


def test_no_vectors_are_fine_when_none_are_expected():
    assert validate_vectors([], 0) == []


@pytest.mark.parametrize("value", [None, "abc", 5, {"a": 1}, b"xy"], ids=lambda v: type(v).__name__)
def test_something_that_is_not_a_list_is_rejected(value):
    with pytest.raises(EmbeddingError, match="list of vectors"):
        validate_vectors(value, 1)


@pytest.mark.parametrize("vectors", [[[1.0]], [], [[1.0], [2.0]]], ids=["short", "none", "extra"])
def test_the_wrong_number_of_vectors_is_rejected(vectors):
    expected = 2 if len(vectors) == 1 else 1
    with pytest.raises(EmbeddingError, match="wrong number"):
        validate_vectors(vectors, expected)


@pytest.mark.parametrize(
    "vector",
    [[], None, "ab", 5, [1.0, "x"], [1.0, None], [True, False], [1.0, [2.0]]],
    ids=["empty", "none", "string", "number", "text_value", "none_value", "booleans", "nested"],
)
def test_empty_or_non_numeric_vectors_are_rejected(vector):
    with pytest.raises(EmbeddingError):
        validate_vectors([vector], 1)


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), 1e7, -1e7, 10**400],
    ids=["nan", "inf", "minus_inf", "huge", "minus_huge", "overflow"],
)
def test_invalid_numbers_are_rejected(value):
    with pytest.raises(EmbeddingError, match="invalid number"):
        validate_vectors([[0.5, value]], 1)


def test_vectors_of_different_sizes_are_rejected():
    with pytest.raises(EmbeddingError, match="different sizes"):
        validate_vectors([[1.0, 2.0], [1.0]], 2)


@pytest.mark.parametrize(
    ("vectors", "expected"),
    [
        ("SECRET resume text", 1),
        (["SECRET resume text"], 1),
        ([["SECRET resume text"]], 1),
        ([[1.0, "SECRET resume text"]], 1),
        ([["SECRET resume text", 1.0], [1.0]], 2),
        ([["SECRET resume text"]], 3),
        ({"SECRET resume text": [1.0]}, 1),
    ],
    ids=["not_a_list", "string_vector", "text_value", "mixed_value", "ragged", "count", "mapping"],
)
def test_error_messages_never_echo_the_values(vectors, expected):
    with pytest.raises(EmbeddingError) as caught:
        validate_vectors(vectors, expected)
    assert "SECRET" not in str(caught.value)
    assert "resume" not in str(caught.value)


# --- cosine similarity and helpers ------------------------------------------------------------


def test_cosine_of_known_pairs():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine_similarity([1, 1], [1, 0]) == pytest.approx(math.sqrt(0.5))


def test_cosine_ignores_length_and_order_of_arguments():
    assert cosine_similarity([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert cosine_similarity([1, 2, 3], [3, 1, 2]) == cosine_similarity([3, 1, 2], [1, 2, 3])


def test_cosine_stays_within_bounds_for_large_vectors():
    result = cosine_similarity([1e5] * 10, [1e5] * 10)
    assert -1.0 <= result <= 1.0
    assert result == pytest.approx(1.0)


def test_a_vector_of_zeros_is_similar_to_nothing():
    assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0
    assert cosine_similarity([0, 0], [0, 0]) == 0.0


@pytest.mark.parametrize(
    ("left", "right"),
    [([1, 2], [1, 2, 3]), ([], [1.0]), ([1.0], []), ([float("nan")], [1.0]), ([1.0], ["a"])],
    ids=["sizes", "empty_left", "empty_right", "nan", "text"],
)
def test_cosine_rejects_unusable_vectors(left, right):
    with pytest.raises(EmbeddingError):
        cosine_similarity(left, right)


def test_unit_vectors_have_length_one_and_zeros_stay_zeros():
    assert math.hypot(*unit_vector([3.0, 4.0])) == pytest.approx(1.0)
    assert unit_vector([0.0, 0.0]) == [0.0, 0.0]
    original = [3.0, 4.0]
    unit_vector(original)
    assert original == [3.0, 4.0]


def test_dot_needs_equal_sizes():
    assert dot([1.0, 0.0], [0.5, 0.5]) == 0.5
    with pytest.raises(EmbeddingError):
        dot([1.0], [1.0, 2.0])


# --- the fake embedder ------------------------------------------------------------------------


def test_the_fake_gives_one_valid_vector_per_text_in_order():
    fake = FakeEmbedder(dimensions=32)
    vectors = fake.embed(["python sql", "forecasting", "python sql"])
    assert validate_vectors(vectors, 3)
    assert all(len(v) == 32 for v in vectors)
    assert vectors[0] == vectors[2]
    assert vectors[0] != vectors[1]


def test_the_fake_is_the_same_every_time_and_in_every_instance():
    first = FakeEmbedder(dimensions=64).embed(["Built dashboards in Power BI"])
    second = FakeEmbedder(dimensions=64).embed(["Built dashboards in Power BI"])
    assert first == second


def test_texts_that_share_words_are_more_similar():
    fake = FakeEmbedder(dimensions=256)
    base, near, far = fake.embed(["python sql dashboards", "python sql reports", "cooking dinner"])
    assert cosine_similarity(base, near) > cosine_similarity(base, far)
    assert cosine_similarity(base, far) == pytest.approx(0.0)


def test_concepts_make_different_words_similar():
    plain = FakeEmbedder(dimensions=256)
    smart = FakeEmbedder(dimensions=256, concepts={"tableau": "bi", "looker": "bi"})
    assert cosine_similarity(*plain.embed(["tableau", "looker"])) == pytest.approx(0.0)
    assert cosine_similarity(*smart.embed(["tableau", "looker"])) == pytest.approx(1.0)


@pytest.mark.parametrize("text", ["", "   ", "the of and"])
def test_text_without_words_gives_a_zero_vector(text):
    vector = FakeEmbedder(dimensions=16).embed([text])[0]
    assert vector == [0.0] * 16
    assert cosine_similarity(vector, [1.0] * 16) == 0.0


def test_the_fake_records_calls_and_can_be_told_to_fail():
    fake = FakeEmbedder(error=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        fake.embed(["a", "b"])
    assert fake.calls == [["a", "b"]]


def test_the_fake_needs_at_least_one_dimension():
    with pytest.raises(ValueError):
        FakeEmbedder(dimensions=0)


def test_the_fake_does_not_depend_on_python_hash_randomization():
    script = (
        "import json\n"
        "from hr_screening.rag.embeddings import FakeEmbedder\n"
        "print(json.dumps(FakeEmbedder(64).embed(['python and sql dashboards', 'power bi'])))\n"
    )
    outputs = []
    for seed in ("1", "2", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(SRC)}
        run = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
        )
        outputs.append(json.loads(run.stdout))
    assert outputs[0] == outputs[1] == outputs[2]