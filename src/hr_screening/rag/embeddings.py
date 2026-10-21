"""The embedding interface, vector checks, cosine similarity and a deterministic fake.

An embedder turns texts into vectors so that passages and requirements can be compared by
meaning instead of by shared words. The contract is strict because the numbers come from a
model: ``embed(texts)`` must return one finite, non empty vector per text, in the same order,
all with the same number of values. ``validate_vectors`` enforces that for any embedder.

Errors use fixed messages and never contain the texts that were embedded.
"""

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Protocol

from hr_screening.errors import EmbeddingError
from hr_screening.rag.lexical import tokenize

MAX_ABS_VALUE = 1e6  # real embedding values are small. Anything larger is treated as invalid.


class Embedder(Protocol):
    """Converts texts to vectors. Implementations must not log or expose the texts."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def validate_vectors(vectors: object, expected_count: int) -> list[list[float]]:
    """Return the vectors as lists of floats, or raise EmbeddingError.

    Rejected: not a list, wrong number of vectors, an empty or non list vector, values that
    are not numbers (booleans included), NaN, infinity, absurdly large values, and vectors of
    different sizes.
    """
    if not isinstance(vectors, list | tuple):
        raise EmbeddingError("The embedding model did not return a list of vectors.")
    if len(vectors) != expected_count:
        raise EmbeddingError("The embedding model returned the wrong number of vectors.")
    checked: list[list[float]] = []
    dimension: int | None = None
    for vector in vectors:
        if not isinstance(vector, list | tuple) or not vector:
            raise EmbeddingError("The embedding model returned an empty or invalid vector.")
        numbers = [_to_float(value) for value in vector]
        if dimension is None:
            dimension = len(numbers)
        elif len(numbers) != dimension:
            raise EmbeddingError("The embedding model returned vectors of different sizes.")
        checked.append(numbers)
    return checked


def _to_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EmbeddingError("The embedding model returned a value that is not a number.")
    try:
        number = float(value)
    except OverflowError:
        raise EmbeddingError("The embedding model returned an invalid number.") from None
    if not math.isfinite(number) or abs(number) > MAX_ABS_VALUE:
        raise EmbeddingError("The embedding model returned an invalid number.")
    return number


def unit_vector(vector: Sequence[float]) -> list[float]:
    """Scale to length 1. A vector of zeros stays zeros (it is similar to nothing)."""
    norm = math.hypot(*vector)
    if norm == 0.0:
        return [0.0] * len(vector)
    return [value / norm for value in vector]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Dot product, clamped to [-1, 1] for unit vectors. Sizes must match."""
    if len(left) != len(right):
        raise EmbeddingError("The vectors have different sizes.")
    return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(left, right, strict=True))))


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine of the angle between two vectors: 1 same direction, 0 unrelated, -1 opposite.

    Raises EmbeddingError for empty vectors, invalid values or different sizes. If either
    vector is all zeros the result is 0.0.
    """
    first = validate_vectors([list(left)], 1)[0]
    second = validate_vectors([list(right)], 1)[0]
    if len(first) != len(second):
        raise EmbeddingError("The vectors have different sizes.")
    return dot(unit_vector(first), unit_vector(second))


class FakeEmbedder:
    """A deterministic embedder for tests and offline demos. It needs no model and no network.

    Each word is hashed (SHA-256, so the result does not depend on Python's hash seed) to a
    position in the vector and counted. Texts that share words are similar. ``concepts`` maps
    words to a shared label, so words that mean the same thing can be made similar to show
    what semantic search adds over keyword search. Two different words can land on the same
    position (a hash collision), so use many dimensions when a test needs exact separation.
    """

    model = "fake-embedder"

    def __init__(
        self,
        dimensions: int = 1024,
        concepts: Mapping[str, str] | None = None,
        error: Exception | None = None,
    ) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be at least 1.")
        self.dimensions = dimensions
        self.error = error
        self.calls: list[list[str]] = []
        self._concepts: dict[str, str] = {}
        for word, label in (concepts or {}).items():
            tokens = tokenize(word)
            if len(tokens) == 1:
                self._concepts[tokens[0]] = label

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.error is not None:
            raise self.error
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            label = self._concepts.get(token, token)
            digest = hashlib.sha256(label.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dimensions] += 1.0
        return vector