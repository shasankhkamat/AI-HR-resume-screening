"""A small deterministic lexical index (BM25) over resume chunks. Standard library only.

This is a keyword baseline, not semantic search. It ranks chunks by how well their words match
a query. A high score means "this passage is about the query terms". It does not mean the
candidate has the skill or any level of proficiency.

Determinism: query terms are processed in sorted order and scores are rounded to 9 decimals,
so the ranking does not depend on hash order or on tiny floating point differences. Equal
scores are ordered by chunk position (earlier first).
"""

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from hr_screening.extraction.skills_vocab import SKILL_ALIASES, TRANSFERABLE
from hr_screening.rag.chunking import Chunk

RELATED_WEIGHT = 0.5
_TOKEN = re.compile(r"[^\W_]+(?:[+#]+)?")  # letters and digits, keeping c++ and c#
_STOPWORDS = frozenset(
    {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it"}
    | {"of", "on", "or", "the", "to", "was", "were", "with"}
)


def _stem(token: str) -> str:
    """Drop a plural s so 'dashboards' and 'APIs' match the singular. 'analysis' stays."""
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "sis")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower case words without stop words and single letters. Case and accents are folded."""
    tokens = []
    for match in _TOKEN.finditer(text.casefold()):
        token = match.group(0)
        if len(token) > 1 and token not in _STOPWORDS:
            tokens.append(_stem(token))
    return tokens


def terms(text: str) -> list[str]:
    """Words plus adjacent word pairs, so the phrase 'power bi' scores above scattered words."""
    words = tokenize(text)
    return words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]


@dataclass(frozen=True)
class Query:
    """Weighted terms in sorted order. Related terms weigh less than the requirement itself."""

    terms: tuple[tuple[str, float], ...]


def make_query(
    primary: Iterable[str], related: Iterable[str] = (), *, related_weight: float = RELATED_WEIGHT
) -> Query:
    weights: dict[str, float] = {}
    for phrase in related:
        for term in terms(phrase):
            weights[term] = related_weight
    for phrase in primary:
        for term in terms(phrase):
            weights[term] = 1.0  # the requirement's own words always win over related ones
    return Query(tuple(sorted(weights.items())))


def skill_query(skill: str) -> Query:
    """A skill, its aliases, and (at lower weight) the related skills the evaluator knows."""
    key = skill.strip().lower()
    if not key:
        return Query(())
    related: list[str] = []
    for other in TRANSFERABLE.get(key, ()):
        related += [other, *SKILL_ALIASES.get(other, ())]
    return make_query([key, *SKILL_ALIASES.get(key, ())], related)


def text_query(text: str) -> Query:
    """Every word of ``text`` at full weight."""
    return make_query([text])


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # relevance of the passage to the query, not a measure of skill
    matched_terms: tuple[str, ...]  # query terms found in the chunk, in sorted order


class LexicalIndex:
    """BM25 over a fixed set of chunks."""

    def __init__(self, chunks: Sequence[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = tuple(chunks)
        self._k1 = k1
        self._b = b
        self._frequencies = [Counter(terms(chunk.text)) for chunk in self.chunks]
        self._lengths = [sum(freq.values()) for freq in self._frequencies]
        self._document_frequency: Counter[str] = Counter()
        for freq in self._frequencies:
            self._document_frequency.update(freq.keys())
        count = len(self.chunks)
        self._average_length = sum(self._lengths) / count if count else 0.0

    def search(self, query: Query, top_k: int = 3) -> list[Hit]:
        """Return up to ``top_k`` chunks with a score above zero, best first."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not self.chunks or not query.terms:
            return []
        total = len(self.chunks)
        scored: list[tuple[float, int, tuple[str, ...]]] = []
        for position, freq in enumerate(self._frequencies):
            score = 0.0
            matched: list[str] = []
            for term, weight in query.terms:
                count = freq.get(term, 0)
                if not count:
                    continue
                df = self._document_frequency[term]
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                length_norm = 1 - self._b + self._b * self._lengths[position] / self._average_length
                score += weight * idf * count * (self._k1 + 1) / (count + self._k1 * length_norm)
                matched.append(term)
            if score > 0:
                scored.append((round(score, 9), position, tuple(matched)))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [Hit(self.chunks[pos], score, matched) for score, pos, matched in scored[:top_k]]