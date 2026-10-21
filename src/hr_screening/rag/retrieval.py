"""Find the resume passages that relate to a requirement. Retrieval only: no verdicts.

A passage is returned because its words match the requirement, not because the candidate has
the skill. A passage may say "no experience with Kubernetes", "currently learning Spark" or
appear only under PROJECTS, and it is still returned so that a human can read it. For skill
requirements each passage carries ``mention_kinds``, the way the skill itself is mentioned
(affirmative, negated or aspirational), using the same rules as the evaluator. Even an
affirmative mention does not show depth, recency or proficiency. Statuses and the coverage
score are decided elsewhere and are not changed by anything in this module.
"""

from dataclasses import dataclass

from hr_screening.extraction.mentions import MentionKind, find_skill_mentions
from hr_screening.rag.chunking import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_CHUNKS,
    DEFAULT_OVERLAP_CHARS,
    chunk_text,
)
from hr_screening.rag.lexical import Hit, LexicalIndex, skill_query, text_query

DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class Passage:
    """An exact slice of the resume that relates to a requirement.

    ``text == resume_text[start:end]``. ``score`` ranks passages for one query. It is not a
    measure of skill. ``matched_terms`` are normalized index terms (lower case, plural
    stripped, so 'kubernetes' shows as 'kubernete'), listed to explain why the passage was
    retrieved. ``mention_kinds`` is empty for text queries, and for skill queries when the
    passage was retrieved only through aliases or related skills.
    """

    chunk_index: int
    start: int
    end: int
    text: str
    section: str | None
    score: float
    matched_terms: tuple[str, ...]
    mention_kinds: tuple[MentionKind, ...] = ()


class LexicalRetriever:
    """Chunks one resume once and answers queries against it. Everything stays in memory."""

    def __init__(
        self,
        resume_text: str,
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
        overlap_chars: int = DEFAULT_OVERLAP_CHARS,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
    ) -> None:
        result = chunk_text(
            resume_text, max_chars=max_chars, overlap_chars=overlap_chars, max_chunks=max_chunks
        )
        self.chunks = result.chunks
        self.truncated = result.truncated
        self._index = LexicalIndex(self.chunks)

    def for_skill(self, skill: str, top_k: int = DEFAULT_TOP_K) -> list[Passage]:
        """Passages about a skill, its aliases and related skills."""
        hits = self._index.search(skill_query(skill), top_k)
        return [_passage(hit, skill) for hit in hits]

    def for_text(self, text: str, top_k: int = DEFAULT_TOP_K) -> list[Passage]:
        """Passages that share words with ``text`` (a degree, a certification, a duty)."""
        return [_passage(hit, None) for hit in self._index.search(text_query(text), top_k)]


def _passage(hit: Hit, skill: str | None) -> Passage:
    chunk = hit.chunk
    kinds = tuple(m.kind for m in find_skill_mentions(chunk.text, skill)) if skill else ()
    return Passage(
        chunk_index=chunk.index,
        start=chunk.start,
        end=chunk.end,
        text=chunk.text,
        section=chunk.section,
        score=hit.score,
        matched_terms=hit.matched_terms,
        mention_kinds=kinds,
    )