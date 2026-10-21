"""Semantic retrieval: rank resume passages by embedding similarity to a requirement.

It has the same interface as ``LexicalRetriever`` (``for_skill``, ``for_text``, ``chunks``,
``truncated``), so the rest of the pipeline treats both the same way. Passages are the exact
chunks from ``chunk_text``. Vectors live in memory only and are never written to disk.

What the scores mean: cosine similarity between the requirement and the passage, from about
0 (unrelated) to 1 (very close). It is model dependent, it is not a probability, and it is
not a measure of skill. Semantic search always has a nearest passage, so a skill that is not
in the resume can still return passages with low scores. Only passages with a similarity
above ``min_similarity`` (default 0) are returned. For skill requirements each passage also
says how the skill itself is mentioned (affirmative, negated or aspirational), exactly as in
keyword retrieval.

Task prefixes: if the embedder has ``query_prefix`` or ``document_prefix`` attributes, they are
put in front of the text that is embedded (queries and resume chunks respectively). Both are
empty by default. The passages that are shown and checked against the resume never include them.

Privacy: when the embedder is an Ollama server, the resume chunks are sent to it. A resume
with no text sends nothing. Errors never contain resume text.
"""

from collections.abc import Sequence
from typing import Literal

from hr_screening.errors import EmbeddingError, LLMError
from hr_screening.extraction.skills_vocab import SKILL_ALIASES
from hr_screening.rag.chunking import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_CHUNKS,
    DEFAULT_OVERLAP_CHARS,
    chunk_text,
)
from hr_screening.rag.embeddings import (
    Embedder,
    dot,
    normalize_prefix,
    unit_vector,
    validate_vectors,
)
from hr_screening.rag.lexical import Hit
from hr_screening.rag.retrieval import DEFAULT_TOP_K, Passage, passage_from_hit

DEFAULT_BATCH_SIZE = 8  # keeps each reply well under the 1 MiB response limit
DEFAULT_MIN_SIMILARITY = 0.0


def skill_query_text(skill: str) -> str:
    """The skill and its aliases, for example 'scikit-learn, sklearn, scikit learn'."""
    key = skill.strip().lower()
    if not key:
        return ""
    return ", ".join([key, *SKILL_ALIASES.get(key, ())])


class SemanticRetriever:
    """Embeds one resume's chunks once, then answers queries against them."""

    method: Literal["lexical", "semantic"] = "semantic"

    def __init__(
        self,
        resume_text: str,
        embedder: Embedder,
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
        overlap_chars: int = DEFAULT_OVERLAP_CHARS,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        result = chunk_text(
            resume_text, max_chars=max_chars, overlap_chars=overlap_chars, max_chunks=max_chunks
        )
        self.chunks = result.chunks
        self.truncated = result.truncated
        self._embedder = embedder
        self._min_similarity = min_similarity
        # None means "use what the embedder says", so the pipeline needs no extra arguments.
        if query_prefix is None:
            query_prefix = getattr(embedder, "query_prefix", "")
        if document_prefix is None:
            document_prefix = getattr(embedder, "document_prefix", "")
        self._query_prefix = normalize_prefix(query_prefix)
        self._document_prefix = normalize_prefix(document_prefix)
        self._vectors: list[list[float]] = []
        for start in range(0, len(self.chunks), batch_size):
            group = self.chunks[start : start + batch_size]
            batch = [self._document_prefix + chunk.text for chunk in group]
            self._vectors += [unit_vector(vector) for vector in self._embed(batch)]
        self._dimension = len(self._vectors[0]) if self._vectors else 0
        if any(len(vector) != self._dimension for vector in self._vectors):
            raise EmbeddingError("The embedding model returned vectors of different sizes.")

    def for_skill(self, skill: str, top_k: int = DEFAULT_TOP_K) -> list[Passage]:
        """Passages closest in meaning to a skill and its aliases."""
        query = skill_query_text(skill)
        if not query:
            return []
        return [passage_from_hit(hit, skill) for hit in self._search(query, top_k)]

    def for_text(self, text: str, top_k: int = DEFAULT_TOP_K) -> list[Passage]:
        """Passages closest in meaning to ``text`` (a degree, a certification, a duty)."""
        if not text.strip():
            return []
        return [passage_from_hit(hit, None) for hit in self._search(text, top_k)]

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            raw = self._embedder.embed(texts)
        except LLMError:
            raise
        except Exception:
            raise EmbeddingError("The embedding model failed.") from None
        return validate_vectors(raw, len(texts))

    def _search(self, query: str, top_k: int) -> list[Hit]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not self.chunks:
            return []
        vector = unit_vector(self._embed([self._query_prefix + query])[0])
        if len(vector) != self._dimension:
            raise EmbeddingError("The query and passage vectors have different sizes.")
        scored = [(round(dot(vector, v), 9), position) for position, v in enumerate(self._vectors)]
        kept = [(score, position) for score, position in scored if score > self._min_similarity]
        kept.sort(key=lambda item: (-item[0], item[1]))  # equal scores: earlier passage first
        return [Hit(self.chunks[position], score, ()) for score, position in kept[:top_k]]