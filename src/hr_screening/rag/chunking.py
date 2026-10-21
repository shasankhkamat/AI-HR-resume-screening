"""Split resume text into bounded, overlapping chunks that keep exact source offsets.

Every chunk is an exact slice of the original text: ``chunk.text == text[chunk.start:chunk.end]``.
Nothing is normalized, so a reviewer can always find a chunk in the resume. Chunking is
deterministic: the same text and settings always give the same chunks.

Rules:
- A chunk never crosses a section heading (SKILLS, EXPERIENCE, ...), so its section label is exact.
- A chunk holds at most ``max_chars`` characters and has no leading or trailing whitespace.
- A chunk ends at a line break where possible, else at a space, else at a hard cut (only for
  very long words or lines).
- The next chunk starts up to ``overlap_chars`` before the previous one ended, moved forward to
  a line start or word start. The overlap is therefore at most ``overlap_chars`` and can be 0.
- At most ``max_chunks`` chunks are produced. If text remains, ``truncated`` is True and the
  rest of the resume is not indexed.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from hr_screening.extraction.resume_parser import SectionIndex

DEFAULT_MAX_CHARS = 500
DEFAULT_OVERLAP_CHARS = 100
DEFAULT_MAX_CHUNKS = 200
MIN_MAX_CHARS = 20


@dataclass(frozen=True)
class Chunk:
    """A passage of the resume. ``text`` is exactly ``resume_text[start:end]``."""

    index: int
    start: int
    end: int
    text: str
    section: str | None  # 'skills', 'experience', ... or None when no heading applies


@dataclass(frozen=True)
class ChunkingResult:
    chunks: tuple[Chunk, ...]
    truncated: bool  # True if the chunk limit was reached and some text was not chunked


def chunk_text(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> ChunkingResult:
    """Chunk ``text``. Raises ValueError for unusable settings (a programming error)."""
    if max_chars < MIN_MAX_CHARS:
        raise ValueError(f"max_chars must be at least {MIN_MAX_CHARS}.")
    if not 0 <= overlap_chars < max_chars:
        raise ValueError("overlap_chars must be at least 0 and less than max_chars.")
    if max_chunks < 1:
        raise ValueError("max_chunks must be at least 1.")

    chunks: list[Chunk] = []
    for section, region_start, region_end in _regions(text):
        for start, end in _region_spans(text, region_start, region_end, max_chars, overlap_chars):
            if len(chunks) == max_chunks:
                return ChunkingResult(tuple(chunks), truncated=True)
            chunks.append(Chunk(len(chunks), start, end, text[start:end], section))
    return ChunkingResult(tuple(chunks), truncated=False)


def _regions(text: str) -> Iterator[tuple[str | None, int, int]]:
    """Yield (section, start, end) for each run of lines that share a section."""
    sections = SectionIndex(text)
    region_start = 0
    current = sections.at(0)
    offset = 0
    for line in text.splitlines(keepends=True):
        if offset > 0:
            section = sections.at(offset)
            if section != current:
                yield current, region_start, offset
                region_start, current = offset, section
        offset += len(line)
    yield current, region_start, len(text)


def _skip_whitespace(text: str, position: int, limit: int) -> int:
    while position < limit and text[position].isspace():
        position += 1
    return position


def _region_spans(
    text: str, region_start: int, region_end: int, max_chars: int, overlap_chars: int
) -> Iterator[tuple[int, int]]:
    start = _skip_whitespace(text, region_start, region_end)
    while start < region_end:
        limit = min(start + max_chars, region_end)
        end = region_end if limit == region_end else _choose_end(text, start, limit)
        trimmed_end = end
        while trimmed_end > start and text[trimmed_end - 1].isspace():
            trimmed_end -= 1
        yield start, trimmed_end
        if end >= region_end:
            return
        start = _next_start(text, start, end, region_end, overlap_chars)


def _choose_end(text: str, start: int, limit: int) -> int:
    """Pick where a chunk that would run past ``limit`` should stop. Always > start."""
    half = start + (limit - start) // 2
    newline = text.rfind("\n", half, limit)
    if newline != -1:
        return newline + 1
    end = limit
    while end > half and not text[end - 1].isspace():
        end -= 1
    return end if end > half else limit


def _next_start(text: str, start: int, end: int, region_end: int, overlap_chars: int) -> int:
    """Where the next chunk begins. Always greater than ``start``."""
    candidate = end
    if overlap_chars > 0 and end - start > overlap_chars:
        candidate = end - overlap_chars
        newline = text.find("\n", candidate, end)
        if newline != -1 and newline + 1 < end:
            candidate = newline + 1
        elif not text[candidate - 1].isspace() and not text[candidate].isspace():
            while candidate < end and not text[candidate].isspace():
                candidate += 1
    return _skip_whitespace(text, candidate, region_end)
    