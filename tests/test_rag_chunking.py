"""Chunking: exact offsets, bounds, determinism, section boundaries and the chunk limit."""

from pathlib import Path

import pytest

from hr_screening.extraction.resume_parser import SectionIndex
from hr_screening.rag.chunking import (
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_CHUNKS,
    DEFAULT_OVERLAP_CHARS,
    chunk_text,
)

FIXTURE = Path(__file__).parent / "fixtures" / "long_synthetic_resume.txt"
WORDS = " ".join(f"word{i}" for i in range(400))  # one long line, about 2,700 characters
LINES = "".join(f"- bullet number {i} about a task\n" for i in range(60))


@pytest.fixture(scope="module")
def resume() -> str:
    return FIXTURE.read_text(encoding="utf-8")


SAMPLES = {
    "fixture": FIXTURE.read_text(encoding="utf-8"),
    "one_long_line": WORDS,
    "many_lines": LINES,
    "crlf": "SKILLS\r\nPython, SQL\r\nEXPERIENCE\r\n" + LINES.replace("\n", "\r\n"),
    "unicode": "SKILLS\nPython, SQL\nÉlève naïve 日本語のスキル " * 40,
    "padded": "   leading spaces\n\n\n  double  spaced   words here  \n" * 30,
}


def assert_exact(text: str, chunks) -> None:
    for chunk in chunks:
        assert chunk.text == text[chunk.start : chunk.end]
        assert chunk.start < chunk.end
        assert chunk.text == chunk.text.strip()
        assert chunk.text


# --- empty and tiny input ----------------------------------------------------------------


@pytest.mark.parametrize(
    "text", ["", " ", "   \n\n\t  ", "\r\n"], ids=["empty", "space", "blank", "crlf"]
)
def test_blank_text_gives_no_chunks(text):
    result = chunk_text(text)
    assert result.chunks == ()
    assert result.truncated is False


def test_short_text_is_one_chunk_with_exact_offsets():
    text = "  Hello world \n"
    result = chunk_text(text)
    assert len(result.chunks) == 1
    chunk = result.chunks[0]
    assert (chunk.start, chunk.end, chunk.text) == (2, 13, "Hello world")
    assert text[chunk.start : chunk.end] == chunk.text
    assert chunk.index == 0


# --- exactness, bounds and coverage ------------------------------------------------------


@pytest.mark.parametrize("name", SAMPLES)
@pytest.mark.parametrize("max_chars", [40, 120, DEFAULT_MAX_CHARS])
def test_every_chunk_is_an_exact_slice_within_the_size_limit(name, max_chars):
    text = SAMPLES[name]
    result = chunk_text(text, max_chars=max_chars, overlap_chars=max_chars // 5)
    assert result.chunks
    assert_exact(text, result.chunks)
    assert all(len(c.text) <= max_chars for c in result.chunks)


@pytest.mark.parametrize("name", SAMPLES)
def test_every_non_whitespace_character_is_covered(name):
    text = SAMPLES[name]
    result = chunk_text(text, max_chars=120, overlap_chars=30)
    assert result.truncated is False
    covered = [False] * len(text)
    for chunk in result.chunks:
        for position in range(chunk.start, chunk.end):
            covered[position] = True
    missing = [i for i, ch in enumerate(text) if not ch.isspace() and not covered[i]]
    assert missing == []


@pytest.mark.parametrize("name", SAMPLES)
def test_chunks_move_forward_and_are_numbered_in_order(name):
    chunks = chunk_text(SAMPLES[name], max_chars=100, overlap_chars=40).chunks
    assert [c.index for c in chunks] == list(range(len(chunks)))
    for earlier, later in zip(chunks, chunks[1:], strict=False):
        assert later.start > earlier.start
        assert later.end > earlier.end


def test_offsets_refer_to_the_original_text_not_a_cleaned_copy():
    text = "  Built   dashboards\n\n   in  Power BI  "
    chunks = chunk_text(text).chunks
    assert [c.text for c in chunks] == ["Built   dashboards\n\n   in  Power BI"]
    assert text[chunks[0].start : chunks[0].end] == chunks[0].text


def test_chunking_is_deterministic(resume):
    first = chunk_text(resume)
    second = chunk_text(resume)
    assert first == second
    assert first.chunks == chunk_text(resume, max_chars=DEFAULT_MAX_CHARS).chunks


def test_fixture_is_long_enough_to_need_many_chunks(resume):
    result = chunk_text(resume)
    assert len(resume) > 2000
    assert len(result.chunks) >= 6
    assert result.truncated is False


# --- overlap -------------------------------------------------------------------------------


def test_overlap_is_never_more_than_requested_and_happens_on_the_fixture(resume):
    result = chunk_text(resume, max_chars=DEFAULT_MAX_CHARS, overlap_chars=DEFAULT_OVERLAP_CHARS)
    chunks = result.chunks
    overlaps = []
    for earlier, later in zip(chunks, chunks[1:], strict=False):
        if earlier.section == later.section:
            overlaps.append(earlier.end - later.start)
    assert overlaps
    assert max(overlaps) <= DEFAULT_OVERLAP_CHARS
    assert any(size > 0 for size in overlaps)


def test_zero_overlap_gives_disjoint_chunks():
    chunks = chunk_text(LINES, max_chars=120, overlap_chars=0).chunks
    for earlier, later in zip(chunks, chunks[1:], strict=False):
        assert later.start >= earlier.end


# --- where chunks start and stop ---------------------------------------------------------


def test_chunks_start_and_stop_on_word_boundaries_when_a_space_is_available():
    chunks = chunk_text(WORDS, max_chars=200, overlap_chars=50).chunks
    assert len(chunks) > 5
    for chunk in chunks:
        assert chunk.start == 0 or WORDS[chunk.start - 1].isspace()
        assert chunk.end == len(WORDS) or WORDS[chunk.end].isspace()


def test_chunks_start_on_line_boundaries_when_lines_are_short():
    chunks = chunk_text(LINES, max_chars=200, overlap_chars=80).chunks
    assert len(chunks) > 5
    for chunk in chunks:
        assert chunk.start == 0 or LINES[chunk.start - 1] == "\n"


def test_a_very_long_word_is_cut_hard_but_still_exact_and_complete():
    text = "x" * 1300
    chunks = chunk_text(text, max_chars=500, overlap_chars=100).chunks
    assert_exact(text, chunks)
    assert all(len(c.text) <= 500 for c in chunks)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    assert len(chunks) == 3


def test_a_long_line_of_words_is_split_in_several_chunks():
    chunks = chunk_text(WORDS).chunks
    assert len(chunks) >= 5
    assert chunks[0].start == 0
    assert chunks[-1].end == len(WORDS)


# --- sections -------------------------------------------------------------------------------


def test_chunks_never_cross_a_section_heading_and_carry_the_right_label(resume):
    sections = SectionIndex(resume)
    chunks = chunk_text(resume, max_chars=150, overlap_chars=30).chunks
    labels = set()
    for chunk in chunks:
        assert sections.at(chunk.start) == chunk.section
        assert sections.at(chunk.end - 1) == chunk.section
        labels.add(chunk.section)
    assert {"skills", "experience", "education", "certifications", "projects"} <= labels
    assert None in labels  # the SUMMARY block belongs to no known section


def test_a_chunk_that_starts_with_a_heading_includes_it(resume):
    heads = [c for c in chunk_text(resume).chunks if c.text.startswith("EDUCATION")]
    assert len(heads) == 1
    assert heads[0].section == "education"
    assert "B.Tech in Computer Science" in heads[0].text


def test_text_without_headings_has_no_section_labels():
    chunks = chunk_text(LINES).chunks
    assert {c.section for c in chunks} == {None}


# --- the chunk limit -------------------------------------------------------------------------


def test_chunk_limit_is_respected_and_reported():
    result = chunk_text(LINES, max_chars=100, overlap_chars=20, max_chunks=3)
    assert len(result.chunks) == 3
    assert result.truncated is True
    assert_exact(LINES, result.chunks)


def test_truncated_is_false_when_the_limit_is_exactly_enough():
    needed = len(chunk_text(LINES, max_chars=100, overlap_chars=20).chunks)
    settings = {"max_chars": 100, "overlap_chars": 20}
    assert chunk_text(LINES, max_chunks=needed, **settings).truncated is False
    assert chunk_text(LINES, max_chunks=needed - 1, **settings).truncated is True


def test_a_huge_input_is_capped_at_the_default_limit():
    text = "word " * 100_000
    result = chunk_text(text)
    assert len(result.chunks) == DEFAULT_MAX_CHUNKS
    assert result.truncated is True
    assert_exact(text, result.chunks)


# --- settings ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_chars": 19},
        {"max_chars": 100, "overlap_chars": -1},
        {"max_chars": 100, "overlap_chars": 100},
        {"max_chunks": 0},
    ],
    ids=["tiny_chunk", "negative_overlap", "overlap_too_big", "no_chunks"],
)
def test_unusable_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        chunk_text("some text", **kwargs)