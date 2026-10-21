"""Detect the highest degree level mentioned in a line of text, and whether it is finished."""

import re
from enum import StrEnum

from hr_screening.models.candidate import DegreeLevel

_PATTERNS = (
    (DegreeLevel.DOCTORATE, re.compile(r"\b(?:ph\.?\s?d|doctorate|doctoral)\b")),
    (DegreeLevel.MASTER, re.compile(r"\b(?:master(?:'?s)?|m\.?sc|m\.?tech|m\.?eng|mba)\b")),
    (
        DegreeLevel.BACHELOR,
        re.compile(r"\b(?:bachelor(?:'?s)?|b\.?sc|b\.?tech|b\.?eng|b\.?com|bba|b\.e)\b"),
    ),
    (DegreeLevel.DIPLOMA, re.compile(r"\b(?:diploma|associate(?:'?s)?)\b")),
)


class DegreeState(StrEnum):
    COMPLETE = "complete"
    IN_PROGRESS = "in_progress"
    INCOMPLETE = "incomplete"
    AMBIGUOUS = "ambiguous"


_IN_PROGRESS = re.compile(
    r"\b(?:pursuing|currently\s+(?:enrolled|studying|pursuing)|in\s+progress|expected|"
    r"anticipated|ongoing|enrolled|candidate|studying|graduating|will\s+graduate|"
    r"(?:working|work)\s+towards?|"
    r"(?:first|second|third|fourth|final|1st|2nd|3rd|4th)\s+year)\b"
    r"|(?:-|\u2013|\u2014|\bto\b)\s*(?:present|current|now|ongoing)\b"
)
_INCOMPLETE = re.compile(
    r"\b(?:dropped\s+out|drop\s?out|did\s+not\s+(?:complete|finish|graduate)|"
    r"didn't\s+(?:complete|finish|graduate)|incomplete|unfinished|discontinued|withdrew|"
    r"withdrawn|not\s+(?:completed|finished)|abandoned)\b"
    # 'Completed coursework toward a Bachelor of Science' is part of a degree, not a degree.
    # 'Partial' and 'some college' say the same. Reading them as unfinished can only understate.
    r"|\bpartial(?:ly)?\b|\bsome\s+(?:college|university|coursework|course\s?work)\b"
    r"|\b(?:coursework|course\s?work|courses|classes|credits?|credit\s+hours|studies)\s+"
    r"(?:completed\s+)?towards?\b"
)
_NEGATION = re.compile(r"(?<![\w'])(?:no|not|never|without|none|lack|lacks|lacking)(?![\w'])")


def detect_degree_level(text: str) -> DegreeLevel | None:
    """Return the highest level found, or None. Bare 'BS'/'MS' are not recognized."""
    lowered = text.lower().replace("\u2019", "'")
    for level, pattern in _PATTERNS:
        if pattern.search(lowered):
            return level
    return None


# For job descriptions. Plain words such as 'Master Data Management', 'Scrum Master' or
# 'associate engineers' must not read as degrees, so 'master' and 'associate' count only in a
# degree like form ("master's", "master of", "master degree", "associate degree").
_REQUEST_PATTERNS = (
    (DegreeLevel.DOCTORATE, re.compile(r"\b(?:ph\.?\s?d|doctorate|doctoral)\b")),
    (
        DegreeLevel.MASTER,
        re.compile(
            r"(?<!scrum )\bmaster's\b|\bmasters\s+(?:degree|in|of)\b|\bmaster\s+(?:degree|of)\b"
            r"|\b(?:m\.?sc|m\.?tech|m\.?eng|mba)\b"
        ),
    ),
    (
        DegreeLevel.BACHELOR,
        re.compile(r"\b(?:bachelor(?:'?s)?|b\.?sc|b\.?tech|b\.?eng|b\.?com|bba|b\.e)\b"),
    ),
    (
        DegreeLevel.DIPLOMA,
        re.compile(r"\bdiploma\b|\bassociate'?s?\s+(?:degree|of)\b|\bassociate's\b"),
    ),
)


def detect_requested_degree_level(text: str) -> DegreeLevel | None:
    """The highest degree level a job line asks for, or None.

    Stricter than ``detect_degree_level``, which is meant for a resume's education section where
    every word is about education. A job line is mostly about other things.
    """
    lowered = text.lower().replace("\u2019", "'")
    for level, pattern in _REQUEST_PATTERNS:
        if pattern.search(lowered):
            return level
    return None


def degree_state(text: str) -> DegreeState:
    """Say whether a degree line describes a finished degree.

    Wording such as 'pursuing' or 'expected 2027' gives IN_PROGRESS. 'Dropped out' gives
    INCOMPLETE. Any other negation gives AMBIGUOUS. Everything else is COMPLETE, which only
    means that no sign of an unfinished degree was found. It does not verify the degree.
    """
    lowered = text.lower().replace("\u2019", "'")
    if _INCOMPLETE.search(lowered):
        return DegreeState.INCOMPLETE
    if _NEGATION.search(lowered):
        return DegreeState.AMBIGUOUS
    if _IN_PROGRESS.search(lowered):
        return DegreeState.IN_PROGRESS
    return DegreeState.COMPLETE