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
    r"(?:first|second|third|fourth|final|1st|2nd|3rd|4th)\s+year)\b"
    r"|(?:-|\u2013|\u2014|\bto\b)\s*(?:present|current|now|ongoing)\b"
)
_INCOMPLETE = re.compile(
    r"\b(?:dropped\s+out|drop\s?out|did\s+not\s+(?:complete|finish|graduate)|"
    r"didn't\s+(?:complete|finish|graduate)|incomplete|unfinished|discontinued|withdrew|"
    r"withdrawn|not\s+(?:completed|finished)|abandoned)\b"
)
_NEGATION = re.compile(r"(?<![\w'])(?:no|not|never|without|none|lack|lacks|lacking)(?![\w'])")


def detect_degree_level(text: str) -> DegreeLevel | None:
    """Return the highest level found, or None. Bare 'BS'/'MS' are not recognized."""
    lowered = text.lower().replace("\u2019", "'")
    for level, pattern in _PATTERNS:
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