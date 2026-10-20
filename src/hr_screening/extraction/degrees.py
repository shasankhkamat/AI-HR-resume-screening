"""Detect the highest degree level mentioned in a line of text."""

import re

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


def detect_degree_level(text: str) -> DegreeLevel | None:
    """Return the highest level found, or None. Bare 'BS'/'MS' are not recognized."""
    lowered = text.lower().replace("\u2019", "'")
    for level, pattern in _PATTERNS:
        if pattern.search(lowered):
            return level
    return None