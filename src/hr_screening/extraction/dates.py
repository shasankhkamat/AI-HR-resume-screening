"""Date range helpers for experience calculation.

Calculation is deliberately conservative: a year with no month counts as January for
both start and end, and the end month is not counted inclusively.
Overlapping roles are merged so they are not counted twice.
"""

import re
from collections.abc import Iterable
from datetime import date

from hr_screening.models.candidate import WorkExperience

_MONTHS = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
MONTH_PATTERN = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
DATE_PATTERN = rf"(?:(?:{MONTH_PATTERN}\.?\s+)?(?:19|20)\d{{2}}|\d{{1,2}}/(?:19|20)\d{{2}})"
PRESENT_PATTERN = r"(?:present|current|now|till date)\b"


def to_month_index(token: str, today: date) -> int | None:
    """Convert 'Jan 2019', '03/2019', '2019' or 'Present' to a month number."""
    value = token.strip().lower().rstrip(".")
    if re.fullmatch(PRESENT_PATTERN, value):
        return today.year * 12 + today.month - 1
    match = re.fullmatch(r"(\d{1,2})/((?:19|20)\d{2})", value)
    if match:
        month = int(match.group(1))
        return int(match.group(2)) * 12 + month - 1 if 1 <= month <= 12 else None
    match = re.fullmatch(r"([a-z]{3,9})\.?\s+((?:19|20)\d{2})", value)
    if match:
        prefix = match.group(1)[:3]
        if prefix in _MONTHS:
            return int(match.group(2)) * 12 + _MONTHS.index(prefix)
        return None
    if re.fullmatch(r"(?:19|20)\d{2}", value):
        return int(value) * 12
    return None


def month_span(start: str | None, end: str | None, today: date) -> int | None:
    if not start or not end:
        return None
    start_index = to_month_index(start, today)
    end_index = to_month_index(end, today)
    if start_index is None or end_index is None or end_index < start_index:
        return None
    return end_index - start_index


def total_years(experience: Iterable[WorkExperience], today: date) -> float | None:
    """Total years across dated roles with overlaps merged. None if no role has usable dates."""
    spans = []
    for item in experience:
        if not item.start or not item.end:
            continue
        start = to_month_index(item.start, today)
        end = to_month_index(item.end, today)
        if start is None or end is None or end < start:
            continue
        spans.append((start, end))
    if not spans:
        return None
    spans.sort()
    months = 0
    current_start, current_end = spans[0]
    for start, end in spans[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            months += current_end - current_start
            current_start, current_end = start, end
    months += current_end - current_start
    return round(months / 12, 1)