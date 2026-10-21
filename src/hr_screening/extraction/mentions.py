"""Find skill mentions in resume text and label how each one is used.

Each mention is AFFIRMATIVE, NEGATED ("no experience with Python") or ASPIRATIONAL
("currently learning Python"). This is a small rule set, not language understanding.
A cue only counts when it sits close to the skill, so "No Java, strong Python" leaves
Python affirmative. Wording the rules do not recognize stays AFFIRMATIVE, which is how
every mention was treated before this module existed. It never infers proficiency.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

from hr_screening.extraction.skills_vocab import SKILL_ALIASES, skill_pattern


class MentionKind(StrEnum):
    AFFIRMATIVE = "affirmative"
    NEGATED = "negated"
    ASPIRATIONAL = "aspirational"


@dataclass(frozen=True)
class SkillMention:
    start: int
    end: int
    text: str  # exact text as written in the resume
    kind: MentionKind


_MAX_FILLER_WORDS = 6
_LOOKBACK_CHARS = 200
_CLAUSE_END = re.compile(r"[.!?;](?=\s|$)|\s(?:but|however|although|though|except|yet|whereas)\s")
_LIST_SPLIT = re.compile(r"\s*(?:,|/|&|\band\b|\bor\b|\bnor\b)\s*")
_CLAUSE_BREAKERS = frozenset(
    {"using", "via", "by", "while", "when", "then", "after", "before", "because"}
    | {"which", "that", "who", "where", "if", "so"}
)
_NEGATION_CUE = re.compile(
    r"(?<![\w'])(?:no|not|never|without|none|nor|cannot|lack|lacks|lacked|lacking|unfamiliar|"
    r"\w+n't)(?![\w'])(?!\s+(?:only|just|longer|limited|to\s+mention))"
)
_ASPIRATION_CUE = re.compile(
    r"\b(?:currently|now|still|actively)\s+(?:\w+\s+)?(?:learning|studying)\b"
    r"|\b(?:am|is|are|i'm)\s+(?:\w+\s+)?(?:learning|studying)\b"
    r"|\b(?:want|wants|wish|wishes|hope|hopes|plan|plans|planning|intend|intends|aim|aims|"
    r"looking|keen|eager|going|about)\s+(?:to\s+)?(?:learn|study|pick\s+up|start\s+learning|"
    r"begin\s+learning|get\s+started\s+(?:with|in))\b"
    r"|\bwill\s+(?:learn|study)\b"
    r"|\binterested\s+in\s+(?:learning|exploring)\b"
    r"|\blearning\s+goals?\b"
    r"|\bto\s+learn\b"
)
_ASPIRATION_START = re.compile(r"^\W*(?:learning|studying)\b")
_SUFFIX_NEGATED = re.compile(
    r"^\s*[:\-\u2013\u2014(]\s*(?:no(?!\s+longer)|none|not(?!\s+(?:only|just|limited))|"
    r"never|n/a|nil)(?![\w'])"
)
_SUFFIX_ASPIRATIONAL = re.compile(
    r"^\s*[:\-\u2013\u2014(]\s*(?:currently\s+)?(?:learning|studying|in\s+progress|planned|"
    r"to\s+learn)(?![\w'])"
)
_SKILL_NAMES = frozenset(
    name.lower() for key, aliases in SKILL_ALIASES.items() for name in (key, *aliases)
)
_VERSION_TAIL = re.compile(r"\d+(?:\.\d+){0,2}$")


def find_skill_mentions(text: str, skill: str) -> list[SkillMention]:
    """Return every whole word mention of ``skill`` (or an alias), in text order."""
    if not skill.strip():
        return []
    return [
        SkillMention(m.start(), m.end(), m.group(0), classify_mention(text, m.start(), m.end()))
        for m in skill_pattern(skill).finditer(text)
    ]


def classify_mention(text: str, start: int, end: int) -> MentionKind:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    window_start = max(line_start, start - _LOOKBACK_CHARS)
    before = _fold(text[window_start:start])
    prefix = _last_clause(before)
    after = _fold(text[end:line_end])
    if _cue_in_scope(prefix, _NEGATION_CUE) or _SUFFIX_NEGATED.match(after):
        return MentionKind.NEGATED
    start_cue = window_start == line_start and bool(_ASPIRATION_START.match(prefix))
    if start_cue or _cue_in_scope(prefix, _ASPIRATION_CUE) or _SUFFIX_ASPIRATIONAL.match(after):
        return MentionKind.ASPIRATIONAL
    return MentionKind.AFFIRMATIVE


def _fold(text: str) -> str:
    return text.lower().replace("\u2019", "'")


def _last_clause(prefix: str) -> str:
    last = 0
    for match in _CLAUSE_END.finditer(prefix):
        last = match.end()
    return prefix[last:]


def _cue_in_scope(prefix: str, cue: re.Pattern[str]) -> bool:
    """True if the nearest ``cue`` in ``prefix`` still governs the skill that follows."""
    matches = list(cue.finditer(prefix))
    if not matches:
        return False
    return _within_scope(prefix[matches[-1].end() :])


def _within_scope(between: str) -> bool:
    chunks = [c.strip(" \t:;-\u2013\u2014()\"'") for c in _LIST_SPLIT.split(between)]
    words = chunks[0].split()
    if len(words) > _MAX_FILLER_WORDS or any(w in _CLAUSE_BREAKERS for w in words):
        return False
    return all(not chunk or _is_skill_name(chunk) for chunk in chunks[1:])


def _is_skill_name(chunk: str) -> bool:
    return _VERSION_TAIL.sub("", chunk).strip() in _SKILL_NAMES