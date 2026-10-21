"""The field named in a years of experience requirement, such as 'data engineering'.

'3+ years of experience in data engineering' asks for time spent in a field, not for any dated
work. This module only reads the field's words from the job line and splits them into terms.
Deciding whether a role is in the field is the evaluator's job, and it can only check whether
those words appear in the role's text.
"""

import re

_AFTER = re.compile(r"\bexperience\s+(?:in|with|as|of|within|using)\s+(?P<field>[^,;.()]+)", re.I)
_BEFORE = re.compile(r"\b(?:years?|yrs?)\b\W*(?:of\s+)?(?P<field>[^,;.()]+?)\s+experience\b", re.I)
_CUT = re.compile(r"\b(?:including|such as|preferably|ideally|e\.g)\b", re.I)
_WORD = re.compile(r"[a-z][a-z0-9+#]*")
MAX_FIELD_CHARS = 100
_ALTERNATIVE = re.compile(r"\s+or\s+|\s*/\s*", re.I)
# A phrase with one of these describes a setting ('a fast paced environment'), not a field.
_SETTING_WORDS = frozenset("environment environments setting settings capacity".split())

# Words that do not name a field. A phrase made only of these ('a related field') names none.
_NOT_A_FIELD = frozenset(
    "a an the of in on at to for with and or as related relevant similar equivalent comparable "
    "field fields area areas role roles position positions industry industries environment "
    "environments setting settings professional work working practical hands handson proven "
    "demonstrated strong solid good prior previous significant substantial progressive "
    "relevant years year yrs experience".split()
)


# Fields whose experience is judged from work history, by the field's own words or by the exact
# job titles listed here. Such a field is an experience requirement, not a skill to look for, so
# the job parser does not also list it as a required skill. 'analysis' is not 'analyst', and the
# bare word 'analyst' never counts, so 'Business Analyst' does not prove data analysis. Each title
# must appear as consecutive words. Add an entry only for a title that plainly names the work.
FIELD_TITLES: dict[tuple[str, ...], tuple[tuple[str, ...], ...]] = {
    ("data", "analysis"): (("data", "analyst"),),
}


def field_terms(field: str | None) -> list[str]:
    """The distinguishing words of a field phrase, lower case, in order, without repeats."""
    if not field:
        return []
    terms: list[str] = []
    for word in _WORD.findall(field.lower()):
        if word not in _NOT_A_FIELD and word not in terms:
            terms.append(word)
    return terms


def field_alternatives(field: str | None) -> list[list[str]]:
    """The terms of each alternative in a field phrase: 'a or b' is satisfied by either one.

    Words joined by 'and' stay together in one alternative, so all of them are required.
    """
    if not field:
        return []
    parts = (field_terms(part) for part in _ALTERNATIVE.split(field))
    return [terms for terms in parts if terms]


def find_experience_field(line: str) -> str | None:
    """The field named in one job line, as written, or None when the line names none."""
    for pattern in (_AFTER, _BEFORE):
        match = pattern.search(line)
        if match is None:
            continue
        phrase = _CUT.split(match.group("field"))[0].strip(" \t-:")
        if _SETTING_WORDS & set(_WORD.findall(phrase.lower())):
            return None
        if field_terms(phrase):
            return phrase[:MAX_FIELD_CHARS].strip()
    return None


def is_work_field(field: str) -> bool:
    """True if the field is one of the work fields in FIELD_TITLES, written as a single phrase."""
    alternatives = field_alternatives(field)
    return len(alternatives) == 1 and tuple(alternatives[0]) in FIELD_TITLES
