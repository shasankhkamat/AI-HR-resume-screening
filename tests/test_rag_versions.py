"""A version number after a skill name (Python3) must not hide the skill from retrieval.

Found by the synthetic retrieval check: the evaluator accepts Python3 as Python, but the
keyword index used to treat python3 as a different word.
"""

import pytest

from hr_screening.rag.lexical import tokenize
from hr_screening.rag.retrieval import LexicalRetriever

RESUME = """SKILLS
Python3, Docker, PostgreSQL

EXPERIENCE
Backend Engineer, Sample Ltd   Mar 2020 - Present
- Maintain REST services in Python3.11 and deploy them with Docker.
"""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Python3", ["python"]),
        ("python3.11", ["python", "11"]),
        ("HTML5", ["html"]),
        ("C++17", ["c++", "17"]),
        ("Python 3", ["python"]),
    ],
)
def test_a_trailing_version_is_dropped(text, expected):
    assert tokenize(text) == expected


@pytest.mark.parametrize("text", ["k8s", "S3", "x86", "log4j", "2022", "3d"])
def test_other_words_with_digits_are_left_alone(text):
    assert tokenize(text) == [text.lower()]


def test_a_skill_is_found_on_a_line_that_writes_a_version():
    passages = LexicalRetriever(RESUME).for_skill("python")
    texts = " ".join(p.text for p in passages)
    assert "Python3, Docker" in texts
    assert "Python3.11" in texts
    assert "python" in passages[0].matched_terms
    assert [kind.value for p in passages for kind in p.mention_kinds] == ["affirmative"] * 2


def test_a_requirement_that_writes_a_version_finds_the_plain_skill():
    plain = LexicalRetriever("SKILLS\nPython, SQL\n").for_skill("python3")
    assert plain
    assert "Python" in plain[0].text