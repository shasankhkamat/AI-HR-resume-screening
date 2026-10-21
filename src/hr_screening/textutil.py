"""Small text helpers shared by extraction and evaluation.

Matching runs on folded text (case folded, whitespace collapsed). Anything shown to a
reviewer is cut from the original text, so folded text is never stored as evidence.
"""


def normalize(text: str) -> str:
    """Case fold and collapse whitespace. Used for comparison only, never for display."""
    return " ".join(text.casefold().split())


class FoldedText:
    """A text plus its folded form, with a map from folded offsets back to the original."""

    def __init__(self, text: str) -> None:
        self.text = text
        chars: list[str] = []
        index: list[int] = []
        pending_space = False
        for position, char in enumerate(text):
            if char.isspace():
                pending_space = bool(chars)
                continue
            if pending_space:
                chars.append(" ")
                index.append(position)
                pending_space = False
            for folded in char.casefold():
                chars.append(folded)
                index.append(position)
        self.folded = "".join(chars)
        self._index = index

    def find_spans(self, needle: str, *, whole_word: bool = True) -> list[tuple[int, int]]:
        """Return (start, end) offsets in the ORIGINAL text for each match of ``needle``.

        Case and whitespace differences are ignored. With ``whole_word`` a match may not
        touch a letter or digit on either side, so 'Present' does not match 'presentation'.
        """
        target = normalize(needle)
        if not target:
            return []
        spans: list[tuple[int, int]] = []
        position = 0
        while True:
            found = self.folded.find(target, position)
            if found == -1:
                break
            position = found + len(target)
            start = self._index[found]
            end = self._index[found + len(target) - 1] + 1
            if whole_word and not self._bounded(start, end, target):
                continue
            spans.append((start, end))
        return spans

    def _bounded(self, start: int, end: int, target: str) -> bool:
        before = self.text[start - 1] if start > 0 else ""
        after = self.text[end] if end < len(self.text) else ""
        if target[0].isalnum() and before.isalnum():
            return False
        return not (target[-1].isalnum() and after.isalnum())


def original_span(text: str, needle: str) -> str | None:
    """Return the first whole word match of ``needle`` as it is written in ``text``."""
    spans = FoldedText(text).find_spans(needle)
    if not spans:
        return None
    start, end = spans[0]
    return text[start:end]


def line_at(text: str, position: int, limit: int = 200) -> str:
    """Return the line of ``text`` containing ``position``, trimmed.

    A line longer than ``limit`` is cut to a window around ``position`` so the match stays
    visible. The result is always an exact slice of ``text``.
    """
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    if end == -1:
        end = len(text)
    line = text[start:end]
    if len(line.strip()) <= limit:
        return line.strip()
    offset = max(position - start, 0)
    window_start = max(0, min(offset - limit // 2, len(line) - limit))
    return line[window_start : window_start + limit].strip()