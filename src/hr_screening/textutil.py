"""Small text helpers shared by extraction and evaluation."""


def line_at(text: str, position: int, limit: int = 200) -> str:
    """Return the line of ``text`` containing ``position``, trimmed and truncated."""
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    if end == -1:
        end = len(text)
    return text[start:end].strip()[:limit]