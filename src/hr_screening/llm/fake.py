"""Deterministic stand in for a model, used by tests and offline demos."""


class FakeLLMClient:
    def __init__(self, response: str = "{}", error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self.error is not None:
            raise self.error
        return self.response