"""The only thing the rest of the application knows about language models."""

from typing import Protocol


class LLMClient(Protocol):
    """A provider adapter must return the raw text reply.

    Adapters must translate provider failures into LLMTimeoutError, LLMRateLimitError
    or LLMError, and must never put prompt or response text in exception messages.
    """

    def complete(self, system_prompt: str, user_prompt: str) -> str: ...