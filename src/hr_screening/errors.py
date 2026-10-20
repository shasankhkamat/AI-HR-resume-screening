"""Exception types for the screening workflow.

Messages must never contain resume content or personal data.
"""


class ScreeningError(Exception):
    """Base class for all expected, user-facing failures."""


class UnsupportedFileTypeError(ScreeningError):
    """The file extension is not supported."""


class EmptyFileError(ScreeningError):
    """The file has zero bytes."""


class UnreadableFileError(ScreeningError):
    """The file is missing, corrupted, protected, too large or not what it claims to be."""


class NoExtractableTextError(ScreeningError):
    """The file was readable but contained no text."""


class InvalidJobDescriptionError(ScreeningError):
    """The job description is empty, too long or unreadable."""


class LLMError(ScreeningError):
    """Base class for language model failures."""


class LLMTimeoutError(LLMError):
    """The provider did not answer in time."""


class LLMRateLimitError(LLMError):
    """The provider rejected the call because of rate limits."""


class LLMResponseError(LLMError):
    """The model answered, but the answer was malformed or failed validation."""