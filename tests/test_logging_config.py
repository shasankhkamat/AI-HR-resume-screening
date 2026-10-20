import io
import logging
import re
from types import SimpleNamespace

import pytest

from hr_screening import logging_config
from hr_screening.logging_config import LOGGER_NAME, configure_logging


@pytest.fixture
def isolated_logger():
    """Restore logger state after each test."""
    logger = logging.getLogger(LOGGER_NAME)
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    saved_propagate = logger.propagate

    logger.handlers.clear()

    try:
        yield logger
    finally:
        new_handlers = [
            handler
            for handler in logger.handlers
            if handler not in saved_handlers
        ]

        logger.handlers.clear()

        for handler in new_handlers:
            handler.close()

        logger.handlers.extend(saved_handlers)
        logger.setLevel(saved_level)
        logger.propagate = saved_propagate


@pytest.fixture
def set_level(monkeypatch):
    """Supply test settings without reading the real .env file."""

    def _set(level: str) -> None:
        monkeypatch.setattr(
            logging_config,
            "get_settings",
            lambda: SimpleNamespace(log_level=level),
        )

    return _set


def _capture(logger: logging.Logger) -> io.StringIO:
    """Capture output from the configured console handler."""
    buffer = io.StringIO()

    handler = next(
        handler
        for handler in logger.handlers
        if handler.get_name() == "hr_screening_console"
    )

    handler.setStream(buffer)
    return buffer


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
    ],
)
def test_level_comes_from_settings(
    isolated_logger, set_level, level, expected
):
    set_level(level)

    logger = configure_logging()

    assert logger.level == expected


def test_repeated_calls_do_not_duplicate_handlers(
    isolated_logger, set_level
):
    set_level("INFO")

    logger = configure_logging()

    configure_logging()
    configure_logging()

    named_handlers = [
        handler
        for handler in logger.handlers
        if handler.get_name() == "hr_screening_console"
    ]

    assert len(named_handlers) == 1


def test_repeated_configuration_emits_message_once(
    isolated_logger, set_level
):
    set_level("INFO")

    logger = configure_logging()
    buffer = _capture(logger)

    configure_logging()
    configure_logging()

    logging.getLogger("hr_screening.demo").info("single message")

    assert buffer.getvalue().count("single message") == 1


def test_reconfigure_updates_level_without_new_handler(
    isolated_logger, set_level
):
    set_level("INFO")

    logger = configure_logging()

    set_level("ERROR")
    configure_logging()

    assert logger.level == logging.ERROR
    assert sum(
        handler.get_name() == "hr_screening_console"
        for handler in logger.handlers
    ) == 1


def test_format_has_timestamp_severity_name_and_message(
    isolated_logger, set_level
):
    set_level("INFO")

    logger = configure_logging()
    buffer = _capture(logger)

    logging.getLogger("hr_screening.demo").info("hello")

    pattern = (
        r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"
        r" \| INFO\s+\| hr_screening\.demo \| hello$"
    )

    assert re.match(pattern, buffer.getvalue().strip())


def test_messages_below_level_are_dropped(
    isolated_logger, set_level
):
    set_level("WARNING")

    logger = configure_logging()
    buffer = _capture(logger)

    child = logging.getLogger("hr_screening.demo")
    child.info("should not appear")
    child.warning("should appear")

    output = buffer.getvalue()

    assert "should not appear" not in output
    assert "should appear" in output


def test_no_file_handlers_are_created(isolated_logger, set_level):
    set_level("DEBUG")

    logger = configure_logging()

    assert not any(
        isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )


def test_root_logger_is_untouched(isolated_logger, set_level):
    set_level("DEBUG")

    root = logging.getLogger()
    handlers_before = list(root.handlers)
    level_before = root.level

    configure_logging()

    assert root.handlers == handlers_before
    assert root.level == level_before


def test_configure_itself_writes_nothing(
    isolated_logger, set_level
):
    set_level("DEBUG")

    logger = configure_logging()
    buffer = _capture(logger)

    configure_logging()

    assert buffer.getvalue() == ""
