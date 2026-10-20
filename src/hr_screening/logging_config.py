
"""Centralized logging configuration.

Call configure_logging() once when the application starts.
Other modules should use logging.getLogger(__name__).

Privacy rule: never log resume text, candidate personal data, or secrets.
Log identifiers and counts instead.
"""

import logging

from hr_screening.config import get_settings

LOGGER_NAME = "hr_screening"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_HANDLER_NAME = "hr_screening_console"


def configure_logging() -> logging.Logger:
    """Configure application logging and return the package logger.

    Repeated calls update the configured level and format without
    adding duplicate console handlers.
    """
    settings = get_settings()

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(settings.log_level)

    # Avoid duplicate output through the root logger.
    logger.propagate = False

    handler = next(
        (
            existing_handler
            for existing_handler in logger.handlers
            if existing_handler.get_name() == _HANDLER_NAME
        ),
        None,
    )

    if handler is None:
        handler = logging.StreamHandler()
        handler.set_name(_HANDLER_NAME)
        logger.addHandler(handler)

    handler.setFormatter(
        logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    )
    return logger
