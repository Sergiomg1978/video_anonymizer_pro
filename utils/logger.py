"""Logging setup for Video Anonymizer Pro."""

import logging
import sys

_initialized = False


def setup_logger(level: str = "INFO") -> logging.Logger:
    """Configure the root logger with Rich handler if available, plain StreamHandler otherwise.

    Args:
        level: Log level string - DEBUG, INFO, WARNING, or ERROR.

    Returns:
        The configured root logger.
    """
    global _initialized

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Avoid adding duplicate handlers on repeated calls
    if _initialized:
        for handler in root_logger.handlers:
            handler.setLevel(numeric_level)
        return root_logger

    # Remove any pre-existing handlers
    root_logger.handlers.clear()

    try:
        from rich.logging import RichHandler

        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
        )
        fmt = "%(message)s"
    except ImportError:
        handler = logging.StreamHandler(sys.stderr)
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    handler.setLevel(numeric_level)
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    root_logger.addHandler(handler)

    _initialized = True
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    If setup_logger() has not been called yet, it is invoked with defaults
    so that every logger has at least one handler.

    Args:
        name: Dotted logger name, e.g. ``"anonymizer.detection"``.

    Returns:
        A :class:`logging.Logger` instance.
    """
    if not _initialized:
        setup_logger()
    return logging.getLogger(name)
