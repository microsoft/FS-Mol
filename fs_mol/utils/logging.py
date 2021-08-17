from contextlib import contextmanager
from typing import Optional
import logging
import logging.config


PROGRESS_LOG_LEVEL = 15


def set_up_logging(log_path: Optional[str] = None) -> None:
    # set up logging - some things go to console, and everything to file
    logging.addLevelName(PROGRESS_LOG_LEVEL, "PROGRESS")

    # Should be moved out into a config file at some point...
    logging.getLogger("").setLevel(logging.NOTSET)
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "format": "%(asctime)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "file": {
                "format": "%(asctime)s %(name)-12s %(levelname)-10s %(message)s",
            },
        },
        "handlers": {
            "console": {
                "level": logging.INFO,
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "console",
            },
            "file": {
                "level": logging.DEBUG,
                "class": "logging.FileHandler",
                "filename": log_path,
                "formatter": "file",
            },
        },
        "root": {
            "handlers": ["console", "file"],
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)


@contextmanager
def prefix_log_msgs(prefix: str):
    # Store old formatters for all handlers and create new ones, inserting the prefix:
    formatter_to_old_style = {}
    for handler in logging._handlerList:
        formatter = handler().formatter
        if formatter is not None:
            if not isinstance(formatter._style, logging.PercentStyle):
                raise ValueError()
            old_fmt = formatter._style._fmt
            formatter_to_old_style[formatter] = old_fmt
            formatter._style._fmt = old_fmt.replace("%(message)s", f"{prefix} %(message)s")

    yield

    # Now restore the old formatters:
    for formatter, old_fmt in formatter_to_old_style.items():
        formatter._style._fmt = old_fmt


@contextmanager
def restrict_console_log_level(log_level: int):
    console_log_handler = None
    for handler in logging._handlerList:
        if handler().get_name() == "console":
            console_log_handler = handler()
            break

    if console_log_handler is not None:
        old_level = console_log_handler.level
        console_log_handler.setLevel(log_level)

    yield

    if console_log_handler is not None:
        console_log_handler.setLevel(old_level)


class FileLikeLogger:
    """Utility to make a logging call look like an io.IOBase writable object."""

    def __init__(self, logger, level):
        self._logger = logger
        self._level = level

    def close(self):
        pass

    def writable(self):
        return True

    def readable(self):
        return False

    def seekable(self):
        return False

    def writelines(self, lines):
        for line in lines:
            self._logger.log(self._level, line)

    def write(self, s):
        self._logger.log(self._level, s.strip())
