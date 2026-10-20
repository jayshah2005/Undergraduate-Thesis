"""Logging set-up shared by every pipeline (see the separate-logging skill).

Computation code never prints; scripts create one logger in their entry point and pass it (or
callbacks) to anything that reports progress:

    from logs import get_logger
    log = get_logger("gmkl", log_file=HERE / "logs" / "12.log")
    log.info("fold %d done", fold)

Messages are written as plain text (no timestamps or levels), so log files look the same as the
printed output they replace.
"""
import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Logger that writes plain messages to the terminal and, if log_file is given, appends them to it.

    Calling it again with the same name returns the same logger; a new log_file is added once."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False                                   # do not duplicate into the root logger
    fmt = logging.Formatter("%(message)s")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(fmt)
        logger.addHandler(h)
    if log_file is not None:
        log_file = Path(log_file).resolve()
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file for h in logger.handlers):
            log_file.parent.mkdir(parents=True, exist_ok=True)
            h = logging.FileHandler(log_file, mode="a")
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger


def close_log_file(logger: logging.Logger) -> None:
    """Detach and close the logger's file handlers (e.g. before starting the next run's log file)."""
    for h in [h for h in logger.handlers if isinstance(h, logging.FileHandler)]:
        logger.removeHandler(h)
        h.close()
