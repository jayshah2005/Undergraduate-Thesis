"""Plain-text logger for clean/ scripts."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Write messages to the terminal and, if given, append them to log_file.

    No timestamps or levels in the file, so logs/run.log is readable as a report.
    Computation code never calls this; use it once in main().

    :param name: Logger name, unique per script.
    :type name: str
    :param log_file: Optional path to append. Parent directories are created.
    :type log_file: Path | None
    :param level: Logging level, default INFO.
    :type level: int
    :return: A logger that writes to stdout and optionally to log_file.
    :rtype: logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    fmt = logging.Formatter("%(message)s")
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(fmt)
        logger.addHandler(h)
    if log_file is not None:
        log_file = Path(log_file).resolve()
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file
                   for h in logger.handlers):
            log_file.parent.mkdir(parents=True, exist_ok=True)
            h = logging.FileHandler(log_file, mode="a")
            h.setFormatter(fmt)
            logger.addHandler(h)
    return logger
