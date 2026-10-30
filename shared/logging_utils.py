"""
shared/logging_utils.py
=======================
Centralized logging configuration for the entire project.

All modules import `get_logger(__name__)` to get a properly configured logger
that writes to both console (INFO+) and a rotating file log (DEBUG+).
"""

import logging
import sys
from pathlib import Path
from typing import Optional


_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "project.log"

_initialized = False


def _setup_root_logger() -> None:
    """Configure the root logger once for the entire project."""
    global _initialized
    if _initialized:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler — DEBUG and above (full trace)
    try:
        from logging.handlers import RotatingFileHandler

        fh = RotatingFileHandler(
            _LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except Exception as e:  # noqa: BLE001
        root.warning(f"Could not set up file logging: {e}")

    _initialized = True


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Return a named logger configured for the project.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.
    level : int, optional
        Override log level for this specific logger.

    Returns
    -------
    logging.Logger
    """
    _setup_root_logger()
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger
