# ─────────────────────────────────────────────
#  src/utils/logger.py — Centralized Logging
# ─────────────────────────────────────────────
import logging
import sys
from pathlib import Path
from config import LOG_DIR


def get_logger(name: str) -> logging.Logger:
    """Return a logger with file + console handlers."""
    logger = logging.getLogger(name)

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    log_file = LOG_DIR / "medical_chatbot.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
