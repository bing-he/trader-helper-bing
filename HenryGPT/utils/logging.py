import logging
from typing import Optional


def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging once for the application."""
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger."""
    return logging.getLogger(name)