from __future__ import annotations

import logging
from pathlib import Path

from .pathing import ROOT


def get_file_logger(script_name: str) -> logging.Logger:
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(script_name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(logs_dir / f"{script_name}.log", encoding="utf-8")
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger
