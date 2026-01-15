"""Input validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

MISSING_REQUIRED_COLUMNS = "missing_required_columns"


def ensure_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found at {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{description} path is not a file: {path}")


def require_columns(df: pd.DataFrame, required: Sequence[str], context: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(
            f"{context} is missing required columns: {missing_list}. "
            f"Expected columns: {', '.join(required)}"
        )


def validate_non_empty(df: pd.DataFrame, context: str) -> None:
    if df.empty:
        raise ValueError(f"{context} is empty.")
