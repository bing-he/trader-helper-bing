"""Utility helpers for the LNG dashboard."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def find_repo_root(start: Optional[Path] = None, marker_name: str = "TraderHelper") -> Path:
    """Find the repository root by walking up until the marker directory is found."""
    start_path = start or Path(__file__).resolve()
    for parent in [start_path, *start_path.parents]:
        if parent.name == marker_name:
            return parent
        if (parent / ".git").exists():
            return parent
    return start_path if start_path.is_dir() else start_path.parent


def resolve_paths(info_dir_arg: Optional[str], out_dir_arg: Optional[str]) -> Tuple[Path, Path]:
    """Resolve default INFO and output directories."""
    repo_root = find_repo_root()
    default_info = repo_root / "INFO"
    default_out = repo_root / "Scripts" / "out"
    info_dir = Path(info_dir_arg).expanduser() if info_dir_arg else default_info
    out_dir = Path(out_dir_arg).expanduser() if out_dir_arg else default_out
    return info_dir, out_dir


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def drop_leap_days(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Remove leap day entries to keep day-of-year alignment stable."""
    if date_col not in df.columns:
        return df
    return df.loc[~((df[date_col].dt.month == 2) & (df[date_col].dt.day == 29))].copy()


def parse_date(value: object) -> Optional[pd.Timestamp]:
    """Parse a date value into a pandas Timestamp."""
    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    return parsed if pd.notnull(parsed) else None


def current_run_timestamp() -> datetime:
    """Return the current UTC timestamp for run metadata."""
    return datetime.utcnow()

