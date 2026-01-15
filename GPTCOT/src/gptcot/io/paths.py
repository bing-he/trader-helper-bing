"""Helpers for well-known input/output paths."""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Return the repository root containing the INFO directory."""

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "INFO").exists():
            return parent
    raise RuntimeError("Unable to locate project root containing an INFO directory.")


def info_dir() -> Path:
    """Return the INFO directory path."""

    return get_project_root() / "INFO"


def cot_path(base_dir: Path | None = None) -> Path:
    base = base_dir or info_dir()
    return base / "NGCommitofTraders.csv"


def forward_curve_path(base_dir: Path | None = None) -> Path:
    base = base_dir or info_dir()
    return base / "HenryForwardCurve.csv"


def eia_totals_path(base_dir: Path | None = None) -> Path:
    base = base_dir or info_dir()
    return base / "EIAtotals.csv"


def absolute_history_path(base_dir: Path | None = None) -> Path:
    base = base_dir or info_dir()
    return base / "HenryHub_Absolute_History.csv"


def output_csv_path(output_dir: Path) -> Path:
    return output_dir / "cot_forward_returns.csv"


def output_parquet_path(output_dir: Path) -> Path:
    return output_dir / "cot_forward_returns.parquet"
