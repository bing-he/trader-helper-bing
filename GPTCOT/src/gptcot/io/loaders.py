"""Data loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from gptcot.features import SERIES_COLUMNS
from gptcot.io.validators import ensure_file, require_columns, validate_non_empty


def _parse_dates(df: pd.DataFrame, column: str, context: str) -> pd.DataFrame:
    df[column] = pd.to_datetime(df[column], errors="coerce")
    if df[column].isna().any():
        raise ValueError(f"{context} contains unparseable dates in column '{column}'.")
    return df


def load_cot(path: Path) -> pd.DataFrame:
    """Load NGCommitofTraders.csv."""

    ensure_file(path, "NGCommitofTraders.csv")
    df = pd.read_csv(path)
    require_columns(df, ["Date", *SERIES_COLUMNS], "NGCommitofTraders.csv")
    validate_non_empty(df, "NGCommitofTraders.csv")
    df = _parse_dates(df, "Date", "NGCommitofTraders.csv")
    for col in SERIES_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(f"NGCommitofTraders.csv contains non-numeric values in '{col}'.")
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_forward_curve(path: Path) -> pd.DataFrame:
    """Load HenryForwardCurve.csv."""

    ensure_file(path, "HenryForwardCurve.csv")
    df = pd.read_csv(path)
    require_columns(df, ["Date", "FrontMonth_Label"], "HenryForwardCurve.csv")
    validate_non_empty(df, "HenryForwardCurve.csv")
    df = _parse_dates(df, "Date", "HenryForwardCurve.csv")
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_absolute_history(path: Path) -> pd.DataFrame:
    """Load HenryHub_Absolute_History.csv."""

    ensure_file(path, "HenryHub_Absolute_History.csv")
    df = pd.read_csv(path)
    if "TradeDate" not in df.columns:
        raise ValueError("HenryHub_Absolute_History.csv is missing column 'TradeDate'.")
    validate_non_empty(df, "HenryHub_Absolute_History.csv")
    df = _parse_dates(df, "TradeDate", "HenryHub_Absolute_History.csv")
    value_columns = [col for col in df.columns if col != "TradeDate"]
    df[value_columns] = df[value_columns].apply(pd.to_numeric, errors="coerce")
    df = df.sort_values("TradeDate").reset_index(drop=True)
    return df
