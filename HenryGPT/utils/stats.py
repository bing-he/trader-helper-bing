from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def rolling_pct_rank(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Rolling percentile rank of the most recent value in each window."""
    def _pct_rank(values: Iterable[float]) -> float:
        window_series = pd.Series(values)
        return float(window_series.rank(pct=True).iloc[-1])

    return series.rolling(window=window, min_periods=min_periods).apply(_pct_rank, raw=False)


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    """Correlation with NaN handling."""
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 3:
        return float("nan")
    if df.iloc[:, 0].std() == 0 or df.iloc[:, 1].std() == 0:
        return float("nan")
    return float(df.iloc[:, 0].corr(df.iloc[:, 1]))


def segment_correlation_stability(
    x: pd.Series,
    y: pd.Series,
    segments: int,
    min_abs_corr: float,
) -> float:
    """Share of segments whose correlation aligns with overall sign and magnitude."""
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < segments * 5:
        return 0.0
    overall = safe_corr(df.iloc[:, 0], df.iloc[:, 1])
    if not np.isfinite(overall) or overall == 0:
        return 0.0
    sign = np.sign(overall)
    splits = np.array_split(df.index, segments)
    valid = 0
    stable = 0
    for idx in splits:
        if len(idx) < 5:
            continue
        sub = df.loc[idx]
        corr = safe_corr(sub.iloc[:, 0], sub.iloc[:, 1])
        if not np.isfinite(corr):
            continue
        valid += 1
        if np.sign(corr) == sign and abs(corr) >= min_abs_corr:
            stable += 1
    if valid == 0:
        return 0.0
    return stable / valid


def gas_year_key(date: pd.Timestamp, start_month: int) -> Tuple[int, int]:
    """Return (gas_year, day_index) for a given date and gas-year start month."""
    year = date.year
    if date.month < start_month:
        year -= 1
    start = pd.Timestamp(year=year, month=start_month, day=1)
    day_index = int((date - start).days)
    return year, day_index
