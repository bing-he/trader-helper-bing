"""Rolling feature engineering for CoT series."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

SERIES_COLUMNS: List[str] = [
    "Total_OI",
    "Total_MM_Net",
    "Total_Prod_Net",
    "Total_Swap_Net",
]


def _rolling_zscore(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std(ddof=0)
    z = (series - mean) / std
    z = z.where(std != 0, 0)
    return z


def compute_rolling_percentile(series: pd.Series, window: int, min_periods: Optional[int] = None) -> pd.Series:
    """
    Rolling percentile using rank(pct=True) on the trailing window.

    Args:
        series: Input series.
        window: Rolling window size.
        min_periods: Minimum periods required; defaults to ``window`` if not provided.
    """

    min_periods_resolved = window if min_periods is None else min_periods

    def _percentile(window_series: pd.Series) -> float:
        ranked = window_series.rank(pct=True)
        return float(ranked.iloc[-1]) if not ranked.empty else np.nan

    return series.rolling(window=window, min_periods=min_periods_resolved).apply(
        _percentile, raw=False
    )


def compute_cot_features(
    cot_df: pd.DataFrame,
    min_periods: int,
    windows: Iterable[int] = (52,),
    percentile_windows: Iterable[int] = (52, 156),
) -> pd.DataFrame:
    """Attach rolling z-scores and percentile ranks to a CoT dataframe."""

    df = cot_df.copy()
    for col in SERIES_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required CoT column '{col}'.")
        for window in windows:
            df[f"{col}_z_{window}"] = _rolling_zscore(df[col], window=window, min_periods=min_periods)
        for window in percentile_windows:
            percentile_min = 26 if window == 52 else 78 if window == 156 else min_periods
            df[f"{col}_pct_{window}"] = compute_rolling_percentile(
                df[col], window=window, min_periods=percentile_min
            )
    return df


def feature_columns() -> Dict[str, List[str]]:
    """Return mapping of base series to their derived feature columns."""

    derived: Dict[str, List[str]] = {}
    for col in SERIES_COLUMNS:
        derived[col] = [
            f"{col}_z_52",
            f"{col}_pct_52",
            f"{col}_pct_156",
        ]
    return derived
