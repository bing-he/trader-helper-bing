"""Analytics and feature engineering for the LNG dashboard."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .config import DashboardConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class TimeSeriesBundle:
    """Container for actual and forecast LNG time series."""

    actual: pd.DataFrame
    forecast: pd.DataFrame
    total_actual: pd.DataFrame
    total_forecast: pd.DataFrame
    combined: pd.DataFrame
    forecast_start: Optional[pd.Timestamp]


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.groupby(["date", "facility"], as_index=False)["value"]
        .sum()
        .sort_values(["facility", "date"])
    )


def build_timeseries_bundle(actual_df: pd.DataFrame, forecast_df: pd.DataFrame) -> TimeSeriesBundle:
    """Aggregate time series for actuals and forecast."""
    actual = _aggregate(actual_df)
    forecast = _aggregate(forecast_df)
    total_actual = (
        actual.groupby("date", as_index=False)["value"].sum().assign(facility="Total")
        if not actual.empty
        else pd.DataFrame(columns=["date", "value", "facility"])
    )
    total_forecast = (
        forecast.groupby("date", as_index=False)["value"].sum().assign(facility="Total")
        if not forecast.empty
        else pd.DataFrame(columns=["date", "value", "facility"])
    )
    forecast_start = forecast["date"].min() if not forecast.empty else None
    combined = pd.concat(
        [
            actual.assign(segment="actual"),
            forecast.assign(segment="forecast"),
        ],
        ignore_index=True,
    )
    return TimeSeriesBundle(
        actual=actual,
        forecast=forecast,
        total_actual=total_actual,
        total_forecast=total_forecast,
        combined=combined,
        forecast_start=forecast_start,
    )


def _delta(series: pd.Series, latest_date: pd.Timestamp, window: int) -> Optional[float]:
    target_date = latest_date - pd.Timedelta(days=window)
    try:
        target_value = series.loc[target_date]
    except KeyError:
        return None
    return float(series.loc[latest_date] - target_value)


def _trailing_stats(series: pd.Series, latest_date: pd.Timestamp, window: int) -> Tuple[Optional[float], Optional[float]]:
    window_start = latest_date - pd.Timedelta(days=window - 1)
    window_series = series.loc[series.index >= window_start]
    if window_series.empty:
        return None, None
    return float(window_series.mean()), float(window_series.std(ddof=0))


def _latest_value(series: pd.Series) -> Tuple[Optional[pd.Timestamp], Optional[float]]:
    if series.empty:
        return None, None
    latest_date = series.index.max()
    return latest_date, float(series.loc[latest_date])


def classify_regime(delta_5d: Optional[float], threshold: float) -> str:
    """Classify Ramp/Flat/Decline using a symmetric threshold."""
    if delta_5d is None:
        return "Flat"
    if delta_5d > threshold:
        return "Ramp"
    if delta_5d < -threshold:
        return "Decline"
    return "Flat"


def detect_shock(
    series: pd.Series, latest_date: pd.Timestamp, window_days: int, quantile: float
) -> Dict[str, Optional[float]]:
    """Detect whether the latest 1d change is a top-decile move."""
    window_start = latest_date - pd.Timedelta(days=window_days)
    recent = series.loc[series.index >= window_start]
    if recent.shape[0] < 2:
        return {"is_shock": False, "threshold": None, "latest_abs_change": None}

    delta_series = recent.diff().dropna().abs()
    threshold = delta_series.quantile(quantile) if not delta_series.empty else None
    latest_abs_change = abs(recent.diff().iloc[-1]) if recent.shape[0] > 1 else None
    is_shock = bool(threshold is not None and latest_abs_change is not None and latest_abs_change >= threshold)
    return {
        "is_shock": is_shock,
        "threshold": float(threshold) if threshold is not None else None,
        "latest_abs_change": float(latest_abs_change) if latest_abs_change is not None else None,
    }


def compute_actual_metrics(actual_df: pd.DataFrame, config: DashboardConfig) -> Dict[str, Dict[str, object]]:
    """Compute level, change, statistics, regime, and shock metrics per facility."""
    metrics: Dict[str, Dict[str, object]] = {}
    for facility, df_fac in actual_df.groupby("facility"):
        series = df_fac.sort_values("date").set_index("date")["value"]
        latest_date, latest_value = _latest_value(series)
        if latest_date is None or latest_value is None:
            continue

        deltas = {
            "delta_1d": _delta(series, latest_date, 1),
            "delta_5d": _delta(series, latest_date, 5),
            "delta_10d": _delta(series, latest_date, 10),
        }

        trailing_mean: Dict[int, Optional[float]] = {}
        trailing_std: Dict[int, Optional[float]] = {}
        for window in config.trailing_windows_list():
            mean_val, std_val = _trailing_stats(series, latest_date, window)
            trailing_mean[window] = mean_val
            trailing_std[window] = std_val

        zscore = None
        mean60 = trailing_mean.get(60)
        std60 = trailing_std.get(60)
        if mean60 is not None and std60 not in (None, 0):
            zscore = (latest_value - mean60) / std60

        regime = classify_regime(deltas.get("delta_5d"), config.regime_delta_threshold)
        shock = detect_shock(series, latest_date, config.shock_window_days, config.shock_quantile)

        metrics[facility] = {
            "latest_date": latest_date,
            "latest_value": latest_value,
            "deltas": deltas,
            "trailing_mean": trailing_mean,
            "trailing_std": trailing_std,
            "zscore_60d": zscore,
            "regime": regime,
            "shock": shock,
        }
    return metrics


def _latest_actual_before(actual_df: pd.DataFrame, facility: str, as_of: pd.Timestamp) -> Optional[float]:
    subset = actual_df[(actual_df["facility"] == facility) & (actual_df["date"] <= as_of)]
    if subset.empty:
        return None
    subset = subset.sort_values("date")
    return float(subset.iloc[-1]["value"])


def compute_forecast_metrics(
    actual_df: pd.DataFrame, forecast_df: pd.DataFrame, config: DashboardConfig
) -> Dict[str, Dict[str, object]]:
    """Compute forecast diagnostics per facility."""
    metrics: Dict[str, Dict[str, object]] = {}
    if forecast_df.empty:
        return metrics

    forecast_start = forecast_df["date"].min()
    for facility, df_fac in forecast_df.groupby("facility"):
        df_fac = df_fac.sort_values("date")
        start_value = _latest_actual_before(actual_df, facility, forecast_start) or float(df_fac.iloc[0]["value"])
        horizon_changes: Dict[int, Optional[float]] = {}
        for horizon in config.forecast_horizons:
            target_date = forecast_start + pd.Timedelta(days=horizon)
            target = df_fac.loc[df_fac["date"] == target_date, "value"]
            horizon_changes[horizon] = float(target.iloc[0] - start_value) if not target.empty else None

        horizon_end = forecast_start + pd.Timedelta(days=config.forecast_maxmin_horizon)
        window_slice = df_fac[df_fac["date"] <= horizon_end]
        max_val = float(window_slice["value"].max()) if not window_slice.empty else None
        min_val = float(window_slice["value"].min()) if not window_slice.empty else None

        metrics[facility] = {
            "forecast_start": forecast_start,
            "start_value": start_value,
            "horizon_changes": horizon_changes,
            "max_next_14d": max_val,
            "min_next_14d": min_val,
        }
    return metrics


def select_top_facilities(actual_df: pd.DataFrame, top_n: int) -> List[str]:
    """Select the top-N facilities by latest value."""
    if actual_df.empty:
        return []
    latest_date = actual_df["date"].max()
    latest_values = actual_df[actual_df["date"] == latest_date].set_index("facility")["value"]
    top = latest_values.nlargest(top_n).index.tolist()
    return top


def build_forecast_contributions(
    forecast_metrics: Dict[str, Dict[str, object]], horizon: int
) -> List[Dict[str, object]]:
    """Convert forecast metrics into a contribution table."""
    contributions: List[Dict[str, object]] = []
    for facility, metrics in forecast_metrics.items():
        change = metrics["horizon_changes"].get(horizon)
        contributions.append(
            {
                "facility": facility,
                "horizon_days": horizon,
                "expected_change": change,
            }
        )
    contributions.sort(key=lambda row: (row["expected_change"] is None, -(row["expected_change"] or 0)))
    return contributions
