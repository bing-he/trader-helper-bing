from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def _zigzag_swings(series: pd.Series, pct_threshold: float = 0.02) -> List[float]:
    """Lightweight ZigZag detector that records swing highs/lows when price reverses by pct_threshold."""

    clean = series.dropna()
    if clean.empty:
        return []
    values = clean.values
    swings: List[float] = []
    last_extreme = values[0]
    direction = 0  # 1 = rising, -1 = falling
    for price in values[1:]:
        change = (price - last_extreme) / last_extreme if last_extreme else 0.0
        if direction >= 0 and change <= -pct_threshold:
            swings.append(last_extreme)
            direction = -1
            last_extreme = price
        elif direction <= 0 and change >= pct_threshold:
            swings.append(last_extreme)
            direction = 1
            last_extreme = price
        else:
            # Update the extreme within the current leg.
            if direction >= 0:
                if price > last_extreme:
                    last_extreme = price
            else:
                if price < last_extreme:
                    last_extreme = price
    swings.append(last_extreme)
    return swings


def identify_sr_levels(
    series: pd.Series,
    n_levels: int = 3,
    outlier_threshold: float = 1.5,
    max_multiple_of_current: float | None = 2.0,
) -> List[Tuple[float, int]]:
    """
    Identify the strongest support/resistance levels in a long-term price series.

    Steps:
      1. Apply a ZigZag algorithm to detect swing highs and lows (e.g. using peak/trough detection with a percentage threshold).
      2. Collect all swing prices and cluster them using a density-based algorithm (e.g. DBSCAN) or hierarchical clustering on the price values.
      3. Score each cluster by the number of swing points it contains (frequency of reversals) and the tightness of the price band.
      4. Penalise clusters where the level was often violated (optional).
      5. Return the top `n_levels` clusters’ median price and score, sorted by score.  Ensure the levels are sufficiently distinct (e.g. at least one standard deviation apart).
    """

    swings = _zigzag_swings(series)
    if not swings:
        return []

    swing_values = np.asarray(swings, dtype=float).reshape(-1, 1)
    eps = float(max(np.std(swings) * 0.1, 0.01))
    clusterer = DBSCAN(eps=eps, min_samples=2)
    labels = clusterer.fit_predict(swing_values)

    clusters: List[Tuple[float, int, float]] = []  # median, count, std
    unique_labels = set(labels)
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        if label == -1:
            # Treat noise as its own thin cluster.
            for idx in indices:
                val = float(swing_values[idx, 0])
                clusters.append((val, 1, 0.0))
            continue
        vals = swing_values[indices, 0]
        clusters.append((float(np.median(vals)), len(vals), float(np.std(vals))))

    if not clusters:
        return []

    # Score: more hits and tighter band -> higher score
    scored: List[Tuple[float, float]] = []
    for median, count, spread in clusters:
        score = count / (1.0 + spread)
        scored.append((median, score))

    # Outlier filtering based on IQR and optional multiple of current
    swings_series = pd.Series(swings)
    overall_median = swings_series.median()
    iqr = swings_series.quantile(0.75) - swings_series.quantile(0.25)
    upper = overall_median + outlier_threshold * iqr if not pd.isna(iqr) else np.inf
    lower = overall_median - outlier_threshold * iqr if not pd.isna(iqr) else -np.inf
    current_val = series.dropna().iloc[-1] if not series.dropna().empty else None
    filtered: List[Tuple[float, float]] = []
    for median, score in scored:
        if median < lower or median > upper:
            continue
        if max_multiple_of_current is not None and current_val is not None:
            if median > current_val * max_multiple_of_current:
                continue
        filtered.append((median, score))

    scored = sorted(filtered, key=lambda x: x[1], reverse=True)

    # Enforce distinctness
    distinct: List[Tuple[float, float]] = []
    level_std = float(np.std(swings)) if len(swings) > 1 else 0.0
    min_gap = level_std if level_std > 0 else max(abs(np.mean(swings)) * 0.05, 0.05)
    for median, score in scored:
        if all(abs(median - existing[0]) >= min_gap for existing in distinct):
            distinct.append((median, score))
        if len(distinct) >= n_levels:
            break

    return [(level, int(round(score))) if score >= 1 else (level, 1) for level, score in distinct]


def last_year_series(series: pd.Series) -> pd.Series:
    """
    Given a pandas Series indexed by date, return the portion covering the last
    365 days relative to series.index.max(). Assumes daily data.
    """

    if series.empty:
        return series
    end_date = series.index.max()
    start_date = end_date - pd.Timedelta(days=365)
    return series.loc[start_date:]
