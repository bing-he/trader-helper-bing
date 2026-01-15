#!/usr/bin/env python3
"""
EIA Weekly Storage Forecaster - Adaptive Feature Selection with Weather

This version automatically selects between baseline and enhanced models
based on validation performance for each region.

Enhanced features:
- Changepoint detection (ruptures)
- Normalized drawdown measure
- Weather features (HDD/CDD) - current, lagged, and rolling averages

Key improvement: Per-region adaptive model selection based on validation MAE.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------


def _get_logger(name: str, quiet: bool = False) -> logging.Logger:
    """Create and return a configured logger."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    level = logging.WARNING if quiet else logging.INFO
    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# ----------------------------------------------------------------------------
# Regions
# ----------------------------------------------------------------------------

REGIONS = {
    "lower48": "Lower 48",
    "east": "East",
    "midwest": "Midwest",
    "southcentral": "South Central",
    "mountain": "Mountain",
    "pacific": "Pacific",
}

REGION_COLORS = {
    "lower48": "#1f77b4",
    "east": "#ff7f0e",
    "midwest": "#2ca02c",
    "southcentral": "#d62728",
    "mountain": "#9467bd",
    "pacific": "#8c564b",
}

# ----------------------------------------------------------------------------
# Data prep
# ----------------------------------------------------------------------------


def load_changes(changes_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load and sanitize weekly storage changes per region from CSV."""
    logger.info("Loading changes from %s", changes_path)
    df = pd.read_csv(changes_path, index_col=0, parse_dates=True)
    df = df.select_dtypes(include=[np.number])
    df.columns = df.columns.str.replace("_Change", "", regex=False).str.lower()
    present = [c for c in df.columns if c in REGIONS]
    if not present:
        raise ValueError("No known region columns found in changes CSV")
    df = df[present]
    df = df.ffill().bfill().sort_index()
    logger.info("Loaded %d rows, columns: %s", len(df), list(df.columns))
    return df


def load_weather(weather_path: Path, logger: logging.Logger) -> pd.DataFrame:
    """Load weather data and aggregate HDD/CDD by week."""
    logger.info("Loading weather from %s", weather_path)
    df = pd.read_csv(weather_path, parse_dates=["Date"])

    # Aggregate HDD/CDD across all cities by date
    daily_agg = df.groupby("Date").agg({"HDD": "sum", "CDD": "sum"}).sort_index()

    # Resample to weekly (Thursday) to match EIA report timing
    weekly = daily_agg.resample("W-THU").sum()
    weekly.index.name = "Date"

    logger.info(
        "Loaded weather: %d weeks, range: %s to %s",
        len(weekly),
        weekly.index.min().date(),
        weekly.index.max().date(),
    )
    return weekly


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------


def detect_changepoints_conservative(
    series: pd.Series, min_size: int = 20, penalty: float = 5.0
) -> np.ndarray:
    """Detect changepoints using ruptures with conservative settings."""
    if len(series) < min_size * 2:
        return np.array([])

    try:
        algo = rpt.Pelt(model="rbf", min_size=min_size).fit(series.values)
        changepoints = algo.predict(pen=penalty)
        changepoints = np.array(changepoints[:-1]) if changepoints else np.array([])
        return changepoints
    except Exception:
        return np.array([])


def compute_normalized_drawdown(series: pd.Series, window: int = 13) -> pd.Series:
    """Compute normalized drawdown measure for storage changes."""
    running_max = series.rolling(window=window, min_periods=1).max()
    drawdown = series - running_max
    rolling_std = series.rolling(window=window, min_periods=1).std()
    normalized_dd = drawdown / (rolling_std + 1e-6)
    return normalized_dd


def make_features_baseline(y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Build BASELINE features (original model)."""
    df = pd.DataFrame({"y": y})

    # lags 1..8
    for lag in range(1, 9):
        df[f"lag{lag}"] = df["y"].shift(lag)

    # seasonal lag 52
    df["lag52"] = df["y"].shift(52)

    # rolling means
    df["roll4"] = df["y"].shift(1).rolling(4).mean()
    df["roll8"] = df["y"].shift(1).rolling(8).mean()

    df = df.dropna()
    X = df.drop(columns=["y"])
    yy = df["y"].copy()
    return X, yy


def make_features_enhanced(
    y: pd.Series, weather: pd.DataFrame = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build ENHANCED features (with changepoints, normalized drawdown, and weather)."""
    df = pd.DataFrame({"y": y})

    # lags 1..8
    for lag in range(1, 9):
        df[f"lag{lag}"] = df["y"].shift(lag)

    # seasonal lag 52
    df["lag52"] = df["y"].shift(52)

    # rolling means
    df["roll4"] = df["y"].shift(1).rolling(4).mean()
    df["roll8"] = df["y"].shift(1).rolling(8).mean()

    # ENHANCED: Normalized drawdown
    df["norm_dd"] = compute_normalized_drawdown(df["y"], window=13)

    # ENHANCED: Changepoint features
    changepoints = detect_changepoints_conservative(df["y"].dropna())
    df["weeks_since_cp"] = 0.0
    if len(changepoints) > 0:
        for i in range(len(df)):
            prior_cps = changepoints[changepoints < i]
            if len(prior_cps) > 0:
                df.iloc[i, df.columns.get_loc("weeks_since_cp")] = i - prior_cps[-1]
            else:
                df.iloc[i, df.columns.get_loc("weeks_since_cp")] = i
    else:
        df["weeks_since_cp"] = np.arange(len(df))

    df["weeks_since_cp"] = df["weeks_since_cp"].clip(upper=52)

    # ENHANCED: Weather features (HDD/CDD)
    if weather is not None:
        # Merge weather data
        weather_aligned = weather.reindex(df.index)
        df["hdd"] = weather_aligned["HDD"].fillna(method="ffill").fillna(0)
        df["cdd"] = weather_aligned["CDD"].fillna(method="ffill").fillna(0)

        # Add lagged weather (previous week's weather affects current storage change)
        df["hdd_lag1"] = df["hdd"].shift(1)
        df["cdd_lag1"] = df["cdd"].shift(1)

        # Rolling averages of weather (4-week trends)
        df["hdd_roll4"] = df["hdd"].shift(1).rolling(4).mean()
        df["cdd_roll4"] = df["cdd"].shift(1).rolling(4).mean()

    df = df.dropna()
    X = df.drop(columns=["y"])
    yy = df["y"].copy()
    return X, yy


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------


@dataclass
class RegionForecast:
    """Container for a single region's forecast and metadata."""

    point: float
    ci95: Tuple[float, float]
    last_date: pd.Timestamp
    next_date: pd.Timestamp
    history: pd.Series
    model_used: str  # 'baseline' or 'enhanced'
    val_mae_baseline: float
    val_mae_enhanced: float
    mae_improvement_pct: float


def next_thursday(after_date: pd.Timestamp) -> pd.Timestamp:
    """Return the next Thursday strictly after the given date."""
    delta = (3 - after_date.weekday()) % 7
    delta = 7 if delta == 0 else delta
    return after_date + pd.Timedelta(days=delta)


def fit_and_forecast_region(
    name: str, series: pd.Series, logger: logging.Logger, weather: pd.DataFrame = None
) -> RegionForecast:
    """Fit baseline and enhanced models, select best based on validation MAE."""
    series = series.dropna().sort_index()
    if len(series) < 70:
        raise ValueError(f"Not enough history for region {name}")

    # Build both feature sets
    X_baseline, y_baseline = make_features_baseline(series)
    X_enhanced, y_enhanced = make_features_enhanced(series, weather=weather)

    if len(X_baseline) < 60:
        raise ValueError(f"Not enough feature rows for region {name}")

    # Validation split
    val_horizon = min(12, max(6, len(X_baseline) // 10))
    split_baseline = len(X_baseline) - val_horizon
    split_enhanced = len(X_enhanced) - val_horizon

    # Baseline splits
    X_tr_base, y_tr_base = (
        X_baseline.iloc[:split_baseline],
        y_baseline.iloc[:split_baseline],
    )
    X_val_base, y_val_base = (
        X_baseline.iloc[split_baseline:],
        y_baseline.iloc[split_baseline:],
    )

    # Enhanced splits
    X_tr_enh, y_tr_enh = (
        X_enhanced.iloc[:split_enhanced],
        y_enhanced.iloc[:split_enhanced],
    )
    X_val_enh, y_val_enh = (
        X_enhanced.iloc[split_enhanced:],
        y_enhanced.iloc[split_enhanced:],
    )

    # ========== BASELINE MODEL ==========
    model_baseline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=3.0, solver="auto")),
        ]
    )
    model_baseline.fit(X_tr_base, y_tr_base)
    val_pred_base = model_baseline.predict(X_val_base)
    mae_baseline = float(mean_absolute_error(y_val_base, val_pred_base))

    # ========== ENHANCED MODEL ==========
    model_enhanced = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=5.0, solver="auto")),
        ]
    )
    model_enhanced.fit(X_tr_enh, y_tr_enh)
    val_pred_enh = model_enhanced.predict(X_val_enh)
    mae_enhanced = float(mean_absolute_error(y_val_enh, val_pred_enh))

    # ========== ADAPTIVE SELECTION ==========
    if mae_enhanced < mae_baseline:
        # Use enhanced model
        model_to_use = model_enhanced
        X_next, _ = make_features_enhanced(series, weather=weather)
        model_type = "enhanced"
        mae_to_use = mae_enhanced
    else:
        # Use baseline model
        model_to_use = model_baseline
        X_next, _ = make_features_baseline(series)
        model_type = "baseline"
        mae_to_use = mae_baseline

    # ========== FORECAST ==========
    last_date = series.index[-1]
    x_last = X_next.iloc[[-1]]
    point = float(model_to_use.predict(x_last)[0])

    half_width = 1.96 * mae_to_use
    ci95 = (point - half_width, point + half_width)

    mae_improvement_pct = ((mae_enhanced - mae_baseline) / mae_baseline) * 100

    return RegionForecast(
        point=point,
        ci95=ci95,
        last_date=last_date,
        next_date=next_thursday(last_date),
        history=series.tail(30),
        model_used=model_type,
        val_mae_baseline=mae_baseline,
        val_mae_enhanced=mae_enhanced,
        mae_improvement_pct=mae_improvement_pct,
    )


# ----------------------------------------------------------------------------
# Plotting and output
# ----------------------------------------------------------------------------


def plot_region_grid(
    forecasts: Dict[str, RegionForecast], output_dir: Path, logger: logging.Logger
) -> None:
    """Render a 2×3 grid plot of forecasts."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for idx, (key, label) in enumerate(REGIONS.items()):
        ax = axes[idx]
        if key not in forecasts:
            ax.axis("off")
            continue
        fc = forecasts[key]
        color = REGION_COLORS.get(key, "#1f77b4")

        ax.plot(fc.history.index, fc.history.values, color=color, lw=2, label="History")
        ax.errorbar(
            [fc.next_date],
            [fc.point],
            yerr=[[fc.point - fc.ci95[0]], [fc.ci95[1] - fc.point]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=5,
            label=f"Forecast ({fc.model_used})",
        )
        ax.set_title(
            f"{label} [{fc.model_used.upper()}]", fontsize=12, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", lw=0.6, alpha=0.4)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        "EIA Weekly Storage Forecast - Adaptive Model + Weather",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "eia_region_forecast_adaptive_weather.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure -> %s", out_path)


def save_outputs(
    forecasts: Dict[str, RegionForecast], output_dir: Path, logger: logging.Logger
) -> None:
    """Save forecast CSV and model selection summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Forecast table
    forecast_df = pd.DataFrame(
        {
            "region": [REGIONS[k] for k in forecasts.keys()],
            "key": list(forecasts.keys()),
            "point": [f.point for f in forecasts.values()],
            "ci95_low": [f.ci95[0] for f in forecasts.values()],
            "ci95_high": [f.ci95[1] for f in forecasts.values()],
            "next_date": [f.next_date for f in forecasts.values()],
            "model_used": [f.model_used for f in forecasts.values()],
        }
    )
    forecast_path = output_dir / "eia_region_forecast_adaptive_weather.csv"
    forecast_df.to_csv(forecast_path, index=False)
    logger.info("Saved forecast -> %s", forecast_path)

    # Model selection summary
    summary_df = pd.DataFrame(
        {
            "region": [REGIONS[k] for k in forecasts.keys()],
            "model_used": [f.model_used for f in forecasts.values()],
            "mae_baseline": [f.val_mae_baseline for f in forecasts.values()],
            "mae_enhanced": [f.val_mae_enhanced for f in forecasts.values()],
            "mae_improvement_pct": [f.mae_improvement_pct for f in forecasts.values()],
        }
    )
    summary_path = output_dir / "model_selection_summary_weather.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved model selection summary -> %s", summary_path)

    # Console summary
    logger.info("\n" + "=" * 70)
    logger.info("ADAPTIVE MODEL SELECTION RESULTS (WITH WEATHER)")
    logger.info("=" * 70)
    for _, row in summary_df.iterrows():
        selected = "✓" if row["model_used"] == "enhanced" else "○"
        logger.info(
            f"{selected} {row['region']:15s} | Model: {row['model_used']:8s} | "
            f"Base MAE: {row['mae_baseline']:6.2f} | "
            f"Enhanced MAE: {row['mae_enhanced']:6.2f} | "
            f"Change: {row['mae_improvement_pct']:+6.1f}%"
        )

    enhanced_count = summary_df[summary_df["model_used"] == "enhanced"].shape[0]
    logger.info("=" * 70)
    logger.info(f"Enhanced model selected: {enhanced_count}/6 regions")
    logger.info("=" * 70 + "\n")

    # Print forecast table
    logger.info("=" * 90)
    logger.info("NEXT WEEK FORECAST - STORAGE CHANGE (BCF)")
    logger.info("=" * 90)
    logger.info(
        f"{'Region':<15s} | {'Forecast':>10s} | {'95% CI Low':>10s} | "
        f"{'95% CI High':>10s} | {'Date':>12s} | {'Model':>8s}"
    )
    logger.info("-" * 90)
    for _, row in forecast_df.iterrows():
        logger.info(
            f"{row['region']:<15s} | {row['point']:>10.1f} | {row['ci95_low']:>10.1f} | "
            f"{row['ci95_high']:>10.1f} | {row['next_date'].strftime('%Y-%m-%d'):>12s} | "
            f"{row['model_used']:>8s}"
        )
    logger.info("=" * 90 + "\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    """Entry point for adaptive EIA forecaster with weather."""
    parser = argparse.ArgumentParser(
        description="Adaptive EIA weekly forecaster with weather"
    )
    parser.add_argument(
        "--changes",
        type=Path,
        default=Path(__file__).parent.parent / "INFO" / "EIAchanges.csv",
        help="CSV with weekly storage changes",
    )
    parser.add_argument(
        "--weather",
        type=Path,
        default=Path(__file__).parent.parent / "INFO" / "WEATHER.csv",
        help="CSV with weather data (HDD/CDD)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "output_adaptive_weather",
        help="Output directory",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logger = _get_logger("EIAAdaptiveWeather", quiet=args.quiet)

    try:
        changes = load_changes(args.changes, logger)
    except Exception as e:
        logger.error(f"Failed to load changes: {e}")
        return 1

    # Load weather data if available
    weather = None
    if args.weather.exists():
        try:
            weather = load_weather(args.weather, logger)
        except Exception as e:
            logger.warning(
                f"Failed to load weather data: {e}. Continuing without weather features."
            )
    else:
        logger.warning(
            f"Weather file not found: {args.weather}. Continuing without weather features."
        )

    forecasts: Dict[str, RegionForecast] = {}
    for key in REGIONS.keys():
        if key not in changes.columns:
            logger.warning("Skipping %s: not in changes file", key)
            continue
        try:
            fc = fit_and_forecast_region(key, changes[key], logger, weather=weather)
            logger.info(
                "%s: %.2f (95%% CI: %.2f, %.2f) for %s [%s model]",
                REGIONS[key],
                fc.point,
                fc.ci95[0],
                fc.ci95[1],
                fc.next_date.date(),
                fc.model_used.upper(),
            )
            forecasts[key] = fc
        except Exception as e:
            logger.warning("Region %s failed: %s", key, e)
            continue

    if forecasts:
        try:
            plot_region_grid(forecasts, args.output, logger)
            save_outputs(forecasts, args.output, logger)
        except Exception as e:
            logger.warning("Output generation failed: %s", e)
    else:
        logger.warning("No regional forecasts produced")

    logger.info("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
