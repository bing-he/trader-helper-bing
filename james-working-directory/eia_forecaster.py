"""EIA storage forecasting playground using combined fundamentals and weather.

This module builds a weekly feature matrix from the wide combined daily
historical data, trains gradient boosting models for EIA storage changes
by region, and applies them to the latest forecast feature set to obtain
next-week EIA predictions suitable for dashboard use.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.ensemble import GradientBoostingRegressor

try:
    from tabulate import tabulate
except Exception:  # pragma: no cover - optional dependency
    tabulate = None

try:
    import optuna
except ImportError:
    optuna = None

# ---------------------------------------------------------------------------
# Configuration (no CLI or environment parsing)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRADER_HELPER_ROOT = PROJECT_ROOT
INFO_ROOT = TRADER_HELPER_ROOT / "INFO"
COMPILED_CSV_DIR = INFO_ROOT / "CompiledInfo" / "csv"

COMBINED_HISTORICAL_PATH = COMPILED_CSV_DIR / "combined_daily_historical_wide.csv"
COMBINED_FORECAST_PATH = COMPILED_CSV_DIR / "combined_daily_forecast_wide.csv"
EIA_TOTALS_PATH = INFO_ROOT / "EIAtotals.csv"
EIA_CHANGES_PATH = INFO_ROOT / "EIAchanges.csv"

EIA_WEEKDAY = 4
MODEL_MAX_DEPTH = 2
MODEL_LEARNING_RATE = 0.05
MODEL_N_ESTIMATORS = 200
MODEL_MIN_SAMPLES_LEAF = 5
TRAIN_FRACTION = 0.75
OPTUNA_TRIALS = 0  # Set >0 to trigger Optuna tuning before training
RNG_SEED = 1234
OPTUNA_SAMPLER_SEED = RNG_SEED
SKLEARN_SEED = 1234
LOG_LEVEL_NAME = "INFO"


def configure_logging() -> logging.Logger:
    """Configure and return a module-level logger."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL_NAME, logging.INFO),
        format="%(asctime)s %(levelname)-8s eia_forecaster | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("eia_forecaster")


LOG = configure_logging()


def configure_determinism(seed: int = RNG_SEED) -> None:
    """Seed Python and NumPy RNGs for reproducible behavior."""
    # Keep RNGs aligned so Optuna sampling and model training remain deterministic across runs.
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class EIAModelBundle:
    """Container for trained EIA models and metadata."""

    feature_columns: List[str]
    target_columns: List[str]
    models: Dict[str, GradientBoostingRegressor]


@dataclass
class EIAForecast:
    """Container for a single-week EIA forecast and levels."""

    week_ending: pd.Timestamp
    changes: Dict[str, np.float64]
    levels: Dict[str, np.float64]


def _require_file(path: Path, label: str) -> None:
    """Validate that a required input file exists."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} file: {path}")


REQUIRED_INPUT_PATHS: Mapping[str, Path] = {
    "combined historical": COMBINED_HISTORICAL_PATH,
    "combined forecast": COMBINED_FORECAST_PATH,
    "EIAtotals": EIA_TOTALS_PATH,
    "EIAchanges": EIA_CHANGES_PATH,
}


def validate_required_paths() -> None:
    """Ensure all upstream CSV inputs are discoverable before running the pipeline."""
    missing = [
        f"{label}: {path}"
        for label, path in REQUIRED_INPUT_PATHS.items()
        if not path.exists()
    ]
    # Raise early so we never enter the training path without all input tables present.
    if missing:
        raise FileNotFoundError("Missing required input paths:\n" + "\n".join(missing))


def _compute_week_ending(dates: pd.Series, weekday: int = EIA_WEEKDAY) -> pd.Series:
    """Map calendar dates to the EIA week-ending date for grouping.

    The EIA storage report uses a Friday week ending. This helper maps
    each daily observation to the upcoming Friday of the same ISO week.
    """
    dt = pd.to_datetime(dates)
    delta = (weekday - dt.dt.weekday) % 7
    return dt + pd.to_timedelta(delta, unit="D")


def _flatten_agg_columns(columns: Iterable[Tuple[str, str]]) -> List[str]:
    """Flatten a two-level aggregation column index into simple labels."""
    return [f"{base}__{stat}" for base, stat in columns]


def load_combined_frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load historical and forecast combined daily feature frames."""
    _require_file(COMBINED_HISTORICAL_PATH, "combined historical wide")
    _require_file(COMBINED_FORECAST_PATH, "combined forecast wide")

    # The historical and forecast frames are expected to share column names so they can be aligned later.
    hist = pd.read_csv(COMBINED_HISTORICAL_PATH)
    fc = pd.read_csv(COMBINED_FORECAST_PATH)

    if "Date" not in hist.columns or "Date" not in fc.columns:
        raise ValueError("Both combined CSVs must contain a 'Date' column.")

    hist["Date"] = pd.to_datetime(hist["Date"])
    fc["Date"] = pd.to_datetime(fc["Date"])

    hist = hist.sort_values("Date").reset_index(drop=True)
    fc = fc.sort_values("Date").reset_index(drop=True)

    LOG.info(
        "Loaded combined frames: historical=%d rows, forecast=%d rows",
        len(hist),
        len(fc),
    )
    return hist, fc


def load_eia_weekly() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load weekly EIA totals and changes."""
    _require_file(EIA_TOTALS_PATH, "EIAtotals")
    _require_file(EIA_CHANGES_PATH, "EIAchanges")

    totals = pd.read_csv(EIA_TOTALS_PATH)
    changes = pd.read_csv(EIA_CHANGES_PATH)

    if "Period" not in totals.columns or "Period" not in changes.columns:
        raise ValueError("EIA CSVs must contain a 'Period' column.")

    totals["Period"] = pd.to_datetime(totals["Period"])
    changes["Period"] = pd.to_datetime(changes["Period"])

    totals = totals.sort_values("Period").reset_index(drop=True)
    changes = changes.sort_values("Period").reset_index(drop=True)

    # Totals and changes remain in lockstep by period so downstream mapping to storage levels is deterministic.
    return totals, changes


def derive_feature_mappings(
    hist: pd.DataFrame,
    fc: pd.DataFrame,
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """Derive mappings between historical and forecast feature columns.

    Historical features use realized fundamentals and weather series such as
    ``Fundy_CONUS_-_Balance__CONUS`` and ``Weather_Min_Temp_Minneapolis__CONUS``.
    The forecast frame exposes analogous forecast series with prefixes like
    ``Fundy_FC_`` and ``Weather_FC_``. This helper links the two so that
    models can be trained on realized history and applied using forecasts.
    """
    hist_cols = set(hist.columns)
    fc_cols = list(fc.columns)

    prefix_pairs: list[Tuple[str, str]] = [
        ("Fundy_FC_", "Fundy_"),
        ("Weather_FC_", "Weather_"),
        ("LNG_Flow_FC_", "LNG_Flow_"),
        ("CriterionNuclear_FC_", "CriterionNuclear_"),
    ]

    fc_to_hist: Dict[str, str] = {}

    for col in fc_cols:
        for fc_prefix, hist_prefix in prefix_pairs:
            if col.startswith(fc_prefix):
                candidate = hist_prefix + col[len(fc_prefix) :]
                if candidate in hist_cols:
                    fc_to_hist[col] = candidate
                break

    if not fc_to_hist:
        raise ValueError("No forecast-to-historical feature mappings could be derived.")

    hist_features = sorted(set(fc_to_hist.values()))
    fc_features = sorted(set(fc_to_hist.keys()))

    LOG.info(
        "Using %d historical driver features with forecast counterparts.",
        len(hist_features),
    )
    # The derived mapping ties forecast-only columns to their historical twins so training/forecast frames align.
    return hist_features, fc_features, fc_to_hist


def aggregate_weekly_features(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    weekday: int = EIA_WEEKDAY,
) -> pd.DataFrame:
    """Aggregate daily forecast features to a weekly panel keyed by week end.

    Parameters
    ----------
    df:
        Daily combined DataFrame containing a 'Date' column and feature columns.
    feature_columns:
        Subset of columns to aggregate.
    weekday:
        Target weekday (0=Mon) that defines the EIA week ending.
    """
    if "Date" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'Date' column.")

    cols = [c for c in feature_columns if c in df.columns]
    if not cols:
        raise ValueError("No requested feature columns are present in the DataFrame.")

    working = df[["Date"] + cols].copy()
    working["WeekEnding"] = _compute_week_ending(working["Date"], weekday=weekday)

    grouped = working.groupby("WeekEnding", as_index=True)[cols].agg(
        ["mean", "min", "max"]
    )
    grouped.columns = _flatten_agg_columns(grouped.columns)  # type: ignore[arg-type]
    grouped = grouped.sort_index()

    return grouped


def build_training_matrix(
    hist: pd.DataFrame,
    totals: pd.DataFrame,
    changes: pd.DataFrame,
    feature_columns: Iterable[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Construct the weekly training matrix with aligned targets."""
    weekly_features = aggregate_weekly_features(
        hist, feature_columns, weekday=EIA_WEEKDAY
    )

    weekly_changes = changes.copy()
    weekly_changes = weekly_changes.rename(columns={"Period": "WeekEnding"})

    target_columns = [c for c in weekly_changes.columns if c.endswith("_Change")]
    if not target_columns:
        raise ValueError("No *_Change target columns found in EIAchanges.")

    merged = weekly_changes.merge(
        weekly_features,
        how="inner",
        left_on="WeekEnding",
        right_index=True,
    )

    merged = merged.sort_values("WeekEnding").set_index("WeekEnding")

    # Ensure targets are present and drop rows without a valid EIA observation.
    merged = merged.dropna(subset=target_columns)

    # Separate targets and candidate feature columns.
    feature_cols = [c for c in merged.columns if c not in target_columns]

    # Drop feature columns that are entirely missing.
    non_empty_features: list[str] = []
    for col in feature_cols:
        if merged[col].notna().any():
            non_empty_features.append(col)

    if not non_empty_features:
        raise ValueError("No non-empty feature columns available for training matrix.")

    # Impute remaining feature NaNs with the column median to keep more weeks.
    features_frame = merged[non_empty_features].copy()
    medians = features_frame.median(axis=0)
    features_frame = features_frame.fillna(medians)

    cleaned = pd.concat(
        [merged[target_columns], features_frame],
        axis=1,
    )

    if cleaned.empty:
        raise ValueError("Training matrix is empty after target alignment.")

    LOG.info(
        "Training matrix shape: %s (targets=%d, features=%d)",
        cleaned.shape,
        len(target_columns),
        len(non_empty_features),
    )
    # After median imputation we log the matrix shape so downstream diagnostics know how much data persisted.
    return cleaned, target_columns


def _prepare_feature_matrix(
    matrix: pd.DataFrame, target_columns: Iterable[str]
) -> tuple[list[str], list[str], np.ndarray, int, int]:
    """Return the ordered targets, feature list, and train/validation split metadata."""
    target_list = list(target_columns)
    feature_cols = [c for c in matrix.columns if c not in target_list]
    X = matrix[feature_cols].values
    n = len(matrix)
    split = int(n * TRAIN_FRACTION)
    if split <= 0 or split >= n:
        split = n
    return target_list, feature_cols, X, split, n


def _optimize_hyperparameters(
    matrix: pd.DataFrame,
    target_columns: Iterable[str],
    n_trials: int,
    seed: int,
) -> Mapping[str, np.float64]:
    """Use Optuna to search for gradient boosting hyperparameters minimizing MAE."""
    if optuna is None:
        raise RuntimeError("Optuna is not available in this environment.")

    target_list, _, X, split, n = _prepare_feature_matrix(matrix, target_columns)
    if split >= n:
        LOG.warning("Insufficient data for a validation split; skipping optimization.")
        return {}

    X_train = X[:split]
    X_val = X[split:]

    def objective(trial: "optuna.trial.Trial") -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "random_state": seed,
        }

        total_mae = 0.0
        count = 0
        for col in target_list:
            y = matrix[col].values
            y_train = y[:split]
            y_val = y[split:]

            model = GradientBoostingRegressor(**params)
            model.fit(X_train, y_train)

            if len(y_val) > 0:
                preds = model.predict(X_val)
                total_mae += float(np.mean(np.abs(y_val - preds)))
                count += 1

        if count == 0:
            return float("inf")
        # Optuna minimizes the mean MAE across regions so no single target dominates.
        return total_mae / count

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)
    LOG.info("Optuna best params (avg MAE across targets): %s", study.best_params)
    return study.best_params


def train_eia_models(
    matrix: pd.DataFrame,
    target_columns: Iterable[str],
    random_state: int = SKLEARN_SEED,
    tuned_params: Mapping[str, Mapping[str, np.float64]] | None = None,
) -> EIAModelBundle:
    """Train gradient boosting models for each regional EIA change target."""
    target_list, feature_cols, X, split, n = _prepare_feature_matrix(
        matrix, target_columns
    )
    X_train = X[:split]
    X_val = X[split:] if split < n else None

    models: Dict[str, GradientBoostingRegressor] = {}

    tuned_params = tuned_params or {}

    for col in target_list:
        y = matrix[col].values
        y_train = y[:split]

        # Apply Optuna/override values when provided; fall back to conservative defaults otherwise.
        overrides = tuned_params.get(col, {}) if tuned_params else {}
        model = GradientBoostingRegressor(
            max_depth=int(overrides.get("max_depth", MODEL_MAX_DEPTH)),
            learning_rate=np.float64(
                overrides.get("learning_rate", MODEL_LEARNING_RATE)
            ),
            n_estimators=int(overrides.get("n_estimators", MODEL_N_ESTIMATORS)),
            min_samples_leaf=int(
                overrides.get("min_samples_leaf", MODEL_MIN_SAMPLES_LEAF)
            ),
            random_state=int(overrides.get("random_state", random_state)),
        )
        model.fit(X_train, y_train)
        models[col] = model

        if X_val is not None and len(X_val) > 0:
            y_val = y[split:]
            pred = model.predict(X_val)
            mse = np.float64(np.mean((y_val - pred) ** 2))
            mae = np.float64(np.mean(np.abs(y_val - pred)))
            LOG.info("Validation %s: MSE=%.3f MAE=%.3f", col, mse, mae)

    bundle = EIAModelBundle(
        feature_columns=feature_cols,
        target_columns=target_list,
        models=models,
    )
    return bundle


def _next_forecast_week(
    fc_weekly: pd.DataFrame,
    last_eia_week: pd.Timestamp,
) -> pd.Timestamp:
    """Determine the next available forecast week-ending date."""
    weeks = fc_weekly.index.to_series().sort_values().unique()
    future_weeks = [w for w in weeks if w > last_eia_week]
    if future_weeks:
        return future_weeks[0]
    return weeks[-1]


def build_forecast_feature_row(
    fc: pd.DataFrame,
    fc_feature_columns: Iterable[str],
    fc_to_hist: Mapping[str, str],
    last_eia_week: pd.Timestamp,
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """Prepare a single-row feature matrix for the next EIA week.

    The returned row uses the same aggregated feature column names as the
    training matrix (based on historical columns) so it can be passed
    directly into the trained models.
    """
    weekly_fc = aggregate_weekly_features(fc, fc_feature_columns, weekday=EIA_WEEKDAY)
    if weekly_fc.empty:
        raise ValueError("No weekly aggregates available from combined forecast frame.")

    target_week = _next_forecast_week(weekly_fc, last_eia_week)
    if target_week not in weekly_fc.index:
        raise ValueError(
            "Target forecast week not present in aggregated forecast frame."
        )

    row = weekly_fc.loc[[target_week]]

    # Rename aggregated forecast columns to align with historical feature names.
    rename_map: Dict[str, str] = {}
    stats = ("mean", "min", "max")
    for fc_col, hist_col in fc_to_hist.items():
        for stat in stats:
            fc_name = f"{fc_col}__{stat}"
            hist_name = f"{hist_col}__{stat}"
            if fc_name in weekly_fc.columns:
                rename_map[fc_name] = hist_name

    row = row.rename(columns=rename_map)
    return row, target_week


def _change_to_level_column(change_col: str) -> str:
    """Map a change column name to its corresponding storage level column."""
    if not change_col.endswith("_Change"):
        raise ValueError(f"Column is not a change series: {change_col}")
    return change_col.replace("_Change", "")


def _latest_levels_by_region(totals: pd.DataFrame) -> Mapping[str, np.float64]:
    """Extract the latest storage level by region from EIAtotals."""
    latest = totals.sort_values("Period").iloc[-1]
    return {col: np.float64(latest[col]) for col in totals.columns if col != "Period"}


def forecast_next_week_eia() -> Tuple[EIAForecast, EIAModelBundle]:
    """Train models on historical data and forecast the next EIA week."""
    configure_determinism()

    hist, fc = load_combined_frames()
    totals, changes = load_eia_weekly()

    hist_features, fc_features, fc_to_hist = derive_feature_mappings(hist, fc)
    matrix, target_cols = build_training_matrix(hist, totals, changes, hist_features)

    tuned_params: Mapping[str, Mapping[str, np.float64]] | None = None
    if OPTUNA_TRIALS > 0:
        best = _optimize_hyperparameters(
            matrix, target_cols, OPTUNA_TRIALS, OPTUNA_SAMPLER_SEED
        )
        if best:
            tuned_params = {col: best for col in target_cols}

    bundle = train_eia_models(
        matrix,
        target_cols,
        random_state=SKLEARN_SEED,
        tuned_params=tuned_params,
    )

    last_eia_week = pd.Timestamp(changes["Period"].max())
    X_future, week_ending = build_forecast_feature_row(
        fc,
        fc_features,
        fc_to_hist,
        last_eia_week,
    )

    predictions: Dict[str, np.float64] = {}
    # Ensure the forecast row has the same feature columns as the training matrix.
    # Align future row columns with the bundle’s training set and pad missing columns with zeros.
    X_future_aligned = X_future.reindex(columns=bundle.feature_columns, fill_value=0.0)

    for col, model in bundle.models.items():
        value = np.float64(model.predict(X_future_aligned.values)[0])
        predictions[col] = value

    latest_levels = _latest_levels_by_region(totals)
    level_forecasts: Dict[str, np.float64] = {}
    for change_col, change_value in predictions.items():
        level_col = _change_to_level_column(change_col)
        base_level = latest_levels.get(level_col)
        if base_level is not None:
            level_forecasts[level_col] = base_level + change_value

    forecast = EIAForecast(
        week_ending=week_ending,
        changes=predictions,
        levels=level_forecasts,
    )

    LOG.info("Forecast week ending %s", week_ending.date())
    for name, value in sorted(forecast.changes.items()):
        LOG.info("Change %s: %.1f", name, value)
    for name, value in sorted(forecast.levels.items()):
        LOG.info("Level %s: %.1f", name, value)

    return forecast, bundle


def plot_backtest(
    matrix: pd.DataFrame,
    bundle: EIAModelBundle,
    region_change_column: str = "Lower48_Change",
    forecast: EIAForecast | None = None,
    window: int | None = 104,
    ax: Axes | None = None,
) -> Axes:
    """Plot a backtest chart for a single change target with forecast."""
    if region_change_column not in bundle.target_columns:
        raise ValueError(f"Target column {region_change_column} not present in bundle.")

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    data = matrix.copy()
    if window is not None and len(data) > window:
        data = data.iloc[-window:]

    X = data[bundle.feature_columns].values
    y = data[region_change_column].values

    model = bundle.models[region_change_column]
    pred = model.predict(X)

    sns.lineplot(x=data.index, y=y, label="Actual", color="black", ax=ax)
    sns.lineplot(x=data.index, y=pred, label="Model", color="tab:blue", ax=ax)

    if forecast is not None and region_change_column in forecast.changes:
        ax.scatter(
            [forecast.week_ending],
            [forecast.changes[region_change_column]],
            color="red",
            label="Next forecast",
            zorder=5,
        )

    ax.set_title(
        f"EIA {region_change_column} backtest (last {len(data)} weeks)", fontsize=11
    )
    ax.set_xlabel("Week ending", fontsize=9)
    ax.set_ylabel("Bcf", fontsize=9)

    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=8)
    return ax


def plot_latest_drivers(
    hist: pd.DataFrame,
    feature_columns: Iterable[str],
    num_features: int = 3,
    ax: Axes | None = None,
) -> Axes:
    """Plot latest daily paths for a subset of key driver features."""
    cols = [c for c in feature_columns if c in hist.columns]
    if not cols:
        raise ValueError("No driver feature columns found in historical frame.")

    selected = cols[:num_features]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    tail = hist.tail(45)
    for col in selected:
        sns.lineplot(data=tail, x="Date", y=col, label=col, ax=ax)

    ax.set_title("Recent forecast driver paths (last 45 days)", fontsize=11)
    ax.set_xlabel("Date", fontsize=9)

    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    return ax


def _backtest_error_stats(
    matrix: pd.DataFrame,
    bundle: EIAModelBundle,
    target_column: str,
) -> Dict[str, np.float64]:
    """Compute simple backtest error statistics for a given target."""
    if target_column not in matrix.columns:
        raise ValueError(f"Target column {target_column} not present in matrix.")

    if target_column not in bundle.models:
        raise ValueError(f"Model for target {target_column} not present in bundle.")

    X = matrix[bundle.feature_columns].values
    y = matrix[target_column].values
    model = bundle.models[target_column]
    pred = model.predict(X)

    errors = y - pred
    mae = np.float64(np.mean(np.abs(errors)))
    rmse = np.float64(np.sqrt(np.mean(errors**2)))
    bias = np.float64(np.mean(errors))
    std = np.float64(np.std(errors))

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "std": std,
    }


def _colorize_delta(value: np.float64) -> str:
    """Apply simple ANSI color to a delta value for console output."""
    if not np.isfinite(value):
        return "n/a"

    text = f"{value:+.1f}"

    # Larger draws (more negative) in green, larger injections (more positive) in red.
    if value <= -5.0:
        return f"\x1b[32m{text}\x1b[0m"
    if value >= 5.0:
        return f"\x1b[31m{text}\x1b[0m"
    return text


def _build_region_summary_table(
    weekly_matrix: pd.DataFrame,
    bundle: EIAModelBundle,
    forecast: EIAForecast,
    changes: pd.DataFrame,
    target_columns: Iterable[str],
) -> str | None:
    """Create a tabulated per-region summary of forecast vs last actual."""
    if tabulate is None:
        return None

    latest_changes = changes.sort_values("Period").iloc[-1]

    rows: list[dict[str, str]] = []
    for col in target_columns:
        if col not in forecast.changes:
            continue

        region = col.replace("_Change", "")
        last_val = np.float64(latest_changes[col])
        fc_val = np.float64(forecast.changes[col])
        delta = fc_val - last_val

        stats = _backtest_error_stats(weekly_matrix, bundle, col)
        band_lo = fc_val - stats["mae"]
        band_hi = fc_val + stats["mae"]

        row = {
            "Region": region,
            "Last": f"{last_val:+.1f}",
            "Forecast": f"{fc_val:+.1f}",
            "Delta": _colorize_delta(delta),
            "MAE": f"{stats['mae']:.1f}",
            "+/- band": f"[{band_lo:+.1f}, {band_hi:+.1f}]",
        }
        rows.append(row)

    if not rows:
        return None

    table = tabulate(
        rows,
        headers="keys",
        tablefmt="github",
        showindex=False,
    )
    return table


def run_eia_playground() -> None:
    """Run the full EIA forecasting playground and display basic plots."""
    # Guardrail ensures required aggregates exist before kicking off training/plotting.
    validate_required_paths()
    forecast, bundle = forecast_next_week_eia()
    hist, fc = load_combined_frames()
    totals, changes = load_eia_weekly()
    hist_features, _, _ = derive_feature_mappings(hist, fc)

    weekly_matrix, target_cols = build_training_matrix(
        hist, totals, changes, hist_features
    )
    sns.set_theme(style="darkgrid", context="paper", font_scale=0.9)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=110)

    plot_backtest(
        weekly_matrix,
        bundle,
        region_change_column=target_cols[0],
        forecast=forecast,
        window=104,
        ax=axes[0],
    )
    plot_latest_drivers(hist, hist_features, num_features=3, ax=axes[1])

    fig.suptitle(
        f"EIA Forecast Playground – week ending {forecast.week_ending.date()}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    # Summarize Lower48 forecast with simple error band using backtest stats.
    target = "Lower48_Change"
    if target in target_cols:
        stats = _backtest_error_stats(weekly_matrix, bundle, target)
        last_actual = np.float64(changes.sort_values("Period").iloc[-1][target])
        fc_val = np.float64(forecast.changes.get(target, np.float64("nan")))
        delta = fc_val - last_actual
        band_lo = fc_val - stats["mae"]
        band_hi = fc_val + stats["mae"]

        LOG.info(
            "Summary Lower48: forecast=%.1f Bcf (last=%.1f, delta=%.1f). "
            "Backtest MAE=%.1f, RMSE=%.1f, bias=%.1f. "
            "Approx +/- band [%.1f, %.1f].",
            fc_val,
            last_actual,
            delta,
            stats["mae"],
            stats["rmse"],
            stats["bias"],
            band_lo,
            band_hi,
        )

    # Per-region tabulated summary using ANSI colors for deltas if available.
    table = _build_region_summary_table(
        weekly_matrix=weekly_matrix,
        bundle=bundle,
        forecast=forecast,
        changes=changes,
        target_columns=target_cols,
    )
    if table:
        LOG.info("Per-region EIA change summary:\n%s", table)

    plt.show()


if __name__ == "__main__":
    run_eia_playground()
    # Comment: logging ensures we can trace Optuna outcomes when used, otherwise defaults remain deterministic.
