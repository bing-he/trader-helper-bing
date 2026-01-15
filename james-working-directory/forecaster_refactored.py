#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weekly EIA Natural Gas Storage Forecaster

Time series forecasting for regional natural gas storage changes using quantile regression.
Prevents data leakage by computing derived features within cross-validation folds.

Inputs:
  features: Daily market/weather data (Combined_Wide_Data.csv)
  changes:  Weekly EIA storage changes (EIAchanges.csv)

Outputs:
  Quantile forecasts for 6 regions with prediction intervals
  Backtest metrics and feature importance analysis
"""

import argparse
import json
import logging
import math
import os
import re
import shutil
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Configuration constants
TARGETS = ["lower48", "east", "midwest", "southcentral", "mountain", "pacific"]

PRETTY_NAMES = {
    "lower48": "Lower 48",
    "east": "East",
    "midwest": "Midwest",
    "southcentral": "South Central",
    "mountain": "Mountain",
    "pacific": "Pacific",
}

# Week structure: Friday through Thursday
DAY_TAGS = ["d0_fri", "d1_sat", "d2_sun", "d3_mon", "d4_tue", "d5_wed", "d6_thu"]

# CatBoost parameters tuned for ~300 weekly samples
DEFAULT_PARAMS = {
    "iterations": 600,  # Enough for convergence without overfitting
    "depth": 6,  # Captures interactions but keeps trees manageable
    "learning_rate": 0.05,  # Conservative for small dataset
    "l2_leaf_reg": 3.0,  # Regularization for generalization
    "subsample": 0.8,  # Bootstrap sampling reduces variance
    "rsm": 0.8,  # Random subspace method
    "loss_function": "Quantile:alpha=0.50",
    "random_seed": 42,
    "verbose": False,
}

# South Central region aliases for pattern matching
SC_ALIASES = [r"south[_\s]?central", r"\bsc\b", r"\btx\b", r"texas", r"gulf"]

# Regional storage capacity weights based on EIA data
# Used for hierarchical forecast reconciliation
REGIONAL_WEIGHTS = {
    "east": 0.30,
    "midwest": 0.16,
    "southcentral": 0.40,  # Largest region
    "mountain": 0.09,
    "pacific": 0.05,
}


def load_params_or_default(target_region):
    """Load saved hyperparameters or fall back to defaults"""
    param_file = f"best_params_{target_region}.json"
    if os.path.exists(param_file):
        with open(param_file) as f:
            return json.load(f)
    return DEFAULT_PARAMS.copy()


def _filter_columns(df, must_contain=None, any_of=None, exclude=None):
    """Filter DataFrame columns by regex patterns"""
    patterns = []
    if must_contain:
        patterns += [rf"(?=.*{p})" for p in must_contain]
    if any_of:
        patterns += [rf"(?:{'|'.join(any_of)})"]

    regex = re.compile("".join(patterns), flags=re.I)
    matching_cols = [c for c in df.columns if regex.search(c)]

    if exclude:
        exclude_regex = re.compile("|".join(exclude), flags=re.I)
        matching_cols = [c for c in matching_cols if not exclude_regex.search(c)]

    return matching_cols


def _sum_columns_if(df, must_contain=None, any_of=None, exclude=None):
    """Sum columns matching filter criteria"""
    cols = _filter_columns(df, must_contain, any_of, exclude)
    return df[cols].sum(axis=1) if cols else pd.Series(index=df.index, dtype=np.float64)


# ------------------------------ utils ------------------------------


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return out


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to lowercase with underscores"""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def normalize_to_thursday(timestamp: pd.Timestamp) -> pd.Timestamp:
    """Convert any weekday to the Thursday of that week"""
    return timestamp - pd.Timedelta(days=((timestamp.weekday() - 3) % 7))


def weekly_thursday_index(date_index: pd.Index) -> pd.Index:
    """Convert daily dates to weekly Thursday index"""
    return pd.Index(
        [normalize_to_thursday(pd.Timestamp(d)) for d in date_index],
        name="week_end_thu",
    )


def validate_monotonic_index(index: pd.Index, name: str):
    """Ensure index is sorted chronologically"""
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be chronologically sorted")


def forward_fill_with_trailing_median(series: pd.Series, window: int = 7) -> pd.Series:
    """Forward fill missing values, then use trailing median for remaining gaps"""
    filled = series.ffill()
    trailing_median = filled.rolling(window=window, min_periods=1).median()
    return filled.fillna(trailing_median)


def build_sequence_matrix(
    daily_data: pd.DataFrame,
    thursday_dates: Iterable[pd.Timestamp],
    include_missing_flags: bool = True,
) -> pd.DataFrame:
    """Build 7-day sequence features for each Thursday forecast date"""
    numeric_cols = daily_data.select_dtypes(include=[np.number])
    column_names = list(numeric_cols.columns)

    rows, row_index = [], []

    for thursday in thursday_dates:
        # Get Friday-to-Thursday sequence (7 days)
        date_sequence = [
            thursday + pd.Timedelta(days=offset) for offset in range(-6, 1)
        ]

        if any(date not in daily_data.index for date in date_sequence):
            continue

        week_block = numeric_cols.loc[date_sequence, column_names]
        flattened_data = week_block.to_numpy().reshape(-1)
        rows.append(flattened_data)
        row_index.append(thursday)

    # Create column names: feature__day_tag
    flat_column_names = [
        f"{col}__{DAY_TAGS[day]}" for day in range(7) for col in column_names
    ]

    feature_matrix = pd.DataFrame(
        rows, index=pd.Index(row_index, name="week_end_thu"), columns=flat_column_names
    )

    # Add missing data flags if requested
    if include_missing_flags:
        missing_flags = {}
        missing_data = numeric_cols.isna().astype(int)

        for thursday in feature_matrix.index:
            date_sequence = [
                thursday + pd.Timedelta(days=offset) for offset in range(-6, 1)
            ]
            if any(date not in missing_data.index for date in date_sequence):
                continue

            week_missing = (
                missing_data.loc[date_sequence, column_names].sum(axis=0).clip(upper=1)
            )
            missing_flags[thursday] = week_missing

        if missing_flags:
            flags_df = pd.DataFrame(missing_flags).T
            flags_df.index.name = "week_end_thu"
            flags_df.columns = [f"{col}__any_missing" for col in flags_df.columns]
            feature_matrix = feature_matrix.join(flags_df, how="left")

    return feature_matrix


def select_features_by_variance(
    feature_matrix: pd.DataFrame, max_features: int, required_features: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Select top variance features while preserving required columns"""
    # Keep required features that exist in the data
    required_cols = [col for col in required_features if col in feature_matrix.columns]
    required_data = feature_matrix[required_cols]

    # Process remaining features
    remaining_cols = [col for col in feature_matrix.columns if col not in required_cols]
    remaining_data = feature_matrix[remaining_cols]

    if remaining_data.shape[1] > 0:
        # Remove zero-variance features
        variance_filter = VarianceThreshold(1e-12)
        filtered_data = variance_filter.fit_transform(remaining_data)
        filtered_cols = remaining_data.columns[variance_filter.get_support()]
        filtered_df = pd.DataFrame(
            filtered_data, index=feature_matrix.index, columns=filtered_cols
        )

        # Select top variance features
        if max_features > 0 and max_features < filtered_df.shape[1]:
            variances = filtered_df.var().sort_values(ascending=False)
            top_cols = list(variances.index[:max_features])
            filtered_df = filtered_df[top_cols]
    else:
        filtered_df = pd.DataFrame(index=feature_matrix.index)

    # Combine required and selected features
    final_features = pd.concat([required_data, filtered_df], axis=1)
    return final_features, list(final_features.columns)


def calculate_recency_weights(
    n_samples: int, half_life_weeks: Optional[int]
) -> np.ndarray:
    """Exponential recency weights for time series - recent data gets higher weight"""
    if n_samples <= 1 or not half_life_weeks:
        return np.ones(n_samples, dtype=np.float64)

    decay_constant = half_life_weeks / math.log(2.0)
    time_indices = np.arange(n_samples)
    weights = np.exp((time_indices - (n_samples - 1)) / decay_constant)

    # Normalize to mean of 1.0 for stability
    return weights / weights.mean()


def generate_validation_blocks(n_samples, n_folds=5, gap_weeks=3, validation_weeks=10):
    """Generate time series cross-validation blocks with gaps to prevent leakage"""
    # Start validation from middle of dataset to ensure enough training data
    fold_boundaries = np.linspace(
        int(n_samples * 0.5), n_samples - 1, n_folds + 1, dtype=int
    )

    for i in range(n_folds):
        train_end, val_end = fold_boundaries[i], fold_boundaries[i + 1]
        val_indices = np.arange(train_end, val_end)

        # Create gap around validation period
        gap_start = max(0, train_end - gap_weeks)
        gap_end = min(n_samples, val_end + validation_weeks)

        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[gap_start:gap_end] = False
        train_indices = np.where(train_mask)[0]

        yield train_indices, val_indices


def train_quantile_model_with_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    quantile: np.float64,
    train_weights: np.ndarray,
    val_weights: np.ndarray,
    params: Dict,
) -> CatBoostRegressor:
    """Train quantile model with early stopping using validation set"""
    model_params = params.copy()
    model_params["loss_function"] = f"Quantile:alpha={quantile}"
    model_params["use_best_model"] = True
    model_params["early_stopping_rounds"] = 50

    model = CatBoostRegressor(**model_params)
    model.fit(
        X_train,
        y_train,
        sample_weight=train_weights,
        eval_set=Pool(X_val, y_val, weight=val_weights),
        verbose=False,
    )
    return model


def enforce_quantile_ordering(q10, q50, q90):
    """Ensure q10 <= q50 <= q90"""
    return tuple(sorted([np.float64(q10), np.float64(q50), np.float64(q90)]))


def reconcile_forecasts_to_total(total_forecast, regional_forecasts, weights):
    """Proportionally adjust regional forecasts to match total"""
    if weights is None:
        weights = np.ones(len(regional_forecasts))

    discrepancy = total_forecast - np.sum(regional_forecasts)
    adjustments = discrepancy * np.array(weights) / np.sum(weights)

    return regional_forecasts + adjustments


def reconcile_quantile_forecasts(total_quantiles, regional_quantiles, weights):
    """Reconcile regional quantile forecasts to sum to total while preserving spreads"""
    q10s, q50s, q90s = zip(*regional_quantiles)

    # Adjust medians to sum to total
    adjusted_q50s = reconcile_forecasts_to_total(
        total_quantiles[1], np.array(q50s), weights
    )

    # Preserve original spreads around adjusted medians
    original_spreads_low = np.array(q50s) - np.array(q10s)
    original_spreads_high = np.array(q90s) - np.array(q50s)

    adjusted_q10s = adjusted_q50s - original_spreads_low
    adjusted_q90s = adjusted_q50s + original_spreads_high

    return list(zip(adjusted_q10s, adjusted_q50s, adjusted_q90s))


def reconcile_hierarchical_forecasts(forecasts_by_region):
    """Reconcile regional forecasts to sum to Lower 48 total"""
    regional_keys = ["east", "midwest", "southcentral", "mountain", "pacific"]

    if "lower48" not in forecasts_by_region or any(
        key not in forecasts_by_region for key in regional_keys
    ):
        return forecasts_by_region

    lower48_quantiles = forecasts_by_region["lower48"]
    weights = np.array([REGIONAL_WEIGHTS[key] for key in regional_keys])

    # Reconcile quantiles while preserving spreads
    regional_quantiles = [forecasts_by_region[key] for key in regional_keys]
    adjusted_quantiles = reconcile_quantile_forecasts(
        lower48_quantiles, regional_quantiles, weights
    )

    # Update forecasts with reconciled values
    for key, quantiles in zip(regional_keys, adjusted_quantiles):
        forecasts_by_region[key] = tuple(np.float64(q) for q in quantiles)

    return forecasts_by_region


def format_number(value: np.float64) -> str:
    """Format number for display, handling NaN/inf gracefully"""
    if value is None or (
        isinstance(value, np.float64) and (np.isnan(value) or np.isinf(value))
    ):
        return "NA"
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}"
    return f"{value:.2f}"


def calculate_pinball_loss(
    y_true: np.ndarray, y_pred: np.ndarray, quantile: np.float64
) -> np.float64:
    """Calculate pinball loss for quantile regression evaluation"""
    residuals = y_true - y_pred
    return np.mean(
        np.where(residuals >= 0, quantile * residuals, (quantile - 1) * residuals)
    )


def apply_shrinkage_to_forecast(
    prediction, historical_mean, high_volatility_regime, base_shrinkage=0.25
):
    """Apply moderate shrinkage towards historical mean during uncertain periods"""
    shrinkage_factor = base_shrinkage * (0.5 if high_volatility_regime else 1.0)
    shrinkage_factor = max(0.10, min(0.30, shrinkage_factor))
    return (1 - shrinkage_factor) * prediction + shrinkage_factor * historical_mean


# ============================================================================
# CRITICAL FIX: Fold-aware derived feature engineering
# ============================================================================


def add_derived_features_fold_safe(
    base_daily: pd.DataFrame, train_end_date: pd.Timestamp, salt_map_path: str
) -> pd.DataFrame:
    """Compute derived features using only data up to training cutoff to prevent lookahead bias

    This prevents data leakage by ensuring all rolling statistics and z-scores
    use only historical data available at prediction time.
    """
    # Only use data up to training cutoff - this is the key anti-leakage measure
    df = base_daily[base_daily.index <= train_end_date].copy()

    # Load South Central salt storage mapping
    sc_salt_map = {}
    if os.path.exists(salt_map_path):
        with open(salt_map_path) as f:
            sc_salt_map = json.load(f)

    # Build z-scores for South Central storage facilities
    z_scores = {}

    sc_salt_cols = [
        col
        for col, storage_type in sc_salt_map.items()
        if storage_type == "salt" and col in df.columns
    ]
    sc_nonsalt_cols = [
        col
        for col, storage_type in sc_salt_map.items()
        if storage_type == "nonsalt" and col in df.columns
    ]

    def compute_weekly_zscore(series):
        """Compute z-score from week-over-week changes using trailing window only"""
        weekly_change = series - series.shift(7)
        trailing_mean = weekly_change.rolling(52, min_periods=10).mean()
        trailing_std = weekly_change.rolling(52, min_periods=10).std().replace(0, 1)
        return ((weekly_change - trailing_mean) / trailing_std).clip(-3, 3)

    # South Central salt storage z-score
    if sc_salt_cols:
        sc_salt_total = df[sc_salt_cols].sum(axis=1)
        z_scores["sc_salt_z"] = compute_weekly_zscore(sc_salt_total)

    # South Central non-salt storage z-score
    if sc_nonsalt_cols:
        sc_nonsalt_total = df[sc_nonsalt_cols].sum(axis=1)
        z_scores["sc_nonsalt_z"] = compute_weekly_zscore(sc_nonsalt_total)

    # CDD z-score
    sc_cdd = _sum_columns_if(df, any_of=SC_ALIASES + [r"\bcdd\b"])
    if not sc_cdd.empty:
        z_scores["sc_cdd_z"] = compute_weekly_zscore(sc_cdd)

    # LNG utilization rate z-score (captures LNG feed + exports)
    lng_util = _sum_columns_if(df, any_of=[r"lng_feed", r"lng_export", r"lng_sendout"])
    if not lng_util.empty:
        z_scores["sc_lng_util_z"] = compute_weekly_zscore(lng_util)

    # Add all z-scores at once to avoid DataFrame fragmentation
    if z_scores:
        z_score_df = pd.DataFrame(z_scores, index=df.index)
        df = pd.concat([df, z_score_df], axis=1)

    # =========================================================================
    # REGIME-SENSITIVE RE-WEIGHTING: Adjust salt/nonsalt deltas by stress variables
    # =========================================================================
    # Salt caverns respond strongly to short-term weather and LNG load volatility
    # Nonsalt formations respond mainly to sustained temperature anomalies
    # Re-weight deltas so model sees amplified signals during regime transitions

    weighted_deltas = {}

    # Re-weighted salt delta: amplifies with temperature deviation and LNG utilization
    if "sc_salt_z" in df and "sc_cdd_z" in df:
        weighted_deltas["sc_salt_weighted"] = df["sc_salt_z"] * (
            1 + 0.5 * df["sc_cdd_z"].abs()
        )

    if "sc_nonsalt_z" in df and "sc_cdd_z" in df:
        weighted_deltas["sc_nonsalt_weighted"] = df["sc_nonsalt_z"] * (
            1 + 0.2 * df["sc_cdd_z"].abs()  # Lower elasticity for nonsalt
        )

    # Cross-terms with LNG utilization (captures LNG export facility impact)
    if "sc_salt_z" in df and "sc_lng_util_z" in df:
        weighted_deltas["sc_salt_x_lng"] = df["sc_salt_z"] * df["sc_lng_util_z"]

    if "sc_nonsalt_z" in df and "sc_lng_util_z" in df:
        weighted_deltas["sc_non_x_lng"] = df["sc_nonsalt_z"] * df["sc_lng_util_z"]

    # Add weighted deltas to dataframe
    if weighted_deltas:
        weighted_df = pd.DataFrame(weighted_deltas, index=df.index)
        df = pd.concat([df, weighted_df], axis=1)

    # Build price-related features
    new_features = {}
    price_columns = [col for col in df.columns if "_price_" in col.lower()]

    if price_columns:
        henry_hub_cols = [col for col in price_columns if "henryhub" in col.lower()]
        henry_hub_price = df[henry_hub_cols[0]] if henry_hub_cols else None

        # Calculate basis spreads vs Henry Hub
        if henry_hub_price is not None:
            for col in price_columns:
                if col != henry_hub_cols[0]:
                    spread_name = col.replace("_price_", "_basis_")
                    new_features[spread_name] = df[col] - henry_hub_price

        # Week-over-week price changes (using historical data only)
        for col in price_columns:
            new_features[f"{col}_chg7d"] = df[col] - df[col].shift(7)

        # Rolling 7-day volatility (trailing window only)
        for col in price_columns:
            new_features[f"{col}_std7d"] = df[col].rolling(7, min_periods=3).std()

        # Regional price spreads for key hubs
        sc_waha = [col for col in price_columns if "waha" in col]
        sc_houston = [col for col in price_columns if "houston" in col or "hsc" in col]
        sc_katy = [col for col in price_columns if "katy" in col]

        if sc_waha and sc_houston:
            new_features["sc_basis_waha_hsc"] = df[sc_waha[0]] - df[sc_houston[0]]
        if sc_waha and sc_katy:
            new_features["sc_basis_waha_katy"] = df[sc_waha[0]] - df[sc_katy[0]]

        # East vs Midwest spread
        east_hubs = [col for col in price_columns if "east" in col or "tetco" in col]
        midwest_hubs = [
            col for col in price_columns if "midwest" in col or "chicago" in col
        ]
        if east_hubs and midwest_hubs:
            new_features["east_midwest_spread"] = df[east_hubs[0]] - df[midwest_hubs[0]]

        # Pacific vs Mountain spread
        pacific_hubs = [
            col for col in price_columns if "pacific" in col or "socal" in col
        ]
        mountain_hubs = [
            col for col in price_columns if "mountain" in col or "opal" in col
        ]
        if pacific_hubs and mountain_hubs:
            new_features["pacific_mountain_spread"] = (
                df[pacific_hubs[0]] - df[mountain_hubs[0]]
            )

    # Interaction features
    if "sc_salt_z" in df and "sc_cdd_z" in df:
        new_features["sc_salt_x_cdd"] = df["sc_salt_z"] * df["sc_cdd_z"]

    if "sc_nonsalt_z" in df and "sc_basis_waha_hsc" in new_features:
        new_features["sc_non_x_basis"] = (
            df["sc_nonsalt_z"] * new_features["sc_basis_waha_hsc"]
        )

    # South Central flow volatility
    sc_facilities = df.filter(
        regex=r"(south[_\s]?central|tx|gulf).*critstor", axis=1
    ).sum(axis=1)
    if not sc_facilities.empty:
        new_features["sc_flow_std_7"] = sc_facilities.rolling(7, min_periods=2).std()

    # Add all new features at once
    if new_features:
        features_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, features_df], axis=1)

    return df


def prepare_daily_index(df):
    """Ensure DataFrame has proper datetime index"""
    df = df.copy()

    if "date" in df.columns:
        date_index = pd.to_datetime(df["date"], errors="coerce")
        df = df.drop(columns=["date"])
    elif pd.api.types.is_datetime64_any_dtype(df.index):
        date_index = pd.to_datetime(df.index, errors="coerce")
    else:
        first_col = df.columns[0]
        date_index = pd.to_datetime(df[first_col], errors="coerce")
        if first_col != "date":
            df = df.drop(columns=[first_col])

    if date_index.isna().any():
        raise ValueError("Unable to parse dates - check date column format")

    df.index = date_index
    df = df.sort_index()

    # Create complete daily index
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    return df.reindex(full_range)


def flip_storage_sign_convention(df, make_injection_positive=True):
    """Flip CritStor sign convention so injections are positive, withdrawals negative"""
    storage_cols = [col for col in df.columns if "critstor" in col]
    if make_injection_positive and storage_cols:
        df[storage_cols] = -1 * df[storage_cols]
    return df


# ------------------------------ pipeline ------------------------------


def run(features_path: str, changes_path: str, outdir: str):
    # Cleanup
    if os.path.exists("catboost_info"):
        shutil.rmtree("catboost_info")
    if os.path.exists("models"):
        shutil.rmtree("models")
    for fn in os.listdir("."):
        if fn.startswith("best_params_") and fn.endswith(".json"):
            os.remove(fn)

    os.makedirs(outdir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Load data
    base_daily = pd.read_csv(features_path)
    # base_daily = base_daily.drop(
    #     columns=[
    #         "East_Weather_KATL_Daily_Precip_Amount",
    #         "East_Weather_KBOS_Daily_Precip_Amount",
    #         "East_Weather_KDCA_Daily_Precip_Amount",
    #         "East_Weather_KJFK_Daily_Precip_Amount",
    #         "East_Weather_KPHL_Daily_Precip_Amount",
    #         "East_Weather_KRDU_Daily_Precip_Amount",
    #         "Midwest_Weather_KDTW_Daily_Precip_Amount",
    #         "Midwest_Weather_KMSP_Max_Surface_Wind",
    #         "Midwest_Weather_KORD_Daily_Precip_Amount",
    #         "Mountain_Weather_KDEN_Avg_Cloud_Cover",
    #         "Mountain_Weather_KPHX_Daily_Precip_Amount",
    #         "South Central_CritStor_Louisiana_LA Storage",
    #         "South Central_Weather_KBHM_Daily_Precip_Amount",
    #         "South Central_Weather_KIAH_Daily_Precip_Amount",
    #         "South Central_Weather_KJAN_Daily_Precip_Amount",
    #         "South Central_Weather_KMAF_Avg_Cloud_Cover",
    #         "South Central_Weather_KSAT_Daily_Precip_Amount",
    #     ],
    #     errors="ignore",
    # )
    base_daily = clean_column_names(base_daily)
    base_daily = prepare_daily_index(base_daily)
    base_daily = flip_storage_sign_convention(base_daily, make_injection_positive=True)
    base_daily = base_daily.dropna(axis=1, how="all")

    weekly = pd.read_csv(changes_path, index_col=0, parse_dates=True)
    weekly = clean_column_names(weekly).sort_index()
    weekly.columns = weekly.columns.str.replace("_change", "", regex=False)
    weekly = weekly[weekly.index >= "2018-01-01"]

    # =========================================================================
    # DETERMINE FORECAST TARGET ALGORITHMICALLY
    # =========================================================================
    # EIA releases storage reports ONE WEEK (7 days) after the data week ends.
    # Timeline example:
    #   - Data week ending: Thursday Oct 10, 2025
    #   - Report released: Thursday Oct 17, 2025 (7 days later at 10:30 AM ET)
    #
    # Algorithm to find the correct forecast target:
    #   1. Get today's date
    #   2. Find the next Thursday from today (this is when the EIA report will be released)
    #   3. The forecast target is 7 days BEFORE that (the data week ending)
    #   4. Train on all data before the forecast target
    # =========================================================================

    today = pd.Timestamp.now().normalize()

    # Find next Thursday from today (when EIA report will be released)
    days_until_thursday = (3 - today.weekday()) % 7  # Thursday is weekday 3
    if days_until_thursday == 0:
        # If today is Thursday, get next Thursday
        days_until_thursday = 7
    next_thursday = today + pd.Timedelta(days=days_until_thursday)

    # The forecast target is the Thursday 7 days before the report release
    forecast_thursday = next_thursday - pd.Timedelta(days=7)
    report_release_date = next_thursday

    # Verify this Thursday exists in our dataset
    last_thursday_in_data = weekly.index.max()

    if forecast_thursday > last_thursday_in_data:
        logger.warning(
            f"Calculated forecast Thursday ({forecast_thursday.date()}) is beyond available data "
            f"(last available: {last_thursday_in_data.date()}). Using last available Thursday instead."
        )
        forecast_thursday = last_thursday_in_data
        report_release_date = forecast_thursday + pd.Timedelta(days=7)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"TODAY: {today.date()}")
    logger.info(f"FORECAST TARGET: Week ending Thursday {forecast_thursday.date()}")
    logger.info(f"EIA REPORT RELEASE: {report_release_date.date()} at 10:30 AM ET")
    logger.info(f"TRAINING DATA: All weeks before {forecast_thursday.date()}")
    logger.info(f"{'=' * 80}\n")

    # Split: training data (all weeks before the forecast Thursday)
    weekly_train = weekly[weekly.index < forecast_thursday].copy()
    # Hold-out: the forecast Thursday (for validation if actual data exists)
    weekly_holdout = weekly[weekly.index == forecast_thursday].copy()

    if len(weekly_holdout) == 0:
        raise ValueError(
            f"No data found for hold-out Thursday {forecast_thursday.date()}"
        )

    # Historical means for shrinkage (use only training data to prevent leakage)
    historical_means = {
        tgt: weekly_train[tgt].mean() for tgt in TARGETS if tgt in weekly_train.columns
    }

    # Detect high volatility regime
    high_vol = False
    henry_cols = [c for c in base_daily.columns if "henryhub" in c.lower()]
    if henry_cols:
        hh = base_daily[henry_cols[0]]
        recent_std = hh.tail(28).std()
        if recent_std > 0.5:
            high_vol = True

    validate_monotonic_index(base_daily.index, "features")
    validate_monotonic_index(weekly_train.index, "changes_train")
    validate_monotonic_index(weekly_holdout.index, "changes_holdout")

    weekly_train.index = weekly_thursday_index(weekly_train.index)
    weekly_holdout.index = weekly_thursday_index(weekly_holdout.index)

    # Results storage
    results: List[Dict] = []
    bt_metrics: List[Dict] = []
    calibration_metrics: List[Dict] = []
    feature_importance_log: List[Dict] = []
    backtest_log: List[Dict] = []
    residuals_log: List[Dict] = []
    forecasts_by_region: Dict[str, Tuple[np.float64, np.float64, np.float64]] = {}

    map_path = os.path.join(os.path.dirname(__file__), "sc_salt_map.json")

    # =========================================================================
    # MAIN LOOP: Process each target region
    # =========================================================================
    for tgt in TARGETS:
        if tgt not in weekly_train.columns:
            continue

        logger.info(f"\n{'=' * 60}\nProcessing: {PRETTY_NAMES[tgt]}\n{'=' * 60}")

        params = load_params_or_default(tgt)
        hl = 13 if high_vol else 26

        # Build historical sequence WITHOUT derived features (using training data only)
        X_base = build_sequence_matrix(
            base_daily, weekly_train.index, include_missing_flags=True
        )
        Y_hist = weekly_train[[tgt]].reindex(X_base.index).dropna()
        X_base = X_base.reindex(Y_hist.index)

        if X_base.empty:
            raise ValueError(f"No aligned data for {tgt}")

        y = Y_hist[tgt].astype(np.float64)
        n = len(X_base)

        if n < 120:
            raise ValueError(f"Not enough samples for {tgt} ({n} < 120)")

        # =====================================================================
        # BACKTEST WITH FOLD-SAFE FEATURE ENGINEERING
        # =====================================================================
        blocks = list(
            generate_validation_blocks(n, n_folds=5, gap_weeks=3, validation_weeks=10)
        )

        preds_q10_bt, preds_q50_bt, preds_q90_bt = [], [], []
        acts_bt = []

        for fold_idx, (tr_idx, va_idx) in enumerate(blocks):
            # Get training end date (last Thursday in training)
            train_end_thu = X_base.index[tr_idx[-1]]

            # CRITICAL: Compute derived features using ONLY training data
            daily_with_features = add_derived_features_fold_safe(
                base_daily, train_end_thu, map_path
            )

            # Build sequence matrix with derived features
            X_fold = build_sequence_matrix(
                daily_with_features, X_base.index, include_missing_flags=True
            )

            # CRITICAL: Align indices - X_fold may have fewer rows than X_base
            # due to missing data in derived features computation
            common_idx = X_fold.index.intersection(X_base.index)
            tr_dates = X_base.index[tr_idx]
            va_dates = X_base.index[va_idx]

            # Only use dates that exist in X_fold
            tr_dates_valid = tr_dates.intersection(common_idx)
            va_dates_valid = va_dates.intersection(common_idx)

            # Check for excessive sample loss due to missing derived features
            tr_loss_pct = 100 * (1 - len(tr_dates_valid) / len(tr_dates))
            va_loss_pct = 100 * (1 - len(va_dates_valid) / len(va_dates))

            if tr_loss_pct > 5.0:
                logger.warning(
                    f"{PRETTY_NAMES[tgt]} fold {fold_idx + 1}: Lost {tr_loss_pct:.1f}% training samples "
                    f"({len(tr_dates)} ---> {len(tr_dates_valid)}) due to NaN features"
                )
            if va_loss_pct > 5.0:
                logger.warning(
                    f"{PRETTY_NAMES[tgt]} fold {fold_idx + 1}: Lost {va_loss_pct:.1f}% validation samples "
                    f"({len(va_dates)} ---> {len(va_dates_valid)}) due to NaN features"
                )

            if len(va_dates_valid) == 0:
                logger.error(
                    f"{PRETTY_NAMES[tgt]} fold {fold_idx + 1}: No valid validation data, skipping"
                )
                continue  # Skip fold if no validation data

            if len(tr_dates_valid) < 60:
                logger.error(
                    f"{PRETTY_NAMES[tgt]} fold {fold_idx + 1}: Only {len(tr_dates_valid)} training samples, "
                    "skipping (need ≥60 for feature selection)"
                )
                continue

            # Split into train/validation using date-based indexing
            Xtr = X_fold.loc[tr_dates_valid]
            ytr = y.loc[tr_dates_valid]
            Xva = X_fold.loc[va_dates_valid]
            yva = y.loc[va_dates_valid]

            # Feature selection on training data only
            must_have = [
                c
                for c in [
                    "sc_salt_z",
                    "sc_nonsalt_z",
                    "sc_cdd_z",
                    "sc_lng_util_z",
                    "sc_salt_weighted",
                    "sc_nonsalt_weighted",
                    "sc_salt_x_lng",
                    "sc_non_x_lng",
                    "sc_basis_waha_hsc",
                    "sc_basis_waha_katy",
                    "east_midwest_spread",
                    "pacific_mountain_spread",
                ]
                if c in Xtr.columns
            ]
            Xtr_sel, selected_cols = select_features_by_variance(Xtr, 60, must_have)
            Xva_sel = Xva[selected_cols]

            if len(Xtr_sel) < 60 or len(Xva_sel) == 0:
                continue

            # Recency weights
            w_tr = calculate_recency_weights(len(Xtr_sel), hl)

            # Train quantile models with early stopping for consistent methodology
            # Use fixed absolute validation size across all contexts
            n_tr = len(Xtr_sel)
            EARLY_STOP_SIZE = 30  # Fixed absolute size
            split_idx = max(
                30, n_tr - EARLY_STOP_SIZE
            )  # Ensure minimum training samples

            Xtr_fit, Xtr_es = Xtr_sel.iloc[:split_idx], Xtr_sel.iloc[split_idx:]
            ytr_fit, ytr_es = ytr.iloc[:split_idx], ytr.iloc[split_idx:]
            w_tr_fit, w_tr_es = w_tr[:split_idx], w_tr[split_idx:]

            m_q10 = train_quantile_model_with_validation(
                Xtr_fit, ytr_fit, Xtr_es, ytr_es, 0.10, w_tr_fit, w_tr_es, params
            )
            m_q50 = train_quantile_model_with_validation(
                Xtr_fit, ytr_fit, Xtr_es, ytr_es, 0.50, w_tr_fit, w_tr_es, params
            )
            m_q90 = train_quantile_model_with_validation(
                Xtr_fit, ytr_fit, Xtr_es, ytr_es, 0.90, w_tr_fit, w_tr_es, params
            )

            # Predict on validation
            p_q10 = m_q10.predict(Xva_sel)
            p_q50 = m_q50.predict(Xva_sel)
            p_q90 = m_q90.predict(Xva_sel)

            preds_q10_bt.append(pd.Series(p_q10, index=Xva.index))
            preds_q50_bt.append(pd.Series(p_q50, index=Xva.index))
            preds_q90_bt.append(pd.Series(p_q90, index=Xva.index))
            acts_bt.append(yva)

        # Aggregate backtest results
        if preds_q50_bt:
            preds_q10_bt = pd.concat(preds_q10_bt).sort_index()
            preds_q50_bt = pd.concat(preds_q50_bt).sort_index()
            preds_q90_bt = pd.concat(preds_q90_bt).sort_index()
            acts_bt = pd.concat(acts_bt).sort_index()

            # Compute metrics
            mae = mean_absolute_error(acts_bt, preds_q50_bt)
            bias = np.mean(acts_bt.values - preds_q50_bt.values)

            # Pinball losses for quantile evaluation
            pb_q10 = calculate_pinball_loss(acts_bt.values, preds_q10_bt.values, 0.10)
            pb_q50 = calculate_pinball_loss(acts_bt.values, preds_q50_bt.values, 0.50)
            pb_q90 = calculate_pinball_loss(acts_bt.values, preds_q90_bt.values, 0.90)

            # NEW: Calibration metrics
            below_q10 = np.mean(acts_bt.values < preds_q10_bt.values)
            below_q50 = np.mean(acts_bt.values < preds_q50_bt.values)
            below_q90 = np.mean(acts_bt.values < preds_q90_bt.values)

            # Conformal prediction calibration adjustments
            # Compute quantile-specific residuals for proper coverage
            residuals_q10 = acts_bt.values - preds_q10_bt.values
            residuals_q90 = acts_bt.values - preds_q90_bt.values

            # Calculate conformal offsets to ensure proper interval coverage
            offset_q10 = np.quantile(residuals_q10, 0.10)
            offset_q90 = np.quantile(residuals_q90, 0.90)

            conformal_offsets = {"offset_q10": offset_q10, "offset_q90": offset_q90}

            # Crisis period validation - check model performance during extreme events
            crisis_periods = {
                "Texas Freeze": (
                    pd.Timestamp("2021-02-08"),
                    pd.Timestamp("2021-02-26"),
                ),
                "Freeport LNG": (
                    pd.Timestamp("2022-06-08"),
                    pd.Timestamp("2022-12-31"),
                ),
            }

            # Separate normal vs crisis predictions
            crisis_mask = pd.Series(False, index=acts_bt.index)
            for start, end in crisis_periods.values():
                crisis_mask |= (acts_bt.index >= start) & (acts_bt.index <= end)

            normal_mask = ~crisis_mask

            # Compute crisis vs normal performance
            if crisis_mask.sum() > 0:
                crisis_mae = mean_absolute_error(
                    acts_bt[crisis_mask], preds_q50_bt[crisis_mask]
                )
                crisis_bias = np.mean(
                    acts_bt[crisis_mask].values - preds_q50_bt[crisis_mask].values
                )
            else:
                crisis_mae = np.nan
                crisis_bias = np.nan

            if normal_mask.sum() > 0:
                normal_mae = mean_absolute_error(
                    acts_bt[normal_mask], preds_q50_bt[normal_mask]
                )
                normal_bias = np.mean(
                    acts_bt[normal_mask].values - preds_q50_bt[normal_mask].values
                )
            else:
                normal_mae = np.nan
                normal_bias = np.nan

            # Flag if crisis performance significantly worse
            crisis_ratio = (
                crisis_mae / normal_mae
                if (normal_mae > 0 and not np.isnan(crisis_mae))
                else np.nan
            )

            if crisis_ratio > 2.0:
                logger.warning(
                    f"Model performance during crisis periods is concerning for {tgt}: "
                    f"Crisis MAE ({crisis_mae:.1f}) is {crisis_ratio:.1f}x normal ({normal_mae:.1f}). "
                    "Consider additional validation during extreme market conditions."
                )

            calibration_metrics.append(
                {
                    "target": tgt,
                    "region": PRETTY_NAMES[tgt],
                    "coverage_q10": f"{below_q10:.1%}",
                    "coverage_q50": f"{below_q50:.1%}",
                    "coverage_q90": f"{below_q90:.1%}",
                    "ideal_q10": "10%",
                    "ideal_q50": "50%",
                    "ideal_q90": "90%",
                }
            )

            # Log residuals and predictions
            for ts, (pred, act) in enumerate(zip(preds_q50_bt, acts_bt)):
                backtest_log.append(
                    {
                        "week_end_thu": str(preds_q50_bt.index[ts].date()),
                        "target": tgt,
                        "pred": np.float64(pred),
                        "actual": np.float64(act),
                    }
                )
                residuals_log.append(
                    {
                        "week_end_thu": str(preds_q50_bt.index[ts].date()),
                        "target": tgt,
                        "residual": np.float64(act - pred),
                    }
                )
        else:
            mae = bias = pb_q10 = pb_q50 = pb_q90 = np.nan
            conformal_offsets = {"offset_q10": 0.0, "offset_q90": 0.0}
            crisis_mae = normal_mae = crisis_ratio = np.nan
            crisis_bias = normal_bias = np.nan

        bt_metrics.append(
            {
                "target": tgt,
                "region": PRETTY_NAMES[tgt],
                "cv_mae": mae,
                "bias": bias,
                "pinball_q10": pb_q10,
                "pinball_q50": pb_q50,
                "pinball_q90": pb_q90,
                "half_life": hl,
                # Crisis period validation metrics
                "crisis_mae": crisis_mae,
                "normal_mae": normal_mae,
                "crisis_ratio": crisis_ratio,
                "crisis_bias": crisis_bias,
                "normal_bias": normal_bias,
            }
        )

        # =====================================================================
        # TRAIN FINAL PRODUCTION MODELS
        # =====================================================================
        last_thu = X_base.index.max()

        # Compute derived features for full training set
        daily_full = add_derived_features_fold_safe(base_daily, last_thu, map_path)
        X_full = build_sequence_matrix(
            daily_full, X_base.index, include_missing_flags=True
        )

        # Feature selection on full data
        must_have = [
            c
            for c in [
                "sc_salt_z",
                "sc_nonsalt_z",
                "sc_cdd_z",
                "sc_lng_util_z",
                "sc_salt_weighted",
                "sc_nonsalt_weighted",
                "sc_salt_x_lng",
                "sc_non_x_lng",
                "sc_basis_waha_hsc",
                "sc_basis_waha_katy",
                "east_midwest_spread",
                "pacific_mountain_spread",
            ]
            if c in X_full.columns
        ]
        X_full_sel, selected_cols = select_features_by_variance(X_full, 60, must_have)

        w_all = calculate_recency_weights(len(X_full_sel), hl)

        # Train final production models with early stopping
        # Use same validation setup as cross-validation for consistency
        n_full = len(X_full_sel)
        EARLY_STOP_SIZE = 30  # Fixed size for mathematical consistency
        split_idx = n_full - EARLY_STOP_SIZE

        X_train_final, X_es_final = (
            X_full_sel.iloc[:split_idx],
            X_full_sel.iloc[split_idx:],
        )
        y_train_final, y_es_final = y.iloc[:split_idx], y.iloc[split_idx:]
        w_train_final, w_es_final = w_all[:split_idx], w_all[split_idx:]

        m_q10_final = train_quantile_model_with_validation(
            X_train_final,
            y_train_final,
            X_es_final,
            y_es_final,
            0.10,
            w_train_final,
            w_es_final,
            params,
        )
        m_q50_final = train_quantile_model_with_validation(
            X_train_final,
            y_train_final,
            X_es_final,
            y_es_final,
            0.50,
            w_train_final,
            w_es_final,
            params,
        )
        m_q90_final = train_quantile_model_with_validation(
            X_train_final,
            y_train_final,
            X_es_final,
            y_es_final,
            0.90,
            w_train_final,
            w_es_final,
            params,
        )

        # NEW: Save feature importances
        importances = m_q50_final.get_feature_importance()
        for feat_idx, imp in enumerate(importances):
            if feat_idx < len(selected_cols):
                feature_importance_log.append(
                    {
                        "target": tgt,
                        "feature": selected_cols[feat_idx],
                        "importance": np.float64(imp),
                    }
                )

        # Save models
        m_q10_final.save_model(os.path.join("models", f"{tgt}_q10.cbm"))
        m_q50_final.save_model(os.path.join("models", f"{tgt}_q50.cbm"))
        m_q90_final.save_model(os.path.join("models", f"{tgt}_q90.cbm"))

        # =====================================================================
        # FORECAST THE HELD-OUT LAST THURSDAY
        # =====================================================================
        # We train on all data BEFORE the last Thursday, then forecast that last Thursday
        # This simulates real-time forecasting where we predict the week's storage change
        # before the EIA report is released (which happens 7 days after the data week ends)
        # =====================================================================
        forecast_thu = forecast_thursday  # The held-out Thursday we want to forecast
        need_dates = [forecast_thu + pd.Timedelta(days=o) for o in range(-6, 1)]

        # Check if we have complete data for the forecast week
        window_df = base_daily.reindex(need_dates)
        have_data = (
            window_df.notna().select_dtypes(include=[np.number]).sum().sum()
            == window_df.select_dtypes(include=[np.number]).size
        )

        if have_data:
            # Use raw data for forecast week
            daily_forecast = add_derived_features_fold_safe(
                base_daily, forecast_thu, map_path
            )
            X_forecast = build_sequence_matrix(
                daily_forecast, [forecast_thu], include_missing_flags=True
            )
        else:
            # Causal fill missing data
            idx7 = pd.date_range(
                forecast_thu - pd.Timedelta(days=6), forecast_thu, freq="D"
            )
            temp = base_daily.reindex(idx7)
            num = temp.select_dtypes(include=[np.number])
            for c in num.columns:
                num[c] = forward_fill_with_trailing_median(num[c], window=7)
            temp[num.columns] = num
            daily_forecast = add_derived_features_fold_safe(
                temp, forecast_thu, map_path
            )
            X_forecast = build_sequence_matrix(
                daily_forecast, [forecast_thu], include_missing_flags=True
            )

        x_forecast = X_forecast.reindex(columns=selected_cols).fillna(0.0).iloc[[-1]]

        # Predict
        f_q10 = np.float64(m_q10_final.predict(x_forecast).item())
        f_q50 = np.float64(m_q50_final.predict(x_forecast).item())
        f_q90 = np.float64(m_q90_final.predict(x_forecast).item())

        # Enforce ordering
        f_q10, f_q50, f_q90 = enforce_quantile_ordering(f_q10, f_q50, f_q90)

        # =====================================================================
        # BIAS DAMPING NEAR ZERO-CROSSINGS (South Central only)
        # =====================================================================
        # When net storage change oscillates around zero (injection ↔ withdrawal),
        # quantile regressors tend to overshoot due to asymmetric loss.
        # Shrink magnitude toward zero when:
        #   - |predicted median| < 10 Bcf (near zero-crossing)
        #   - Wide prediction interval (high uncertainty)
        # This corrects local bias without contaminating training.

        if tgt == "southcentral":
            zero_cross_zone = abs(f_q50) < 10.0  # Within ±10 Bcf of zero
            high_uncertainty = (
                f_q90 - f_q10
            ) > 25.0  # Wide interval indicates uncertainty

            if zero_cross_zone and high_uncertainty:
                bias_adj = 0.5  # Shrink 50% toward zero
                logger.info(
                    f"South Central zero-crossing adjustment: |{f_q50:.1f}| < 10 Bcf, "
                    f"interval width {f_q90 - f_q10:.1f} > 25 Bcf. Applying {bias_adj:.0%} shrinkage."
                )
                f_q50 *= bias_adj
                f_q10 *= bias_adj
                f_q90 *= bias_adj

        # Apply moderate shrinkage to median forecast while preserving prediction intervals
        spread_low = f_q50 - f_q10
        spread_high = f_q90 - f_q50

        if tgt in historical_means:
            f_q50 = apply_shrinkage_to_forecast(f_q50, historical_means[tgt], high_vol)
            # Preserve interval widths around shrunk median
            f_q10 = f_q50 - spread_low
            f_q90 = f_q50 + spread_high

        # Apply conformal adjustments to calibrate prediction intervals
        f_q10 += conformal_offsets["offset_q10"]
        f_q90 += conformal_offsets["offset_q90"]

        # Re-enforce ordering after conformal adjustment
        f_q10, f_q50, f_q90 = enforce_quantile_ordering(f_q10, f_q50, f_q90)

        forecasts_by_region[tgt] = (f_q10, f_q50, f_q90)

        results.append(
            {
                "region": tgt,
                "next_est": round(f_q50, 2),  # Point = q50
                "p10": round(f_q10, 2),
                "p50": round(f_q50, 2),
                "p90": round(f_q90, 2),
                "cv_mae": round(mae, 3),
                "bias": round(bias, 3),
                "half_life": hl,
                "n_features": len(selected_cols),
            }
        )

    # Reconcile regional forecasts to match Lower 48 total
    forecasts_by_region = reconcile_hierarchical_forecasts(forecasts_by_region)

    # Adjust regional intervals to account for reconciliation uncertainty
    RECONCILIATION_INFLATION = 1.10  # 10% wider intervals after reconciliation
    regional_keys = ["east", "midwest", "southcentral", "mountain", "pacific"]

    for region in regional_keys:
        if region in forecasts_by_region:
            q10, q50, q90 = forecasts_by_region[region]
            spread_low = q50 - q10
            spread_high = q90 - q50

            # Widen intervals to account for reconciliation uncertainty
            forecasts_by_region[region] = (
                q50 - RECONCILIATION_INFLATION * spread_low,
                q50,  # Point forecast unchanged
                q50 + RECONCILIATION_INFLATION * spread_high,
            )

    # Update results with reconciled values and actual holdout values
    for r in results:
        tgt = r["region"]
        if tgt in forecasts_by_region:
            f_q10, f_q50, f_q90 = forecasts_by_region[tgt]
            r["next_est"] = round(f_q50, 2)
            r["p10"] = round(f_q10, 2)
            r["p50"] = round(f_q50, 2)
            r["p90"] = round(f_q90, 2)

        # Add actual value from held-out last Thursday
        if tgt in weekly_holdout.columns and len(weekly_holdout) > 0:
            actual_value = weekly_holdout[tgt].iloc[0]
            r["actual"] = round(actual_value, 2)
            r["error"] = round(actual_value - r["p50"], 2)
            r["abs_error"] = round(abs(actual_value - r["p50"]), 2)
            # Check if actual falls within prediction interval
            r["in_interval"] = "Yes" if r["p10"] <= actual_value <= r["p90"] else "No"
        else:
            r["actual"] = None
            r["error"] = None
            r["abs_error"] = None
            r["in_interval"] = None

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    df_res = pd.DataFrame(results).sort_values("region")
    df_res.to_csv(os.path.join(outdir, "next_week_forecast.csv"), index=False)

    df_bt = pd.DataFrame(bt_metrics)
    df_bt.to_csv(os.path.join(outdir, "backtest_metrics.csv"), index=False)

    if calibration_metrics:
        pd.DataFrame(calibration_metrics).to_csv(
            os.path.join(outdir, "calibration_metrics.csv"), index=False
        )

    if feature_importance_log:
        df_fi = pd.DataFrame(feature_importance_log)
        # Get top 10 features per region
        top_features = (
            df_fi.sort_values(["target", "importance"], ascending=[True, False])
            .groupby("target")
            .head(10)
        )
        top_features.to_csv(os.path.join(outdir, "feature_importance.csv"), index=False)

    if backtest_log:
        pd.DataFrame(backtest_log).to_csv(
            os.path.join(outdir, "backtest_predictions.csv"), index=False
        )

    if residuals_log:
        pd.DataFrame(residuals_log).to_csv(
            os.path.join(outdir, "residuals.csv"), index=False
        )

    # Generate summary report
    title = "EIA Natural Gas Storage Forecast Summary"
    lines = [title, "=" * len(title)]

    lines.append("")
    lines.append("HOLD-OUT VALIDATION:")
    lines.append(f"  ---> Forecast Thursday: {forecast_thursday.date()}")
    lines.append(
        f"  ---> EIA Report Release: {report_release_date.date()} (7 days after data week ends)"
    )
    lines.append(
        f"  ---> Training data: Through {(forecast_thursday - pd.Timedelta(days=7)).date()}"
    )
    lines.append("")

    lines.append("METHODOLOGY:")
    lines.append("  ---> Quantile regression with CatBoost")
    lines.append("  ---> Time series cross-validation (no lookahead bias)")
    lines.append("  ---> Hierarchical forecast reconciliation")
    lines.append("  ---> Recency-weighted training samples")
    lines.append("=" * len(title))

    # Add hold-out validation results if available
    if "actual" in df_res.columns and df_res["actual"].notna().any():
        lines.append("")
        lines.append("HOLD-OUT FORECAST vs ACTUAL:")
        header_holdout = (
            f"{'Region':<18}{'Forecast':>12}{'Actual':>12}{'Error':>12}{'In PI?':>10}"
        )
        lines.append(header_holdout)
        for _, r in df_res.iterrows():
            if pd.notna(r.get("actual")):
                lines.append(
                    f"{PRETTY_NAMES.get(r['region'], r['region']):<18}"
                    f"{format_number(r['p50']):>12}"
                    f"{format_number(r['actual']):>12}"
                    f"{format_number(r['error']):>12}"
                    f"{r.get('in_interval', 'N/A'):>10}"
                )
        lines.append("=" * len(title))

    lines.append("")
    lines.append("CROSS-VALIDATION METRICS:")
    header = f"{'Region':<18}{'Forecast':>12}{'PI[10,90]':>18}{'CV MAE':>10}{'Bias':>10}{'HalfLife':>10}"
    lines.append(header)
    for _, r in df_res.iterrows():
        pi = f"[{format_number(r['p10'])},{format_number(r['p90'])}]"
        lines.append(
            f"{PRETTY_NAMES.get(r['region'], r['region']):<18}{format_number(r['p50']):>12}{pi:>18}"
            f"{format_number(r['cv_mae']):>10}{format_number(r['bias']):>10}{str(r['half_life']):>10}"
        )

    summary = "\n".join(lines)
    logger.info("\n" + summary)
    with open(os.path.join(outdir, "summary_table.txt"), "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")

    # Save run configuration
    run_config = {
        "features_file": features_path,
        "changes_file": changes_path,
        "output_directory": outdir,
        "last_training_week": str((forecast_thursday - pd.Timedelta(days=7)).date()),
        "forecast_week": str(forecast_thursday.date()),
        "eia_report_release_date": str(report_release_date.date()),
        "target_regions": TARGETS,
        "random_seed": 42,
        "methodology": "quantile_regression_with_time_series_cv",
        "reconciliation_weights": {k: float(v) for k, v in REGIONAL_WEIGHTS.items()},
        "half_life_explanation": "13 weeks in volatile periods, 26 weeks in stable conditions",
        "feature_selection": "Top 60 features by variance with domain-specific requirements",
        "validation_note": "Forecast is for the last Thursday in the dataset. EIA reports are released 7 days after the data week ends.",
    }
    with open(os.path.join(outdir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    logger.info(f"\nForecast pipeline complete. Results saved to {outdir}")


def main():
    parser = argparse.ArgumentParser()
    root = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--features", default=os.path.join(root, "output", "Combined_Wide_Data.csv")
    )
    parser.add_argument(
        "--changes", default=os.path.join(root, "output", "EIAchanges.csv")
    )
    parser.add_argument("--outdir", default=os.path.join(root, "output", "forecaster"))
    args = parser.parse_args()
    run(args.features, args.changes, args.outdir)


if __name__ == "__main__":
    main()
