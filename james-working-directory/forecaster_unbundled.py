#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Weekly EIA Forecaster — Regional, Fri→Thu windows, fold-safe selection, recency weighting, WLS reconciliation.

Inputs (defaults):
  features: ./output/Combined_Wide_Data.csv    # DAILY features, calendar-indexed
  changes : ./output/EIAchanges.csv            # WEEKLY changes, any weekday; will be normalized to week_end_thu

Artifacts:
  output/summary_table.txt
  output/next_week_forecast.csv
  output/backtest_metrics.csv
  output/backtest_predictions.csv
  output/residuals.csv
  output/run_config.json
  models/<region>_{point|q10|q50|q90}.cbm
"""

import argparse
import json
import logging
import math
import os
import re
import shutil
from typing import Dict, Iterable, List, Optional, Tuple

import colorama
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_absolute_error

colorama.init()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    colorama.Fore.GREEN
    + "%(asctime)s"
    + colorama.Fore.RESET
    + " - %(levelname)s - %(message)s"
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# ------------------------------ config ------------------------------

# Define target regions for forecasting EIA weekly changes
TARGETS = ["lower48", "east", "midwest", "southcentral", "mountain", "pacific"]

# Pretty names for display purposes
PRETTY = {
    "lower48": "Lower 48",
    "east": "East",
    "midwest": "Midwest",
    "southcentral": "South Central",
    "mountain": "Mountain",
    "pacific": "Pacific",
}

# Tags for days in the Fri→Thu week window (d0_fri to d6_thu)
DAY_TAGS = ["d0_fri", "d1_sat", "d2_sun", "d3_mon", "d4_tue", "d5_wed", "d6_thu"]

# Default hyperparameters for CatBoost models, tuned for MAE loss and stability
DEFAULT_PARAMS = {
    "iterations": 600,  # Number of boosting iterations
    "depth": 6,  # Maximum tree depth
    "learning_rate": 0.05,  # Learning rate for gradient descent
    "l2_leaf_reg": 3.0,  # L2 regularization strength
    "subsample": 0.8,  # Fraction of samples used for training each tree
    "rsm": 0.8,  # Random subspace method fraction
    "loss_function": "MAE",  # Mean Absolute Error as the objective
    "early_stopping_rounds": 50,  # Stop if no improvement after 50 rounds
    "random_seed": 42,  # For reproducibility
    "verbose": False,  # Suppress training output
    "use_best_model": True,  # Use best model from early stopping
}

# Regex aliases for South Central region to handle variations in naming
REG_ALIASES_SC = [r"south[_\s]?central", r"\bsc\b", r"\btx\b", r"texas", r"gulf"]


# Load best parameters from JSON file if exists, else use defaults
def load_params_or_default(tgt):
    fn = f"best_params_{tgt}.json"
    if os.path.exists(fn):
        with open(fn) as f:
            return json.load(f)
    return DEFAULT_PARAMS.copy()


# Utility function to select columns based on regex patterns
# must: patterns that must be present, anyof: at least one must match, none: exclude these
def _cols(df, must=None, anyof=None, none=None):
    pats = []
    if must:
        pats += [rf"(?=.*{p})" for p in must]  # Positive lookaheads for must patterns
    if anyof:
        pats += [rf"(?:{'|'.join(anyof)})"]  # Alternation for anyof
    rx = re.compile("".join(pats), flags=re.I)  # Case-insensitive regex
    keep = [c for c in df.columns if rx.search(c)]
    if none:
        rxn = re.compile("|".join(none), flags=re.I)
        keep = [c for c in keep if not rxn.search(c)]
    return keep


# Sum columns matching the patterns
def _sum_if(df, must=None, anyof=None, none=None):
    cols = _cols(df, must, anyof, none)
    return df[cols].sum(axis=1) if cols else pd.Series(index=df.index, dtype=np.float64)


# ------------------------------ utils ------------------------------


# Clean column names: lowercase, replace non-alphanum with _, strip
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


# Convert any date to the Thursday of its week (EIA reports on Thursdays)
def to_thu(ts: pd.Timestamp) -> pd.Timestamp:
    return ts - pd.Timedelta(days=((ts.weekday() - 3) % 7))


# Convert index to Thursday week ends
def week_thu_index(idx: pd.Index) -> pd.Index:
    return pd.Index([to_thu(pd.Timestamp(d)) for d in idx], name="week_end_thu")


# Assert that the index is monotonically increasing
def assert_monotone(idx: pd.Index, name: str):
    if not idx.is_monotonic_increasing:
        raise ValueError(f"{name} index not monotone increasing")


# Causal fill: forward fill, then replace NaNs with trailing median over window
# This avoids lookahead bias by only using past data
def causal_fill(s: pd.Series, window: int = 7) -> pd.Series:
    x = s.ffill()  # Forward fill existing NaNs
    med = x.rolling(window=window, min_periods=1).median()  # Trailing median
    return x.fillna(med)


# Build sequence matrix: flatten 7-day windows into rows for each Thursday
# This creates features from the past week's daily data
def build_seq_matrix(
    daily: pd.DataFrame, thursdays: Iterable[pd.Timestamp], include_flags: bool = True
) -> pd.DataFrame:
    # Select only numeric columns for flattening
    num = daily.select_dtypes(include=[np.number])
    cols = list(num.columns)
    rows, rindex = [], []
    for thu in thursdays:
        # Get dates from Fri to Thu (6 days back to current Thu)
        dates = [thu + pd.Timedelta(days=o) for o in range(-6, 1)]
        if any(d not in daily.index for d in dates):
            continue  # Skip if any date missing
        block = num.loc[dates, cols]
        flat = block.to_numpy().reshape(-1)  # Flatten the 7xN matrix to 1x(7*N)
        rows.append(flat)
        rindex.append(thu)
    # Column names: feature__day_tag for each day
    flat_cols = [f"{c}__{DAY_TAGS[d]}" for d in range(7) for c in cols]
    X = pd.DataFrame(
        rows, index=pd.Index(rindex, name="week_end_thu"), columns=flat_cols
    )
    if include_flags:
        # Add flags for any missing values in the 7-day window per feature
        # This helps the model know when data was imputed
        missing = num.isna().astype(int)
        win = {}
        for thu in X.index:
            dates = [thu + pd.Timedelta(days=o) for o in range(-6, 1)]
            if any(d not in missing.index for d in dates):
                continue
            w = missing.loc[dates, cols].sum(axis=0).clip(upper=1)  # 1 if any missing
            win[thu] = w
        if win:
            flags_df = pd.DataFrame(win).T
            flags_df.index.name = "week_end_thu"
            flags_df.columns = [f"{c}__any_missing" for c in flags_df.columns]
            X = X.join(flags_df, how="left")
    return X


def variance_topk_with_must(
    X: pd.DataFrame, top_k: int, must_have: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    must_have = [c for c in must_have if c in X.columns]
    X_must = X[must_have]

    remaining = [c for c in X.columns if c not in must_have]
    X_rem = X[remaining]

    if X_rem.shape[1] > 0:
        vt = VarianceThreshold(1e-12)
        Xv = vt.fit_transform(X_rem)
        cols = X_rem.columns[vt.get_support()]
        Xv = pd.DataFrame(Xv, index=X.index, columns=cols)

        if top_k > 0 and top_k < Xv.shape[1]:
            var = Xv.var().sort_values(ascending=False)
            keep = list(var.index[:top_k])
            Xv = Xv[keep]
    else:
        Xv = pd.DataFrame(index=X.index)

    X_final = pd.concat([X_must, Xv], axis=1)
    return X_final, list(X_final.columns)


# Compute recency weights: exponential decay with half-life in weeks
# Recent data gets higher weight to adapt to recent trends
def recency_weights(n: int, half_life_weeks: Optional[int]) -> np.ndarray:
    if n <= 1 or not half_life_weeks:
        return np.ones(n, dtype=np.float64)  # Uniform if no half-life
    tau = half_life_weeks / math.log(2.0)  # Time constant for exponential decay
    t = np.arange(n)  # Time indices
    w = np.exp((t - (n - 1)) / tau)  # Exponential weights, most recent highest
    return w / w.mean()  # Normalize so mean is 1


# Generate holdout validation blocks: K folds with gaps to avoid data leakage
# h: horizon gap before validation, v: validation length
def hv_blocks(n, K=5, h=3, v=3):
    # start folds in the latter half to keep regimes comparable
    edges = np.linspace(int(n * 0.5), n - 1, K + 1, dtype=int)
    for i in range(K):
        tr_end, va_end = edges[i], edges[i + 1]
        va_idx = np.arange(tr_end, va_end)  # validation window
        # training strictly BEFORE validation, with a gap of h
        tr_hi = max(0, tr_end - h)
        tr_idx = np.arange(0, tr_hi)
        yield tr_idx, va_idx


# Tune CatBoost hyperparameters using Optuna for MAE minimization
def tune_catboost_params(
    X_train,
    y_train,
    X_valid,
    y_valid,
    w_train,
    n_trials=50,  # Increased trials for more params
):
    def objective(trial):
        # Core training loop params
        depth = trial.suggest_int("depth", 4, 8)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.15)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1e-4, 100.0, log=True)
        iterations = trial.suggest_int("iterations", 400, 1200)

        # Bootstrap / sampling
        bootstrap_type = trial.suggest_categorical(
            "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
        )
        bagging_temperature = None
        subsample = None
        if bootstrap_type == "Bayesian":
            bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 2.0)
        elif bootstrap_type == "Bernoulli":
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
        # For MVS, no additional params

        rsm = trial.suggest_float("rsm", 0.5, 1.0)

        # Tree building
        grow_policy = trial.suggest_categorical(
            "grow_policy", ["SymmetricTree", "Lossguide"]
        )
        max_leaves = None
        if grow_policy == "Lossguide":
            max_leaves = trial.suggest_int("max_leaves", 16, 128)
        min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 1, 20)
        sampling_frequency = "PerTree"  # Default for Lossguide
        if grow_policy == "SymmetricTree":
            sampling_frequency = trial.suggest_categorical(
                "sampling_frequency", ["PerTree", "PerTreeLevel"]
            )

        # Loss and optimization
        leaf_estimation_method = "Gradient"  # Newton not supported for MAE
        leaf_estimation_iterations = trial.suggest_int(
            "leaf_estimation_iterations", 1, 10
        )
        eval_metric = trial.suggest_categorical("eval_metric", ["MAE", "RMSE"])

        # Regularization
        random_strength = trial.suggest_float("random_strength", 0.0, 5.0)
        feature_border_type = trial.suggest_categorical(
            "feature_border_type", ["GreedyLogSum", "Median"]
        )

        # Build params dict
        params = {
            "depth": depth,
            "learning_rate": learning_rate,
            "l2_leaf_reg": l2_leaf_reg,
            "rsm": rsm,
            "bootstrap_type": bootstrap_type,
            "grow_policy": grow_policy,
            "min_data_in_leaf": min_data_in_leaf,
            "sampling_frequency": sampling_frequency,
            "leaf_estimation_method": leaf_estimation_method,
            "leaf_estimation_iterations": leaf_estimation_iterations,
            "random_strength": random_strength,
            "feature_border_type": feature_border_type,
            "eval_metric": eval_metric,
            "iterations": iterations,
            "early_stopping_rounds": 50,
            "use_best_model": True,
            "loss_function": "MAE",
            "verbose": False,
            "random_seed": 42,
        }

        # Conditional params
        if bagging_temperature is not None:
            params["bagging_temperature"] = bagging_temperature
        if subsample is not None:
            params["subsample"] = subsample
        if max_leaves is not None:
            params["max_leaves"] = max_leaves

        model = CatBoostRegressor(**params)
        model.fit(
            X_train,
            y_train,
            sample_weight=w_train,
            eval_set=Pool(X_valid, y_valid),
        )
        preds = model.predict(X_valid)
        return mean_absolute_error(y_valid, preds)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


# Fit a quantile regression model for a specific quantile alpha
def fit_quantile(
    X: pd.DataFrame, y: pd.Series, alpha: np.float64, w: np.ndarray, params: Dict
) -> CatBoostRegressor:
    base_params = params.copy()
    base_params["loss_function"] = f"Quantile:alpha={alpha}"  # Set quantile loss
    m = CatBoostRegressor(**base_params)
    m.fit(X, y, sample_weight=w)
    return m


# Enforce quantile order: ensure q10 <= q50 <= q90, and contain point if provided
def enforce_quantile_order(p10, p50, p90, point=None):
    q10, q50, q90 = sorted(
        [np.float64(p10), np.float64(p50), np.float64(p90)]
    )  # Sort to enforce order
    if point is not None:
        if point < q10:
            q10 = point  # Adjust if point is below q10
        if point > q90:
            q90 = point  # Adjust if point is above q90
    return np.float64(q10), np.float64(q50), np.float64(q90)


# Reconcile parts to sum to total using weights
def reconcile_sum_to_total(total, parts, weights=None):
    if weights is None:
        weights = np.ones(len(parts))  # Equal weights if none provided
    delta = total - np.sum(parts)
    adj = delta * np.array(weights) / np.sum(weights)  # Proportional adjustment
    return parts + adj


# Reconcile quantiles while preserving spreads
def reconcile_quantiles(total, children_qs, weights=None):
    q10s, q50s, q90s = zip(*children_qs)  # Unzip quantiles
    adj_q50s = reconcile_sum_to_total(
        total[1], np.array(q50s), weights
    )  # Adjust medians
    spreads_low = np.array(q50s) - np.array(q10s)  # Lower spreads
    spreads_high = np.array(q90s) - np.array(q50s)  # Upper spreads
    adj_q10s = adj_q50s - spreads_low  # Adjust q10s
    adj_q90s = adj_q50s + spreads_high  # Adjust q90s
    return list(zip(adj_q10s, adj_q50s, adj_q90s))  # Fixed: was adj_q90s duplicated


# Reconcile all levels: adjust regions to sum to Lower 48
def reconcile_all_levels(next_point_raw, next_quantiles):
    kids = ["east", "midwest", "southcentral", "mountain", "pacific"]
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]  # Neutral weights
    if "lower48" not in next_point_raw or any(k not in next_point_raw for k in kids):
        return next_point_raw, next_quantiles
    parts = np.array([next_point_raw[k] for k in kids], dtype=np.float64)
    parts_adj = reconcile_sum_to_total(next_point_raw["lower48"], parts, weights)
    for k, v in zip(kids, parts_adj):
        next_point_raw[k] = np.float64(v)
    children_qs = [next_quantiles[k] for k in kids]
    adj_qs = reconcile_quantiles(next_quantiles["lower48"], children_qs, weights)
    for k, qs in zip(kids, adj_qs):
        next_quantiles[k] = tuple(np.float64(q) for q in qs)
    return next_point_raw, next_quantiles


# Format numbers for display: int if whole, else 2 decimals
def fmt_num(v: np.float64) -> str:
    if v is None or (isinstance(v, np.float64) and (np.isnan(v) or np.isinf(v))):
        return "NA"
    if abs(v - round(v)) < 1e-9:  # Check if essentially integer
        return f"{int(round(v))}"
    return f"{v:.2f}"


# Compute pinball loss for quantile evaluation
def pinball_loss(
    y_true: np.ndarray, y_pred: np.ndarray, alpha: np.float64
) -> np.float64:
    diff = y_true - y_pred
    return np.mean(
        np.where(diff >= 0, alpha * diff, (alpha - 1) * diff)
    )  # Asymmetric loss


def bias_corrected_shrink(pred, hist_mean, recent_bias, base=0.25):
    lam = base + min(0.25, abs(recent_bias) / 10)  # heavier shrink if bias is large
    lam = max(0.10, min(0.50, lam))
    return (1 - lam) * pred + lam * hist_mean


# Add South Central specific composite features
def add_sc_composites(df):
    sc_fac = df.filter(regex=r"(south[_\s]?central|tx|gulf).*critstor", axis=1).sum(
        axis=1
    )  # Sum critical storage for SC
    if not sc_fac.empty:
        df["sc_flow_std_7"] = sc_fac.rolling(
            7, min_periods=2
        ).std()  # 7-day rolling std

    # Proxy for SC CDD: use CDD if available, else temps
    sc_cdd = df.filter(regex=r"(south[_\s]?central|tx|gulf).*\bcdd\b", axis=1).sum(
        axis=1
    )
    if sc_cdd.empty:
        sc_cdd = df.filter(regex=r"(south[_\s]?central|tx).*temp", axis=1).mean(axis=1)
    if not sc_cdd.empty:
        dow = df.index.dayofweek  # Day of week
        fri_mon = ((dow == 4) | (dow == 5) | (dow == 6) | (dow == 0)).astype(
            int
        )  # Fri-Mon
        tue_thu = ((dow == 1) | (dow == 2) | (dow == 3)).astype(int)  # Tue-Thu
        num = (sc_cdd * fri_mon).rolling(7, min_periods=1).sum()  # Fri-Mon sum
        den = sc_cdd.rolling(7, min_periods=1).sum().replace(0, np.nan)  # Total sum
        df["sc_fri_mon_frac"] = num / den  # Fraction of CDD in Fri-Mon
        df["sc_midweek_skew"] = (sc_cdd * tue_thu).rolling(
            7, min_periods=1
        ).sum() - num  # Midweek excess

    return df


# ---------- helpers ----------
def enforce_index(df):
    df = df.copy()
    if "date" in df.columns:
        idx = pd.to_datetime(df["date"], errors="coerce")
        df = df.drop(columns=["date"])
    elif pd.api.types.is_datetime64_any_dtype(df.index):
        idx = pd.to_datetime(df.index, errors="coerce")
    else:
        # fall back: try first column
        first = df.columns[0]
        idx = pd.to_datetime(df[first], errors="coerce")
        if first != "date":
            df = df.drop(columns=[first])
    if idx.isna().any():
        raise ValueError("Unparseable dates in features; check the first/date column.")
    df.index = idx
    df = df.sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    return df.reindex(full_idx)


def flip_critstor(df, make_injection_positive=True):
    """If True, multiply CritStor by -1 so that injections are positive and withdrawals negative."""
    crit_cols = [c for c in df.columns if "critstor" in c]
    if make_injection_positive and crit_cols:
        df[crit_cols] = -1 * df[crit_cols]
    return df


# === Price feature engineering ===
def add_price_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer rich set of price-based features for forecasting accuracy.

    For each *_price_* series:
      - Raw daily levels (flattened by build_seq_matrix)
      - Basis vs Henry Hub
      - 7-day change (week-over-week)
      - 7-day rolling volatility
      - Cross spreads between key regional hubs
    """
    df = daily.copy()

    # Identify price columns
    price_cols = [c for c in df.columns if "_price_" in c.lower()]
    if not price_cols:
        return df

    # --- Reference hub (Henry Hub if present) ---
    henry_cols = [c for c in price_cols if "henryhub" in c.lower()]
    henry = df[henry_cols[0]] if henry_cols else None

    new_cols = []

    # Basis vs Henry
    if henry is not None:
        for col in price_cols:
            if col == henry_cols[0]:
                continue
            spread_name = col.replace("_price_", "_basis_")
            new_cols.append(df[col] - henry)
            new_cols[-1].name = spread_name

    # Week-over-week changes
    for col in price_cols:
        new_cols.append(df[col] - df[col].shift(7))
        new_cols[-1].name = f"{col}_chg7d"

    # Rolling 7d volatility
    for col in price_cols:
        new_cols.append(df[col].rolling(7, min_periods=3).std())
        new_cols[-1].name = f"{col}_std7d"

    # --- Cross-region spreads (key structural signals) ---
    hub_groups = {
        "southcentral": [c for c in price_cols if "southcentral" in c],
        "midwest": [c for c in price_cols if "midwest" in c or "chicago" in c],
        "east": [
            c for c in price_cols if "east" in c or "tetco" in c or "transco" in c
        ],
        "pacific": [
            c for c in price_cols if "pacific" in c or "socal" in c or "pge" in c
        ],
        "mountain": [c for c in price_cols if "mountain" in c or "opal" in c],
    }

    # Within South Central, capture Waha vs HSC / Katy
    sc_waha = [c for c in hub_groups["southcentral"] if "waha" in c]
    sc_hsc = [c for c in hub_groups["southcentral"] if "houston" in c or "hsc" in c]
    sc_katy = [c for c in hub_groups["southcentral"] if "katy" in c]

    if sc_waha and sc_hsc:
        new_cols.append(df[sc_waha[0]] - df[sc_hsc[0]])
        new_cols[-1].name = "sc_basis_waha_hsc"
    if sc_waha and sc_katy:
        new_cols.append(df[sc_waha[0]] - df[sc_katy[0]])
        new_cols[-1].name = "sc_basis_waha_katy"

    # East vs Midwest (market connectivity spreads)
    if hub_groups["east"] and hub_groups["midwest"]:
        new_cols.append(df[hub_groups["east"][0]] - df[hub_groups["midwest"][0]])
        new_cols[-1].name = "east_midwest_spread"

    # Pacific vs Mountain (west spreads)
    if hub_groups["pacific"] and hub_groups["mountain"]:
        new_cols.append(df[hub_groups["pacific"][0]] - df[hub_groups["mountain"][0]])
        new_cols[-1].name = "pacific_mountain_spread"

    if new_cols:
        new_df = pd.concat(new_cols, axis=1)
        df = pd.concat([df, new_df], axis=1)

    return df


# Add synthetic features: z-scores for weekly changes in salt, nonsalt, CDD
def add_synthetic_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()

    # Load South Central salt vs nonsalt mapping
    sc_salt_map = {}
    map_path = os.path.join(os.path.dirname(__file__), "sc_salt_map.json")
    if os.path.exists(map_path):
        with open(map_path) as f:
            sc_salt_map = json.load(f)

    # Separate SC facilities into salt and nonsalt based on mapping
    sc_salt_cols = [
        col for col, typ in sc_salt_map.items() if typ == "salt" and col in df.columns
    ]
    sc_nonsalt_cols = [
        col
        for col, typ in sc_salt_map.items()
        if typ == "nonsalt" and col in df.columns
    ]

    sc_salt = (
        df[sc_salt_cols].sum(axis=1)
        if sc_salt_cols
        else pd.Series(index=df.index, dtype=np.float64)
    )
    sc_nonsalt = (
        df[sc_nonsalt_cols].sum(axis=1)
        if sc_nonsalt_cols
        else pd.Series(index=df.index, dtype=np.float64)
    )

    if not sc_salt.empty:
        delta = sc_salt - sc_salt.shift(7)  # Weekly change
        mean_delta = delta.rolling(52, min_periods=10).mean()  # 52-week rolling mean
        std_delta = delta.rolling(52, min_periods=10).std().replace(0, 1)  # Rolling std
        z_score = (delta - mean_delta) / std_delta  # Z-score
        df["sc_salt_z"] = z_score.clip(-3, 3)  # Clip outliers
    if not sc_nonsalt.empty:
        delta = sc_nonsalt - sc_nonsalt.shift(7)
        mean_delta = delta.rolling(52, min_periods=10).mean()
        std_delta = delta.rolling(52, min_periods=10).std().replace(0, 1)
        z_score = (delta - mean_delta) / std_delta
        df["sc_nonsalt_z"] = z_score.clip(-3, 3)
    sc_cdd = _sum_if(df, anyof=REG_ALIASES_SC + [r"\bcdd\b"])  # SC cooling degree days
    if not sc_cdd.empty:
        delta = sc_cdd - sc_cdd.shift(7)
        mean_delta = delta.rolling(52, min_periods=10).mean()
        std_delta = delta.rolling(52, min_periods=10).std().replace(0, 1)
        z_score = (delta - mean_delta) / std_delta
        df["sc_cdd_z"] = z_score.clip(-3, 3)
    if "sc_salt_z" in df and "sc_cdd_z" in df:
        df["sc_salt_x_cdd"] = df["sc_salt_z"] * df["sc_cdd_z"]
    if "sc_nonsalt_z" in df and "sc_basis_waha_hsc" in df:
        df["sc_non_x_basis"] = df["sc_nonsalt_z"] * df["sc_basis_waha_hsc"]
    return df


# ------------------------------ pipeline ------------------------------


def split_into_regions(df):
    regions = ["east", "midwest", "southcentral", "mountain", "pacific"]
    region_dfs = {}
    for region in regions:
        prefix = f"{region}_"
        cols = [c for c in df.columns if c.startswith(prefix)]
        if cols:
            region_dfs[region] = df[cols].copy()
    return region_dfs


# Main forecasting pipeline
def run(features_path: str, changes_path: str, outdir: str):
    # Cleanup temp files and caches before each run
    if os.path.exists("catboost_info"):
        shutil.rmtree("catboost_info")
    if os.path.exists("models"):
        shutil.rmtree("models")
    # remove best_params_*.json files
    for fn in os.listdir("."):
        if fn.startswith("best_params_") and fn.endswith(".json"):
            os.remove(fn)

    os.makedirs(outdir, exist_ok=True)  # Create output directory
    os.makedirs("models", exist_ok=True)  # Create models directory

    # Load daily features and weekly changes
    daily = pd.read_csv(features_path)
    daily = daily.drop(
        columns=[  # Drop irrelevant or problematic columns
            "East_Weather_KATL_Daily_Precip_Amount",
            "East_Weather_KBOS_Daily_Precip_Amount",
            "East_Weather_KDCA_Daily_Precip_Amount",
            "East_Weather_KJFK_Daily_Precip_Amount",
            "East_Weather_KPHL_Daily_Precip_Amount",
            "East_Weather_KRDU_Daily_Precip_Amount",
            "Midwest_Weather_KDTW_Daily_Precip_Amount",
            "Midwest_Weather_KMSP_Max_Surface_Wind",
            "Midwest_Weather_KORD_Daily_Precip_Amount",
            "Mountain_Weather_KDEN_Avg_Cloud_Cover",
            "Mountain_Weather_KPHX_Daily_Precip_Amount",
            "South Central_CritStor_Louisiana_LA Storage",
            "South Central_Weather_KBHM_Daily_Precip_Amount",
            "South Central_Weather_KIAH_Daily_Precip_Amount",
            "South Central_Weather_KJAN_Daily_Precip_Amount",
            "South Central_Weather_KMAF_Avg_Cloud_Cover",
            "South Central_Weather_KSAT_Daily_Precip_Amount",
        ],
        errors="ignore",
    )
    daily = clean_cols(daily)
    daily = enforce_index(daily)
    daily = flip_critstor(daily, make_injection_positive=True)
    daily = daily.dropna(axis=1, how="all")  # drop all-NaN cols
    weekly = pd.read_csv(changes_path, index_col=0, parse_dates=True)

    # Clean and sort data
    weekly = clean_cols(weekly).sort_index()
    weekly.columns = weekly.columns.str.replace(
        "_change", "", regex=False
    )  # Remove _change suffix

    # Restrict to post-2018 to reduce regime shift noise
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
    days_until_thu = (3 - today.weekday()) % 7 or 7
    next_thursday = today + pd.Timedelta(days=days_until_thu)

    # The forecast target is the Thursday 7 days before the report release
    forecast_thursday = next_thursday - pd.Timedelta(days=7)  # data week end
    report_release_date = next_thursday

    # Normalize weekly index to Thursday week ends BEFORE checking forecast target
    weekly.index = week_thu_index(weekly.index)

    # Verify this Thursday exists in our dataset
    last_thu_in_data = weekly.index.max()

    # Determine if we're in live mode (forecasting future) or backtest mode (validating past)
    live_mode = forecast_thursday > last_thu_in_data

    if live_mode:
        logger.info(f"\n{'=' * 80}")
        logger.info("LIVE FORECAST MODE")
        logger.info(f"TODAY: {today.date()}")
        logger.info(f"FORECAST TARGET: Week ending Thursday {forecast_thursday.date()}")
        logger.info(f"EIA REPORT RELEASE: {report_release_date.date()} at 10:30 AM ET")
        logger.info(
            f"TRAINING DATA: All available data through {last_thu_in_data.date()}"
        )
        logger.info("Note: No actual data available for validation (live forecast)")
        logger.info(f"{'=' * 80}\n")
    else:
        logger.info(f"\n{'=' * 80}")
        logger.info("BACKTEST MODE")
        logger.info(f"TODAY: {today.date()}")
        logger.info(f"FORECAST TARGET: Week ending Thursday {forecast_thursday.date()}")
        logger.info(f"EIA REPORT RELEASE: {report_release_date.date()} at 10:30 AM ET")
        logger.info(f"TRAINING DATA: All weeks before {forecast_thursday.date()}")
        logger.info(f"{'=' * 80}\n")

    # Split: training data (all weeks before the forecast Thursday)
    weekly_train = weekly[weekly.index < forecast_thursday].copy()
    # Hold-out: the forecast Thursday (may be empty in live_mode)
    weekly_holdout = weekly[weekly.index == forecast_thursday].copy()

    # Compute historical means for shrinkage (use only training data to prevent leakage)
    historical_means = {
        tgt: weekly_train[tgt].mean() for tgt in TARGETS if tgt in weekly_train.columns
    }

    # Compute market volatility for regime-aware half-life
    high_vol = False
    henry_cols = [c for c in daily.columns if "henryhub" in c.lower()]
    if henry_cols:
        hh = daily[henry_cols[0]]
        recent_std = hh.tail(28).std()
        if recent_std > 0.5:  # Threshold for high volatility
            high_vol = True

    # Assert monotonic indices
    assert_monotone(daily.index, "features")
    assert_monotone(weekly_train.index, "changes_train")
    assert_monotone(weekly_holdout.index, "changes_holdout")

    # Add synthetic and composite features
    daily = add_synthetic_features(daily)
    daily = add_sc_composites(daily)
    daily = add_price_features(daily)

    # Split into region-specific feature sets
    region_dfs = split_into_regions(daily)
    for region, rdf in region_dfs.items():
        out_path = os.path.join(
            os.path.dirname(features_path), f"Combined_Clean_Features_{region}.csv"
        )
        rdf.to_csv(out_path)
        print(f"Saved region features: {out_path} shape={rdf.shape}")

    # Preserve raw NaNs for flags, prepare for causal filling
    daily_raw = daily.copy()

    # Build historical sequence matrix for training (using training data only)
    X_hist = build_seq_matrix(daily_raw, weekly_train.index, include_flags=True)
    Y_hist = weekly_train.reindex(X_hist.index).dropna()  # Align targets
    X_hist = X_hist.reindex(Y_hist.index)

    if X_hist.empty:
        raise ValueError("No aligned Fri→Thu windows between features and targets")

    # Determine forecast Thursday (this is the held-out week we want to forecast)
    forecast_thu = forecast_thursday
    need_forecast = [
        forecast_thu + pd.Timedelta(days=o) for o in range(-6, 1)
    ]  # Forecast week's dates
    window_df = daily_raw.reindex(need_forecast)
    have_forecast_data = (
        window_df.notna().select_dtypes(include=[np.number]).sum().sum()
        == window_df.select_dtypes(include=[np.number]).size
    )  # Check if all data available

    # Build forecast week's features: use raw if complete, else causal fill
    if have_forecast_data:
        X_next = build_seq_matrix(
            daily_raw, [forecast_thu], include_flags=True
        ).reindex(columns=X_hist.columns)
    else:
        win_start = forecast_thu - pd.Timedelta(days=6)
        idx7 = pd.date_range(win_start, forecast_thu, freq="D")
        temp = daily_raw.reindex(idx7)
        num = temp.select_dtypes(include=[np.number])
        for c in num.columns:
            num[c] = causal_fill(num[c], window=7)  # Causal fill missing data
        temp[num.columns] = num
        # Drop existing synthetic features to recompute on filled data
        temp = temp.drop(
            columns=["sc_salt_z", "sc_nonsalt_z", "sc_cdd_z"], errors="ignore"
        )
        temp = add_synthetic_features(temp)  # Recompute synthetics on filled data
        # Drop existing composite features to recompute on filled data
        temp = temp.drop(
            columns=["sc_flow_std_7", "sc_fri_mon_frac", "sc_midweek_skew"],
            errors="ignore",
        )
        temp = add_sc_composites(temp)  # Recompute composites on filled data
        # Drop existing price features to recompute on filled data
        price_cols_to_drop = [
            c
            for c in temp.columns
            if "_basis_" in c
            or "_chg7d" in c
            or "_std7d" in c
            or c
            in [
                "sc_basis_waha_hsc",
                "sc_basis_waha_katy",
                "east_midwest_spread",
                "pacific_mountain_spread",
            ]
        ]
        temp = temp.drop(columns=price_cols_to_drop, errors="ignore")
        temp = add_price_features(temp)  # Recompute price features on filled data
        X_next = build_seq_matrix(temp, [forecast_thu], include_flags=True).reindex(
            columns=X_hist.columns
        )

    # Initialize result collections
    results: List[Dict] = []
    bt_metrics: List[Dict] = []
    residuals_log: List[Dict] = []
    backtest_log: List[Dict] = []
    inv_res_var: Dict[str, np.float64] = {}
    next_point_raw: Dict[str, np.float64] = {}
    next_quantiles: Dict[str, Tuple[np.float64, np.float64, np.float64]] = {}
    params_dict: Dict[str, Dict] = {}
    half_lives: Dict[str, Optional[int]] = {}
    final_features: Dict[str, List[str]] = {}
    recent_bias: Dict[str, np.float64] = {}
    offsets_q10: Dict[str, np.float64] = {}
    offsets_q90: Dict[str, np.float64] = {}

    # Loop over each target region
    for tgt in TARGETS:
        if tgt not in Y_hist.columns:
            continue

        y = Y_hist[tgt].astype(np.float64)
        X = X_hist.copy()

        n = len(X)
        if n < 120:  # Minimum samples for robust training
            raise ValueError(f"Not enough samples for {tgt} ({n} < 120)")

        # Load or use default params
        params = load_params_or_default(tgt)
        params_dict[tgt] = params

        # Fixed half-life for recency weighting
        hl = 13 if high_vol else 26
        half_lives[tgt] = hl
        w_all = recency_weights(n, hl)

        # Perform holdout backtest with hv-blocks
        blocks = list(hv_blocks(n, K=5, h=3, v=3))

        # Tune hyperparameters if not already done
        if not os.path.exists(f"best_params_{tgt}.json"):
            tr_idx, va_idx = blocks[0]
            Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
            Xva, yva = X.iloc[va_idx], y.iloc[va_idx]
            must_have = [
                "sc_salt_z",
                "sc_nonsalt_z",
                "sc_cdd_z",
                "sc_basis_waha_hsc",
                "sc_basis_waha_katy",
                "east_midwest_spread",
                "pacific_mountain_spread",
            ]
            Xtr_v, _ = variance_topk_with_must(Xtr, 60, must_have)
            Xva_v = Xva[Xtr_v.columns]
            if len(Xva_v) > 0 and len(Xtr_v) >= 60:
                w_tr = w_all[tr_idx]
                w_va = w_all[va_idx]
                best_params = tune_catboost_params(
                    Xtr_v, ytr, Xva_v, yva, w_tr, w_va, n_trials=10
                )
                with open(f"best_params_{tgt}.json", "w") as f:
                    json.dump(best_params, f)
                params = best_params
                params_dict[tgt] = params

        preds_bt, acts_bt = [], []
        res_q10_list = []
        res_q90_list = []
        pinball_10: List[np.float64] = []
        pinball_50: List[np.float64] = []
        pinball_90: List[np.float64] = []
        for tr_idx, va_idx in blocks:
            Xtr, ytr = X.iloc[tr_idx], y.iloc[tr_idx]
            Xva, yva = X.iloc[va_idx], y.iloc[va_idx]
            must_have = [
                "sc_salt_z",
                "sc_nonsalt_z",
                "sc_cdd_z",
                "sc_basis_waha_hsc",
                "sc_basis_waha_katy",
                "east_midwest_spread",
                "pacific_mountain_spread",
            ]
            Xtr_v, _ = variance_topk_with_must(
                Xtr, 60, must_have
            )  # Select top 60 by variance
            Xva_v = Xva[Xtr_v.columns]
            if len(Xva_v) == 0 or len(Xtr_v) < 60:
                continue
            w = recency_weights(len(Xtr_v), hl)  # Recency weights for training
            w = w / np.mean(w)  # Normalize weights
            params_backtest = params.copy()
            params_backtest["loss_function"] = "MAE"
            m = CatBoostRegressor(**params_backtest)
            m.fit(
                Xtr_v,
                ytr,
                sample_weight=w,
                eval_set=Pool(Xva_v, yva),
                use_best_model=True,
            )
            p = m.predict(Xva_v)
            preds_bt.append(pd.Series(p, index=Xva.index))
            acts_bt.append(yva)
            # Train quantile models for evaluation
            m_q10_bt = fit_quantile(Xtr_v, ytr, 0.10, w_all[tr_idx], params)
            m_q50_bt = fit_quantile(Xtr_v, ytr, 0.50, w_all[tr_idx], params)
            m_q90_bt = fit_quantile(Xtr_v, ytr, 0.90, w_all[tr_idx], params)
            p_q10 = m_q10_bt.predict(Xva_v)
            p_q50 = m_q50_bt.predict(Xva_v)
            p_q90 = m_q90_bt.predict(Xva_v)
            pinball_10.append(pinball_loss(yva.values, p_q10, 0.10))
            pinball_50.append(pinball_loss(yva.values, p_q50, 0.50))
            pinball_90.append(pinball_loss(yva.values, p_q90, 0.90))
            res_q10_list.extend(yva.values - p_q10)
            res_q90_list.extend(yva.values - p_q90)

        # Aggregate backtest results
        if preds_bt:
            preds_bt = pd.concat(preds_bt).sort_index()
            acts_bt = pd.concat(acts_bt).sort_index()
            res = acts_bt - preds_bt
            mae = np.float64(np.mean(np.abs(res.values)))
            bias = np.float64(np.mean(res.values))
            var = np.float64(np.var(res.values, ddof=1)) if len(res) > 1 else 1.0
            recent_bias[tgt] = res.rolling(12, min_periods=4).mean().iloc[-1]
            for ts, e in res.items():
                residuals_log.append(
                    {
                        "week_end_thu": str(pd.Timestamp(ts).date()),
                        "target": tgt,
                        "residual": np.float64(e),
                    }
                )
            for ts in preds_bt.index:
                backtest_log.append(
                    {
                        "week_end_thu": str(pd.Timestamp(ts).date()),
                        "target": tgt,
                        "pred": np.float64(preds_bt.loc[ts]),
                        "actual": np.float64(acts_bt.loc[ts]),
                    }
                )
        else:
            mae = bias = np.float64("nan")
            var = 1.0
            avg_pinball_10 = avg_pinball_50 = avg_pinball_90 = np.float64("nan")
            recent_bias[tgt] = 0.0

        if pinball_10:
            avg_pinball_10 = np.float64(np.mean(pinball_10))
            avg_pinball_50 = np.float64(np.mean(pinball_50))
            avg_pinball_90 = np.float64(np.mean(pinball_90))
        else:
            avg_pinball_10 = avg_pinball_50 = avg_pinball_90 = np.float64("nan")

        if res_q10_list:
            offsets_q10[tgt] = np.percentile(res_q10_list, 10)
            offsets_q90[tgt] = np.percentile(res_q90_list, 90)
        else:
            offsets_q10[tgt] = 0.0
            offsets_q90[tgt] = 0.0
        inv_res_var[tgt] = 0.0 if (not np.isfinite(var) or var <= 0) else 1.0 / var
        bt_metrics.append(
            {
                "target": tgt,
                "region": PRETTY[tgt],
                "cv_mae": mae,
                "bias": bias,
                "half_life": hl,
                "pinball_q10": avg_pinball_10,
                "pinball_q50": avg_pinball_50,
                "pinball_q90": avg_pinball_90,
            }
        )

        # Select final features for production model
        must_have = [
            "sc_salt_z",
            "sc_nonsalt_z",
            "sc_cdd_z",
            "sc_basis_waha_hsc",
            "sc_basis_waha_katy",
            "east_midwest_spread",
            "pacific_mountain_spread",
        ]
        X_full_sel, _ = variance_topk_with_must(X, 60, must_have)
        final_features[tgt] = list(X_full_sel.columns)

        # Train final models: point and quantiles
        params_point = params.copy()
        params_point["train_dir"] = f"models/{tgt}_point"
        m_point = CatBoostRegressor(**params_point)
        m_point.fit(X_full_sel, y, sample_weight=w_all)
        params_q10 = params.copy()
        params_q10["loss_function"] = "Quantile:alpha=0.10"
        params_q10["train_dir"] = f"models/{tgt}_point"
        m_q10 = CatBoostRegressor(**params_q10)
        m_q10.fit(X_full_sel, y, sample_weight=w_all)
        params_q50 = params.copy()
        params_q50["loss_function"] = "Quantile:alpha=0.50"
        params_q50["train_dir"] = f"models/{tgt}_point"
        m_q50 = CatBoostRegressor(**params_q50)
        m_q50.fit(X_full_sel, y, sample_weight=w_all)
        params_q90 = params.copy()
        params_q90["loss_function"] = "Quantile:alpha=0.90"
        params_q90["train_dir"] = f"models/{tgt}_point"
        m_q90 = CatBoostRegressor(**params_q90)
        m_q90.fit(X_full_sel, y, sample_weight=w_all)

        # Predict next week
        if X_next is not None:
            x_next = X_next.reindex(columns=X_full_sel.columns).fillna(0.0).iloc[[-1]]
        else:
            x_next = X_full_sel.iloc[[-1]]  # Fallback to last historical

        f_point = np.float64(m_point.predict(x_next).item())
        f_q10 = np.float64(m_q10.predict(x_next).item())
        f_q50 = np.float64(m_q50.predict(x_next).item())
        f_q90 = np.float64(m_q90.predict(x_next).item())

        # Enforce quantile ordering
        f_q10, f_q50, f_q90 = enforce_quantile_order(f_q10, f_q50, f_q90, f_point)

        if tgt == "lower48":
            f_point = f_q50  # Use median as point for Lower 48

        # Apply bias-corrected shrinkage to all regions
        if tgt in historical_means and tgt in recent_bias:
            f_point = bias_corrected_shrink(
                f_point, historical_means[tgt], recent_bias[tgt]
            )
            f_q10 = bias_corrected_shrink(
                f_q10, historical_means[tgt], recent_bias[tgt]
            )
            f_q50 = bias_corrected_shrink(
                f_q50, historical_means[tgt], recent_bias[tgt]
            )
            f_q90 = bias_corrected_shrink(
                f_q90, historical_means[tgt], recent_bias[tgt]
            )

        # Apply conformal offsets to prediction intervals
        f_q10 += offsets_q10.get(tgt, 0.0)
        f_q90 += offsets_q90.get(tgt, 0.0)

        # Save models
        mdir = "models"
        m_point.save_model(os.path.join(mdir, f"{tgt}_point.cbm"))
        m_q10.save_model(os.path.join(mdir, f"{tgt}_q10.cbm"))
        m_q50.save_model(os.path.join(mdir, f"{tgt}_q50.cbm"))
        m_q90.save_model(os.path.join(mdir, f"{tgt}_q90.cbm"))

        next_point_raw[tgt] = f_point
        next_quantiles[tgt] = (f_q10, f_q50, f_q90)

        results.append(
            {
                "region": tgt,
                "next_est": round(f_point, 2),
                "p10": round(f_q10, 2),
                "p50": round(f_q50, 2),
                "p90": round(f_q90, 2),
                "cv_mae": round(mae, 3),
                "bias": round(bias, 3),
                "half_life": hl,
                "n_features": len(X_full_sel.columns),
            }
        )

    # Post-processing: reconciliation
    if "lower48" in next_point_raw and all(
        k in next_point_raw
        for k in ["east", "midwest", "southcentral", "mountain", "pacific"]
    ):
        next_point_raw, next_quantiles = reconcile_all_levels(
            next_point_raw, next_quantiles
        )

    # Update results with reconciled values and actual holdout values
    for r in results:
        tgt = r["region"]
        if tgt in next_point_raw:
            r["next_est"] = round(next_point_raw[tgt], 2)
        if tgt in next_quantiles:
            f_q10, f_q50, f_q90 = enforce_quantile_order(
                *next_quantiles[tgt], next_point_raw[tgt]
            )
            r["p10"], r["p50"], r["p90"] = [round(q, 2) for q in (f_q10, f_q50, f_q90)]

        # Add actual value from held-out forecast Thursday
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

    # Output results
    df_res = pd.DataFrame(results).sort_values("region")
    df_res.to_csv(os.path.join(outdir, "next_week_forecast.csv"), index=False)

    df_bt = pd.DataFrame(bt_metrics)
    df_bt.to_csv(os.path.join(outdir, "backtest_metrics.csv"), index=False)

    if residuals_log:
        pd.DataFrame(residuals_log).to_csv(
            os.path.join(outdir, "residuals.csv"), index=False
        )
    if backtest_log:
        pd.DataFrame(backtest_log).to_csv(
            os.path.join(outdir, "backtest_predictions.csv"), index=False
        )

    # Create summary table
    lines = []
    title = "Operational Forecast Summary"
    lines.append(title)
    lines.append("-" * len(title))

    lines.append("")
    lines.append("FORECAST SETUP:" if live_mode else "HOLD-OUT VALIDATION:")
    lines.append(f"  ---> Today: {today.date()}")
    lines.append(f"  ---> Forecast Thursday: {forecast_thursday.date()}")
    lines.append(
        f"  ---> EIA Report Release: {report_release_date.date()} at 10:30 AM ET"
    )
    lines.append(
        f"  ---> Training data: Through {(forecast_thursday - pd.Timedelta(days=7)).date()}"
    )
    if live_mode:
        lines.append("  ---> Mode: LIVE (no actual data available for validation)")
    else:
        lines.append("  ---> Mode: BACKTEST (actual data available for validation)")
    lines.append("")

    # Add hold-out validation results if available (only in backtest mode)
    if (
        (not live_mode)
        and "actual" in df_res.columns
        and df_res["actual"].notna().any()
    ):
        lines.append("HOLD-OUT FORECAST vs ACTUAL:")
        hdr_holdout = (
            f"{'Region':<18}{'Forecast':>12}{'Actual':>12}{'Error':>12}{'In PI?':>10}"
        )
        lines.append(hdr_holdout)
        for _, r in df_res.iterrows():
            if pd.notna(r.get("actual")):
                lines.append(
                    f"{PRETTY.get(r['region'], r['region']):<18}"
                    f"{fmt_num(r['p50']):>12}"
                    f"{fmt_num(r['actual']):>12}"
                    f"{fmt_num(r['error']):>12}"
                    f"{r.get('in_interval', 'N/A'):>10}"
                )
        lines.append("-" * len(title))
        lines.append("")

    lines.append("CROSS-VALIDATION METRICS:")
    hdr = f"{'Region':<18}{'Forecast':>12}{'PI[10,90]':>18}{'CV MAE':>10}{'Bias':>10}{'HalfLife':>10}{'Nfeat':>8}"
    lines.append(hdr)
    for _, r in df_res.iterrows():
        pi = f"[{fmt_num(r['p10'])},{fmt_num(r['p90'])}]"
        lines.append(
            f"{PRETTY.get(r['region'], r['region']):<18}{fmt_num(r['next_est']):>12}{pi:>18}"
            f"{fmt_num(r['cv_mae']):>10}{fmt_num(r['bias']):>10}{str(r['half_life']):>10}{int(r['n_features']):>8}"
        )

    summary = "\n".join(lines)
    logger.info(summary)
    with open(os.path.join(outdir, "summary_table.txt"), "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")

    # Save run configuration
    run_cfg = {
        "features": features_path,
        "changes": changes_path,
        "outdir": outdir,
        "today": str(today.date()),
        "forecast_thursday": str(forecast_thursday.date()),
        "eia_report_release_date": str(report_release_date.date()),
        "last_train_week_end_thu": str(
            (forecast_thursday - pd.Timedelta(days=7)).date()
        ),
        "live_mode": live_mode,
        "have_forecast_week_inputs": have_forecast_data,
        "targets": TARGETS,
        "tuned_params": params_dict,
        "half_lives": half_lives,
        "random_seed": 42,
        "final_features": final_features,
        "validation_note": "Forecast is for the Thursday whose EIA report will be released on the next Thursday from today. EIA reports are released 7 days after the data week ends.",
    }
    with open(os.path.join(outdir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(run_cfg, fh, indent=2)


# Main entry point with argument parsing
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
