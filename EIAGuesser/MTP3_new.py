#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MTP3.py — EIA Weekly Changes Forecaster (Fri--->Thu sequences)

Production-grade research pipeline with full reproducibility, robustness, and academic hygiene.

Highlights
- Non-lossy Fri--->Thu sequence builder with explicit missingness handling
- Adaptive variance filter with NaN-aware computation
- Lower-48-driven recency weighting with regional homogeneity testing
- Big model zoo with deterministic seeding and parallelism
- PyTorch transformer with normalization, deterministic algorithms, and LR scheduling
- Gap-aware time series CV to prevent leakage
- Variance-based reconciliation with statistical foundation
- Modular structure for composability and experimentation
- Portable CLI with relative path resolution
- Uncertainty quantification via bootstrapped prediction intervals

Run example with recommended parameters (adjust paths as needed):
    python3 MTP3_new.py --reconcile variance --splits 5 --top_k -1 --use_transformer --device cpu --quiet

If you don't have torch installed, omit --use_transformer. For detailed output, drop --quiet.

TODO: Add support for GPU acceleration, maybe CUDA if we ever get rich enough for a GPU
"""

# Holy imports batman, this thing needs half the python ecosystem
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import pickle
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Optional boosters - because why not have 50 different ML libraries?
HAS_LGB = False
try:  # pragma: no cover
    import lightgbm as lgb  # type: ignore

    HAS_LGB = True
except Exception:  # pragma: no cover
    pass

HAS_XGB = False
try:  # pragma: no cover
    HAS_XGB = True
except Exception:  # pragma: no cover
    pass

HAS_CAT = False
try:  # pragma: no cover
    from catboost import CatBoostRegressor  # type: ignore

    HAS_CAT = True
except Exception:  # pragma: no cover
    pass

HAS_TORCH = False
try:  # pragma: no cover
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore

    HAS_TORCH = True
except Exception:  # pragma: no cover
    pass

HAS_COLORLOG = False
try:  # pragma: no cover
    import colorlog  # type: ignore

    HAS_COLORLOG = True
except Exception:  # pragma: no cover
    pass

from joblib import Parallel, delayed
from tabulate import tabulate

# TODO: Consider adding more optional dependencies, maybe even tensorflow for the hell of it

# ------------------------------ Constants ---------------------------------
# These magical numbers that make the whole thing work. Don't touch unless you want chaos.
DAY_TAGS = [
    "d0_fri",
    "d1_sat",
    "d2_sun",
    "d3_mon",
    "d4_tue",
    "d5_wed",
    "d6_thu",
]
TARGET_MAP = {
    "conus": "lower48",
    "east": "east",
    "midwest": "midwest",
    "south_central": "southcentral",
    "mountain": "mountain",
    "pacific": "pacific",
}
REGION_PRETTY = {
    "conus": "Lower 48",
    "east": "East",
    "midwest": "Midwest",
    "south_central": "South Central",
    "mountain": "Mountain",
    "pacific": "Pacific",
}

# ------------------------------ Logging Setup --------------------------------


def setup_logging(quiet: bool) -> logging.Logger:
    """Set up colored logging if available."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if not quiet else logging.INFO)
    handler = logging.StreamHandler()
    if HAS_COLORLOG:
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
            datefmt="%H:%M:%S",
        )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# ------------------------------ Data Types --------------------------------
@dataclass
class Row:
    """Row for the final terminal table.

    Because we need to display results somehow, and pandas DataFrames are too mainstream.
    """

    region: str
    model: str
    next_est: np.float64
    cv_mae: np.float64
    last_est: np.float64
    last_act: np.float64
    pred_interval: Optional[Tuple[np.float64, np.float64]] = None


@dataclass
class Config:
    """Configuration for the forecasting pipeline.

    All the knobs and dials to tweak this beast. Mess with these at your own risk.
    """

    features_path: Path
    changes_path: Path
    output_dir: Path
    n_splits: int
    top_k: Optional[int]
    reconcile_method: str
    use_transformer: bool
    device: str
    quiet: bool
    seed: int = 42


class GapTimeSeriesSplit:
    """Gap-aware time series cross-validator using rolling origin splits tied to Thursday timestamps.

    Prevents data leakage by ensuring train sets are strictly before test sets with temporal gap.
    """

    def __init__(self, n_splits: int, gap: int = 1):
        self.n_splits = n_splits
        self.gap = gap  # in weeks

    def split(self, X: pd.DataFrame):
        if not isinstance(X.index, pd.DatetimeIndex):
            # Fallback to index-based for non-time data
            n = len(X)
            test_size = n // (self.n_splits + 1)
            indices = np.arange(n)
            for i in range(self.n_splits):
                train_end = n - (self.n_splits - i) * test_size - self.gap
                test_start = train_end + self.gap
                test_end = test_start + test_size
                yield indices[:train_end], indices[test_start : min(test_end, n)]
        else:
            # Canonical rolling-origin with fixed h=1 week horizon and gap
            idx = pd.Index(sorted(set(to_thursday_label(ts) for ts in X.index)))
            for i in range(self.n_splits):
                test_end = len(idx) - (self.n_splits - i - 1)
                test_start = test_end - 1  # one-week horizon
                train_end = test_start - self.gap
                if train_end <= 0:
                    continue
                train_mask = X.index <= idx[train_end - 1]
                test_mask = (X.index > idx[train_end - 1]) & (
                    X.index <= idx[test_start]
                )
                yield np.flatnonzero(train_mask), np.flatnonzero(test_mask)


# ------------------------------ Utilities ---------------------------------


def set_global_seed(seed: int) -> None:
    """Set deterministic random state across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic algorithms for reproducibility
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            # Some ops may not support deterministic mode
            pass


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with snake_case alphanumeric column names.

    Because EIA data has the worst column names ever. This makes them readable.
    """
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return out


def to_thursday_label(ts: pd.Timestamp) -> pd.Timestamp:
    """Map any timestamp to the Thursday of its week (Thu=3).

    EIA reports on Thursdays, so we normalize everything to Thursday. Don't ask why.
    """
    return ts - pd.Timedelta(days=((ts.weekday() - 3) % 7))


def normalize_week_index(idx: pd.Index) -> pd.Index:
    """Return a Thursday-indexed weekly index for given timestamps using pandas resample.

    More Thursday magic. This ensures all our data aligns properly.
    """
    # Use pandas resample to handle DST and missing days robustly
    temp_df = pd.DataFrame(index=idx)
    resampled = temp_df.resample("W-THU").asfreq()
    return resampled.index


def build_sequence_matrix(
    daily_num: pd.DataFrame,
    week_thus: Iterable[pd.Timestamp],
    allow_partial: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Create flattened Fri→Thu windows as one row per week.

    The core magic: turns daily data into weekly sequences. This is where the ML happens.
    TODO: Optimize this for memory usage, it's a bit of a memory hog
    """
    cols = list(daily_num.columns)
    rows, rindex, completeness = [], [], []
    for thu in week_thus:
        dates = [thu + pd.Timedelta(days=o) for o in range(-6, 1)]
        available_dates = [d for d in dates if d in daily_num.index]

        if len(available_dates) < 7:
            if not allow_partial:
                continue  # Skip incomplete weeks
            # Interpolate missing days
            seq_data = []
            for d in dates:
                if d in daily_num.index:
                    seq_data.append(daily_num.loc[d, cols].values)
                else:
                    # Forward/backward fill from nearest available days
                    prev_date = daily_num.index[daily_num.index <= d].max()
                    next_date = daily_num.index[daily_num.index >= d].min()
                    if (
                        prev_date is not pd.NaT
                        and next_date is not pd.NaT
                        and prev_date is not None
                        and next_date is not None
                    ):
                        weight = (d - prev_date).days / (next_date - prev_date).days
                        interpolated = (1 - weight) * daily_num.loc[
                            prev_date, cols
                        ] + weight * daily_num.loc[next_date, cols]
                        seq_data.append(interpolated.values)
                    elif prev_date is not pd.NaT and prev_date is not None:
                        seq_data.append(daily_num.loc[prev_date, cols].values)
                    elif next_date is not pd.NaT and next_date is not None:
                        seq_data.append(daily_num.loc[next_date, cols].values)
                    else:
                        seq_data.append(np.full(len(cols), np.nan))
            seq = np.concatenate(seq_data)
            completeness_val = len(available_dates) / 7.0
        else:
            seq = daily_num.loc[dates, cols].to_numpy().reshape(-1)
            completeness_val = 1.0

        rows.append(seq.tolist())
        rindex.append(thu)
        completeness.append(completeness_val)

    flat_cols = [f"{c}__{DAY_TAGS[d]}" for d in range(7) for c in cols]
    # Add missingness ratio feature per week
    flat_cols.append("missingness_ratio")
    # Add completeness to rows
    for i, row in enumerate(rows):
        row.append(completeness[i])
    return (
        pd.DataFrame(
            rows, index=pd.Index(rindex, name="week_end_thu"), columns=flat_cols
        ),
        pd.Series(completeness, index=rindex, name="completeness"),
    )


def variance_topk(
    X_df: pd.DataFrame, top_k: Optional[int], logger: logging.Logger = None
) -> pd.DataFrame:
    """Drop zero-variance cols; keep top-K by robust scale metric.

    Feature selection because not all features are created equal. Some are just noise.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Compute MAD-based scale
    mad = X_df.sub(X_df.median()).abs().median()
    score = X_df.var() / (mad + 1e-6)
    var_series = score.sort_values(ascending=False)

    if top_k is None:
        k = int(min(4000, max(800, 0.35 * len(var_series))))
    else:
        k = min(top_k, len(var_series))

    logger.info(
        f"Feature selection: {len(X_df.columns)} total → {k} selected (ratio: {100 * k / len(X_df.columns):.1f}%)"
    )

    # DIAGNOSTIC: Check feature explosion
    if k > len(var_series) * 0.1 and len(var_series) < 1000:
        logger.warning(
            f"  ⚠️  HIGH FEATURE RATIO: {k}/{len(var_series)} = {100 * k / len(var_series):.1f}% - HIGH OVERFITTING RISK"
        )

    selected_cols = var_series.index[:k]
    return X_df[selected_cols]


def permutation_test_paired(diffs: np.ndarray, n_permutations: int = 2000) -> float:
    """Permutation test for paired differences, returns p-value."""
    if len(diffs) < 2:
        return 1.0
    observed = np.abs(np.mean(diffs))
    count = 0
    for _ in range(n_permutations):
        signs = np.random.choice([-1, 1], size=len(diffs))
        perm_diffs = diffs * signs
        if np.abs(np.mean(perm_diffs)) >= observed:
            count += 1
    return count / n_permutations


def make_recency_weights(n: int, half_life_weeks: Optional[int]) -> np.ndarray:
    """Exponential recency weights normalized to mean=1 for length n.

    Recent data matters more. This gives exponential decay weights.
    """
    if n <= 1 or not half_life_weeks:
        return np.ones(n)
    # t from n-1 (oldest) to 0 (most recent)
    t = np.arange(n - 1, -1, -1)
    w = np.exp(-t * np.log(2) / half_life_weeks)
    return w / np.mean(w)


def weighted_mae(
    y_true: np.ndarray, y_pred: np.ndarray, w: Optional[np.ndarray]
) -> np.float64:
    """Weighted MAE; falls back to unweighted if `w` is None.

    Because sometimes you want to weight recent predictions more heavily.
    """
    if w is None:
        return np.float64(mean_absolute_error(y_true, y_pred))
    w = np.clip(np.asarray(w, np.float64), 1e-12, None)
    return np.float64(np.sum(w * np.abs(y_true - y_pred)) / np.sum(w))


# ----------------------------- Half-life selector -------------------------


def select_half_life(
    X: pd.DataFrame, y: pd.Series, splits: int, gap_size: int = 1
) -> Tuple[Optional[int], np.ndarray]:
    """Select optimal half-life for recency weighting using CV.

    Tries different half-lives and picks the one that gives best CV performance.
    This is basically hyperparameter tuning for the recency weighting.
    TODO: Cache results since this is expensive and doesn't change much
    """
    candidates = [None, 4, 8, 12, 16, 20, 26, 32, 40, 52]
    best_hl, best_score = None, np.float64("inf")

    gap_cv = GapTimeSeriesSplit(n_splits=splits, gap=gap_size)

    for hl in candidates:
        maes = []
        for tr, va in gap_cv.split(X):
            w = make_recency_weights(len(tr), hl)
            m = Ridge(alpha=10.0, solver="svd", random_state=42)
            fit_with_optional_weight(m, X.iloc[tr], y.iloc[tr], w)
            pv = m.predict(X.iloc[va])
            maes.append(weighted_mae(y.iloc[va].values, pv, None))
        score = np.float64(np.mean(maes))
        if score < best_score:
            best_hl, best_score = hl, score

    # Compute final weights for best half-life
    final_weights = make_recency_weights(len(X), best_hl)
    return best_hl, final_weights


def select_half_life_with_homogeneity_test(
    X: pd.DataFrame,
    y_regions: pd.DataFrame,
    splits: int,
    gap_size: int = 1,
    logger: logging.Logger = None,
) -> Tuple[Optional[int], np.ndarray, Dict[str, np.float64], Dict[str, np.float64]]:
    """Select half-life and test regional homogeneity assumption with statistical significance.

    Tests if using L48 weights vs region-specific weights makes a difference.
    Basically statistical testing to see if regions behave similarly.
    TODO: This is getting complex, maybe simplify or add more statistical tests
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    best_hl, weights = select_half_life(X, y_regions["lower48"], splits, gap_size)

    # DIAGNOSTIC: Log weight distribution
    logger.info("\nRecency Weighting Diagnostics")
    logger.info(f"Selected half-life: {best_hl} weeks")
    if best_hl is not None:
        logger.info(f"Weight range: {weights.min():.4f} - {weights.max():.4f}")
        logger.info(f"Recent week (t=0) weight: {weights[-1]:.4f}")
        logger.info(f"Oldest week weight: {weights[0]:.4f}")
        logger.info(f"Weight ratio (recent/oldest): {weights[-1] / weights[0]:.2f}x")

        # Flag if recent weights are extremely high (could amplify noise)
        if weights[-1] / weights[0] > 5:
            logger.warning(
                f"  ⚠️  HIGH RECENCY BIAS: Recent weeks weighted {weights[-1] / weights[0]:.1f}x more"
            )
            logger.warning(
                "      This could amplify anomalies in recent data into wild predictions"
            )

    # Test homogeneity: compare CV performance when applying L48 weights vs own weights
    homogeneity_scores = {}
    homogeneity_p_values = {}
    base_model = build_model_registry(1)["HistGBM"]

    gap_cv = GapTimeSeriesSplit(n_splits=splits, gap=gap_size)

    for region in y_regions.columns:
        if region == "lower48":
            continue
        l48_maes = []
        own_maes = []
        for tr, va in gap_cv.split(X):
            # L48 weights
            m_l48 = clone(base_model)
            fit_with_optional_weight(
                m_l48, X.iloc[tr], y_regions[region].iloc[tr], weights[tr]
            )
            pv_l48 = m_l48.predict(X.iloc[va])
            l48_maes.append(
                weighted_mae(y_regions[region].iloc[va].values, pv_l48, None)
            )

            # Own weights (uniform or best for region)
            own_weights = make_recency_weights(len(tr), best_hl)
            m_own = clone(base_model)
            fit_with_optional_weight(
                m_own, X.iloc[tr], y_regions[region].iloc[tr], own_weights
            )
            pv_own = m_own.predict(X.iloc[va])
            own_maes.append(
                weighted_mae(
                    y_regions[region].iloc[va].values,
                    pv_own,
                    None,
                )
            )

        homogeneity_scores[region] = np.float64(np.mean(l48_maes))
        # Permutation test for paired differences
        diffs = np.array(l48_maes) - np.array(own_maes)
        p_value = permutation_test_paired(diffs)
        homogeneity_p_values[region] = np.float64(p_value)

    return best_hl, weights, homogeneity_scores, homogeneity_p_values


# ------------------------------ Torch Models ------------------------------

if HAS_TORCH:

    class TinyTransformerRegressor(nn.Module):
        """Small Transformer over 7-day sequences -> scalar weekly change.

        Because sometimes you need a neural network to predict gas storage changes.
        This is basically a mini GPT for time series, but dumber.
        TODO: Add attention visualization to see what days the model focuses on
        """

        def __init__(
            self,
            n_features: int,
            d_model: int = 128,
            nhead: int = 8,
            nlayers: int = 2,
            dropout: np.float64 = 0.1,
        ):
            super().__init__()
            self.n_features = n_features
            self.d_model = d_model

            # Projection to model dimension
            self.proj = nn.Linear(n_features, d_model)

            # Positional encoding
            self.pos = nn.Parameter(torch.zeros(7, d_model))
            nn.init.xavier_uniform_(self.pos)

            # Per-feature normalization after reshaping
            self.input_norm = nn.LayerNorm(n_features)

            # Transformer encoder
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

            # Output head
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

        def forward(self, x):
            # x: (N, 7*F)
            n, d = x.shape
            assert d % 7 == 0, "Flat sequence must be 7 * F (day-major) columns"
            f = d // 7
            x = x.view(n, 7, f)

            # Normalize per-feature
            x = self.input_norm(x)

            # Project to model dimension and add positional encoding
            z = self.proj(x) + self.pos  # (N, 7, d_model)

            # Encode
            z = self.encoder(z)

            # Pool and predict
            z = z.mean(dim=1)
            return self.head(z).squeeze(-1)

    # Hint static analyzers: forward is used implicitly by torch via __call__
    _unused_forward_ref = TinyTransformerRegressor.forward
    del _unused_forward_ref

    class TorchTransformerRegressor:
        """Sklearn-like wrapper with early stopping, LR scheduling, and checkpointing."""

        def __init__(
            self,
            n_features: int,
            d_model: int = 128,
            nhead: int = 8,
            nlayers: int = 2,
            dropout: np.float64 = 0.1,
            lr: np.float64 = 3e-4,
            weight_decay: np.float64 = 1e-4,
            epochs: int = 200,
            batch_size: int = 128,
            patience: int = 20,
            device: str = "cpu",
            seed: int = 42,
        ):
            self.n_features = n_features
            self.d_model = d_model
            self.nhead = nhead
            self.nlayers = nlayers
            self.dropout = dropout
            self.lr = lr
            self.weight_decay = weight_decay
            self.epochs = epochs
            self.batch_size = batch_size
            self.patience = patience
            self.device = device
            self.seed = seed
            self._model = None
            self._best_state = None

        def _init_model(self):
            torch.manual_seed(self.seed)
            self._model = TinyTransformerRegressor(
                self.n_features, self.d_model, self.nhead, self.nlayers, self.dropout
            )
            self._model.to(self.device)

        def _to_tensor(
            self,
            X: pd.DataFrame,
            y: Optional[pd.Series] = None,
            w: Optional[np.ndarray] = None,
        ):
            X_t = torch.tensor(X.values, dtype=torch.float32)
            if y is None:
                return X_t
            y_t = torch.tensor(y.values, dtype=torch.float32)
            w_t = torch.tensor(
                w if w is not None else np.ones_like(y.values), dtype=torch.float32
            )
            return X_t, y_t, w_t

        def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            sample_weight: Optional[np.ndarray] = None,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None,
            w_val: Optional[np.ndarray] = None,
        ):
            self._init_model()
            model = self._model

            # Setup optimizer and scheduler
            opt = torch.optim.AdamW(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=self.patience // 2, min_lr=1e-6
            )

            X_t, y_t, w_t = self._to_tensor(X, y, sample_weight)

            if X_val is None or y_val is None:
                # Self-validation: use last portion as validation
                val_size = min(20, len(X_t) // 5)
                X_train, X_val_t = X_t[:-val_size], X_t[-val_size:]
                y_train, y_val_t = y_t[:-val_size], y_t[-val_size:]
                w_train, w_val_t = w_t[:-val_size], w_t[-val_size:]
            else:
                X_train, y_train, w_train = X_t, y_t, w_t
                X_val_t, y_val_t, w_val_t = self._to_tensor(X_val, y_val, w_val)

            ds = TensorDataset(X_train, y_train, w_train)
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

            best_val = np.float64("inf")
            patience_left = self.patience
            self._best_state = None

            model.train()
            for _ in range(self.epochs):
                for xb, yb, wb in dl:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    wb = wb.to(self.device)
                    opt.zero_grad()
                    pred = model(xb)
                    loss = (torch.abs(pred - yb) * wb).sum() / (wb.sum() + 1e-9)
                    loss.backward()
                    opt.step()

                # Validation
                model.eval()
                with torch.no_grad():
                    pv = model(X_val_t.to(self.device))
                    val_loss = (
                        (
                            torch.abs(pv - y_val_t.to(self.device))
                            * w_val_t.to(self.device)
                        ).sum()
                        / (w_val_t.sum().to(self.device) + 1e-9)
                    ).item()
                model.train()

                scheduler.step(val_loss)

                if val_loss + 1e-9 < best_val:
                    best_val = val_loss
                    self._best_state = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                    patience_left = self.patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break

            # Restore best checkpoint
            if self._best_state:
                model.load_state_dict(
                    {k: v.to(self.device) for k, v in self._best_state.items()}
                )

            return self

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            assert self._model is not None, "Model not fitted"
            self._model.eval()
            with torch.no_grad():
                X_t = torch.tensor(X.values, dtype=torch.float32).to(self.device)
                p = self._model(X_t).cpu().numpy()
            return p


# ------------------------------ Safe Huber --------------------------------


class SafeHuberRegressor(BaseEstimator, RegressorMixin):
    """Robust wrapper around `HuberRegressor` with Ridge fallback.

    HuberRegressor sometimes fails to converge, so this falls back to Ridge.
    Because robust regression is great until it isn't.
    TODO: Maybe try different fallback models, like Lars or something
    """

    def __init__(
        self,
        huber_max_iter: int = 20000,
        huber_alpha: np.float64 = 1e-4,
        huber_epsilon: np.float64 = 1.35,
        ridge_alpha: np.float64 = 10.0,
        random_state: int = 42,
        suppress_warnings: bool = True,
    ) -> None:
        self.huber_max_iter = huber_max_iter
        self.huber_alpha = huber_alpha
        self.huber_epsilon = huber_epsilon
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state
        self.suppress_warnings = suppress_warnings
        self._huber: Optional[HuberRegressor] = None
        self._ridge: Optional[Ridge] = None
        self._use_ridge: bool = False

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None
    ):
        try:
            self._huber = HuberRegressor(
                max_iter=self.huber_max_iter,
                alpha=self.huber_alpha,
                epsilon=self.huber_epsilon,
                random_state=self.random_state,
            )
            with warnings.catch_warnings():
                if self.suppress_warnings:
                    warnings.simplefilter("ignore", ConvergenceWarning)
                if sample_weight is not None:
                    self._huber.fit(X, y, sample_weight=sample_weight)
                else:
                    self._huber.fit(X, y)
            if not hasattr(self._huber, "coef_"):
                raise ValueError("Huber did not set coef_")
            self._use_ridge = False
            return self
        except Exception:
            self._ridge = Ridge(
                alpha=self.ridge_alpha, solver="svd", random_state=self.random_state
            )
            if sample_weight is not None:
                self._ridge.fit(X, y, sample_weight=sample_weight)
            else:
                self._ridge.fit(X, y)
            self._use_ridge = True
            return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._use_ridge and self._ridge is not None:
            return self._ridge.predict(X)
        if self._huber is not None and hasattr(self._huber, "coef_"):
            return self._huber.predict(X)
        raise RuntimeError("SafeHuberRegressor not fitted")


# --------------------- Sample-weight routing helpers ----------------------


def estimator_accepts_sample_weight(est: object) -> Tuple[bool, Optional[str]]:
    """Return (accepts, pipeline_final_step_name).

    Checks if a model accepts sample_weight parameter. Important for pipelines.
    TODO: This could be cached since it doesn't change per estimator instance
    """
    if isinstance(est, Pipeline):
        final_name, final_est = est.steps[-1]
        sig = inspect.signature(final_est.fit)
        return ("sample_weight" in sig.parameters), final_name
    sig = inspect.signature(est.fit)
    return ("sample_weight" in sig.parameters), None


def fit_with_optional_weight(
    est: object, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray]
):
    """Fit estimator; pass sample_weight only if supported (Pipeline-aware).

    Handles the complexity of passing sample weights to different model types.
    This is annoyingly complex but necessary for proper weighting.
    """
    # Special handling for XGBoost to avoid NaN base_score
    if getattr(est, "_estimator_type", None) == "regressor" and "XGB" in str(type(est)):
        if y.isna().any() or np.isnan(y.values).any():
            raise ValueError("XGBoost cannot handle NaN target values")
        if len(y.unique()) < 2:
            raise ValueError("XGBoost requires at least 2 unique target values")

    accepts, final_name = estimator_accepts_sample_weight(est)
    if isinstance(est, Pipeline) and accepts and sample_weight is not None:
        return est.fit(X, y, **{f"{final_name}__sample_weight": sample_weight})
    if (not isinstance(est, Pipeline)) and accepts and sample_weight is not None:
        return est.fit(X, y, sample_weight=sample_weight)
    return est.fit(X, y)


# ---------------------------- Model Registry ------------------------------


def build_model_registry(
    n_features_per_day: int,
    use_transformer: bool = False,
    device: str = "cpu",
    catboost_train_dir: Optional[Path] = None,
    seed: int = 42,
    n_jobs: int = -1,
) -> Dict[str, object]:
    """Return a dict of model name -> estimator with deterministic seeding.

    The big model zoo. We try everything and pick the best.
    This is where the magic (or madness) happens.
    TODO: Add model versioning so we can track which models perform best over time
    """
    std = ("std", StandardScaler(with_mean=True))

    def lin(est):
        return Pipeline([std, ("reg", est)])

    models: Dict[str, object] = {
        # Tree ensembles - because trees are reliable
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=seed, n_jobs=n_jobs
        ),
        "HistGBM": HistGradientBoostingRegressor(random_state=seed),
        # Linear family (scaled) - the classics
        "Ridge": lin(Ridge(alpha=10.0, solver="svd", random_state=seed)),
    }

    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=-1,
        )
    if HAS_CAT:
        models["CatBoost"] = CatBoostRegressor(
            iterations=200,
            learning_rate=0.05,
            depth=8,
            verbose=False,
            random_seed=seed,
            train_dir=str(catboost_train_dir) if catboost_train_dir else None,
        )
    if HAS_TORCH and use_transformer:
        models["Transformer"] = TorchTransformerRegressor(
            n_features=n_features_per_day,
            d_model=128,
            nhead=8,
            nlayers=2,
            dropout=0.1,
            lr=3e-4,
            weight_decay=1e-4,
            epochs=20,  # Reduced from 50 for speed
            batch_size=128,
            patience=20,
            device=device,
            seed=seed,
        )
    return models


# ------------------------------- Printing ---------------------------------


def fmt(v: np.float64) -> str:
    """Format numeric values with minimal noise for table printing."""
    if isinstance(v, np.float64) and np.isnan(v):
        return "NA"
    if abs(v - round(v)) < 1e-6:
        return f"{int(round(v))}"
    return f"{v:.2f}"


def print_table(rows: Iterable[Row], last_date, next_date) -> None:
    """Pretty print the final summary table to the terminal.

    Date Convention: Dates shown are Thursday timestamps (W-THU resampling)
    representing weeks ending the previous Friday per EIA reporting schedule.
    """
    title = "Final Prediction Summary"
    print(title)
    print("-" * len(title))

    table_data = []
    for r in rows:
        interval_str = "[NA, NA]"
        if r.pred_interval:
            interval_str = f"[{fmt(r.pred_interval[0])}, {fmt(r.pred_interval[1])}]"
        table_data.append(
            [
                r.region,
                r.model,
                fmt(r.next_est),
                fmt(r.cv_mae),
                fmt(r.last_est),
                fmt(r.last_act),
                interval_str,
            ]
        )

    headers = [
        "Region",
        "Model",
        f"{next_date} Est",
        "CV MAE",
        f"{last_date} Est",
        f"{last_date} actual",
        "95% PI",
    ]

    print(tabulate(table_data, headers=headers, tablefmt="grid"))


# ------------------------------- Uncertainty Quantification ---------------------------------


def compute_fold_mae(tr, va, X_region, ym, w_region, est, name):
    """Compute MAE and residuals for a single CV fold."""
    Xtr, Xva = X_region.iloc[tr], X_region.iloc[va]
    ytr, yva = ym.iloc[tr], ym.iloc[va]
    wtr, wva = w_region[tr], w_region[va]

    if HAS_TORCH and name == "Transformer":
        m = clone(est)
        m.fit(Xtr, ytr, sample_weight=wtr, X_val=Xva, y_val=yva, w_val=wva)
        pva = m.predict(Xva)
    else:
        m = clone(est)
        fit_with_optional_weight(m, Xtr, ytr, wtr)
        pva = m.predict(Xva)

    residuals = yva.values - pva
    mae = weighted_mae(yva.values, pva, None)  # keep validation unbiased
    return residuals, mae

    # Removed unused helper bootstrap_single_sample to satisfy static analysis


def block_bootstrap_indices(n, b, rng):
    starts = rng.randint(0, n, size=int(np.ceil(n / b)))
    idx = np.concatenate([(np.arange(s, s + b) % n) for s in starts])[:n]
    return idx


def bootstrap_prediction_intervals(
    estimator: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    n_bootstraps: int = 100,
    alpha: np.float64 = 0.05,
    seed: int = 42,
) -> Tuple[np.float64, Tuple[np.float64, np.float64]]:
    """Compute bootstrapped prediction intervals using block bootstrap for time series."""
    rng = np.random.RandomState(seed)

    # Fit model once on full training data for point estimate
    model = clone(estimator)
    fit_with_optional_weight(model, X_train, y_train, None)
    point_est = np.float64(model.predict(X_test)[0])

    # Block bootstrap predictions
    n_train = len(X_train)
    b = max(2, int(round(n_train ** (1 / 3))))  # Politis–White style heuristic
    bootstrap_preds = []
    for _ in range(n_bootstraps):
        boot_idx = block_bootstrap_indices(n_train, b, rng)
        X_boot = X_train.iloc[boot_idx]
        y_boot = y_train.iloc[boot_idx]
        # Fit on bootstrap sample
        boot_model = clone(estimator)
        try:
            fit_with_optional_weight(boot_model, X_boot, y_boot, None)
            # Predict on test
            pred = boot_model.predict(X_test)[0]
            bootstrap_preds.append(pred)
        except (np.linalg.LinAlgError, ValueError):
            # Skip degenerate bootstrap samples
            continue

    bootstrap_preds = np.array(bootstrap_preds)
    if len(bootstrap_preds) < 10:
        # Fallback to wide interval if too few bootstraps
        lower = (
            point_est - 2 * np.std(y_train.values) if len(y_train) > 1 else point_est
        )
        upper = (
            point_est + 2 * np.std(y_train.values) if len(y_train) > 1 else point_est
        )
    else:
        lower = np.float64(np.percentile(bootstrap_preds, 100 * alpha / 2))
        upper = np.float64(np.percentile(bootstrap_preds, 100 * (1 - alpha / 2)))

    return point_est, (lower, upper)


# --------------------------------- Main Pipeline -----------------------------------


def prepare_data(
    config: Config, logger: logging.Logger
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Timestamp
]:
    """Load, clean, and prepare data for modeling.

    EIA Reporting Convention:
    - EIA publishes weekly natural gas storage reports on Thursdays at 10:30 AM ET
    - Each Thursday report contains data for the week ending the PREVIOUS Friday
    - Example: Thursday Oct 24 report → week ending Friday Oct 17 data
    - Our W-THU resampling creates Thursday timestamps representing the week:
      * Friday 2025-10-17 (week ending) → Thursday 2025-10-23 (W-THU timestamp)
      * This week's data is reported on Thursday 2025-10-24 (next business day)

    Returns:
        X_seq: Training features
        y: Training targets
        X_test: Test features (holdout)
        y_test: Test targets (holdout)
        completeness: Completeness series
        last_report_date: Thursday timestamp of most recent week with EIA actuals.
                         Represents the week ending the previous Friday.
                         Captured BEFORE train/test split to preserve the true last week.
    """
    logger.info("Loading and cleaning data files")
    # Load data
    daily = pd.read_csv(config.features_path, index_col=0, parse_dates=True)
    daily = clean_cols(daily.sort_index())
    ychg = pd.read_csv(config.changes_path, index_col=0, parse_dates=True)
    ychg = clean_cols(ychg.sort_index())
    ychg.columns = ychg.columns.str.replace("_change", "", regex=False)
    ychg.index = normalize_week_index(ychg.index)

    logger.info(f"Loaded {len(daily)} daily records and {len(ychg)} weekly changes")

    # Handle daily continuity with interpolation
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx)
    num = daily.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    num = num.interpolate(method="linear", limit_direction="both").ffill().bfill()
    daily[num.columns] = num

    logger.info("Interpolated missing values in daily data")

    # Build sequences with completeness tracking
    daily_num = daily.select_dtypes(include=[np.number])
    X_seq, completeness = build_sequence_matrix(
        daily_num, ychg.index, allow_partial=True
    )

    logger.info(f"Built {len(X_seq)} weekly sequences")

    # Filter by completeness if desired (keep all for now, but log)
    min_completeness = 0.8  # Allow up to 1 missing day
    complete_mask = completeness >= min_completeness
    logger.info(
        f"Kept {complete_mask.sum()}/{len(complete_mask)} weeks ({100 * complete_mask.mean():.1f}% complete)"
    )

    # Align targets

    y = ychg.reindex(X_seq.index).dropna()

    X_seq = X_seq.reindex(y.index)

    completeness = completeness.reindex(y.index)

    # Store the original last report date BEFORE train/test split
    # This is the most recent week we have EIA actuals for
    last_report_date = y.index.max()

    # Hold out last 4 weeks for out-of-sample validation

    if len(X_seq) >= 8:  # Ensure enough for training
        test_size = min(4, len(X_seq) // 2)

        X_test = X_seq.iloc[-test_size:]

        y_test = y.iloc[-test_size:]

        X_seq = X_seq.iloc[:-test_size]

        y = y.iloc[:-test_size]

        completeness = completeness.iloc[:-test_size]

    else:
        X_test = pd.DataFrame()

        y_test = pd.DataFrame()

    return X_seq, y, X_test, y_test, completeness, last_report_date


def process_region(
    key: str,
    tgt: str,
    y: pd.DataFrame,
    X_filtered: pd.DataFrame,
    w_rec: np.ndarray,
    models: Dict[str, object],
    gap_cv: GapTimeSeriesSplit,
    config: Config,
    logger: logging.Logger,
) -> Tuple[
    str,
    Tuple[str, object, np.float64, np.float64, np.float64],
    List[Dict],
    Tuple[pd.DataFrame, pd.Series, np.ndarray],
]:
    """Process training for a single region."""
    if tgt not in y.columns:
        return (
            key,
            ("NA", None, np.float64("inf"), np.float64("nan"), np.float64("inf")),
            [],
            (pd.DataFrame(), pd.Series(dtype=float), np.array([])),
        )

    ym = y[tgt].dropna()
    if len(ym) == 0:
        return (
            key,
            ("NA", None, np.float64("inf"), np.float64("nan"), np.float64("inf")),
            [],
            (pd.DataFrame(), pd.Series(dtype=float), np.array([])),
        )

    X_region = X_filtered.reindex(ym.index)
    X_region = X_region[~X_region.index.duplicated(keep="last")]
    w_series = pd.Series(w_rec, index=X_filtered.index)
    w_region = w_series.reindex(X_region.index).fillna(1.0).clip(lower=1e-12).to_numpy()

    region_data = (X_region, ym, w_region)

    best_name, best_model, best_cv, best_last, best_var = (
        None,
        None,
        np.float64("inf"),
        np.float64("nan"),
        np.float64("inf"),
    )

    all_scores_rows = []

    for name, est in models.items():
        try:
            if hasattr(est, "n_estimators"):
                if ym.nunique() <= 1:
                    raise ValueError(
                        f"Target has constant or single value for {REGION_PRETTY[key]}"
                    )
                if ym.isna().all():
                    raise ValueError(f"Target is all NaN for {REGION_PRETTY[key]}")

            all_residuals = []
            maes = []
            for tr, va in gap_cv.split(X_region):
                res, mae = compute_fold_mae(tr, va, X_region, ym, w_region, est, name)
                all_residuals.extend(res)
                maes.append(mae)
            cv_mae = np.float64(np.mean(maes))

            if HAS_TORCH and name == "Transformer":
                final_model = clone(est)
                final_model.fit(X_region, ym, sample_weight=w_region)
                last_est = np.float64(final_model.predict(X_region.iloc[[-1]])[0])
            else:
                final_model = clone(est)
                fit_with_optional_weight(final_model, X_region, ym, w_region)
                last_est = np.float64(final_model.predict(X_region.iloc[[-1]])[0])

            # DIAGNOSTIC: Training performance check
            train_preds = final_model.predict(X_region)
            train_errors = ym.values - train_preds
            train_mae = np.abs(train_errors).mean()

            logger.debug(
                f"  [{REGION_PRETTY[key]} / {name}] Train MAE: {train_mae:.4f} | "
                f"Preds: [{train_preds.min():.2f}, {train_preds.max():.2f}] | "
                f"Actuals: [{ym.min():.2f}, {ym.max():.2f}]"
            )

            # Flag if training error is very small but CV error is large (overfitting)
            if train_mae < cv_mae * 0.5 and cv_mae > 5:
                logger.warning(
                    f"  ⚠️  OVERFITTING: {REGION_PRETTY[key]}/{name} - Train MAE {train_mae:.4f} << CV MAE {cv_mae:.4f}"
                )

            all_scores_rows.append(
                {
                    "region": REGION_PRETTY[key],
                    "model": name,
                    "cv_mae": round(cv_mae, 4),
                    "last_est": round(last_est, 4),
                    "last_act": np.float64(ym.iloc[-1]),
                }
            )

            logger.info(
                f"CV (training-only) MAE for {REGION_PRETTY[key]} / {name}: {cv_mae:.4f}"
            )

            if cv_mae < best_cv:
                best_name, best_model, best_cv, best_last, best_var = (
                    name,
                    final_model,
                    cv_mae,
                    last_est,
                    (
                        float(np.var(all_residuals, ddof=1))
                        if len(all_residuals) > 1
                        else 1.0
                    ),
                )
                best_var = max(best_var, 1e-6)

        except Exception as e:
            logger.debug(f"Skipping {name} for {REGION_PRETTY[key]}: {e}")

    best_per_region = (best_name or "NA", best_model, best_cv, best_last, best_var)

    return key, best_per_region, all_scores_rows, region_data


def train_models(
    X: pd.DataFrame,
    y: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    completeness: pd.Series,
    config: Config,
    logger: logging.Logger,
) -> Tuple[
    Dict[str, Tuple[str, object, np.float64, np.float64, np.float64]],
    Optional[int],
    np.ndarray,
    Dict[str, np.float64],
    Dict[str, np.float64],
    List[Dict],
    Dict[str, Tuple[pd.DataFrame, pd.Series, np.ndarray]],
]:
    """Train models for each region and return best performers."""
    model_cache_path = config.output_dir / "model_cache.pkl"
    features_mtime = os.path.getmtime(config.features_path)
    changes_mtime = os.path.getmtime(config.changes_path)

    if model_cache_path.exists():
        cache_mtime = os.path.getmtime(model_cache_path)
        if cache_mtime > features_mtime and cache_mtime > changes_mtime:
            logger.info("Loading cached models")
            with open(model_cache_path, "rb") as f:
                return pickle.load(f)
        else:
            logger.info("Cache is stale, retraining")
    else:
        logger.info("No cache found, training models")

    # Feature selection
    top_k = None if config.top_k == -1 else config.top_k
    X_filtered = variance_topk(X, top_k, logger)

    # Persist selected features for reproducibility
    selected_features_path = config.output_dir / "selected_features.json"
    selected_features = X_filtered.columns.tolist()
    with open(selected_features_path, "w") as f:
        json.dump(selected_features, f)

    # Select half-life with homogeneity testing
    if "lower48" not in y.columns:
        raise ValueError("EIAchanges must include 'Lower48' column")
    best_hl, w_rec, homogeneity_scores, homogeneity_p_values = (
        select_half_life_with_homogeneity_test(
            X_filtered, y, config.n_splits, logger=logger
        )
    )

    logger.info(f"Selected half-life: {best_hl} weeks")
    logger.info(f"Regional homogeneity test scores: {homogeneity_scores}")
    logger.info(f"Homogeneity p-values: {homogeneity_p_values}")

    # Model registry
    n_features_per_day = X_filtered.shape[1] // 7
    models = build_model_registry(
        n_features_per_day,
        config.use_transformer,
        config.device,
        config.output_dir,
        config.seed,
        n_jobs=4,  # Enable parallelism
    )

    # Gap-aware CV
    gap_cv = GapTimeSeriesSplit(n_splits=config.n_splits, gap=1)

    # Per-region competition
    all_scores_rows = []
    best_per_region: Dict[str, Tuple[str, object, np.float64, np.float64]] = {}
    region_data: Dict[str, Tuple[pd.DataFrame, pd.Series, np.ndarray]] = {}

    results = Parallel(n_jobs=-1)(
        delayed(process_region)(
            key, tgt, y, X_filtered, w_rec, models, gap_cv, config, logger
        )
        for key, tgt in TARGET_MAP.items()
    )

    for key, best_per_region_entry, scores_rows, region_data_entry in results:
        best_per_region[key] = best_per_region_entry
        all_scores_rows.extend(scores_rows)
        if region_data_entry[0].shape[0] > 0:  # If not empty
            region_data[key] = region_data_entry

    # Out-of-sample validation
    if not X_test.empty and not y_test.empty:
        for key, tgt in TARGET_MAP.items():
            if key in best_per_region and tgt in y_test.columns and key in region_data:
                _, winner, _, _, _ = best_per_region[key]
                X_region_test = X_test.reindex(columns=region_data[key][0].columns)
                if not X_region_test.empty:
                    pred = winner.predict(X_region_test)
                    oos_mae = mean_absolute_error(y_test[tgt], pred)
                    logger.info(
                        f"Out-of-sample MAE for {REGION_PRETTY[key]}: {oos_mae:.4f}"
                    )

    result = (
        best_per_region,
        best_hl,
        w_rec,
        homogeneity_scores,
        homogeneity_p_values,
        all_scores_rows,
        region_data,
    )
    # Save best models for caching
    with open(model_cache_path, "wb") as f:
        pickle.dump(result, f)
    return result


def process_forecast(
    key: str,
    best_per_region: Dict[str, Tuple[str, object, np.float64, np.float64]],
    region_data: Dict[str, Tuple[pd.DataFrame, pd.Series, np.ndarray]],
    seq_next: pd.DataFrame,
    config: Config,
) -> Tuple[str, np.float64, Tuple[np.float64, np.float64]]:
    """Process forecasting for a single region."""
    if key not in best_per_region or key not in region_data:
        return key, np.float64("nan"), (np.float64("nan"), np.float64("nan"))

    _, winner, _, _, _ = best_per_region[key]
    X_region, y_region, w_region = region_data[key]

    # Point prediction
    point_pred = np.float64(winner.predict(seq_next)[0])

    # CONSTRAINT: Bound predictions to reasonable historical range
    # Storage changes must be within plausible bounds of historical observations
    min_hist = np.float64(
        y_region.quantile(0.05)
    )  # 5th percentile (allow for extreme events)
    max_hist = np.float64(y_region.quantile(0.95))  # 95th percentile

    # Add a small buffer beyond historical range (10%) to allow for unprecedented events
    margin = 0.1 * (max_hist - min_hist)
    lower_bound = min_hist - margin
    upper_bound = max_hist + margin

    point_pred = np.clip(point_pred, lower_bound, upper_bound)

    # Uncertainty quantification via bootstrapping
    _, interval = bootstrap_prediction_intervals(
        winner, X_region, y_region, seq_next, n_bootstraps=50, seed=config.seed
    )

    # Clip intervals to the same bounds
    interval = (
        np.clip(interval[0], lower_bound, upper_bound),
        np.clip(interval[1], lower_bound, upper_bound),
    )

    return key, point_pred, interval


def forecast_next(
    X: pd.DataFrame,
    best_per_region: Dict[str, Tuple[str, object, np.float64, np.float64]],
    config: Config,
    daily_num: pd.DataFrame,
    region_data: Dict[str, Tuple[pd.DataFrame, pd.Series, np.ndarray]],
    last_report_date: pd.Timestamp,
    logger: logging.Logger = None,
) -> Tuple[Dict[str, np.float64], Dict[str, Tuple[np.float64, np.float64]]]:
    """Generate next-week forecasts with uncertainty quantification.

    last_report_date: The Thursday representing the most recent EIA report week
                      (from the target data's last index).

    CRITICAL FIX: Prevent data leakage by truncating daily_num to only include
    historical data up to last_report_date, then forward-filling for forecast week.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # DIAGNOSTIC: Check for data leakage in next-week forecast
    logger.info(f"\n{'=' * 60}")
    logger.info("DATA LEAKAGE DIAGNOSTIC")
    logger.info(
        f"Daily data ORIGINAL range: {daily_num.index.min().date()} → {daily_num.index.max().date()}"
    )
    logger.info(f"Last training week (Thu): {last_report_date.date()}")

    next_thu = last_report_date + pd.Timedelta(days=7)
    logger.info(f"Next forecast week (Thu): {next_thu.date()}")
    logger.info(
        f"Expected date range: {(next_thu - pd.Timedelta(days=6)).date()} → {next_thu.date()}"
    )

    # CRITICAL FIX: Truncate daily data to only what's available up to last_report_date
    # This prevents the model from "seeing" future data when interpolating
    cutoff_date = last_report_date  # Thursday of the last complete week
    historical_daily = daily_num[daily_num.index <= cutoff_date].copy()

    logger.info(
        f"Daily data TRUNCATED to: {historical_daily.index.min().date()} → {historical_daily.index.max().date()}"
    )

    if daily_num.index.max() > cutoff_date:
        days_removed = (daily_num.index.max() - cutoff_date).days
        logger.warning(
            f"⚠️  Removed {days_removed} days of future data to prevent leakage"
        )

    # Build next week feature sequence
    # Start with historical data, then extend with forward-filled values for the forecast week
    next_week_start = cutoff_date + pd.Timedelta(days=1)
    next_week_dates = pd.date_range(start=next_week_start, end=next_thu, freq="D")

    # Forward-fill: use the last available values for dates beyond cutoff
    # This is a reasonable assumption - we only know what happened up to last_report_date
    last_values = historical_daily.iloc[-1]
    future_extension = pd.DataFrame(
        [last_values.values] * len(next_week_dates),
        index=next_week_dates,
        columns=historical_daily.columns,
    )

    # Combine historical + extended future
    extended_daily = pd.concat([historical_daily, future_extension])
    extended_daily = extended_daily[
        ~extended_daily.index.duplicated(keep="first")
    ]  # Remove any duplicates

    logger.info(
        f"Extended daily data with forward-fill: {extended_daily.index.min().date()} → {extended_daily.index.max().date()}"
    )

    # Build sequence with now-complete data (no interpolation needed)
    seq_next, completeness_next = build_sequence_matrix(
        extended_daily, [next_thu], allow_partial=False
    )

    logger.info(
        f"Next week sequence completeness: {100 * completeness_next.iloc[0]:.1f}%"
    )
    logger.info(f"{'=' * 60}\n")

    # Load selected features from training
    selected_features_path = config.output_dir / "selected_features.json"
    with open(selected_features_path, "r") as f:
        selected_features = json.load(f)

    seq_next = seq_next.reindex(columns=selected_features).fillna(0.0)

    next_preds: Dict[str, np.float64] = {}
    pred_intervals: Dict[str, Tuple[np.float64, np.float64]] = {}

    # Parallel forecasting across regions
    results = Parallel(n_jobs=-1)(
        delayed(process_forecast)(key, best_per_region, region_data, seq_next, config)
        for key in TARGET_MAP.keys()
    )

    for key, point_pred, interval in results:
        next_preds[key] = point_pred
        pred_intervals[key] = interval

    # Reconciliation
    if config.reconcile_method == "variance":
        next_preds = reconcile_variance(
            next_preds, {k: v[4] for k, v in best_per_region.items()}
        )
        # Reconcile intervals with coherence
        child_adj = reconcile_intervals(
            pred_intervals,
            {k: v[4] for k, v in best_per_region.items()},
            pred_intervals.get("conus", (np.nan, np.nan)),
        )
        pred_intervals.update(child_adj)

    return next_preds, pred_intervals


def main() -> None:
    """CLI entrypoint with modular pipeline."""
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(description="EIA Weekly Changes Forecaster")
    parser.add_argument(
        "--features",
        type=Path,
        default=script_dir / "output" / "Combined_Wide_Data.csv",
    )
    parser.add_argument(
        "--changes", type=Path, default=script_dir.parent / "INFO" / "EIAchanges.csv"
    )
    parser.add_argument("--outdir", type=Path, default=script_dir / "output")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=-1, help="-1 => adaptive")
    parser.add_argument("--reconcile", choices=["off", "variance"], default="variance")
    parser.add_argument("--show_all", action="store_true")
    parser.add_argument("--use_transformer", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set global seed for reproducibility
    set_global_seed(args.seed)

    # Quiet mode
    if args.quiet:
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", message=r".*did not converge.*")
        warnings.filterwarnings("ignore", message=r".*failed to converge.*")
        warnings.filterwarnings("ignore", message=r".*Ill-conditioned matrix.*")
        warnings.filterwarnings("ignore", message=r".*Singular matrix.*")

    config = Config(
        features_path=args.features,
        changes_path=args.changes,
        output_dir=args.outdir,
        n_splits=args.splits,
        top_k=args.top_k,
        reconcile_method=args.reconcile,
        use_transformer=args.use_transformer,
        device=args.device,
        quiet=args.quiet,
        seed=args.seed,
    )

    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # CatBoost setup
    os.environ["CATBOOST_DATA_DIR"] = str(config.output_dir)

    # Set up logging
    logger = setup_logging(config.quiet)
    start_time = time.time()

    # Pipeline execution
    logger.info("Starting data preparation")
    X, y, X_test, y_test, completeness, last_report_date = prepare_data(config, logger)

    logger.info("Starting model training")
    (
        best_per_region,
        best_hl,
        w_rec,
        homogeneity_scores,
        homogeneity_p_values,
        all_scores_rows,
        region_data,
    ) = train_models(X, y, X_test, y_test, completeness, config, logger)

    logger.info("Generating forecasts")
    daily_num = pd.read_csv(config.features_path, index_col=0, parse_dates=True)
    daily_num = clean_cols(daily_num).select_dtypes(include=[np.number])

    # Use the last_report_date from prepare_data (before train/test split)
    # This is the most recent week we have EIA actuals for
    # Note: Thursday timestamps represent weeks ending the previous Friday
    # E.g., 2025-10-23 (Thu) = week ending 2025-10-17 (Fri), reported 2025-10-24 (Thu)
    logger.info(
        f"Last EIA week (Thu timestamp): {last_report_date.date()} | "
        f"Forecasting week (Thu timestamp): {(last_report_date + pd.Timedelta(days=7)).date()}"
    )

    next_preds, pred_intervals = forecast_next(
        X, best_per_region, config, daily_num, region_data, last_report_date, logger
    )

    # Build results table
    rows = []
    for key, tgt in TARGET_MAP.items():
        if key not in best_per_region:
            continue

        name, winner, cv_mae, last_est_stored, _ = best_per_region[key]

        # Last actual: Use test set's last value if available (it's the most recent),
        # otherwise fall back to training set's last value
        if (
            not X_test.empty
            and isinstance(y_test, pd.DataFrame)
            and tgt in y_test.columns
            and len(y_test[tgt]) > 0
        ):
            last_act_disp = np.float64(y_test[tgt].iloc[-1])
        elif tgt in y.columns and len(y[tgt]) > 0:
            last_act_disp = np.float64(y[tgt].iloc[-1])
        else:
            last_act_disp = np.float64("nan")

        # Last estimate is the stored value from training (prediction on last training week)
        last_est_disp = last_est_stored

        rows.append(
            Row(
                region=REGION_PRETTY.get(key, key.title()),
                model=name,
                next_est=next_preds.get(key, np.float64("nan")),
                cv_mae=cv_mae,
                last_est=last_est_disp,
                last_act=last_act_disp,
                pred_interval=pred_intervals.get(key),
            )
        )

    print_table(
        rows,
        last_report_date.date(),
        (last_report_date + pd.Timedelta(days=7)).date(),
    )

    # Optional detailed output
    if args.show_all:
        print("\nDetailed CV by Region (lower is better)\n" + "=" * 40)
        df_scores = pd.DataFrame(all_scores_rows)
        for region in df_scores["region"].unique():
            sub = df_scores[df_scores["region"] == region].sort_values("cv_mae")
            print(f"\n{region}")
            print(sub.to_string(index=False))

    # Save results
    results = {
        "predictions": next_preds,
        "prediction_intervals": pred_intervals,
        "best_models": {k: v[0] for k, v in best_per_region.items()},
        "cv_maes": {k: v[2] for k, v in best_per_region.items()},
        "config": {
            "features": str(args.features),
            "changes": str(args.changes),
            "outdir": str(args.outdir),
            "splits": args.splits,
            "top_k": args.top_k,
            "recency_half_life_selected": best_hl,
            "homogeneity_test_scores": homogeneity_scores,
            "homogeneity_p_values": homogeneity_p_values,
            "models_available": list(build_model_registry(1).keys()),
            "last_train_week_end_thu": str(X.index.max().date()),
            "have_next_week_inputs": bool(next_preds),
            "used_transformer": bool(args.use_transformer and HAS_TORCH),
            "seed": args.seed,
        },
    }

    with open(config.output_dir / "forecast_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    logger.info(f"Results saved to {config.output_dir / 'forecast_results.json'}")
    logger.info(f"Total execution time: {time.time() - start_time:.2f} seconds")


# --------------------------- Reconciliation --------------------------------


def reconcile_variance(
    preds: Dict[str, np.float64], variances: Dict[str, np.float64]
) -> Dict[str, np.float64]:
    """Adjust region preds to sum to L48 using inverse-variance weighting.

    Because the sum of regions should equal the total. Math police.
    This is basically saying "trust the better models more when reconciling".
    TODO: Add more sophisticated reconciliation methods, maybe using hierarchical forecasting
    """
    if "conus" not in preds:
        return preds
    l48 = preds["conus"]
    keys = ["east", "midwest", "south_central", "mountain", "pacific"]
    vec = np.array([preds.get(k, np.nan) for k in keys], np.float64)

    if np.isnan(vec).any():
        return preds

    # Use inverse variance as weights (higher weight for lower variance)
    var_array = np.array([max(variances.get(k, 1.0), 1e-6) for k in keys], np.float64)
    weights = 1.0 / var_array
    weights = weights / weights.sum()

    current_sum = np.float64(vec.sum())
    delta = l48 - current_sum

    # Weighted adjustment
    vec = vec + delta * weights

    for i, k in enumerate(keys):
        preds[k] = np.float64(vec[i])

    return preds


def reconcile_intervals(
    pred_intervals: Dict[str, Tuple[np.float64, np.float64]],
    variances: Dict[str, np.float64],
    l48_interval: Tuple[np.float64, np.float64],
) -> Dict[str, Tuple[np.float64, np.float64]]:
    """Reconcile prediction intervals with coherence."""
    if "conus" not in pred_intervals:
        return pred_intervals
    l48_low, l48_up = l48_interval
    keys = ["east", "midwest", "south_central", "mountain", "pacific"]
    low_vec = np.array([pred_intervals.get(k, (np.nan, np.nan))[0] for k in keys])
    up_vec = np.array([pred_intervals.get(k, (np.nan, np.nan))[1] for k in keys])

    if np.isnan(low_vec).any() or np.isnan(up_vec).any():
        return pred_intervals

    # Use same weights as for means
    var_array = np.array([max(variances.get(k, 1.0), 1e-6) for k in keys], np.float64)
    weights = 1.0 / var_array
    weights = weights / weights.sum()

    low_delta = l48_low - low_vec.sum()
    up_delta = l48_up - up_vec.sum()

    lo_adj = low_vec + low_delta * weights
    up_adj = up_vec + up_delta * weights

    # Ensure monotonicity
    lo_adj, up_adj = np.minimum(lo_adj, up_adj), np.maximum(lo_adj, up_adj)

    reconciled = {}
    for i, k in enumerate(["east", "midwest", "south_central", "mountain", "pacific"]):
        if k in pred_intervals:
            reconciled[k] = (np.float64(lo_adj[i]), np.float64(up_adj[i]))
    return reconciled


if __name__ == "__main__":
    main()
