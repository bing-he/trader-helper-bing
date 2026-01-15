#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MTP3.py — EIA Weekly Changes Forecaster (Fri→Thu sequences)

Full model competition, robust CV, and clean terminal output.

Highlights
- Non-lossy Fri→Thu sequence builder (no weekly summing)
- Adaptive top-K feature filter by variance
- Lower‑48-driven **recency weighting** with CV-chosen half-life
- Big model zoo: RF/ExtraTrees/GBM/HistGBM, Ridge/Lasso/ElasticNet/SGD,
  **HuberSafe** (fallback to Ridge), SVR/KNN/MLP, + optional LGBM/XGB/CatBoost
  and optional **PyTorch Transformer** head
- **Safe sample_weight routing** for Pipelines
- **Quiet mode** to suppress ConvergenceWarnings (no wall-of-text)
- Terminal summary exactly like your screenshot: Region | Model | Next Est |
  CV MAE | Last Est | Last Actual

Run (from repo root):
    python -m EIAGuesser.MTP3 \\
        --reconcile variance --use_transformer --device cpu --show_all --quiet

If you don't have torch, omit --use_transformer. To keep warnings visible, drop
--quiet.
"""

from __future__ import annotations

# --- minimal import safety for direct execution (repo root is one level up from /EIAGuesser)
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import inspect
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from common.logs import get_file_logger
from common.pathing import ROOT
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, HuberRegressor, Lasso, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

logger = get_file_logger(Path(__file__).stem)

CONFIG = {
    "features": ROOT / "EIAGuesser" / "output" / "Combined_Wide_Data.csv",
    "changes": ROOT / "INFO" / "EIAchanges.csv",
    "outdir": ROOT / "EIAGuesser" / "output",
}
# Optional boosters
HAS_LGB = False
try:  # pragma: no cover
    import lightgbm as lgb  # type: ignore

    HAS_LGB = True
except Exception:  # pragma: no cover
    pass

HAS_XGB = False
try:  # pragma: no cover
    import xgboost as xgb  # type: ignore

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


def _resolve_path(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


# ------------------------------ Constants ---------------------------------
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


# ------------------------------ Data Types --------------------------------
@dataclass
class Row:
    """Row for the final terminal table."""

    region: str
    model: str
    next_est: float
    cv_mae: float
    last_est: float
    last_act: float


# ------------------------------ Utilities ---------------------------------


def clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with snake_case alphanumeric column names."""
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
    """Map any timestamp to the Thursday of its week (Thu=3)."""
    return ts - pd.Timedelta(days=((ts.weekday() - 3) % 7))


def normalize_week_index(idx: pd.Index) -> pd.Index:
    """Return a Thursday-indexed weekly index for given timestamps."""
    return pd.Index(
        [to_thursday_label(pd.Timestamp(d)) for d in idx], name="week_end_thu"
    )


def build_sequence_matrix(
    daily_num: pd.DataFrame, week_thus: Iterable[pd.Timestamp]
) -> pd.DataFrame:
    """Create flattened Fri..Thu windows as one row per week.

    Each row contains 7 consecutive days for every numeric feature, preserving
    intraweek shape.
    """
    cols = list(daily_num.columns)
    rows, rindex = [], []
    for thu in week_thus:
        dates = [thu + pd.Timedelta(days=o) for o in range(-6, 1)]
        if any(d not in daily_num.index for d in dates):
            continue
        seq = daily_num.loc[dates, cols].to_numpy().reshape(-1)
        rows.append(seq)
        rindex.append(thu)
    flat_cols = [f"{c}__{DAY_TAGS[d]}" for d in range(7) for c in cols]
    return pd.DataFrame(
        rows, index=pd.Index(rindex, name="week_end_thu"), columns=flat_cols
    )


def variance_topk(X_df: pd.DataFrame, top_k: Optional[int]) -> pd.DataFrame:
    """Drop zero‑variance cols; keep top‑K by variance (adaptive if None)."""
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vt = VarianceThreshold(1e-12)
    Xv = vt.fit_transform(X_df.values)
    cols = X_df.columns[vt.get_support()]
    Xv = pd.DataFrame(Xv, index=X_df.index, columns=cols)
    var = Xv.var().sort_values(ascending=False)
    if top_k is None:
        k = int(min(4000, max(800, 0.35 * len(var))))
    else:
        k = min(top_k, len(var))
    return Xv[var.index[:k]]


# ---- Recency weighting & selection ---------------------------------------


def make_recency_weights(n: int, half_life_weeks: Optional[int]) -> np.ndarray:
    """Exponential recency weights normalized to mean=1 for length n."""
    if n <= 1 or not half_life_weeks:
        return np.ones(n)
    tau = half_life_weeks / math.log(2)
    t = np.arange(n)
    w = np.exp((t - (n - 1)) / tau)
    return w / np.mean(w)


def weighted_mae(
    y_true: np.ndarray, y_pred: np.ndarray, w: Optional[np.ndarray]
) -> float:
    """Weighted MAE; falls back to unweighted if `w` is None."""
    if w is None:
        return float(mean_absolute_error(y_true, y_pred))
    w = np.clip(np.asarray(w, float), 1e-12, None)
    return float(np.sum(w * np.abs(y_true - y_pred)) / np.sum(w))


# ------------------------------ Torch Models ------------------------------

if HAS_TORCH:

    class TinyTransformerRegressor(nn.Module):  # type: ignore[misc]
        """Small Transformer over 7-day sequences -> scalar weekly change.

        Expects flattened (N, 7*F) and internally reshapes to (N,7,F).
        """

        def __init__(
            self,
            n_features: int,
            d_model: int = 128,
            nhead: int = 8,
            nlayers: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.n_features = n_features
            self.d_model = d_model
            self.proj = nn.Linear(n_features, d_model)
            enc = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc, num_layers=nlayers)
            self.pos = nn.Parameter(torch.zeros(7, d_model))
            nn.init.xavier_uniform_(self.pos)
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
            )

        def forward(self, x):  # x: (N, 7*F)
            n, d = x.shape
            f = max(1, d // 7)
            x = x.view(n, 7, f)
            z = self.proj(x) + self.pos  # (N,7,d_model)
            z = self.encoder(z)
            z = z.mean(dim=1)
            return self.head(z).squeeze(-1)

    # Hint static analyzers: forward is used implicitly by torch via __call__
    _unused_forward_ref = TinyTransformerRegressor.forward
    del _unused_forward_ref

    class TorchTransformerRegressor:
        """Sklearn-like wrapper with early stopping and recency weighting."""

        def __init__(
            self,
            n_features: int,
            d_model: int = 128,
            nhead: int = 8,
            nlayers: int = 2,
            dropout: float = 0.1,
            lr: float = 3e-4,
            weight_decay: float = 1e-4,
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

        def get_params(self, deep: bool = True):
            # mark parameter as used to satisfy static analyzers
            del deep
            return {
                "n_features": self.n_features,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "nlayers": self.nlayers,
                "dropout": self.dropout,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "patience": self.patience,
                "device": self.device,
                "seed": self.seed,
            }

        def set_params(self, **params):
            for k, v in params.items():
                setattr(self, k, v)
            return self

        def _init_model(self):
            torch.manual_seed(self.seed)
            self._model = TinyTransformerRegressor(
                self.n_features, self.d_model, self.nhead, self.nlayers, self.dropout
            )
            self._model.to(self.device)

        def _to_tensor(
            self,
            Xdf: pd.DataFrame,
            y: Optional[pd.Series] = None,
            w: Optional[np.ndarray] = None,
        ):
            X_t = torch.tensor(Xdf.values, dtype=torch.float32)
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
            opt = torch.optim.AdamW(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            if X_val is None or y_val is None:
                X_t, y_t, w_t = self._to_tensor(X, y, sample_weight)
                ds = TensorDataset(X_t, y_t, w_t)
                dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
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
                return self

            X_t, y_t, w_t = self._to_tensor(X, y, sample_weight)
            Xv_t, yv_t, wv_t = self._to_tensor(X_val, y_val, w_val)
            ds = TensorDataset(X_t, y_t, w_t)
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

            best_val = float("inf")
            patience_left = self.patience
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
                # validate
                model.eval()
                with torch.no_grad():
                    pv = model(Xv_t.to(self.device))
                    val = float(
                        (torch.abs(pv - yv_t.to(self.device)) * wv_t.to(self.device))
                        .sum()
                        .cpu()
                        .numpy()
                        / (wv_t.sum().cpu().numpy() + 1e-9)
                    )
                model.train()
                if val + 1e-9 < best_val:
                    best_val = val
                    best_state = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                    patience_left = self.patience
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        break
            # restore best
            model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
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

    If Huber converges, we use it. If it issues a ConvergenceWarning or ends up
    without usable attributes, we fall back to Ridge. This keeps the model
    competition robust and the terminal clean when `--quiet` is used.
    """

    def __init__(
        self,
        huber_max_iter: int = 20000,
        huber_alpha: float = 1e-4,
        huber_epsilon: float = 1.35,
        ridge_alpha: float = 1.0,
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

    def get_params(self, deep: bool = True):
        # mark parameter as used to satisfy static analyzers
        del deep
        return {
            "huber_max_iter": self.huber_max_iter,
            "huber_alpha": self.huber_alpha,
            "huber_epsilon": self.huber_epsilon,
            "ridge_alpha": self.ridge_alpha,
            "random_state": self.random_state,
            "suppress_warnings": self.suppress_warnings,
        }


# Static references to mark methods as used for vulture
_unused_get_params1 = SafeHuberRegressor.get_params
del _unused_get_params1
if "TorchTransformerRegressor" in globals():
    _unused_get_params2 = TorchTransformerRegressor.get_params
    del _unused_get_params2

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray] = None
    ):
        try:
            self._huber = HuberRegressor(
                max_iter=self.huber_max_iter,
                alpha=self.huber_alpha,
                epsilon=self.huber_epsilon,
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
            self._ridge = Ridge(alpha=self.ridge_alpha, random_state=self.random_state)
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
    """Return (accepts, pipeline_final_step_name)."""
    if isinstance(est, Pipeline):
        final_name, final_est = est.steps[-1]
        sig = inspect.signature(final_est.fit)
        return ("sample_weight" in sig.parameters), final_name
    sig = inspect.signature(est.fit)
    return ("sample_weight" in sig.parameters), None


def fit_with_optional_weight(
    est: object, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[np.ndarray]
):
    """Fit estimator; pass sample_weight only if supported (Pipeline‑aware)."""
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
    catboost_train_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Return a dict of model name -> estimator with sane defaults."""
    std = ("std", StandardScaler(with_mean=False))
    imp = ("imp", SimpleImputer(strategy="median"))

    models: Dict[str, object] = {
        # Tree ensembles
        "RandomForest": RandomForestRegressor(
            n_estimators=900, random_state=42, n_jobs=1
        ),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=900, random_state=42, n_jobs=1),
        "GBM": GradientBoostingRegressor(random_state=42),
        "HistGBM": HistGradientBoostingRegressor(random_state=42),
        # Linear family (scaled)
        "Ridge": Pipeline([imp, std, ("reg", Ridge(alpha=1.0, random_state=42))]),
        "Lasso": Pipeline(
            [
                imp,
                std,
                ("reg", Lasso(alpha=0.002, tol=1e-3, random_state=42, max_iter=50000)),
            ]
        ),
        "ElasticNet": Pipeline(
            [
                imp,
                std,
                (
                    "reg",
                    ElasticNet(
                        alpha=0.002,
                        l1_ratio=0.25,
                        tol=1e-3,
                        random_state=42,
                        max_iter=50000,
                    ),
                ),
            ]
        ),
        "SGD": Pipeline(
            [
                imp,
                std,
                (
                    "reg",
                    SGDRegressor(
                        loss="huber",
                        alpha=1e-4,
                        random_state=42,
                        max_iter=10000,
                        tol=1e-3,
                    ),
                ),
            ]
        ),
        # Robust linear with safe fallback
        "HuberSafe": Pipeline([imp, std, ("reg", SafeHuberRegressor())]),
        # SVM / KNN / NN
        "SVR": Pipeline([imp, std, ("reg", SVR(C=2.0, epsilon=0.1))]),
        "KNN": Pipeline(
            [imp, std, ("reg", KNeighborsRegressor(n_neighbors=8, weights="distance"))]
        ),
        "MLP": Pipeline(
            [
                imp,
                std,
                (
                    "reg",
                    MLPRegressor(
                        hidden_layer_sizes=(256, 128),
                        activation="relu",
                        alpha=1e-4,
                        random_state=42,
                        max_iter=1000,
                        tol=1e-4,
                    ),
                ),
            ]
        ),
    }
    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=1600,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        )
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=1400,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
    if HAS_CAT:
        models["CatBoost"] = CatBoostRegressor(
            iterations=1400,
            learning_rate=0.05,
            depth=8,
            verbose=False,
            random_seed=42,
            train_dir=catboost_train_dir or os.getcwd(),
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
            epochs=200,
            batch_size=128,
            patience=20,
            device=device,
        )
    return models


# ----------------------------- Half‑life selector -------------------------


def select_half_life(
    X: pd.DataFrame, y_l48: pd.Series, splits: int
) -> Tuple[Optional[int], np.ndarray]:
    """Pick recency half‑life by CV on Lower‑48 over a small grid."""
    grid = [None, 8, 13, 26, 39, 52]
    scores: Dict[str, float] = {}
    tscv = TimeSeriesSplit(n_splits=splits)
    base = build_model_registry(1)["LightGBM" if HAS_LGB else "RandomForest"]
    for hl in grid:
        w = make_recency_weights(len(X), hl)
        maes = []
        for tr, va in tscv.split(X):
            m = clone(base)
            m.fit(X.iloc[tr], y_l48.iloc[tr], sample_weight=w[tr])
            pv = m.predict(X.iloc[va])
            maes.append(weighted_mae(y_l48.iloc[va].values, pv, w[va]))
        scores[str(hl)] = float(np.mean(maes))
    best = min(grid, key=lambda h: scores[str(h)])
    return best, make_recency_weights(len(X), best)


# ------------------------------- Printing ---------------------------------


def fmt(v: float) -> str:
    """Format numeric values with minimal noise for table printing."""
    if isinstance(v, float) and np.isnan(v):
        return "NA"
    if abs(v - round(v)) < 1e-6:
        return f"{int(round(v))}"
    return f"{v:.2f}"


def print_table(rows: Iterable[Row], last_date, next_date) -> None:
    """Pretty print the final summary table to the terminal."""
    w_region, w_model = 15, 14
    w_est, w_mae, w_last_est, w_last_act = 14, 8, 14, 16

    title = "Final Prediction Summary"
    print(title)
    print("-" * len(title))
    header = (
        f"{'Region'.ljust(w_region)}"
        f"{'Model'.ljust(w_model)}"
        f"{str(next_date) + ' Est':>{w_est}}"
        f"{'CV MAE':>{w_mae}}"
        f"{str(last_date) + ' Est':>{w_last_est}}"
        f"{str(last_date) + ' actual':>{w_last_act}}"
    )
    print(header)
    for r in rows:
        print(
            f"{r.region.ljust(w_region)}"
            f"{r.model.ljust(w_model)}"
            f"{fmt(r.next_est):>{w_est}}"
            f"{fmt(r.cv_mae):>{w_mae}}"
            f"{fmt(r.last_est):>{w_last_est}}"
            f"{fmt(r.last_act):>{w_last_act}}"
        )


# --------------------------------- Main -----------------------------------


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=CONFIG["features"])
    parser.add_argument("--changes", type=Path, default=CONFIG["changes"])
    parser.add_argument("--outdir", type=Path, default=CONFIG["outdir"])
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--top_k", type=int, default=-1, help="-1 => adaptive")
    parser.add_argument("--reconcile", choices=["off", "variance"], default="variance")
    parser.add_argument(
        "--dry-run", action="store_true", help="resolve key paths then exit"
    )
    parser.add_argument("--show_all", action="store_true")
    parser.add_argument("--use_transformer", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress ConvergenceWarnings and similar noise",
    )
    args = parser.parse_args()

    features = _resolve_path(args.features)
    changes = _resolve_path(args.changes)
    outdir = _resolve_path(args.outdir)

    logger.info(
        "Args parsed | DRY_RUN=%s | features=%s | changes=%s | outdir=%s | splits=%s | top_k=%s",
        args.dry_run,
        features,
        changes,
        outdir,
        args.splits,
        args.top_k,
    )
    print(f"[dry] ROOT={ROOT}")
    print(f"[dry] features={features}")
    print(f"[dry] changes={changes}")
    print(f"[dry] outdir={outdir}")
    if args.dry_run:
        return

    # Ensure CatBoost writes catboost_info to EIAGuesser
    catboost_train_dir = ROOT / "EIAGuesser"
    os.environ["CATBOOST_DATA_DIR"] = str(catboost_train_dir)

    # Quiet mode: suppress convergence chatter
    if args.quiet:
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", message=r".*did not converge.*")
        warnings.filterwarnings("ignore", message=r".*failed to converge.*")

    # Load & clean
    daily = pd.read_csv(features, index_col=0, parse_dates=True)
    daily = clean_cols(daily.sort_index())
    ychg = pd.read_csv(changes, index_col=0, parse_dates=True)
    ychg = clean_cols(ychg.sort_index())
    ychg.columns = ychg.columns.str.replace("_change", "", regex=False)
    ychg.index = normalize_week_index(ychg.index)

    # Daily continuity
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx)
    num = daily.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    num = num.interpolate(limit_direction="both").ffill().bfill()
    daily[num.columns] = num

    # Build Fri→Thu sequences aligned to target weeks
    daily_num = daily.select_dtypes(include=[np.number])
    X_seq = build_sequence_matrix(daily_num, ychg.index)
    y = ychg.reindex(X_seq.index).dropna()
    X_seq = X_seq.reindex(y.index)

    # Feature filter
    top_k = None if args.top_k == -1 else args.top_k
    X = variance_topk(X_seq, top_k)

    # Recency weights from Lower‑48
    if "lower48" not in y.columns:
        raise ValueError("EIAchanges must include 'Lower48' column")
    best_hl, w_rec = select_half_life(X, y["lower48"], args.splits)

    # Model registry and splitter
    n_features_per_day = X.shape[1] // 7
    models = build_model_registry(
        n_features_per_day,
        use_transformer=args.use_transformer,
        device=args.device,
        catboost_train_dir=catboost_train_dir,
    )
    tscv = TimeSeriesSplit(n_splits=args.splits)

    # Per-region competition
    all_scores_rows = []
    best_per_region: Dict[str, Tuple[str, object, float, float]] = {}
    last_thu = X.index.max()
    next_thu = last_thu + pd.Timedelta(days=7)

    def cv_for_estimator(
        name: str, est: object, Xm: pd.DataFrame, yv: pd.Series
    ) -> Tuple[float, float, object]:
        """Return (cv_mae, last_est, fitted_model)."""
        maes = []
        for tr, va in tscv.split(Xm):
            Xtr, Xva = Xm.iloc[tr], Xm.iloc[va]
            ytr, yva = yv.iloc[tr], yv.iloc[va]
            wtr, wva = w_rec[tr], w_rec[va]
            if HAS_TORCH and name == "Transformer":
                m = clone(est)
                m.fit(Xtr, ytr, sample_weight=wtr, X_val=Xva, y_val=yva, w_val=wva)
                pva = m.predict(Xva)
            else:
                m = clone(est)
                fit_with_optional_weight(m, Xtr, ytr, wtr)
                pva = m.predict(Xva)
            maes.append(weighted_mae(yva.values, pva, wva))
        cv_mae = float(np.mean(maes))
        # fit on all
        if HAS_TORCH and name == "Transformer":
            est.fit(Xm, yv, sample_weight=w_rec)
            last_est = float(est.predict(Xm.iloc[[-1]])[0])
            fitted = est
        else:
            m_final = clone(est)
            fit_with_optional_weight(m_final, Xm, yv, w_rec)
            last_est = float(m_final.predict(Xm.iloc[[-1]])[0])
            fitted = m_final
        return cv_mae, last_est, fitted

    for key, tgt in TARGET_MAP.items():
        if tgt not in y.columns:
            continue
        ym = y[tgt]
        best_name, best_model, best_cv, best_last = (
            None,
            None,
            float("inf"),
            float("nan"),
        )
        for name, est in models.items():
            cv_mae, last_est, fitted = cv_for_estimator(name, est, X, ym)
            all_scores_rows.append(
                {
                    "region": REGION_PRETTY[key],
                    "model": name,
                    "cv_mae": round(cv_mae, 4),
                    "last_est": round(last_est, 4),
                    "last_act": float(ym.iloc[-1]),
                }
            )
            if cv_mae < best_cv:
                best_name, best_model, best_cv, best_last = (
                    name,
                    fitted,
                    cv_mae,
                    last_est,
                )
        best_per_region[key] = (best_name or "NA", best_model, best_cv, best_last)

    # Next‑week prediction (if we have Fri→Thu daily window)
    need = pd.date_range(last_thu + pd.Timedelta(days=1), next_thu, freq="D")
    have_all = all(d in daily.index for d in need)

    next_preds: Dict[str, float] = {}
    cv_mae_map: Dict[str, float] = {k: v[2] for k, v in best_per_region.items()}

    if have_all:
        seq_next = (
            build_sequence_matrix(daily_num, [next_thu])
            .reindex(columns=X.columns)
            .fillna(0.0)
        )
        for key, tgt in TARGET_MAP.items():
            if tgt not in y.columns or key not in best_per_region:
                continue
            _, winner, _, _ = best_per_region[key]
            next_preds[key] = float(winner.predict(seq_next)[0])
        if args.reconcile == "variance":
            next_preds = reconcile_variance(next_preds, cv_mae_map)

    # Winner table
    rows = []
    for key, tgt in TARGET_MAP.items():
        if tgt not in y.columns or key not in best_per_region:
            continue
        name, _, cv_mae, last_est = best_per_region[key]
        rows.append(
            Row(
                region=REGION_PRETTY[key],
                model=name,
                next_est=next_preds.get(key, float("nan")),
                cv_mae=cv_mae,
                last_est=last_est,
                last_act=float(y[tgt].iloc[-1]),
            )
        )

    print_table(rows, last_thu.date(), next_thu.date())

    # Optional: full per-region model comparison in terminal
    if args.show_all:
        print("\nDetailed CV by Region (lower is better)\n" + "=" * 40)
        df_scores = pd.DataFrame(all_scores_rows)
        for region in df_scores["region"].unique():
            sub = df_scores[df_scores["region"] == region].sort_values("cv_mae")
            print(f"\n{region}")
            print(sub.to_string(index=False))

    # Save detailed scoreboard & run config
    outdir.mkdir(parents=True, exist_ok=True)
    score_path = outdir / "cv_scoreboard.csv"
    pd.DataFrame(all_scores_rows).to_csv(score_path, index=False)

    with open(outdir / "run_config.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "features": features.as_posix(),
                "changes": changes.as_posix(),
                "outdir": outdir.as_posix(),
                "splits": args.splits,
                "top_k": top_k,
                "recency_half_life_selected": best_hl,
                "models": list(build_model_registry(n_features_per_day=1).keys()),
                "last_train_week_end_thu": str(last_thu.date()),
                "have_next_week_inputs": bool(have_all),
                "cv_scoreboard": score_path.as_posix(),
                "used_transformer": bool(args.use_transformer and HAS_TORCH),
            },
            fh,
            indent=2,
        )


# --------------------------- Reconciliation --------------------------------


def reconcile_variance(
    preds: Dict[str, float], cv_mae: Dict[str, float]
) -> Dict[str, float]:
    """Adjust region preds to sum to L48 using MAE‑based weights (proxy variance).

    If any region missing, we skip reconciliation.
    """
    if "conus" not in preds:
        return preds
    l48 = preds["conus"]
    keys = ["east", "midwest", "south_central", "mountain", "pacific"]
    vec = np.array([preds.get(k, np.nan) for k in keys], float)
    if np.isnan(vec).any():
        return preds
    w = np.array([max(cv_mae.get(k, 1.0), 1e-6) for k in keys], float)
    delta = l48 - float(vec.sum())
    vec = vec + delta * (w / w.sum())
    for i, k in enumerate(keys):
        preds[k] = float(vec[i])
    return preds


if __name__ == "__main__":  # pragma: no cover
    main()

# python MTP3.py --reconcile variance --show_all --quiet
# python MTP3.py --reconcile variance --use_transformer --device cpu --show_all --quiet

