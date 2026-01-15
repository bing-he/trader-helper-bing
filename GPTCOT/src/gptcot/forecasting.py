from __future__ import annotations

import logging
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TRAIN_SPLIT_DATE = pd.Timestamp("2024-01-01")
FEATURE_PRIORITY: List[str] = [
    "Total_MM_Net_z_52",
    "Total_Prod_Net_z_52",
    "Total_Swap_Net_z_52",
    "Total_MM_Net_pct_52",
    "Total_Prod_Net_pct_52",
    "Total_Swap_Net_pct_52",
    "Total_MM_Net_pct_156",
    "Total_Prod_Net_pct_156",
    "Total_Swap_Net_pct_156",
]
EXTRA_FEATURES: List[str] = [
    "storage_dev",
    "prompt_minus_strip",
    "winter_summer_spread",
    "curve_pca_1",
    "curve_pca_2",
    "curve_pca_3",
    "sin_doy",
    "cos_doy",
    "storage_x_mm_net",
]


def _parse_date_columns(path: Path, candidates: List[str]) -> List[str]:
    try:
        sample_cols = pd.read_csv(path, nrows=0).columns
    except Exception:  # pragma: no cover - defensive
        return []
    return [col for col in candidates if col in sample_cols]


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    """Choose an ordered, numeric feature set for model training."""

    feature_cols: List[str] = [col for col in FEATURE_PRIORITY if col in df.columns]
    dynamic = [
        col
        for col in df.columns
        if any(col.endswith(suffix) for suffix in ("_z_52", "_pct_52", "_pct_156"))
        and col not in feature_cols
    ]
    feature_cols.extend(sorted(dynamic))
    for extra in EXTRA_FEATURES:
        if extra in df.columns and extra not in feature_cols:
            feature_cols.append(extra)
    feature_cols = [col for col in feature_cols if df[col].notna().any()]
    return feature_cols


def _fmt_metric(value: float | None) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.4f}"


def load_data(info_dir: Path, output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load required datasets for training with helpful console feedback.

    Returns CoT forward returns, EIA storage totals, and forward curve dataframes.
    """

    cot_path = output_dir / "cot_forward_returns.csv"
    eia_path = info_dir / "EIAtotals.csv"
    curve_path = info_dir / "HenryForwardCurve.csv"

    for required in (cot_path, eia_path, curve_path):
        if not required.exists():
            raise FileNotFoundError(f"Required input not found: {required}")

    print(f"Loading forward returns from {cot_path}...")
    cot_df = pd.read_csv(cot_path, parse_dates=["cot_date", "horizon_date"])
    print(f" - {len(cot_df)} rows loaded.")

    eia_parse = _parse_date_columns(eia_path, ["Date", "Period"])
    print(f"Loading EIA storage from {eia_path} (parse_dates={eia_parse})...")
    eia_df = pd.read_csv(eia_path, parse_dates=eia_parse)
    print(f" - {len(eia_df)} rows loaded.")

    curve_parse = _parse_date_columns(curve_path, ["Date"])
    print(f"Loading forward curve from {curve_path} (parse_dates={curve_parse})...")
    curve_df = pd.read_csv(curve_path, parse_dates=curve_parse)
    print(f" - {len(curve_df)} rows loaded.")
    return cot_df, eia_df, curve_df


def _storage_deviation_frame(eia_df: pd.DataFrame) -> pd.DataFrame:
    """Compute storage deviation vs dynamically calculated 5Y average."""

    df = eia_df.copy()
    date_col = "Date" if "Date" in df.columns else "Period" if "Period" in df.columns else None
    storage_col = "Lower48" if "Lower48" in df.columns else "Total" if "Total" in df.columns else None
    if date_col is None or storage_col is None:
        return pd.DataFrame(columns=["storage_date", "storage_dev"])

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df[storage_col] = pd.to_numeric(df[storage_col], errors="coerce")
    df["weekofyear"] = df[date_col].dt.isocalendar().week.astype(int)

    avg_5y: List[float] = []
    min_10y: List[float] = []
    max_10y: List[float] = []
    for _, row in df.iterrows():
        year = int(pd.to_datetime(row[date_col]).year)
        week = int(row["weekofyear"])
        hist5 = df.loc[
            (df["weekofyear"] == week) & (df[date_col].dt.year < year) & (df[date_col].dt.year >= year - 5),
            storage_col,
        ]
        hist10 = df.loc[
            (df["weekofyear"] == week) & (df[date_col].dt.year < year) & (df[date_col].dt.year >= year - 10),
            storage_col,
        ]
        avg_5y.append(float(hist5.mean()) if not hist5.empty else np.nan)
        min_10y.append(float(hist10.min()) if not hist10.empty else np.nan)
        max_10y.append(float(hist10.max()) if not hist10.empty else np.nan)
    df["avg_5y"] = avg_5y
    df["min_10y"] = min_10y
    df["max_10y"] = max_10y
    df["storage_dev"] = (df[storage_col] - df["avg_5y"]) / df["avg_5y"]
    df.loc[df["avg_5y"] == 0, "storage_dev"] = np.nan
    return df[[date_col, storage_col, "avg_5y", "min_10y", "max_10y", "storage_dev"]].rename(
        columns={date_col: "storage_date"}
    )


def _curve_metrics(curve_df: pd.DataFrame) -> pd.DataFrame:
    """Build prompt-strip and winter-summer spreads per curve date."""

    if "Date" not in curve_df.columns:
        return pd.DataFrame(columns=["curve_date", "prompt_minus_strip", "winter_summer_spread"])
    metrics: List[Dict[str, Any]] = []
    for _, row in curve_df.iterrows():
        curve_date = pd.to_datetime(row.get("Date"))
        prompt = row.get("FWD_00", np.nan)
        strip_prices = [row.get(f"FWD_{i:02d}", np.nan) for i in range(12)]
        strip = float(np.nanmean(strip_prices)) if strip_prices else np.nan
        prompt_minus_strip = float(prompt - strip) if pd.notna(prompt) and pd.notna(strip) else np.nan

        front_label = str(row.get("FrontMonth_Label", ""))
        front_start = _parse_front_label(front_label) or curve_date
        winter_vals: List[float] = []
        summer_vals: List[float] = []
        for i, price in enumerate(strip_prices):
            if pd.isna(price):
                continue
            month_date = front_start + relativedelta(months=i)
            if month_date.month in (11, 12, 1, 2, 3):
                winter_vals.append(price)
            else:
                summer_vals.append(price)
        winter_avg = float(np.nanmean(winter_vals)) if winter_vals else np.nan
        summer_avg = float(np.nanmean(summer_vals)) if summer_vals else np.nan
        winter_summer_spread = (
            float(winter_avg - summer_avg) if pd.notna(winter_avg) and pd.notna(summer_avg) else np.nan
        )
        metrics.append(
            {
                "curve_date": curve_date,
                "prompt_minus_strip": prompt_minus_strip,
                "winter_summer_spread": winter_summer_spread,
            }
        )
    return pd.DataFrame(metrics).dropna(subset=["curve_date"]).sort_values("curve_date")


def _curve_pca_components(curve_df: pd.DataFrame) -> pd.DataFrame:
    """Extract PCA components (Level, Slope, Curvature) from the forward curve."""

    if "Date" not in curve_df.columns:
        return pd.DataFrame(columns=["curve_date", "curve_pca_1", "curve_pca_2", "curve_pca_3"])
    fwd_cols = [f"FWD_{i:02d}" for i in range(12) if f"FWD_{i:02d}" in curve_df.columns]
    if len(fwd_cols) < 3:
        return pd.DataFrame(columns=["curve_date", "curve_pca_1", "curve_pca_2", "curve_pca_3"])

    df = curve_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    curve_values = df[fwd_cols].apply(pd.to_numeric, errors="coerce")
    standardized = (curve_values - curve_values.mean()) / curve_values.std(ddof=0)
    standardized = standardized.replace([np.inf, -np.inf], np.nan).fillna(0)
    try:
        pca = PCA(n_components=3)
        components = pca.fit_transform(standardized)
    except Exception:  # pragma: no cover - defensive
        return pd.DataFrame(columns=["curve_date", "curve_pca_1", "curve_pca_2", "curve_pca_3"])

    pca_df = pd.DataFrame(components, columns=["curve_pca_1", "curve_pca_2", "curve_pca_3"])  # Level, Slope, Curvature
    pca_df["curve_date"] = df["Date"].values
    return pca_df[["curve_date", "curve_pca_1", "curve_pca_2", "curve_pca_3"]]


def engineer_features(
    cot_df: pd.DataFrame, eia_df: pd.DataFrame, curve_df: pd.DataFrame, horizon: int, target_column: Optional[str] = "strip_pct_change"
) -> pd.DataFrame:
    """
    Build a clean, model-ready frame for a given horizon.

    - Filters to the requested horizon_days
    - Adds storage deviation via merge_asof on cot_date
    - Adds forward-curve metrics via merge_asof on cot_date
    - Adds PCA curve shape (Level, Slope, Curvature), seasonality, and interaction features
    - Drops NaN targets (when provided) and imputes remaining feature gaps with ffill/bfill
    """

    print(f"\nEngineering features for {horizon}-day horizon...")
    df = cot_df.loc[cot_df["horizon_days"] == horizon].copy()
    if df.empty:
        print(f" - No rows found for horizon {horizon}.")
        return df

    df["cot_date"] = pd.to_datetime(df["cot_date"])
    if "horizon_date" in df.columns:
        df["horizon_date"] = pd.to_datetime(df["horizon_date"])
    else:
        df["horizon_date"] = pd.NaT
    df = df.sort_values("cot_date")

    storage_features = _storage_deviation_frame(eia_df)
    if not storage_features.empty:
        df = pd.merge_asof(
            df,
            storage_features.sort_values("storage_date"),
            left_on="cot_date",
            right_on="storage_date",
            direction="backward",
        ).drop(columns=["storage_date"])
    else:
        df["storage_dev"] = np.nan
        print(" - Storage deviation unavailable; filling with NaN.")

    curve_metrics = _curve_metrics(curve_df)
    if not curve_metrics.empty:
        df = pd.merge_asof(
            df,
            curve_metrics,
            left_on="cot_date",
            right_on="curve_date",
            direction="backward",
        ).drop(columns=["curve_date"])
    else:
        df["prompt_minus_strip"] = np.nan
        df["winter_summer_spread"] = np.nan
        print(" - Forward curve metrics unavailable; filling with NaN.")

    curve_pca = _curve_pca_components(curve_df)
    if not curve_pca.empty:
        df = pd.merge_asof(
            df,
            curve_pca.sort_values("curve_date"),
            left_on="cot_date",
            right_on="curve_date",
            direction="backward",
        ).drop(columns=["curve_date"])
    else:
        df["curve_pca_1"] = np.nan
        df["curve_pca_2"] = np.nan
        df["curve_pca_3"] = np.nan
        print(" - Curve PCA unavailable; filling with NaN.")

    day_of_year = df["cot_date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365)
    df["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365)

    storage_series = (
        pd.to_numeric(df["storage_dev"], errors="coerce") if "storage_dev" in df.columns else pd.Series(np.nan, index=df.index)
    )
    mm_net_series = (
        pd.to_numeric(df["Total_MM_Net_z_52"], errors="coerce")
        if "Total_MM_Net_z_52" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    df["storage_x_mm_net"] = storage_series * mm_net_series

    if "prompt_minus_strip" in df.columns:
        prompt_series = pd.to_numeric(df["prompt_minus_strip"], errors="coerce")
        df["spread_change"] = prompt_series.shift(-horizon) - prompt_series
    else:
        df["spread_change"] = np.nan

    resolved_target = target_column
    if resolved_target and resolved_target not in df.columns:
        fallback = "pct_change" if "pct_change" in df.columns else None
        if fallback:
            print(f" - Target column '{resolved_target}' missing; using '{fallback}' instead.")
        else:
            print(f" - Target column '{resolved_target}' missing; skipping target drop.")
        resolved_target = fallback

    before_drop = len(df)
    if resolved_target:
        df = df.dropna(subset=[resolved_target])
    dropped = before_drop - len(df)
    if dropped:
        print(f" - Dropped {dropped} rows with NaN {resolved_target or 'target'}.")
    print(f" - {len(df)} rows remain after cleaning.")
    if df.empty:
        return df

    feature_cols = _select_feature_columns(df)
    print(f" - Feature set ({len(feature_cols)}): {feature_cols}")

    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[feature_cols] = df[feature_cols].ffill().bfill()
    if resolved_target:
        df[resolved_target] = pd.to_numeric(df[resolved_target], errors="coerce")
    df["horizon_days"] = horizon

    target_cols: List[str] = []
    for col in [resolved_target, "strip_pct_change", "pct_change", "spread_change"]:
        if col and col in df.columns and col not in target_cols:
            target_cols.append(col)
    ordered_cols = ["cot_date", "horizon_date", "horizon_days"] + target_cols + feature_cols
    return df[ordered_cols]


def train_model(df: pd.DataFrame, target_column: str = "strip_pct_change") -> Tuple[RegressorMixin, List[str], Dict[str, float]]:
    """
    Train a robust RandomForest model with time-aware CV and imputation.

    Returns the fitted model, the feature names, and validation metrics.
    """

    if df.empty:
        raise ValueError("Received empty dataframe; cannot train model.")

    df = df.sort_values("cot_date")
    if target_column not in df.columns:
        fallback = "pct_change" if "pct_change" in df.columns else None
        if fallback is None:
            raise ValueError(f"Target column '{target_column}' not found in dataframe.")
        print(f" - Target '{target_column}' missing; falling back to '{fallback}'.")
        target_column = fallback
    df = df.dropna(subset=[target_column])
    if df.empty:
        raise ValueError(f"No rows available after dropping NaNs for target '{target_column}'.")
    print(f" - Training target: {target_column}")
    feature_cols = _select_feature_columns(df)
    if not feature_cols:
        raise ValueError("No usable feature columns found for training.")
    X_all = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_all = df[target_column].astype(float)

    train_mask = df["cot_date"] < TRAIN_SPLIT_DATE
    if not train_mask.any():
        print(f" - No rows before {TRAIN_SPLIT_DATE.date()}; using all data for training.")
        train_mask = pd.Series([True] * len(df), index=df.index)
    X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
    X_test, y_test = X_all.loc[~train_mask], y_all.loc[~train_mask]
    print(f" - Training rows: {len(X_train)}, Test rows: {len(X_test)}")

    param_grid = [
        {"n_estimators": 200, "max_depth": 4},
        {"n_estimators": 400, "max_depth": 6},
        {"n_estimators": 300, "max_depth": None},
    ]
    best_params = param_grid[0]
    cv_mae = np.nan
    cv_rmse = np.nan

    cv_splits = min(5, len(X_train) - 1) if len(X_train) > 1 else 0
    if cv_splits >= 2:
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        best_score = np.inf
        for params in param_grid:
            fold_mae: List[float] = []
            fold_rmse: List[float] = []
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                pipeline = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", RandomForestRegressor(random_state=42, **params)),
                    ]
                )
                pipeline.fit(X_tr, y_tr)
                preds = pipeline.predict(X_val)
                fold_mae.append(mean_absolute_error(y_val, preds))
                fold_rmse.append(mean_squared_error(y_val, preds) ** 0.5)
            cv_mae = float(np.mean(fold_mae)) if fold_mae else np.nan
            cv_rmse = float(np.mean(fold_rmse)) if fold_rmse else np.nan
            print(f"   Params {params} -> CV MAE={_fmt_metric(cv_mae)}, RMSE={_fmt_metric(cv_rmse)}")
            if fold_mae and cv_mae < best_score:
                best_score = cv_mae
                best_params = params
    else:
        print(" - Not enough data for cross-validation; using default hyperparameters.")

    print(f" - Selected params: {best_params}")
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(random_state=42, **best_params)),
        ]
    )
    model.fit(X_train, y_train)
    # Attach feature metadata for downstream prediction alignment.
    try:
        model.feature_names_in_ = np.array(feature_cols)
    except Exception:  # pragma: no cover - best effort
        pass

    metrics: Dict[str, float] = {"cv_mae": cv_mae, "cv_rmse": cv_rmse}
    if not X_test.empty:
        test_preds = model.predict(X_test)
        holdout_mae = mean_absolute_error(y_test, test_preds)
        holdout_rmse = mean_squared_error(y_test, test_preds) ** 0.5
        metrics["holdout_mae"] = float(holdout_mae)
        metrics["holdout_rmse"] = float(holdout_rmse)
        print(f" - Holdout MAE={holdout_mae:.4f}, RMSE={holdout_rmse:.4f}")
    return model, feature_cols, metrics


def save_model(model: RegressorMixin, feature_names: List[str], output_path: Path) -> None:
    """Persist the trained model and feature metadata."""

    payload = {"model": model, "feature_names": feature_names}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(payload, f)
    print(f"Model saved to {output_path}")


def load_price_model(path: Optional[Path] = None, target: str = "strip") -> Optional[Tuple[RegressorMixin, Optional[List[str]]]]:
    """Load the trained price forecasting model (and feature names) if present."""

    default_dir = Path(__file__).resolve().parents[2] / "output"
    candidates: List[Path] = []
    if path is not None:
        candidates.append(path)
    if target:
        candidates.append(default_dir / f"price_forecast_model_{target}.pkl")
    candidates.append(default_dir / "price_forecast_model.pkl")

    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        logging.warning("Price forecast model not found for target %s", target or "unknown")
        return None
    try:
        with model_path.open("rb") as f:
            loaded = pickle.load(f)
        feature_names: Optional[List[str]] = None
        model_obj: Optional[RegressorMixin] = None
        if isinstance(loaded, dict) and "model" in loaded:
            model_obj = loaded["model"]
            feature_names = loaded.get("feature_names")
        else:
            model_obj = loaded
        if model_obj is None:
            return None
        if feature_names is not None and not hasattr(model_obj, "feature_names_in_"):
            try:
                model_obj.feature_names_in_ = np.array(feature_names)
            except Exception:  # pragma: no cover - defensive
                pass
        return model_obj, feature_names
    except Exception as exc:  # pragma: no cover - defensive
        logging.warning("Failed to load price forecast model: %s", exc)
        return None


def _resolve_output_dir(info_dir: Path) -> Path:
    candidates = [
        info_dir.parent / "GPTCOT" / "output",
        info_dir.parent / "output",
        info_dir / "output",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _latest_cot_row(output_dir: Path) -> Optional[pd.Series]:
    cot_path = output_dir / "cot_forward_returns.csv"
    if not cot_path.exists():
        logging.warning("cot_forward_returns.csv not found at %s", cot_path)
        return None
    cot_df = pd.read_csv(cot_path)
    if cot_df.empty:
        return None
    cot_df["cot_date"] = pd.to_datetime(cot_df["cot_date"])
    cot_df = cot_df.sort_values("cot_date")
    latest_date = cot_df["cot_date"].max()
    latest_rows = cot_df.loc[cot_df["cot_date"] == latest_date]
    if latest_rows.empty:
        return None
    return latest_rows.iloc[0]


def _storage_deviation(info_dir: Path) -> float:
    storage_path = info_dir / "EIAtotals.csv"
    if not storage_path.exists():
        logging.warning("EIAtotals.csv not found at %s", storage_path)
        return np.nan
    df = pd.read_csv(storage_path)
    if df.empty:
        return np.nan
    date_col = "Date" if "Date" in df.columns else "Period"
    df["date"] = pd.to_datetime(df[date_col])
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df = df.sort_values("date")
    latest = df.tail(1).iloc[0]
    latest_week = int(latest["weekofyear"])
    latest_year = int(latest["date"].year)
    storage_col = "Lower48" if "Lower48" in df.columns else "Total" if "Total" in df.columns else None
    if storage_col is None:
        return np.nan
    history = df.loc[
        (df["weekofyear"] == latest_week)
        & (df["date"].dt.year < latest_year)
        & (df["date"].dt.year >= latest_year - 5),
        storage_col,
    ]
    if history.empty or latest[storage_col] == 0:
        return np.nan
    avg_5y = float(history.mean())
    return float((latest[storage_col] - avg_5y) / avg_5y) if avg_5y else np.nan


def _parse_front_label(label: str) -> Optional[pd.Timestamp]:
    for fmt in ("%b-%Y", "%b-%y"):
        try:
            return pd.to_datetime(label, format=fmt)
        except Exception:
            continue
    return None


def _forward_curve_features(info_dir: Path, curve_df: Optional[pd.DataFrame] = None) -> tuple[float, float]:
    curve_path = info_dir / "HenryForwardCurve.csv"
    if curve_df is None:
        if not curve_path.exists():
            logging.warning("HenryForwardCurve.csv not found at %s", curve_path)
            return np.nan, np.nan
        df = pd.read_csv(curve_path)
    else:
        df = curve_df.copy()
    if df.empty:
        return np.nan, np.nan
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    latest = df.tail(1).iloc[0]
    fwd_cols = [col for col in df.columns if col.startswith("FWD_")]
    if not fwd_cols:
        return np.nan, np.nan
    prompt = latest.get("FWD_00", np.nan)
    strip_prices = [latest.get(f"FWD_{i:02d}", np.nan) for i in range(12)]
    strip = float(np.nanmean(strip_prices)) if strip_prices else np.nan
    prompt_minus_strip = float(prompt - strip) if pd.notna(prompt) and pd.notna(strip) else np.nan

    front_label = str(latest.get("FrontMonth_Label", ""))
    front_start = _parse_front_label(front_label)
    if front_start is None:
        return prompt_minus_strip, np.nan

    winter_values: list[float] = []
    summer_values: list[float] = []
    for col in fwd_cols:
        try:
            idx = int(col.split("_")[1])
        except Exception:
            continue
        month_date = front_start + relativedelta(months=idx)
        price = latest.get(col, np.nan)
        if pd.isna(price):
            continue
        if month_date.month in (11, 12, 1, 2, 3):
            winter_values.append(price)
        elif month_date.month in (4, 5, 6, 7, 8, 9, 10):
            summer_values.append(price)
    winter_avg = float(np.nanmean(winter_values)) if winter_values else np.nan
    summer_avg = float(np.nanmean(summer_values)) if summer_values else np.nan
    winter_summer_spread = (
        float(winter_avg - summer_avg) if pd.notna(winter_avg) and pd.notna(summer_avg) else np.nan
    )
    return prompt_minus_strip, winter_summer_spread


def _available_horizons(output_dir: Path) -> List[int]:
    default = [7, 14, 28, 30]
    cot_path = output_dir / "cot_forward_returns.csv"
    if not cot_path.exists():
        return default
    try:
        df = pd.read_csv(cot_path, usecols=["horizon_days"])
        horizons = sorted({int(h) for h in df["horizon_days"].dropna().unique()})
        return horizons or default
    except Exception:
        return default


def prepare_latest_features(info_dir: Path) -> Optional[pd.Series]:
    """Build the latest feature vector used for price forecasting."""

    output_dir = _resolve_output_dir(info_dir)
    cot_row = _latest_cot_row(output_dir)
    if cot_row is None:
        return None

    storage_dev = _storage_deviation(info_dir)
    prompt_minus_strip = np.nan
    winter_summer_spread = np.nan
    curve_pca_1 = np.nan
    curve_pca_2 = np.nan
    curve_pca_3 = np.nan

    curve_path = info_dir / "HenryForwardCurve.csv"
    curve_df: Optional[pd.DataFrame] = None
    if curve_path.exists():
        curve_parse = _parse_date_columns(curve_path, ["Date"])
        curve_df = pd.read_csv(curve_path, parse_dates=curve_parse)
        curve_df = curve_df.sort_values("Date")
        prompt_minus_strip, winter_summer_spread = _forward_curve_features(info_dir, curve_df)
        pca_df = _curve_pca_components(curve_df)
        if not pca_df.empty:
            latest_pca = pca_df.sort_values("curve_date").iloc[-1]
            curve_pca_1 = latest_pca.get("curve_pca_1", np.nan)
            curve_pca_2 = latest_pca.get("curve_pca_2", np.nan)
            curve_pca_3 = latest_pca.get("curve_pca_3", np.nan)
    else:
        logging.warning("HenryForwardCurve.csv not found at %s", curve_path)

    cot_date = pd.to_datetime(cot_row.get("cot_date"))
    day_of_year = cot_date.dayofyear if pd.notna(cot_date) else np.nan
    sin_doy = np.sin(2 * np.pi * day_of_year / 365) if not pd.isna(day_of_year) else np.nan
    cos_doy = np.cos(2 * np.pi * day_of_year / 365) if not pd.isna(day_of_year) else np.nan
    mm_net = pd.to_numeric(pd.Series([cot_row.get("Total_MM_Net_z_52", np.nan)]), errors="coerce").iloc[0]
    storage_x_mm_net = storage_dev * mm_net if pd.notna(storage_dev) and pd.notna(mm_net) else np.nan

    features = {
        "Total_MM_Net_z_52": cot_row.get("Total_MM_Net_z_52", np.nan),
        "Total_Prod_Net_z_52": cot_row.get("Total_Prod_Net_z_52", np.nan),
        "Total_Swap_Net_z_52": cot_row.get("Total_Swap_Net_z_52", np.nan),
        "Total_MM_Net_pct_52": cot_row.get("Total_MM_Net_pct_52", np.nan),
        "Total_Prod_Net_pct_52": cot_row.get("Total_Prod_Net_pct_52", np.nan),
        "Total_Swap_Net_pct_52": cot_row.get("Total_Swap_Net_pct_52", np.nan),
        "storage_dev": storage_dev,
        "prompt_minus_strip": prompt_minus_strip,
        "winter_summer_spread": winter_summer_spread,
        "curve_pca_1": curve_pca_1,
        "curve_pca_2": curve_pca_2,
        "curve_pca_3": curve_pca_3,
        "sin_doy": sin_doy,
        "cos_doy": cos_doy,
        "storage_x_mm_net": storage_x_mm_net,
    }
    return pd.Series(features, dtype=float)


def predict_returns(
    info_dir: Path, horizons: Optional[List[int]] = None, target_column: str = "strip_pct_change"
) -> Dict[int, Optional[Dict[str, Any]]]:
    """Predict returns for multiple horizons using horizon- and target-specific models when present."""

    output_dir = _resolve_output_dir(info_dir)
    if horizons is None:
        horizons = _available_horizons(output_dir)

    features = prepare_latest_features(info_dir)
    results: Dict[int, Optional[Dict[str, Any]]] = {}
    if features is None:
        for h in horizons:
            results[h] = None
        return results

    if target_column == "spread_change":
        target_suffix = "spread_change"
    elif target_column in (None, "", "strip_pct_change"):
        target_suffix = "strip"
    elif target_column == "pct_change":
        target_suffix = "pct"
    else:
        target_suffix = str(target_column)

    for horizon in horizons:
        horizon_model_path = output_dir / f"price_forecast_model_{horizon}.pkl"
        preferred_path = output_dir / f"price_forecast_model_{target_suffix}_{horizon}.pkl"
        legacy_spread_path = output_dir / f"price_forecast_model_spread_{horizon}.pkl"
        candidates = [preferred_path]
        if target_suffix == "strip":
            candidates.append(horizon_model_path)
        if target_suffix == "spread_change":
            candidates.append(legacy_spread_path)
        loaded = None
        for candidate in candidates:
            if not candidate.exists():
                continue
            loaded = load_price_model(candidate, target_suffix)
            if loaded is not None:
                break
        if loaded is None:
            loaded = load_price_model(None, target_suffix)  # fallback to generic model if available
        if loaded is None:
            results[horizon] = None
            continue
        model, feature_names = loaded

        feature_vector = features.copy()
        model_features = feature_names or list(feature_vector.index)
        if hasattr(model, "feature_names_in_"):
            model_features = list(getattr(model, "feature_names_in_"))
            feature_vector = feature_vector.reindex(model_features, fill_value=np.nan)
        feature_vector = feature_vector.fillna(0)
        try:
            X_pred = pd.DataFrame([feature_vector], columns=model_features)
            prediction = float(model.predict(X_pred)[0])
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning("Prediction failed for horizon %s: %s", horizon, exc)
            results[horizon] = None
            continue
        direction = "up" if prediction > 0 else "down"

        importances: Optional[List[tuple[str, float]]] = None
        if hasattr(model, "feature_importances_"):
            imps = getattr(model, "feature_importances_", [])
            names = model_features if len(model_features) == len(imps) else list(range(len(imps)))
            pairs = list(zip(names, imps))
            importances = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)[:3]
        results[horizon] = {"pred_return": prediction, "direction": direction, "importances": importances}
    return results


def train_and_save_models(horizons: List[int], info_dir: Path, output_dir: Path) -> None:
    """
    Train and persist a regression model for each requested horizon.

    Prints progress for each stage to help users follow the pipeline.
    """

    print(f"Training models for horizons {horizons}...")
    try:
        cot_df, eia_df, curve_df = load_data(info_dir, output_dir)
    except Exception as exc:
        logging.error("Unable to load training data: %s", exc)
        print(f" ! Failed to load training data: {exc}")
        return

    for horizon in horizons:
        print(f"\nStarting training for {horizon}-day horizon...")
        try:
            horizon_df = engineer_features(cot_df, eia_df, curve_df, horizon, target_column=None)
        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("Feature engineering failed for %s-day horizon", horizon)
            print(f" ! Feature engineering failed for {horizon}-day horizon: {exc}")
            continue

        if horizon_df.empty:
            logging.warning("No usable rows for %s-day horizon; skipping.", horizon)
            print(f" ! No usable rows for {horizon}-day horizon; skipping.")
            continue

        print(f" - Rows available after feature engineering: {len(horizon_df)}")

        strip_target = "strip_pct_change" if "strip_pct_change" in horizon_df.columns else None
        if strip_target is None and "pct_change" in horizon_df.columns:
            strip_target = "pct_change"
            print(" - strip_pct_change missing; falling back to pct_change for strip model.")

        strip_df = horizon_df if strip_target is None else horizon_df.dropna(subset=[strip_target])
        if strip_target and not strip_df.empty:
            try:
                print(f" - Fitting Strip Model (target={strip_target})")
                model, feature_names, metrics = train_model(strip_df, target_column=strip_target)
                model_path = output_dir / f"price_forecast_model_strip_{horizon}.pkl"
                save_model(model, feature_names, model_path)
                generic_path = output_dir / f"price_forecast_model_{horizon}.pkl"
                try:
                    shutil.copyfile(model_path, generic_path)
                except Exception as exc:  # pragma: no cover - defensive
                    logging.warning("Failed to write generic model alias %s: %s", generic_path, exc)
                print(
                    f"Finished Strip Model for {horizon}-day horizon; "
                    f"MAE={_fmt_metric(metrics.get('cv_mae'))}, RMSE={_fmt_metric(metrics.get('cv_rmse'))}"
                )
            except Exception as exc:  # pylint: disable=broad-except
                logging.exception("Strip model training failed for %s-day horizon", horizon)
                print(f" ! Strip model training failed for {horizon}-day horizon: {exc}")
        else:
            print(f" ! No usable rows for strip model at {horizon}-day horizon; skipping.")

        if "spread_change" in horizon_df.columns:
            spread_df = horizon_df.dropna(subset=["spread_change"])
        else:
            spread_df = pd.DataFrame()
        if not spread_df.empty:
            try:
                print(" - Fitting Spread Model (target=spread_change)")
                model, feature_names, metrics = train_model(spread_df, target_column="spread_change")
                model_path = output_dir / f"price_forecast_model_spread_change_{horizon}.pkl"
                save_model(model, feature_names, model_path)
                print(
                    f"Finished Spread Model for {horizon}-day horizon; "
                    f"MAE={_fmt_metric(metrics.get('cv_mae'))}, RMSE={_fmt_metric(metrics.get('cv_rmse'))}"
                )
            except Exception as exc:  # pylint: disable=broad-except
                logging.exception("Spread model training failed for %s-day horizon", horizon)
                print(f" ! Spread model training failed for {horizon}-day horizon: {exc}")
        else:
            print(f" ! No usable rows for spread model at {horizon}-day horizon; skipping.")
