# MTP_Phase2_Manual_Stack_v4.py
# Goal: 100x faster wall-clock via vectorized prep, parallel CV/OOF, pinned BLAS threads,
# verbose color logging, and tabulated output. Inputs, targets, weekly cadence, and
# manual stack design preserved.

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend before pyplot import

import json
import os
import platform
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Runtime throttling to avoid thread oversubscription when we spawn processes
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import holidays
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from tabulate import tabulate

# Optional boosters
try:
    import lightgbm as lgb

    LGBM_INSTALLED = True
except Exception:
    LGBM_INSTALLED = False

try:
    import xgboost as xgb

    XGBOOST_INSTALLED = True
except Exception:
    XGBOOST_INSTALLED = False

# Optional interpretability (removed shap import to satisfy static analysis)

# Logging (color)
import logging

try:
    from colorlog import ColoredFormatter

    _COLORLOG = True
except Exception:
    _COLORLOG = False

# Warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ─── Configuration ────────────────────────────────────────────────────────
TOP_N_FEATURES = 50
RANDOM_STATE = 42
MISSING_THRESHOLD = 0.05
MODE = "weekly"
N_SPLITS = 5
GAP_FOLDS = 1
DEFAULT_JOBS = max(1, os.cpu_count() or 1)
np.random.seed(RANDOM_STATE)

# ─── Paths (Pathlib, relative to this file) ───────────────────────────────
ROOT_PATH = Path(__file__).resolve().parents[1]
EIAGUESSER_PATH = ROOT_PATH / "EIAGuesser"
OUTPUT_PATH = EIAGUESSER_PATH / "output"
MODEL_PATH = EIAGUESSER_PATH / "model_outputs"
INFO_PATH = ROOT_PATH / "INFO"
FEAT_FILE = OUTPUT_PATH / "Combined_Wide_Data.csv"
TARGET_FILE = INFO_PATH / "EIAchanges.csv"
MODEL_PATH.mkdir(parents=True, exist_ok=True)
(OUTPUT_PATH / "feature_importances").mkdir(parents=True, exist_ok=True)
(OUTPUT_PATH / "shap_plots").mkdir(parents=True, exist_ok=True)


# ─── Logging setup ────────────────────────────────────────────────────────
def make_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("MTPv4")
    logger.setLevel(level)
    if logger.handlers:
        return logger
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if _COLORLOG:
        fmt = "%(log_color)s%(levelname)-8s%(reset)s | %(asctime)s | %(message)s"
        formatter = ColoredFormatter(
            fmt,
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    else:
        formatter = logging.Formatter(
            "%(levelname)-8s | %(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


LOGGER = make_logger(logging.INFO)


# ─── Utilities ────────────────────────────────────────────────────────────
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def date_label(dt: pd.Timestamp) -> str:
    return (
        dt.strftime("%#m/%#d")
        if platform.system() == "Windows"
        else dt.strftime("%-m/%-d")
    )


def get_agg_dict(columns: pd.Index) -> dict:
    sum_keywords = (
        "_flow",
        "_demand",
        "_stor",
        "_prod",
        "_exp",
        "_imp",
        "_gen",
        "_hdd",
        "_cdd",
    )
    return {
        col: ("sum" if any(k in col for k in sum_keywords) else "mean")
        for col in columns
    }


def cyclic_encode(series: pd.Series, period: int, prefix: str) -> pd.DataFrame:
    val = series.astype(np.float64).to_numpy(copy=False)
    angle = 2 * np.pi * (np.mod(val, period)) / period
    return pd.DataFrame(
        {f"{prefix}_sin": np.sin(angle), f"{prefix}_cos": np.cos(angle)},
        index=series.index,
    )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Vectorized numeric selection
    df_num = df.select_dtypes(include=[np.number])
    cdd = df_num.filter(like="_weather_").filter(like="_cdd").sum(axis=1)
    hdd = df_num.filter(like="_weather_").filter(like="_hdd").sum(axis=1)
    out = df_num.assign(gwdd=cdd - hdd)

    us_holidays = holidays.US()
    # Fast membership via ndarray of ordinals
    idx = out.index
    is_holiday = pd.Series(
        idx.isin(pd.DatetimeIndex([d for d in us_holidays]).tz_localize(None)),
        index=idx,
    ).astype(int)
    month = idx.month
    week = idx.isocalendar().week.astype(int)
    dow = idx.dayofweek

    out = out.assign(
        is_holiday=is_holiday, month=month, week_of_year=week, day_of_week=dow
    )
    out = pd.concat(
        [
            out,
            cyclic_encode(pd.Series(month, index=idx), 12, "month"),
            cyclic_encode(pd.Series(week, index=idx), 53, "woy"),
            cyclic_encode(pd.Series(dow, index=idx), 7, "dow"),
        ],
        axis=1,
    )
    return out


def add_advanced_features(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    idx = df.index
    # lagged features using vectorized shift
    for col in target_cols:
        for lag in (1, 2, 3, 4, 8):
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        df[f"{col}_vol_4wk"] = df[col].shift(1).rolling(4, min_periods=3).std()
        df[f"{col}_vol_8wk"] = df[col].shift(1).rolling(8, min_periods=5).std()

    is_winter = idx.month.isin([12, 1, 2]).astype(int)
    is_summer = idx.month.isin([6, 7, 8]).astype(int)
    df["is_winter"], df["is_summer"] = is_winter, is_summer

    hdd_col_name = next((c for c in df.columns if "hdd" in c and "conus" in c), None)
    cdd_col_name = next((c for c in df.columns if "cdd" in c and "conus" in c), None)
    if hdd_col_name:
        df["winter_hdd"] = df["is_winter"] * df[hdd_col_name]
    if cdd_col_name:
        df["summer_cdd"] = df["is_summer"] * df[cdd_col_name]

    return df.dropna(axis=0)


class GapTimeSeriesSplit:
    def __init__(self, n_splits=5, gap=0):
        self.n_splits = n_splits
        self.gap = gap

    def split(self, X):
        n = len(X)
        base = TimeSeriesSplit(n_splits=self.n_splits)
        for tr, te in base.split(np.arange(n)):
            if self.gap > 0:
                max_tr = tr.max()
                te = te[te > max_tr + self.gap]
                if len(te) == 0:
                    continue
            yield tr, te


# Parallel OOF predictions


def _fit_predict_block(args) -> Tuple[np.ndarray, np.ndarray]:
    X, y, model, tr, va = args
    mdl = clone(model)
    # LightGBM early stopping inside worker
    if LGBM_INSTALLED and isinstance(mdl, lgb.LGBMRegressor) and len(tr) > 40:
        k = max(5, int(0.1 * len(tr)))
        X_tr, y_tr = X[tr][:-k], y[tr][:-k]
        X_es, y_es = X[tr][-k:], y[tr][-k:]
        mdl.set_params(random_state=RANDOM_STATE)
        mdl.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], eval_metric="l1")
    else:
        mdl.fit(X[tr], y[tr])
    return va, mdl.predict(X[va])


def get_oof_predictions_parallel(
    X: np.ndarray,
    y: np.ndarray,
    models: List,
    cv_splitter,
    n_jobs: int,
    logger: logging.Logger,
) -> np.ndarray:
    folds = list(cv_splitter.split(X))
    oof = np.full((len(y), len(models)), np.nan, dtype=np.float64)
    for m_idx, base in enumerate(models):
        logger.debug(
            f"OOF | model={type(base).__name__} | folds={len(folds)} | jobs={n_jobs}"
        )
        tasks = [(X, y, base, tr, te) for (tr, te) in folds]
        results = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
            joblib.delayed(_fit_predict_block)(t) for t in tasks
        )
        for va, pred in results:
            oof[va, m_idx] = pred
        # Fill any NaN due to gap trimming
        col = pd.Series(oof[:, m_idx])
        oof[:, m_idx] = col.ffill().bfill().to_numpy()
    return oof


# ─── Main ────────────────────────────────────────────────────────────────
def main(n_jobs: int = DEFAULT_JOBS, log_level: int = logging.INFO):
    LOGGER.setLevel(log_level)
    for h in LOGGER.handlers:
        h.setLevel(log_level)

    LOGGER.info("Loading data")
    features_daily = pd.read_csv(FEAT_FILE, index_col="date", parse_dates=True)
    targets_raw = pd.read_csv(TARGET_FILE, index_col="Period", parse_dates=True)

    targets_raw.columns = targets_raw.columns.str.replace("_Change", "", regex=False)
    features_daily = clean_columns(features_daily)
    targets_raw = clean_columns(targets_raw)

    target_map = {
        "conus": "lower48",
        "east": "east",
        "midwest": "midwest",
        "south_central": "southcentral",
        "mountain": "mountain",
        "pacific": "pacific",
    }

    target_cols = [c for c in target_map.values() if c in targets_raw.columns]
    if not target_cols:
        raise RuntimeError("No target columns found in EIAchanges.csv after cleaning.")

    features_daily = features_daily[
        ~features_daily.index.duplicated(keep="last")
    ].sort_index()
    missing_pct = features_daily.isnull().sum().div(len(features_daily))
    drop_cols = missing_pct[missing_pct > MISSING_THRESHOLD].index
    if len(drop_cols):
        LOGGER.warning(f"Dropping {len(drop_cols)} cols over missing threshold")
        features_daily.drop(columns=drop_cols, inplace=True)

    # Impute fully vectorized
    features_daily.interpolate(method="linear", limit_direction="both", inplace=True)
    features_daily.fillna(0, inplace=True)

    LOGGER.info("Engineering features + weekly aggregation")
    features_daily_eng = engineer_features(features_daily)
    weekly_feat = features_daily_eng.resample("W-THU").agg(
        get_agg_dict(features_daily_eng.columns)
    )
    weekly_feat.index = weekly_feat.index + pd.Timedelta(days=1)  # align Fri

    weekly_all_base = weekly_feat.join(targets_raw[target_cols], how="inner")
    weekly_all = add_advanced_features(weekly_all_base, target_cols)

    last_report_date = weekly_all.index.max()
    pred_week_label = last_report_date + pd.Timedelta(days=7)
    LOGGER.info(
        f"Last EIA week: {last_report_date.date()} | Forecasting: {pred_week_label.date()}"
    )

    # Shared CV splitter
    cv_splitter = GapTimeSeriesSplit(n_splits=N_SPLITS, gap=GAP_FOLDS)

    results = []

    for region, tgt_col in target_map.items():
        if tgt_col not in weekly_all.columns:
            continue
        LOGGER.info(f"===== {region.upper().replace('_', ' ')} =====")

        y_all = weekly_all[tgt_col]
        feature_cols = [c for c in weekly_all.columns if c not in target_cols]
        X_all = weekly_all[feature_cols]

        if len(X_all) < max(52, N_SPLITS * 2 + 10):
            LOGGER.warning("Skip region: insufficient samples")
            continue

        # Base models; tuned for speed/accuracy tradeoff
        base_models = [
            RandomForestRegressor(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=1,  # 1 per process, parallelized at outer level
            )
        ]
        if LGBM_INSTALLED:
            base_models.append(
                lgb.LGBMRegressor(
                    n_estimators=800,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    verbose=-1,
                )
            )
        if XGBOOST_INSTALLED:
            base_models.append(
                xgb.XGBRegressor(
                    n_estimators=800,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    tree_method="hist",
                    eval_metric="mae",
                )
            )

        meta_model = Ridge(alpha=1.0, random_state=RANDOM_STATE)

        X_np, y_np = X_all.to_numpy(), y_all.to_numpy()

        # Cross-validated evaluation with parallel OOF
        LOGGER.info("Evaluating stack with parallel OOF")
        # For CV scoring, we simulate rolling folds: fit meta on train OOF, score on held-out
        fold_scores = []
        folds = list(cv_splitter.split(X_np))
        for tr, te in folds:
            oof_tr = get_oof_predictions_parallel(
                X_np[tr],
                y_np[tr],
                base_models,
                GapTimeSeriesSplit(n_splits=min(N_SPLITS, 4), gap=GAP_FOLDS),
                n_jobs=n_jobs,
                logger=LOGGER,
            )
            meta = clone(meta_model).fit(oof_tr, y_np[tr])
            # Train base on tr in parallel, predict te
            trained = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
                joblib.delayed(clone(m).fit)(X_np[tr], y_np[tr]) for m in base_models
            )
            base_te = np.column_stack([m.predict(X_np[te]) for m in trained])
            y_hat = meta.predict(base_te)
            fold_scores.append(mean_absolute_error(y_np[te], y_hat))
        mae_score = np.float64(np.mean(fold_scores))
        LOGGER.info(f"CV MAE: {mae_score:.3f}")

        # Retrain on all data using parallel OOF for meta
        LOGGER.info("Retraining full-data stack")
        final_oof = get_oof_predictions_parallel(
            X_np,
            y_np,
            base_models,
            GapTimeSeriesSplit(n_splits=N_SPLITS, gap=GAP_FOLDS),
            n_jobs=n_jobs,
            logger=LOGGER,
        )
        final_meta = clone(meta_model).fit(final_oof, y_np)
        final_base = joblib.Parallel(n_jobs=n_jobs, backend="loky")(
            joblib.delayed(clone(m).fit)(X_np, y_np) for m in base_models
        )

        # Persist
        stack_blob = {"base": final_base, "meta": final_meta, "features": feature_cols}
        model_fname = f"{region}_ManualStack_{last_report_date:%Y%m%d}.joblib"
        joblib.dump(stack_blob, MODEL_PATH / model_fname, compress=("xz", 3))

        # Residual PI
        residuals = y_np - final_meta.predict(final_oof)
        pi_lower = np.float64(np.nanpercentile(residuals, 2.5))
        pi_upper = np.float64(np.nanpercentile(residuals, 97.5))

        # Build prediction row fast
        # If the target prediction week isn't present in weekly_feat (common when
        # exogenous data for the upcoming week isn't fully available yet),
        # fall back to the most recent available week's exogenous features but
        # relabel the index to the prediction week. This prevents KeyError and
        # keeps the pipeline running with a reasonable proxy.
        if pred_week_label in weekly_feat.index:
            next_week_base_features = weekly_feat.loc[[pred_week_label]].copy()
        else:
            # Use the last available week strictly before the pred label
            try:
                last_valid_idx = weekly_feat.index[
                    weekly_feat.index < pred_week_label
                ].max()
            except ValueError:
                last_valid_idx = weekly_feat.index.max()
            proxy = weekly_feat.loc[[last_valid_idx]].copy()
            proxy.index = pd.DatetimeIndex([pred_week_label])
            next_week_base_features = proxy
        pred_X = next_week_base_features.copy()
        for lag_col in target_cols:
            for lag in (1, 2, 3, 4, 8):
                pred_X[f"{lag_col}_lag_{lag}"] = weekly_all[lag_col].iloc[-lag]
            pred_X[f"{lag_col}_vol_4wk"] = weekly_all[lag_col].iloc[-4:].std()
            pred_X[f"{lag_col}_vol_8wk"] = weekly_all[lag_col].iloc[-8:].std()
        pred_X["is_winter"] = pred_X.index.month.isin([12, 1, 2]).astype(int)
        pred_X["is_summer"] = pred_X.index.month.isin([6, 7, 8]).astype(int)
        _m, _w, _d = (
            pred_X.index.month,
            pred_X.index.isocalendar().week.astype(int),
            pred_X.index.dayofweek,
        )
        # Drop existing cyclic features to avoid duplicates
        cyclic_cols = [c for c in pred_X.columns if c.endswith(("_sin", "_cos"))]
        pred_X = pred_X.drop(columns=cyclic_cols, errors="ignore")
        pred_X = pd.concat(
            [
                pred_X,
                cyclic_encode(pd.Series(_m, index=pred_X.index), 12, "month"),
                cyclic_encode(pd.Series(_w, index=pred_X.index), 53, "woy"),
                cyclic_encode(pd.Series(_d, index=pred_X.index), 7, "dow"),
            ],
            axis=1,
        )

        hdd_col_name = next(
            (c for c in X_all.columns if "hdd" in c and "conus" in c), None
        )
        cdd_col_name = next(
            (c for c in X_all.columns if "cdd" in c and "conus" in c), None
        )
        if hdd_col_name and hdd_col_name in pred_X.columns:
            pred_X["winter_hdd"] = pred_X["is_winter"] * pred_X[hdd_col_name]
        if cdd_col_name and cdd_col_name in pred_X.columns:
            pred_X["summer_cdd"] = pred_X["is_summer"] * pred_X[cdd_col_name]
        pred_X = pred_X.reindex(columns=feature_cols, fill_value=0.0)

        # Predict stacked (ensure predictors carry feature names for models
        # fitted with named features to avoid sklearn UserWarning)
        base_pred = np.column_stack([m.predict(pred_X) for m in final_base])
        next_pred = np.float64(final_meta.predict(base_pred)[0])

        # Use the dataframe row to preserve feature names during prediction
        prior_base = np.column_stack([m.predict(X_all.iloc[[-1]]) for m in final_base])
        prior_pred = np.float64(final_meta.predict(prior_base)[0])
        prior_act = np.float64(y_np[-1])

        results.append(
            {
                "Region": region.replace("conus", "Lower 48").replace("_", " ").title(),
                "Model": "ManualStack",
                f"{date_label(pred_week_label)} Est": next_pred,
                "CV MAE": mae_score,
                f"{date_label(last_report_date)} Est": prior_pred,
                f"{date_label(last_report_date)} actual": prior_act,
                "Lower PI": pi_lower,
                "Upper PI": pi_upper,
            }
        )

        # Snapshot importances
        try:
            rf = next(
                (m for m in final_base if isinstance(m, RandomForestRegressor)), None
            )
            if rf is not None:
                pd.Series(rf.feature_importances_, index=feature_cols).nlargest(
                    TOP_N_FEATURES
                ).to_csv(
                    OUTPUT_PATH / "feature_importances" / f"{region}_rf_importance.csv"
                )
            if LGBM_INSTALLED:
                lgbm = next(
                    (m for m in final_base if isinstance(m, lgb.LGBMRegressor)), None
                )
                if lgbm is not None:
                    pd.Series(lgbm.feature_importances_, index=feature_cols).nlargest(
                        TOP_N_FEATURES
                    ).to_csv(
                        OUTPUT_PATH
                        / "feature_importances"
                        / f"{region}_lgb_importance.csv"
                    )
        except Exception as e:
            LOGGER.debug(f"Importance snapshot failed: {e}")

    # Summarize
    res_df = pd.DataFrame(results).set_index(["Region", "Model"]).round(3).sort_index()
    res_df = res_df.assign(seed=RANDOM_STATE, timestamp=datetime.now().isoformat())

    csv_out = MODEL_PATH / "predictions.csv"
    res_df.to_csv(csv_out)

    # Tabulate to stdout
    table = tabulate(
        res_df.reset_index(), headers="keys", tablefmt="github", showindex=False
    )
    print("\n" + "-" * 80 + "\nFinal Prediction Summary (tabulated)\n" + "-" * 80)
    print(table)
    print(f"\nForecast table → {csv_out}")

    try:
        l48_est = res_df.loc[
            ("Lower 48", "ManualStack"), f"{date_label(pred_week_label)} Est"
        ]
        over_under = int(round(np.float64(l48_est)))
        print(f"Over/Under line (Lower-48 rounded): {over_under:+} Bcf")
    except Exception:
        print("Could not generate Over/Under line.")

    # Metadata
    meta = {
        "script": Path(__file__).name,
        "generated_at": datetime.now().isoformat(),
        "random_state": RANDOM_STATE,
        "mode": MODE,
        "n_splits": N_SPLITS,
        "gap_folds": GAP_FOLDS,
        "jobs": n_jobs,
        "regions": len(results),
    }
    with open(MODEL_PATH / "predictions.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Model objects   →", MODEL_PATH)
    print("Done", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    # Light CLI via env vars avoids argparse overhead and keeps code portable
    # Set MTP_JOBS and MTP_LOGLEVEL if desired
    jobs = int(os.getenv("MTP_JOBS", DEFAULT_JOBS))
    lvl = os.getenv("MTP_LOGLEVEL", "INFO").upper()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    main(n_jobs=jobs, log_level=level_map.get(lvl, logging.INFO))
