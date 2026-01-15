# MTP_Phase2_Manual_Stack_v2.py
# This version fixes the KeyError by correctly creating seasonality features for the prediction row.

import matplotlib

matplotlib.use("Agg")  # Set non-interactive backend BEFORE pyplot is imported
import os
import platform
import warnings
from datetime import datetime

import holidays
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# --- ML/DL Frameworks ---
from sklearn.model_selection import TimeSeriesSplit

# --- Optional Tree Boosters ---
try:
    import lightgbm as lgb

    LGBM_INSTALLED = True
except ImportError:
    LGBM_INSTALLED = False

try:
    import xgboost as xgb

    XGBOOST_INSTALLED = True
except ImportError:
    XGBOOST_INSTALLED = False

# --- Optional Interpretability ---
# Removed SHAP optional import as it was unused to satisfy static analysis (vulture)

# --- Suppress Warnings ---
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow INFO messages

# ─── Configuration ────────────────────────────────────────────────────────
TOP_N_FEATURES = 50
RANDOM_STATE = 42
MISSING_THRESHOLD = 0.05
MODE = "weekly"

np.random.seed(RANDOM_STATE)

# ─── Paths ────────────────────────────────────────────────────────────────
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
EIAGUESSER_PATH = os.path.join(ROOT_PATH, "EIAGuesser")
OUTPUT_PATH = os.path.join(EIAGUESSER_PATH, "output")
MODEL_PATH = os.path.join(EIAGUESSER_PATH, "model_outputs")
INFO_PATH = os.path.join(ROOT_PATH, "INFO")
FEAT_FILE = os.path.join(OUTPUT_PATH, "Combined_Wide_Data.csv")
TARGET_FILE = os.path.join(INFO_PATH, "EIAchanges.csv")

# Create output directories
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "feature_importances"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "shap_plots"), exist_ok=True)


# ─── Helper Functions ─────────────────────────────────────────────────────
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip()
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
    agg_dict = {}
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
    for col in columns:
        agg_dict[col] = "sum" if any(k in col for k in sum_keywords) else "mean"
    return agg_dict


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_eng = df.select_dtypes(include=np.number).copy()
    df_eng["gwdd"] = df_eng.filter(like="_weather_").filter(like="_cdd").sum(
        axis=1
    ) - df_eng.filter(like="_weather_").filter(like="_hdd").sum(axis=1)
    us_holidays = holidays.US()
    df_eng["is_holiday"] = (
        df.index.to_series().apply(lambda x: x in us_holidays).astype(int)
    )
    df_eng["month"] = df.index.month
    df_eng["week_of_year"] = df.index.isocalendar().week.astype(int)
    return df_eng


def add_advanced_features(df: pd.DataFrame, target_cols: list) -> pd.DataFrame:
    df_adv = df.copy()
    for col in target_cols:
        for lag in [1, 2, 3, 4]:
            df_adv[f"{col}_lag_{lag}"] = df_adv[col].shift(lag)
        df_adv[f"{col}_vol_4wk"] = df_adv[col].shift(1).rolling(4).std()
    df_adv["is_winter"] = df_adv.index.month.isin([12, 1, 2]).astype(int)
    df_adv["is_summer"] = df_adv.index.month.isin([6, 7, 8]).astype(int)

    # Add interaction features
    # Use a generic name that will exist after aggregation
    hdd_col_name = next(
        (c for c in df_adv.columns if "hdd" in c and "conus" in c), None
    )
    cdd_col_name = next(
        (c for c in df_adv.columns if "cdd" in c and "conus" in c), None
    )
    if hdd_col_name:
        df_adv["winter_hdd"] = df_adv["is_winter"] * df_adv[hdd_col_name]
    if cdd_col_name:
        df_adv["summer_cdd"] = df_adv["is_summer"] * df_adv[cdd_col_name]

    return df_adv.dropna(axis=0)


def get_oof_predictions(X, y, models, cv_splitter):
    """
    Generates out-of-fold predictions for a list of models.
    This is the core of manual stacking.
    """
    oof_preds_full = np.zeros((len(y), len(models)))

    for i, model in enumerate(models):
        oof_cv = TimeSeriesSplit(n_splits=5)
        for train_idx, val_idx in oof_cv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train = y[train_idx]

            cloned_model = clone(model)
            cloned_model.fit(X_train, y_train)
            oof_preds_full[val_idx, i] = cloned_model.predict(X_val)

    return oof_preds_full


# ─── Load & Prepare Data ──────────────────────────────────────────────────
print("-" * 60 + "\nLoading data…")
features_daily = pd.read_csv(FEAT_FILE, index_col="date", parse_dates=True)
targets_raw = pd.read_csv(TARGET_FILE, index_col="Period", parse_dates=True)

# ---> FIX #1: Remove the '_Change' suffix from the target columns to match the new format
targets_raw.columns = targets_raw.columns.str.replace("_Change", "", regex=False)

features_daily, targets_raw = clean_columns(features_daily), clean_columns(targets_raw)

# ---> FIX #2: Update the dictionary to use the new lowercase column names after cleaning
target_map = {
    "conus": "lower48",
    "east": "east",
    "midwest": "midwest",
    "south_central": "southcentral",
    "mountain": "mountain",
    "pacific": "pacific",
}
target_cols = list(target_map.values())
targets_raw = targets_raw[target_cols]  # This should now work without a KeyError

missing_pct = features_daily.isnull().sum() / len(features_daily)
features_daily.drop(
    columns=missing_pct[missing_pct > MISSING_THRESHOLD].index, inplace=True
)
print(f"Dropped {len(missing_pct[missing_pct > MISSING_THRESHOLD])} columns.")
features_daily.interpolate(method="linear", limit_direction="both", inplace=True)
features_daily.fillna(0, inplace=True)
print("Imputation complete.")

features_daily_eng = engineer_features(features_daily)
agg_dict = get_agg_dict(features_daily_eng.columns)
weekly_feat = features_daily_eng.resample("W-THU").agg(agg_dict)
weekly_feat.index = weekly_feat.index + pd.Timedelta(days=1)
print("Aggregation and feature engineering complete.")

weekly_all_base = weekly_feat.join(targets_raw, how="inner")
weekly_all = add_advanced_features(weekly_all_base, target_cols)

print("Advanced features created.")

last_report_date = weekly_all.index.max()
pred_week_label = last_report_date + pd.Timedelta(days=7)
print(
    f"Last EIA week: {last_report_date.date()} | Forecasting for: {pred_week_label.date()}"
)

# ─── Main Loop ────────────────────────────────────────────────────────────
results = []
for region, tgt_col in target_map.items():
    print(f"\n{'=' * 22} {region.upper().replace('_', ' ')} {'=' * 22}")

    y_all = weekly_all[tgt_col]
    feature_cols = [c for c in weekly_all.columns if c not in target_cols]
    X_all = weekly_all[feature_cols]

    if len(X_all) < 52:
        print("   Skipping region (not enough data).")
        continue

    # --- Define Models for Manual Stack ---
    base_models = [
        RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=1),
        lgb.LGBMRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=1, verbose=-1
        ),
    ]
    meta_model = Ridge(alpha=1.0)

    # --- Manual CV Loop for Evaluation ---
    print("   Evaluating Manual Stack with CV loop...")
    eval_tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []

    X_all_np, y_all_np = X_all.values, y_all.values

    for train_idx, test_idx in eval_tscv.split(X_all_np):
        X_train, X_test = X_all_np[train_idx], X_all_np[test_idx]
        y_train, y_test = y_all_np[train_idx], y_all_np[test_idx]

        trained_base_models = [clone(m).fit(X_train, y_train) for m in base_models]
        oof_preds = get_oof_predictions(
            X_train,
            y_train,
            [clone(m) for m in base_models],
            TimeSeriesSplit(n_splits=5),
        )

        cloned_meta_model = clone(meta_model)
        cloned_meta_model.fit(oof_preds, y_train)

        test_base_preds = np.column_stack(
            [m.predict(X_test) for m in trained_base_models]
        )
        final_preds = cloned_meta_model.predict(test_base_preds)

        fold_scores.append(mean_absolute_error(y_test, final_preds))

    mae_score = np.mean(fold_scores)
    print(f" → Manual Stack CV MAE: {mae_score:5.2f}")

    # --- Retrain Final Models on All Data ---
    print("   Retraining final models on all data...")
    final_base_models = [clone(m).fit(X_all_np, y_all_np) for m in base_models]
    final_oof_preds = get_oof_predictions(
        X_all_np,
        y_all_np,
        [clone(m) for m in base_models],
        TimeSeriesSplit(n_splits=5),
    )
    final_meta_model = clone(meta_model).fit(final_oof_preds, y_all_np)

    final_stack = {"base": final_base_models, "meta": final_meta_model}
    model_fname = f"{region}_ManualStack_{last_report_date:%Y%m%d}.joblib"
    model_fpath = os.path.join(MODEL_PATH, model_fname)
    joblib.dump(final_stack, model_fpath)
    print(f"   Saved champion stack to {model_fpath}")

    residuals = y_all_np - final_meta_model.predict(final_oof_preds)
    pi_lower = np.percentile(residuals, 2.5)
    pi_upper = np.percentile(residuals, 97.5)

    # --- Prepare Prediction Inputs ---
    next_week_base_features = weekly_feat.loc[[pred_week_label]]
    pred_X = next_week_base_features.copy()

    for lag_col in target_cols:
        for lag in [1, 2, 3, 4]:
            pred_X[f"{lag_col}_lag_{lag}"] = weekly_all[lag_col].iloc[-lag]
        pred_X[f"{lag_col}_vol_4wk"] = weekly_all[lag_col].iloc[-4:].std()

    # Re-create the seasonality and interaction features for the prediction row
    pred_X["is_winter"] = pred_X.index.month.isin([12, 1, 2]).astype(int)
    pred_X["is_summer"] = pred_X.index.month.isin([6, 7, 8]).astype(int)

    hdd_col_name = next((c for c in X_all.columns if "hdd" in c and "conus" in c), None)
    cdd_col_name = next((c for c in X_all.columns if "cdd" in c and "conus" in c), None)
    if hdd_col_name and hdd_col_name in pred_X.columns:
        pred_X["winter_hdd"] = pred_X["is_winter"] * pred_X[hdd_col_name]
    if cdd_col_name and cdd_col_name in pred_X.columns:
        pred_X["summer_cdd"] = pred_X["is_summer"] * pred_X[cdd_col_name]

    # Ensure columns are in the same order as training data
    pred_X = pred_X[X_all.columns]

    # --- Generate Predictions with the Final Stack ---
    pred_base_preds = np.column_stack(
        [m.predict(pred_X.values) for m in final_stack["base"]]
    )
    next_pred = final_stack["meta"].predict(pred_base_preds)[0]

    prior_base_preds = np.column_stack(
        [m.predict(X_all.iloc[[-1]].values) for m in final_stack["base"]]
    )
    prior_pred = final_stack["meta"].predict(prior_base_preds)[0]
    prior_act = y_all.iloc[-1]

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

# ─── Save Summary ─────────────────────────────────────────────────────────
res_df = pd.DataFrame(results).set_index(["Region", "Model"]).round(2).sort_index()
res_df = res_df.assign(seed=RANDOM_STATE, timestamp=datetime.now().isoformat())
print("\n" + "-" * 80 + "\nFinal Prediction Summary\n" + "-" * 80)
print(res_df)
csv_out = os.path.join(MODEL_PATH, "predictions.csv")
res_df.to_csv(csv_out)
print(f"\nForecast table → {csv_out}")

try:
    l48_est = res_df.loc[
        ("Lower 48", "ManualStack"), f"{date_label(pred_week_label)} Est"
    ]
    over_under = int(round(l48_est))
    print(f"Over/Under line (Lower-48 rounded): {over_under:+} Bcf")
except (KeyError, IndexError):
    print("Could not generate Over/Under line.")

print("Model objects   →", MODEL_PATH)
print("Done", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
