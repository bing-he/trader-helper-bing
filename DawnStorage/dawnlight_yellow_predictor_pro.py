r"""
Dawnlight Yellow Light Predictor (portable)
===========================================

Terminal-first runner that resolves paths from the repo root using `common.pathing.ROOT`.
Defaults:
- Input CSV:  ROOT / "DawnStorage" / "INFO" / "DawnLight.csv"
- Output JSON: ROOT / "DawnStorage" / "DawnYellowForecast.json"
- Output CSV:  ROOT / "DawnStorage" / "DawnYellowForecast.csv"

Run
---
```powershell
python DawnStorage/dawnlight_yellow_predictor_pro.py --dry-run
```
"""



import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance

from common.pathing import ROOT
from common.logs import get_file_logger

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# -------- Paths (resolved from repo root) --------
CONFIG = {
    "input": ROOT / "DawnStorage" / "INFO" / "DawnLight.csv",
    "output_json": ROOT / "DawnStorage" / "DawnYellowForecast.json",
    "output_csv": ROOT / "DawnStorage" / "DawnYellowForecast.csv",
}

logger = get_file_logger(Path(__file__).stem)


# ------------------------------ Data Classes ------------------------------

@dataclass
class ForecastResult:
    date_asof: str
    model_kind: str
    auc_binary: Optional[float]
    n_events: int
    trend_flow: float
    last_storage_total: Optional[float]
    prob_tomorrow_yellow_turn_on: float
    prob_tomorrow_inj_turn_on: float
    prob_tomorrow_wd_turn_on: float
    prob_week_yellow_turn_on: float
    prob_week_inj_turn_on: float
    prob_week_wd_turn_on: float
    top_features: List[Tuple[str, float]]


# ------------------------------ Helpers ------------------------------

def _resolve_path(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _winsorize_series(s: pd.Series, low=0.01, high=0.99) -> pd.Series:
    x = s.astype(float)
    if x.notna().sum() < 10:
        return x
    lo, hi = np.nanpercentile(x, [low*100, high*100])
    return x.clip(lo, hi)


def _safe_bool_from_yellow_cols(row) -> int:
    val = row.get("Yellow", np.nan)
    if pd.isna(val):
        return 1 if pd.notna(row.get("YellowType", np.nan)) and str(row["YellowType"]).lower() not in {"nan", ""} else 0
    if isinstance(val, (int, float)):
        return int(val == 1)
    if isinstance(val, str):
        v = val.strip().lower()
        return 1 if v in {"1", "y", "yes", "true", "on"} else 0
    if isinstance(val, bool):
        return int(val)
    return 0


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "Date" not in df.columns or "DailyChange" not in df.columns:
        raise ValueError("Input must contain at least 'Date' and 'DailyChange' columns.")
    df["Date"] = pd.to_datetime(df["Date"])  # ensure datetime
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)

    # Targets
    if "Yellow" in df.columns or "YellowType" in df.columns:
        df["YellowOn"] = df.apply(_safe_bool_from_yellow_cols, axis=1)
    else:
        df["YellowOn"] = np.nan

    if "YellowType" in df.columns:
        df["YellowType"] = df["YellowType"].astype(str)
    else:
        df["YellowType"] = np.nan

    if "StorageTotal" not in df.columns:
        df["StorageTotal"] = np.nan

    return df


def _interp_storage(df: pd.DataFrame) -> pd.Series:
    """Interpolate StorageTotal over time using a DatetimeIndex (required by method='time').
    Returns a Series aligned to df.index.
    """
    st_by_date = pd.Series(df["StorageTotal"].values, index=pd.to_datetime(df["Date"]))
    out = st_by_date.interpolate(method="time", limit_direction="both")
    aligned = pd.Series(out.values, index=df.index)
    if aligned.isna().any():
        aligned = pd.Series(df["StorageTotal"].values, index=df.index).interpolate(limit_direction="both")
    return aligned


def _rolling_365_storage_pos(df: pd.DataFrame) -> pd.Series:
    """Storage position normalized by rolling 365-day min/max, with graceful fallback.
    Produces a 0..1 band that moves with the most recent year of history.
    """
    s = pd.Series(df["StorageInterp"].values, index=pd.to_datetime(df["Date"]))
    # If we have at least ~180 days, do rolling; else fallback to expanding
    if len(s) >= 180:
        rol_min = s.rolling("365D", min_periods=30).min()
        rol_max = s.rolling("365D", min_periods=30).max()
        rng = (rol_max - rol_min).replace(0, np.nan)
        pos = ((s - rol_min) / rng).clip(0, 1)
        pos = pos.ffill().bfill()
    else:
        exp_min = s.expanding().min(); exp_max = s.expanding().max()
        rng = (exp_max - exp_min).replace(0, np.nan)
        pos = ((s - exp_min) / rng).clip(0, 1).fillna(0.5)
    # align back to row order
    return pd.Series(pos.values, index=df.index)


def _rolling_slope(x: np.ndarray) -> float:
    n = len(x)
    if n < 3 or np.all(np.isnan(x)):
        return 0.0
    x = np.array(x, dtype=float)
    mask = ~np.isnan(x)
    if mask.sum() < 3:
        return 0.0
    x = np.where(mask, x, np.nanmean(x[mask]))
    t = np.arange(n, dtype=float)
    t -= t.mean()
    x = x - x.mean()
    denom = (t**2).sum()
    if denom == 0:
        return 0.0
    return float((t * x).sum() / denom)


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Robust flows
    df["FlowClipped"] = _winsorize_series(df["DailyChange"])  # used for stats

    # Interpolated storage & rolling position
    df["StorageInterp"] = _interp_storage(df)
    df["StoragePos"] = _rolling_365_storage_pos(df)

    # Rolling flow stats (lagged by shift(1)) based on clipped flows
    for w in (3, 5, 7, 14, 28):
        df[f"FlowMean_{w}"] = df["FlowClipped"].rolling(w, min_periods=1).mean().shift(1)
    for w in (7, 14, 28):
        df[f"FlowStd_{w}"] = df["FlowClipped"].rolling(w, min_periods=2).std().shift(1)

    df["FlowSlope_7"] = (
        df["FlowClipped"].rolling(7, min_periods=3)
        .apply(lambda s: _rolling_slope(np.asarray(s)), raw=False)
        .shift(1)
    )

    df["StorageDiff_3"] = df["StorageInterp"].diff().rolling(3, min_periods=1).mean().shift(1)
    df["StorageDiff_7"] = df["StorageInterp"].diff().rolling(7, min_periods=1).mean().shift(1)

    # Seasonality & calendar
    doy = df["Date"].dt.dayofyear
    df["DOY_sin"] = np.sin(2 * np.pi * doy / 366.0)
    df["DOY_cos"] = np.cos(2 * np.pi * doy / 366.0)
    df["Month"] = df["Date"].dt.month
    df["DOW"] = df["Date"].dt.dayofweek

    # Lags
    df["StoragePos_lag1"] = df["StoragePos"].shift(1)
    df["StoragePos_lag7"] = df["StoragePos"].shift(7)

    # Directional flags
    df["RecentInjection"] = (df["FlowMean_7"] > 0).astype(int)
    df["RecentWithdrawal"] = (df["FlowMean_7"] < 0).astype(int)
    
    # --- NEW: Stress Features based on user feedback ---
    # High stress = high storage level AND high injection rate
    df['InjectionStress'] = df['StoragePos_lag1'] * df['FlowMean_7']
    # High stress = low storage level AND high withdrawal rate
    # Note: FlowMean_7 is negative for withdrawals, so we multiply by -1 to make the stress value positive
    df['WithdrawalStress'] = (1 - df['StoragePos_lag1']) * (-1 * df['FlowMean_7'])


    # --- NEW TARGET: YellowTurnOn ---
    # This is 1 only on the first day of a yellow light event.
    df['YellowTurnOn'] = ((df['YellowOn'] == 1) & (df['YellowOn'].shift(1).fillna(0) == 0)).astype(int)

    # Multiclass type: 0=Off, 1=YellowInj, 2=YellowWd (NaN-safe)
    def _type_code(row):
        val = row.get("YellowOn", np.nan)
        if pd.isna(val) or val == 0:
            return 0
        yt = str(row.get("YellowType", "")).strip().lower()
        if yt.startswith("yellowinj"):
            return 1
        if yt.startswith("yellowwd"):
            return 2
        return 0

    if "YellowOn" in df.columns and not df["YellowOn"].isna().all():
        df["TypeCode"] = df.apply(_type_code, axis=1)
    else:
        df["TypeCode"] = np.nan

    return df


def _feature_list(cols: List[str]) -> List[str]:
    preferred = [
        "StoragePos_lag1", "StoragePos_lag7",
        "FlowMean_3", "FlowMean_5", "FlowMean_7", "FlowMean_14", "FlowMean_28",
        "FlowStd_7", "FlowStd_14", "FlowStd_28",
        "FlowSlope_7",
        "StorageDiff_3", "StorageDiff_7",
        "DOY_sin", "DOY_cos",
        "Month", "DOW",
        "RecentInjection", "RecentWithdrawal",
        "InjectionStress", "WithdrawalStress", # <-- New features
    ]
    return [f for f in preferred if f in cols]


def _fit_binary_model_timeaware(X: pd.DataFrame, y: pd.Series):
    """Time-aware calibrated Gradient Boosting model."""
    # TimeSeriesSplit for calibration
    n_splits = 5 if len(y) >= 500 else max(3, len(y)//100 or 3)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Use GradientBoostingClassifier - it's more powerful for this type of data
    base = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.7, random_state=42)
    
    model = CalibratedClassifierCV(base, cv=tscv, method="isotonic")
    model.fit(X, y)

    # Simple holdout AUC (last 20%)
    split_idx = int(len(y) * 0.8)
    auc = None
    if len(np.unique(y.iloc[split_idx:])) > 1:
        probs = model.predict_proba(X.iloc[split_idx:])[:, 1]
        auc = roc_auc_score(y.iloc[split_idx:], probs)
    return model, auc


def _fit_models(df_feat: pd.DataFrame):
    features = _feature_list(df_feat.columns)
    X = df_feat[features].astype(float)
    y_bin = df_feat["YellowTurnOn"] # <-- Use the new target

    mask = ~X.isna().any(axis=1) & y_bin.notna()
    Xb, yb = X[mask], y_bin[mask].astype(int)

    if len(yb) < 80 or yb.sum() < 5 or yb.nunique() < 2:
        return None, features, None, None, None

    model_bin, auc_bin = _fit_binary_model_timeaware(Xb, yb)
    
    # Get in-sample predictions for lookback
    historical_probs = model_bin.predict_proba(Xb)[:, 1]
    df_with_probs = df_feat.copy()
    df_with_probs['PredictedProb'] = np.nan
    df_with_probs.loc[Xb.index, 'PredictedProb'] = historical_probs
    
    # Calculate Feature Importance
    perm_importance = permutation_importance(
        model_bin, Xb, yb, n_repeats=10, random_state=42, n_jobs=1 # Set n_jobs=1 to avoid error
    )
    sorted_idx = perm_importance.importances_mean.argsort()
    top_features = [
        (features[i], perm_importance.importances_mean[i])
        for i in sorted_idx[-5:]
    ][::-1]
    
    return model_bin, features, auc_bin, df_with_probs, top_features


def _trend_flow_estimate(flow: pd.Series) -> float:
    recent = flow.dropna().tail(7)
    if len(recent) == 0:
        recent = flow.dropna().tail(3)
    if len(recent) == 0:
        return 0.0
    return float(np.median(recent.values))


def _simulate_week_ahead(df_feat: pd.DataFrame, trend_flow: float) -> Dict[str, float]:
    sim = df_feat[["Date", "FlowClipped", "StorageInterp", "StoragePos"]].copy()
    last_date = sim["Date"].iloc[-1]
    last_storage = sim["StorageInterp"].iloc[-1]

    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
    future_flows = [trend_flow] * 7
    future_storage = [last_storage]
    for f in future_flows:
        future_storage.append(future_storage[-1] + f)
    future_storage = future_storage[1:]

    future = pd.DataFrame({
        "Date": future_dates,
        "FlowClipped": future_flows,
        "StorageInterp": future_storage,
    })

    # Rebuild rolling storage position over history+future
    hist = sim[["StorageInterp"]].copy()
    combo = pd.concat([hist, future[["StorageInterp"]]], ignore_index=True)
    combo_dates = list(df_feat["Date"]) + future_dates
    tmp = pd.DataFrame({
        "Date": combo_dates,
        "StorageInterp": combo["StorageInterp"].values,
    })
    tmp["StoragePos"] = _rolling_365_storage_pos(tmp)

    combo_df = pd.concat([df_feat.drop(columns=["StoragePos"], errors="ignore"), future], ignore_index=True)
    combo_df["StoragePos"] = tmp["StoragePos"].values
    combo_df = _add_features(combo_df)
    last_row = combo_df.iloc[-1]

    feat_names = _feature_list(combo_df.columns)
    return {f: float(last_row.get(f, np.nan)) for f in feat_names}


def _heuristic_probability(storage_pos: float, flow_mean_7: float) -> float:
    if np.isnan(storage_pos): storage_pos = 0.5
    if np.isnan(flow_mean_7): flow_mean_7 = 0.0
    inj_risk = max(0.0, storage_pos - 0.65) * (1.2 if flow_mean_7 > 0 else 1.0)
    wd_risk  = max(0.0, 0.35 - storage_pos) * (1.2 if flow_mean_7 < 0 else 1.0)
    return float(np.clip(inj_risk + wd_risk, 0, 1))


def _predict_horizons(df_feat: pd.DataFrame, model_bin, features: List[str], trend_flow: float):
    last_row = df_feat.iloc[-1]
    row_t = {f: float(last_row.get(f, np.nan)) for f in features}
    feat_future = _simulate_week_ahead(df_feat, trend_flow)

    def _ensure(d: Dict[str, float]):
        X = pd.DataFrame([d])[features].astype(float)
        return X.fillna(0.0)

    if model_bin is not None:
        p_t = float(model_bin.predict_proba(_ensure(row_t))[:, 1][0])
        p_w = float(model_bin.predict_proba(_ensure(feat_future))[:, 1][0])
    else:
        p_t = _heuristic_probability(float(last_row.get("StoragePos_lag1", np.nan)), float(last_row.get("FlowMean_7", np.nan)))
        p_w = _heuristic_probability(float(feat_future.get("StoragePos_lag1", np.nan)), float(feat_future.get("FlowMean_7", np.nan)))

    # Tilt heuristic from flow direction (since we did not retain a multi-class model in v2)
    def tilt_prob_from_row(d: Dict[str, float]):
        inj = 0.6 if d.get("RecentInjection", 0) >= 1 else 0.2
        wd  = 0.6 if d.get("RecentWithdrawal", 0) >= 1 else 0.2
        s = inj + wd + 1e-9
        return float(inj/s), float(wd/s)

    inj_t, wd_t = tilt_prob_from_row(row_t)
    inj_w, wd_w = tilt_prob_from_row(feat_future)

    return (dict(p_yellow=p_t, p_inj=inj_t, p_wd=wd_t), dict(p_yellow=p_w, p_inj=inj_w, p_wd=wd_w))


def _scenario_table(df_feat: pd.DataFrame, model_bin, features: List[str], base_trend: float, deltas: List[int]):
    rows = []
    for d in deltas:
        feat_future = _simulate_week_ahead(df_feat, base_trend + d)
        X = pd.DataFrame([feat_future])[features].astype(float).fillna(0.0)
        if model_bin is not None:
            p = float(model_bin.predict_proba(X)[:, 1][0])
        else:
            p = _heuristic_probability(float(feat_future.get("StoragePos_lag1", np.nan)), float(feat_future.get("FlowMean_7", np.nan)))
        rows.append({"trend_delta": d, "p_week_yellow": p})
    return pd.DataFrame(rows)


# ------------------------------ CLI & Defaults ------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=CONFIG["input"], help="Path to DawnLight.csv")
    p.add_argument("--output_json", type=Path, default=CONFIG["output_json"], help="Path to JSON output")
    p.add_argument("--output_csv", type=Path, default=CONFIG["output_csv"], help="Path to CSV snapshot")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--perm_importance", action="store_true", help="(Ignored in v2) placeholder for compatibility")
    p.add_argument("--dry-run", action="store_true", help="resolve key paths then exit")
    return p.parse_args()


def main():
    args = _parse_args()

    logger.info(
        "Args parsed | DRY_RUN=%s | input=%s | output_json=%s | output_csv=%s",
        args.dry_run,
        args.input,
        args.output_json,
        args.output_csv,
    )
    input_path = _resolve_path(args.input)
    json_path = _resolve_path(args.output_json)
    csv_path = _resolve_path(args.output_csv)

    print(f"[dry] ROOT={ROOT}")
    print(f"[dry] input={input_path}")
    print(f"[dry] output_json={json_path}")
    print(f"[dry] output_csv={csv_path}")
    if args.dry_run:
        return

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    # Load & features
    df = _read_csv(input_path)
    df_feat_base = _add_features(df)

    # Trend based on CLIPPED flows
    trend_flow = _trend_flow_estimate(df_feat_base["FlowClipped"]) if df_feat_base["FlowClipped"].notna().any() else 0.0
    last_storage = float(df_feat_base["StorageInterp"].iloc[-1]) if pd.notna(df_feat_base["StorageInterp"].iloc[-1]) else None

    # Fit model (time-aware calibrated logistic) and get historical probabilities
    model_bin, features, auc_bin, df_feat, top_features = _fit_models(df_feat_base)
    if df_feat is None: # Happens if _fit_models returns early
        df_feat = df_feat_base
        top_features = []

    model_kind = "heuristic" if model_bin is None else "GradientBoosting"

    # Predict horizons
    pred_t, pred_w = _predict_horizons(df_feat, model_bin, features, trend_flow)

    # Scenario table (week ahead) across deltas (k = thousand units)
    deltas = [-500_000, -250_000, 0, 250_000, 500_000]
    scen_df = _scenario_table(df_feat, model_bin, features, trend_flow, deltas)

    # Build result (with joint probabilities)
    p_t = pred_t["p_yellow"]; inj_t = pred_t["p_inj"]; wd_t = pred_t["p_wd"]
    p_w = pred_w["p_yellow"]; inj_w = pred_w["p_inj"]; wd_w = pred_w["p_wd"]

    res = ForecastResult(
        date_asof=str(df_feat["Date"].iloc[-1].date()),
        model_kind=model_kind,
        auc_binary=float(auc_bin) if auc_bin is not None else None,
        n_events=int(df_feat["YellowTurnOn"].fillna(0).sum()),
        trend_flow=float(trend_flow),
        last_storage_total=last_storage,
        prob_tomorrow_yellow_turn_on=float(p_t),
        prob_tomorrow_inj_turn_on=float(p_t*inj_t),
        prob_tomorrow_wd_turn_on=float(p_t*wd_t),
        prob_week_yellow_turn_on=float(p_w),
        prob_week_inj_turn_on=float(p_w*inj_w),
        prob_week_wd_turn_on=float(p_w*wd_w),
        top_features=top_features,
    )

    # Console output
    print("\n=== Dawn Yellow Light Forecast (Pro, Auto-Path) – v6 ===")
    print(f"Data through:               {res.date_asof}")
    print(f"Model:                      {res.model_kind} (events in history: {res.n_events})")
    if res.auc_binary is not None:
        print(f"Binary AUC (holdout):       {res.auc_binary:.3f}")
    print(f"Recent trend flow:          {res.trend_flow:.0f}  (+injection / -withdrawal)")
    if res.last_storage_total is not None:
        print(f"Last Storage (interp):      {res.last_storage_total:.0f}")

    print("\n--- Tomorrow ---")
    print(f"P(Yellow Turn On)           {100*res.prob_tomorrow_yellow_turn_on:5.1f}%")
    print(f"P(YellowInj Turn On)        {100*res.prob_tomorrow_inj_turn_on:5.2f}%")
    print(f"P(YellowWd Turn On)         {100*res.prob_tomorrow_wd_turn_on:5.2f}%")

    print("\n--- In 1 Week (trend persists) ---")
    print(f"P(Yellow Turn On)           {100*res.prob_week_yellow_turn_on:5.1f}%")
    print(f"P(YellowInj Turn On)        {100*res.prob_week_inj_turn_on:5.2f}%")
    print(f"P(YellowWd Turn On)         {100*res.prob_week_wd_turn_on:5.2f}%")

    # Scenario table
    print("\n--- 1-Week Scenario (flow trend deltas) ---")
    for _, r in scen_df.iterrows():
        print(f"delta={int(r['trend_delta']):>8} -> P(Yellow Turn On)={100*r['p_week_yellow']:5.1f}%")

    # Historical Lookback Section
    print("\n--- Historical Lookback (In-Sample Accuracy) ---")
    if 'PredictedProb' in df_feat.columns:
        # YellowInj
        inj_events = df_feat[(df_feat['YellowTurnOn'] == 1) & (df_feat['TypeCode'] == 1)].tail(3)
        print("Last 3 YellowInj Turn-On Events:")
        if not inj_events.empty:
            for _, row in inj_events.iterrows():
                prob_str = f"{row['PredictedProb']*100:.1f}%" if pd.notna(row['PredictedProb']) else "N/A"
                print(f"  - {row['Date'].date()}: Model chance was {prob_str}")
        else:
            print("  - None found in history.")

        # YellowWd
        wd_events = df_feat[(df_feat['YellowTurnOn'] == 1) & (df_feat['TypeCode'] == 2)].tail(3)
        print("\nLast 3 YellowWd Turn-On Events:")
        if not wd_events.empty:
            for _, row in wd_events.iterrows():
                prob_str = f"{row['PredictedProb']*100:.1f}%" if pd.notna(row['PredictedProb']) else "N/A"
                print(f"  - {row['Date'].date()}: Model chance was {prob_str}")
        else:
            print("  - None found in history.")
    else:
        print("  - Model was not trained (insufficient data), lookback unavailable.")

    # Top Features Section
    print("\n--- Top 5 Predictive Features (Permutation Importance) ---")
    if res.top_features:
        for feature, importance in res.top_features:
            print(f"  - {feature:<20} | Importance: {importance:.4f}")
    else:
        print("  - Feature importance not calculated (insufficient data).")


    # Save JSON/CSV
    out_json = json_path
    out_csv  = csv_path
    try:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res.__dict__, f, indent=2)
        print(f"\nSaved JSON -> {out_json}")
    except Exception as e:
        print(f"Could not save JSON: {e}")

    try:
        row = {
            "asof": res.date_asof,
            "model": res.model_kind,
            "auc_binary": res.auc_binary,
            "n_events": res.n_events,
            "trend_flow": res.trend_flow,
            "last_storage": res.last_storage_total,
            "p_tmr_yellow_turn_on": res.prob_tomorrow_yellow_turn_on,
            "p_tmr_inj_turn_on": res.prob_tomorrow_inj_turn_on,
            "p_tmr_wd_turn_on": res.prob_tomorrow_wd_turn_on,
            "p_wk_yellow_turn_on": res.prob_week_yellow_turn_on,
            "p_wk_inj_turn_on": res.prob_week_inj_turn_on,
            "p_wk_wd_turn_on": res.prob_week_wd_turn_on,
        }
        pd.DataFrame([row]).to_csv(out_csv, index=False)
        print(f"Saved CSV -> {out_csv}")
    except Exception as e:
        print(f"Could not save CSV: {e}")


if __name__ == "__main__":
    main()

