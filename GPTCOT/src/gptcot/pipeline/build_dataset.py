"""Pipeline to build the CoT forward returns dataset."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gptcot.config import PipelineConfig
from gptcot.contracts import (
    INVALID_CONTRACT_COLUMN,
    INVALID_ENTRY_PRICE,
    INVALID_ENTRY_ZERO,
    INVALID_EXIT_PRICE,
    INVALID_FORWARD_CURVE,
    INVALID_LABEL,
    resolve_contract,
    snap_prices,
)
from gptcot.features import SERIES_COLUMNS, compute_cot_features, feature_columns
from gptcot.io import (
    absolute_history_path,
    cot_path,
    forward_curve_path,
    load_absolute_history,
    load_cot,
    load_forward_curve,
    output_csv_path,
    output_parquet_path,
)


def _fmt_date(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _prepare_cot_dataframe(config: PipelineConfig) -> pd.DataFrame:
    cot_df = load_cot(cot_path(config.info_dir))
    return compute_cot_features(cot_df, min_periods=config.min_periods)


def _prepare_forward_curve(config: PipelineConfig) -> pd.DataFrame:
    return load_forward_curve(forward_curve_path(config.info_dir))


def _prepare_absolute_history(config: PipelineConfig) -> pd.DataFrame:
    return load_absolute_history(absolute_history_path(config.info_dir))


def _base_row(cot_row: pd.Series) -> Dict[str, object]:
    row = {
        "cot_date": _fmt_date(cot_row["Date"]),
        "Total_OI": cot_row["Total_OI"],
        "Total_MM_Net": cot_row["Total_MM_Net"],
        "Total_Prod_Net": cot_row["Total_Prod_Net"],
        "Total_Swap_Net": cot_row["Total_Swap_Net"],
    }
    for features in feature_columns().values():
        for col in features:
            row[col] = cot_row.get(col, np.nan)
    return row


def _compute_return(entry_price: float, exit_price: float) -> Tuple[float, float]:
    abs_change = exit_price - entry_price
    pct_change = (exit_price / entry_price) - 1
    return abs_change, pct_change


FWD_COLUMNS: List[str] = [f"FWD_{i:02d}" for i in range(12)]


def _compute_strip_price(row: Optional[pd.Series]) -> Optional[float]:
    if row is None:
        return None
    prices = np.array([pd.to_numeric(row.get(col, np.nan), errors="coerce") for col in FWD_COLUMNS], dtype=float)
    if np.isnan(prices).all():
        return None
    return float(np.nanmean(prices))


def _snap_curve_row(forward_curve: pd.DataFrame, target_date: pd.Timestamp, direction: str) -> Optional[pd.Series]:
    if "Date" not in forward_curve.columns:
        return None
    eligible = forward_curve[forward_curve["Date"] >= target_date] if direction == "forward" else forward_curve[
        forward_curve["Date"] <= target_date
    ]
    if eligible.empty:
        return None
    return eligible.iloc[0] if direction == "forward" else eligible.iloc[-1]


def build_dataset(config: PipelineConfig, logger: logging.Logger | None = None) -> pd.DataFrame:
    """Run the full pipeline and return the assembled dataframe."""

    log = logger or logging.getLogger(__name__)
    forward_curve = _prepare_forward_curve(config)
    absolute_history = _prepare_absolute_history(config)
    cot_df = _prepare_cot_dataframe(config)

    results: List[Dict[str, object]] = []
    horizons = config.horizons

    for _, cot_row in cot_df.iterrows():
        signal_date = pd.to_datetime(cot_row["Date"])
        base = _base_row(cot_row)

        for horizon in horizons:
            horizon_date = signal_date + timedelta(days=int(horizon))
            entry_curve_row = _snap_curve_row(forward_curve, signal_date, "forward")
            exit_curve_row = _snap_curve_row(forward_curve, horizon_date, "backward")
            entry_strip_price = _compute_strip_price(entry_curve_row)
            exit_strip_price = _compute_strip_price(exit_curve_row)
            valid_entry_strip = entry_strip_price is not None and not pd.isna(entry_strip_price) and entry_strip_price != 0
            valid_exit_strip = exit_strip_price is not None and not pd.isna(exit_strip_price)
            strip_pct_change = (
                (exit_strip_price - entry_strip_price) / entry_strip_price if valid_entry_strip and valid_exit_strip else None
            )
            record: Dict[str, object] = {
                **base,
                "horizon_days": int(horizon),
                "horizon_date": _fmt_date(horizon_date),
                "fc_snap_date": None,
                "frontmonth_label": None,
                "target_contract_month": None,
                "entry_date": None,
                "exit_date": None,
                "entry_price": None,
                "exit_price": None,
                "abs_change": None,
                "pct_change": None,
                "entry_strip_price": entry_strip_price,
                "exit_strip_price": exit_strip_price,
                "strip_pct_change": strip_pct_change,
                "is_valid": False,
                "invalid_reason": "",
            }

            resolution, reason = resolve_contract(forward_curve, horizon_date)
            if reason is not None or resolution is None:
                record["invalid_reason"] = reason or INVALID_FORWARD_CURVE
                results.append(record)
                continue

            record["fc_snap_date"] = _fmt_date(resolution.fc_snap_date)
            record["frontmonth_label"] = resolution.frontmonth_label
            record["target_contract_month"] = resolution.target_contract_month

            entry_date, exit_date, entry_price, exit_price, price_reason = snap_prices(
                absolute_history,
                resolution.target_contract_month,
                signal_date,
                horizon_date,
                config.max_price_snap_days,
            )

            if price_reason is not None:
                record["entry_date"] = _fmt_date(entry_date)
                record["exit_date"] = _fmt_date(exit_date)
                record["entry_price"] = entry_price
                record["exit_price"] = exit_price
                record["invalid_reason"] = price_reason
                results.append(record)
                continue

            if entry_price is None or exit_price is None:
                record["invalid_reason"] = INVALID_ENTRY_PRICE
                results.append(record)
                continue

            if entry_price == 0:
                record["invalid_reason"] = INVALID_ENTRY_ZERO
                results.append(record)
                continue

            abs_change, pct_change = _compute_return(entry_price, exit_price)

            record.update(
                {
                    "entry_date": _fmt_date(entry_date),
                    "exit_date": _fmt_date(exit_date),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "abs_change": abs_change,
                    "pct_change": pct_change,
                    "is_valid": True,
                    "invalid_reason": "",
                }
            )
            results.append(record)

    df = pd.DataFrame(results)
    df = df.sort_values(["cot_date", "horizon_days"]).reset_index(drop=True)
    return df


def write_outputs(df: pd.DataFrame, config: PipelineConfig, logger: logging.Logger | None = None) -> None:
    """Write CSV (and optional parquet) outputs."""

    log = logger or logging.getLogger(__name__)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_csv_path(config.output_dir)
    df.to_csv(csv_path, index=False)
    log.info("Wrote CSV to %s", csv_path)
    if config.write_parquet:
        parquet_path = output_parquet_path(config.output_dir)
        df.to_parquet(parquet_path, index=False)
        log.info("Wrote Parquet to %s", parquet_path)


def summarize(df: pd.DataFrame) -> Dict[str, object]:
    """Compute summary statistics for console reporting."""

    total_rows = len(df)
    valid_rows = int(df["is_valid"].sum()) if "is_valid" in df.columns else 0
    invalid_rows = total_rows - valid_rows

    reason_counts = (
        df.loc[df["invalid_reason"] != "", "invalid_reason"]
        .value_counts()
        .to_dict()
        if "invalid_reason" in df.columns
        else {}
    )

    unique_cot_dates = df["cot_date"].nunique() if "cot_date" in df.columns else 0
    horizon_count = df["horizon_days"].nunique() if "horizon_days" in df.columns else 0
    horizons_processed = unique_cot_dates * horizon_count

    return {
        "cot_rows": unique_cot_dates,
        "horizons_processed": horizons_processed,
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "invalid_reasons": reason_counts,
    }
