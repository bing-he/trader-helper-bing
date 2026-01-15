"""Forward curve construction and plotting helpers for the market report."""

from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

from gptcot.io import load_absolute_history, load_forward_curve
from gptcot.io.paths import absolute_history_path, forward_curve_path

LOOKBACK_DAYS: Sequence[int] = (0, 7, 14, 28, 63)
DEFAULT_MAX_MONTHS = 24


def _available_contract_columns(columns: Iterable[str], max_months: int) -> List[pd.Timestamp]:
    """Return sorted contract-month columns (as Timestamps)."""

    contract_cols: List[pd.Timestamp] = []
    for col in columns:
        if col == "TradeDate":
            continue
        try:
            ts = pd.to_datetime(col)
        except Exception:
            continue
        contract_cols.append(ts)
    return sorted(contract_cols)


def _parse_front_month(label: str) -> pd.Timestamp:
    """Convert labels like 'Jan-2026' to month-start Timestamp."""

    return pd.to_datetime(label, format="%b-%Y")


def _fill_tail_from_prior(curve: pd.Series, prior_curve: pd.Series | None) -> pd.Series:
    """Fill missing tail points using the prior day's shape when available."""

    if prior_curve is None or prior_curve.empty:
        return curve
    filled = curve.copy()
    for idx in filled.index:
        if pd.isna(filled.loc[idx]) and idx in prior_curve.index:
            prior_val = prior_curve.loc[idx]
            if pd.notna(prior_val):
                filled.loc[idx] = prior_val
    filled = filled.sort_index()
    for i in range(1, len(filled)):
        if pd.isna(filled.iloc[i]):
            prev_val = filled.iloc[i - 1]
            if pd.isna(prev_val):
                continue
            prev_idx = filled.index[i - 1]
            curr_idx = filled.index[i]
            if prev_idx in prior_curve.index and curr_idx in prior_curve.index:
                carry = prior_curve.loc[curr_idx] - prior_curve.loc[prev_idx]
                if pd.notna(carry):
                    filled.iloc[i] = prev_val + carry
    return filled


def load_absolute_curve_history(info_dir: Path) -> pd.DataFrame:
    """Load HenryHub_Absolute_History.csv as a TradeDate-indexed dataframe."""

    path = absolute_history_path(info_dir)
    df = load_absolute_history(path)
    return df.set_index("TradeDate").sort_index()


def load_forward_curve_history(info_dir: Path) -> pd.DataFrame:
    """Load HenryForwardCurve.csv with parsed dates."""

    path = forward_curve_path(info_dir)
    df = load_forward_curve(path)
    return df


def latest_front_label(info_dir: Path) -> Optional[str]:
    """Return the latest FrontMonth_Label as a Mon-YY string."""

    df = load_forward_curve_history(info_dir)
    if df.empty:
        return None
    latest = df.loc[df["Date"].idxmax()]
    label = latest.get("FrontMonth_Label")
    try:
        dt = pd.to_datetime(label, format="%b-%Y")
        return dt.strftime("%b-%y")
    except Exception:
        return str(label) if label else None


def build_forward_curve_table_absolute(
    absolute_history: pd.DataFrame,
    *,
    lookbacks: Sequence[int] = LOOKBACK_DAYS,
    max_months: int = DEFAULT_MAX_MONTHS,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Construct forward curves from absolute contract-month history."""

    log = logger or logging.getLogger(__name__)
    if "TradeDate" in absolute_history.columns:
        history = absolute_history.set_index("TradeDate")
    else:
        history = absolute_history.copy()
    history = history.sort_index()
    contract_months = _available_contract_columns(history.columns, max_months)
    if not contract_months:
        raise ValueError("No contract-month columns found in absolute history.")

    latest_date = history.index.max()
    selected_dates: List[pd.Timestamp] = []
    for days in lookbacks:
        target = latest_date - pd.Timedelta(days=days)
        subset = history.loc[history.index <= target]
        if subset.empty:
            log.warning("No forward curve available on/before %s (lookback %s)", target.date(), days)
            continue
        selected_dates.append(subset.index[-1])

    curves: List[pd.Series] = []
    labels: List[str] = []
    prior_curve: pd.Series | None = None
    for as_of in sorted(set(selected_dates)):
        row = history.loc[as_of]
        prompt_month = (as_of + pd.offsets.MonthBegin(1)).normalize()
        try:
            prompt_idx = next(i for i, m in enumerate(contract_months) if m >= prompt_month)
        except StopIteration:
            log.warning("No contract months on/after prompt %s for %s", prompt_month.date(), as_of.date())
            continue
        usable_months = contract_months[prompt_idx : prompt_idx + max_months]
        if not usable_months:
            log.warning("Unable to build curve for %s; no usable months.", as_of.date())
            continue
        curve = pd.Series(
            {m: row.get(m.strftime("%Y-%m-%d"), np.nan) for m in usable_months},
            dtype="float64",
        )
        curve = _fill_tail_from_prior(curve, prior_curve)
        prior_curve = curve
        curves.append(curve)
        labels.append(as_of.strftime("%Y-%m-%d"))

    if not curves:
        return pd.DataFrame()

    table = pd.concat(curves, axis=1)
    table.columns = labels
    return table


def build_forward_curve_table_rolling(
    rolling_history: pd.DataFrame,
    *,
    lookbacks: Sequence[int] = LOOKBACK_DAYS,
    max_months: int = DEFAULT_MAX_MONTHS,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Fallback: construct forward curves from rolling forward curve table."""

    log = logger or logging.getLogger(__name__)
    history = rolling_history.sort_values("Date").reset_index(drop=True)
    fwd_cols = [col for col in history.columns if col.startswith("FWD_")]
    if not fwd_cols:
        raise ValueError("No FWD columns found in rolling forward curve history.")

    latest_date = history["Date"].max()
    selected_dates: List[pd.Timestamp] = []
    for days in lookbacks:
        target = latest_date - pd.Timedelta(days=days)
        subset = history.loc[history["Date"] <= target]
        if subset.empty:
            log.warning("No rolling forward curve available on/before %s (lookback %s)", target.date(), days)
            continue
        selected_dates.append(subset["Date"].iloc[-1])

    curves: List[pd.Series] = []
    labels: List[str] = []
    prior_curve: pd.Series | None = None
    for as_of in sorted(set(selected_dates)):
        row = history.loc[history["Date"] == as_of].iloc[0]
        prompt_month = _parse_front_month(row["FrontMonth_Label"])
        months = [prompt_month + pd.DateOffset(months=i) for i in range(len(fwd_cols))]
        usable_months = months[:max_months]
        curve = pd.Series({m: row[f"FWD_{i:02d}"] for i, m in enumerate(usable_months)}, dtype="float64")
        curve = _fill_tail_from_prior(curve, prior_curve)
        prior_curve = curve
        curves.append(curve)
        labels.append(as_of.strftime("%Y-%m-%d"))

    if not curves:
        return pd.DataFrame()

    table = pd.concat(curves, axis=1)
    table.columns = labels
    return table


def build_forward_curve_table(
    info_dir: Path,
    *,
    lookbacks: Sequence[int] = LOOKBACK_DAYS,
    max_months: int = DEFAULT_MAX_MONTHS,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Build forward curve table preferring absolute history; fallback to rolling if needed.
    """

    log = logger or logging.getLogger(__name__)
    try:
        abs_df = load_absolute_curve_history(info_dir)
        table = build_forward_curve_table_absolute(
            abs_df.reset_index(),
            lookbacks=lookbacks,
            max_months=max_months,
            logger=log,
        )
        if not table.empty:
            return table
        log.warning("Absolute history produced empty forward-curve table; attempting rolling fallback.")
    except FileNotFoundError:
        log.warning("HenryHub_Absolute_History.csv missing; falling back to rolling forward curve.")
    except Exception as exc:  # pylint: disable=broad-except
        log.warning("Absolute history unusable (%s); falling back to rolling forward curve.", exc)

    rolling_df = load_forward_curve_history(info_dir)
    return build_forward_curve_table_rolling(
        rolling_df, lookbacks=lookbacks, max_months=max_months, logger=log
    )


def normalize_forward_curves(table: pd.DataFrame) -> pd.DataFrame:
    """Normalize each forward curve to its prompt month (prompt=100)."""

    normalized = {}
    for col in table.columns:
        series = table[col]
        anchor_candidates = series.dropna()
        if anchor_candidates.empty:
            normalized[col] = series
            continue
        anchor = anchor_candidates.iloc[0]
        normalized[col] = (series / anchor) * 100
    return pd.DataFrame(normalized)


def _format_contract_labels(index: pd.Index) -> List[str]:
    """Format contract month labels for plotting."""

    return [pd.to_datetime(ts).strftime("%b-%y") for ts in index]


def plot_forward_curves(
    table: pd.DataFrame,
    *,
    output_dir: Path,
    filename: str,
    title: str,
    ylabel: str,
    normalize: bool = False,
) -> str:
    """Plot forward curves and save to file."""

    if table.empty:
        raise ValueError("Forward curve table is empty; cannot plot.")
    data = normalize_forward_curves(table) if normalize else table
    colors = cm.get_cmap("viridis", len(data.columns))
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, col in enumerate(data.columns):
        ax.plot(data.index, data[col], label=col, color=colors(i))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Contract Month")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(data.index)
    ax.set_xticklabels(_format_contract_labels(data.index), rotation=45, ha="right")
    ax.legend(title="As of")
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return filename


def generate_forward_curve_charts(
    table: pd.DataFrame, *, output_dir: Path
) -> Tuple[str, str]:
    """
    Generate absolute and normalized forward curve comparison charts.

    Both charts trim leading rows with missing values so all curves anchor on
    the same front month; the normalized chart then indexes to 100.
    """

    common = table.dropna(how="any")
    if common.empty:
        raise ValueError("Forward curve table has no common contract months to plot.")

    abs_name = plot_forward_curves(
        common,
        output_dir=output_dir,
        filename="forward_curve_comparison.png",
        title="NYMEX Henry Hub Forward Curve Comparison",
        ylabel="Price ($/MMBtu)",
        normalize=False,
    )

    rel_name = plot_forward_curves(
        common,
        output_dir=output_dir,
        filename="forward_curve_relative.png",
        title="NYMEX Henry Hub Relative Value (Normalized to Prompt Month)",
        ylabel="Relative Price (Prompt Month = 100)",
        normalize=True,
    )
    return abs_name, rel_name
