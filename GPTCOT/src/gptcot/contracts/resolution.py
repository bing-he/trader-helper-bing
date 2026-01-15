"""Contract resolution and price snapping logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from gptcot.contracts.month_label import parse_month_label

INVALID_FORWARD_CURVE = "missing_forward_curve_date"
INVALID_LABEL = "unparseable_frontmonth_label"
INVALID_CONTRACT_COLUMN = "missing_contract_month_column"
INVALID_ENTRY_PRICE = "missing_entry_price"
INVALID_EXIT_PRICE = "missing_exit_price"
INVALID_ENTRY_ZERO = "entry_price_zero"


@dataclass(frozen=True)
class ContractResolution:
    """Details for a resolved contract at a given horizon."""

    fc_snap_date: pd.Timestamp
    frontmonth_label: str
    target_contract_month: str


def snap_forward_curve_date(
    forward_curve: pd.DataFrame, horizon_date: pd.Timestamp
) -> Optional[pd.Timestamp]:
    """Snap backward to the latest forward-curve date <= horizon_date."""

    eligible = forward_curve[forward_curve["Date"] <= horizon_date]
    if eligible.empty:
        return None
    return pd.to_datetime(eligible.iloc[-1]["Date"])


def resolve_contract(
    forward_curve: pd.DataFrame, horizon_date: pd.Timestamp
) -> tuple[Optional[ContractResolution], Optional[str]]:
    """Resolve the target contract month for a horizon date."""

    snap_date = snap_forward_curve_date(forward_curve, horizon_date)
    if snap_date is None:
        return None, INVALID_FORWARD_CURVE

    snapped_rows = forward_curve[forward_curve["Date"] == snap_date]
    row = snapped_rows.iloc[-1]
    try:
        contract_month = parse_month_label(str(row["FrontMonth_Label"]))
    except ValueError:
        return None, INVALID_LABEL

    resolution = ContractResolution(
        fc_snap_date=snap_date,
        frontmonth_label=str(row["FrontMonth_Label"]),
        target_contract_month=contract_month,
    )
    return resolution, None


def _snap_entry_date(prices: pd.DataFrame, signal_date: pd.Timestamp) -> Optional[pd.Timestamp]:
    eligible = prices[prices["TradeDate"] >= signal_date]
    if eligible.empty:
        return None
    return pd.to_datetime(eligible.iloc[0]["TradeDate"])


def _snap_exit_date(prices: pd.DataFrame, horizon_date: pd.Timestamp) -> Optional[pd.Timestamp]:
    eligible = prices[prices["TradeDate"] <= horizon_date]
    if eligible.empty:
        return None
    return pd.to_datetime(eligible.iloc[-1]["TradeDate"])


def _search_price_window(
    rows: pd.DataFrame, contract_column: str, direction: str, max_steps: int
) -> tuple[Optional[pd.Timestamp], Optional[float]]:
    """
    Search within a limited window of trading days for a non-NaN price.

    Args:
        rows: Filtered price rows (already limited to direction).
        contract_column: Column to read.
        direction: 'forward' or 'backward'.
        max_steps: Max number of trading-day hops (inclusive of anchor).
    """

    window = rows.head(max_steps + 1) if direction == "forward" else rows.tail(max_steps + 1)
    iterable = window.iterrows()
    if direction == "backward":
        iterable = reversed(list(window.iterrows()))
    for _, row in iterable:
        trade_date = pd.to_datetime(row["TradeDate"])
        price = row.get(contract_column)
        if pd.notna(price):
            return trade_date, float(price)
    return None, None


def snap_prices(
    prices: pd.DataFrame,
    contract_column: str,
    signal_date: pd.Timestamp,
    horizon_date: pd.Timestamp,
    max_price_snap_days: int,
) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], Optional[float], Optional[float], Optional[str]]:
    """Resolve entry/exit dates and prices with snapping rules."""

    if contract_column not in prices.columns:
        return None, None, None, None, INVALID_CONTRACT_COLUMN

    entry_anchor = _snap_entry_date(prices, signal_date)
    if entry_anchor is None:
        return None, None, None, None, INVALID_ENTRY_PRICE

    exit_anchor = _snap_exit_date(prices, horizon_date)
    if exit_anchor is None:
        return entry_anchor, None, None, None, INVALID_EXIT_PRICE

    forward_rows = prices[prices["TradeDate"] >= entry_anchor]
    entry_date, entry_price = _search_price_window(
        forward_rows, contract_column, "forward", max_price_snap_days
    )
    if entry_price is None:
        return entry_anchor, exit_anchor, None, None, INVALID_ENTRY_PRICE
    if entry_price == 0:
        return entry_date, exit_anchor, entry_price, None, INVALID_ENTRY_ZERO

    backward_rows = prices[prices["TradeDate"] <= exit_anchor]
    exit_date, exit_price = _search_price_window(
        backward_rows, contract_column, "backward", max_price_snap_days
    )
    if exit_price is None:
        return entry_date, exit_anchor, entry_price, None, INVALID_EXIT_PRICE

    return entry_date, exit_date, entry_price, exit_price, None
