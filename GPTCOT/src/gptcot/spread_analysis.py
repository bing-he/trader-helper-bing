"""NYMEX spread chart generation for the market report."""

from __future__ import annotations

import logging
import calendar
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import icepython as ice
except ImportError:  # pragma: no cover - optional dependency
    ice = None
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

YEARS_OF_HISTORY = 10
BASE_SYMBOL = "HNG"
API_SUFFIX = "-IUS"

MONTH_CODES: Dict[int, str] = {
    1: "F",
    2: "G",
    3: "H",
    4: "J",
    5: "K",
    6: "M",
    7: "N",
    8: "Q",
    9: "U",
    10: "V",
    11: "X",
    12: "Z",
}
MONTH_NAMES_TO_NUM: Dict[str, int] = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def parse_contract_string(contract_str: str) -> Optional[datetime]:
    """Parse a Mon-YY string (e.g., 'Aug-25') into a datetime."""

    try:
        month_str, year_str = contract_str.strip().split("-")
        month = MONTH_NAMES_TO_NUM[month_str.title()]
        year = 2000 + int(year_str)
        return datetime(year, month, 1)
    except Exception:
        return None


def _spread_symbol(contract: str) -> str:
    dt = parse_contract_string(contract)
    if dt is None:
        raise ValueError(f"Invalid contract string: {contract}")
    return f"{BASE_SYMBOL} {MONTH_CODES[dt.month]}{str(dt.year)[-2:]}"


def fetch_historical_data(symbols: List[str], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Fetch daily settles for the given symbols."""

    if ice is None:
        return None
    try:
        ts_data = ice.get_timeseries(symbols, ["Settle"], "D", start_date, end_date)
        if not ts_data or len(ts_data) < 2:
            return None
        header = [h.replace(".Settle", "") for h in ts_data[0]]
        df = pd.DataFrame(ts_data[1:], columns=header)
        df["Time"] = pd.to_datetime(df["Time"])
        df = df.set_index("Time").drop_duplicates()
        return df.apply(pd.to_numeric, errors="coerce")
    except Exception:
        return None


def fetch_live_quote(contract1: str, contract2: str) -> Optional[float]:
    """Fetch live bid/ask for the spread and return midpoint."""

    if ice is None:
        return None
    date1 = parse_contract_string(contract1)
    date2 = parse_contract_string(contract2)
    if not date1 or not date2:
        return None

    leg1 = _spread_symbol(contract1)
    leg2 = _spread_symbol(contract2).replace(" ", "")
    spread_symbol = f"{leg1}:{leg2}{API_SUFFIX}"
    try:
        quote_data = ice.get_quotes([spread_symbol], ["bid", "Ask"])
        if not quote_data or len(quote_data) < 2 or "<NotEnt>" in quote_data[1]:
            return None
        bid = pd.to_numeric(quote_data[1][1], errors="coerce")
        ask = pd.to_numeric(quote_data[1][2], errors="coerce")
        if pd.notna(bid) and pd.notna(ask):
            return float((bid + ask) / 2)
    except Exception:
        return None
    return None


def normalize_contract_pair(contract1: str, contract2: str) -> Tuple[str, str]:
    """Ensure the first contract is earlier than the second."""

    date1 = parse_contract_string(contract1)
    date2 = parse_contract_string(contract2)
    if date1 is None or date2 is None:
        return contract1, contract2
    if date1 > date2:
        return contract2, contract1
    return contract1, contract2


def generate_plot(
    df: pd.DataFrame,
    contract1: str,
    contract2: str,
    live_mark: Optional[float],
    output_path: Path,
    cutoff_offset: int,
) -> str:
    """Generate and save the historical spread comparison chart."""

    fig, ax = plt.subplots(figsize=(18, 10))
    latest_year = df.columns.max()
    colors = plt.cm.tab10.colors
    last_valid_day_current_year = df[latest_year].last_valid_index()
    expiry_date = parse_contract_string(contract1)

    for i, year in enumerate(sorted(df.columns)):
        series = df[year]
        label = year
        if year == latest_year and live_mark is not None:
            label = f"{year} (Live Mark: {live_mark:.3f})"
        elif last_valid_day_current_year is not None:
            historical_value = df.loc[last_valid_day_current_year, year]
            if pd.notna(historical_value):
                label = f"{year} ({historical_value:.3f})"

        is_latest_year = year == latest_year
        ax.plot(
            series.index,
            series,
            label=label,
            color="red" if is_latest_year else colors[i % len(colors)],
            linewidth=3.0 if is_latest_year else 1.5,
            zorder=10 if is_latest_year else 5,
            alpha=1.0 if is_latest_year else 0.8,
        )

    ax.set_title(f"Historical Spread Analysis: {contract1} vs {contract2}", fontsize=20, pad=20, weight="bold")
    ax.set_ylabel("Spread Value (USD/MMBtu)", fontsize=14)
    ax.set_xlabel("Days from Expiry", fontsize=14)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6)
    ax.legend(title="Contract Year (Value at Current Date)", fontsize=11, loc="upper left")

    tick_locations: List[int] = []
    tick_labels: List[str] = []
    if expiry_date is not None:
        expiry = _contract_expiry(expiry_date)
        for month_offset in range(13, -1, -1):
            tick_date = (expiry - relativedelta(months=month_offset)).replace(day=1)
            tick_offset = (tick_date - expiry).days
            if df.index.min() <= tick_offset <= df.index.max():
                tick_locations.append(tick_offset)
                tick_labels.append(tick_date.strftime("%b"))
    if tick_locations:
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(tick_labels, rotation=0, fontsize=10)
    ax.set_xlim(df.index.min(), cutoff_offset)
    fig.tight_layout(pad=2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path.name


def _contract_expiry(contract_dt: datetime) -> pd.Timestamp:
    """Approximate contract expiry as month-end minus five trading days."""

    ts = pd.Timestamp(contract_dt)
    month_end = ts + pd.offsets.MonthEnd(0)
    return month_end - pd.offsets.BDay(5)


def _cutoff_offset(contract1: str) -> int:
    """Compute day offset for end of month preceding the earlier contract."""

    dt = parse_contract_string(contract1)
    if dt is None:
        return 0
    expiry_date = _contract_expiry(dt)
    prev_month = dt.month - 1 or 12
    prev_year = dt.year if dt.month > 1 else dt.year - 1
    last_day = calendar.monthrange(prev_year, prev_month)[1]
    cutoff_date = pd.Timestamp(prev_year, prev_month, last_day)
    return int((cutoff_date - expiry_date).days)


def _prepare_spread_history(contract1: str, contract2: str, years: int) -> Dict[int, pd.Series]:
    """Fetch historical spreads by subtracting leg settles and align by days to expiry."""

    base_date1 = parse_contract_string(contract1)
    base_date2 = parse_contract_string(contract2)
    if base_date1 is None or base_date2 is None:
        return {}

    historical_spreads: Dict[int, pd.Series] = {}
    for i in range(years + 1):
        d1 = base_date1 - relativedelta(years=i)
        d2 = base_date2 - relativedelta(years=i)
        symbol1 = f"{_spread_symbol(_fmt_contract(d1))}{API_SUFFIX}"
        symbol2 = f"{_spread_symbol(_fmt_contract(d2))}{API_SUFFIX}"

        expiry_date = _contract_expiry(d1)
        end_date = expiry_date.date()
        start_date = (expiry_date - relativedelta(years=1)).date()

        df_hist = fetch_historical_data(
            [symbol1, symbol2],
            start_date.isoformat(),
            end_date.isoformat(),
        )
        if df_hist is None or df_hist.empty:
            continue
        if symbol1 not in df_hist.columns or symbol2 not in df_hist.columns:
            continue

        spread = df_hist[symbol1] - df_hist[symbol2]
        if spread.empty:
            continue
        spread = spread.sort_index()
        spread.index = (pd.to_datetime(spread.index) - expiry_date).days
        historical_spreads[d1.year] = spread

    return historical_spreads


def _pivot_by_year(spreads: Dict[int, pd.Series], cutoff_offset: int) -> pd.DataFrame:
    """Pivot spread history by contract year, preserving day offsets up to cutoff."""

    if not spreads:
        return pd.DataFrame()

    df = pd.DataFrame(spreads)
    df = df.sort_index()
    df = df.loc[df.index <= cutoff_offset]
    if df.empty:
        return pd.DataFrame()
    df = df[sorted(df.columns)]

    latest_year = df.columns.max()
    last_valid_idx = df[latest_year].last_valid_index()
    df = df.ffill()
    if last_valid_idx is not None:
        df.loc[df.index > last_valid_idx, latest_year] = np.nan
    return df


def generate_spread_chart(contract1: str, contract2: str, output_dir: Path, logger: Optional[logging.Logger] = None) -> Optional[str]:
    """Generate a spread chart using ICE data; returns filename or None on failure."""

    log = logger or logging.getLogger(__name__)
    contract1, contract2 = normalize_contract_pair(contract1, contract2)
    if ice is None:
        log.warning("icepython not installed; skipping spread chart for %s vs %s", contract1, contract2)
        return None

    cutoff_offset = _cutoff_offset(contract1)
    live_mark = fetch_live_quote(contract1, contract2)
    hist = _prepare_spread_history(contract1, contract2, YEARS_OF_HISTORY)
    if not hist:
        log.warning("No spread history for %s vs %s", contract1, contract2)
        return None
    pivoted = _pivot_by_year(hist, cutoff_offset)
    if pivoted.empty:
        log.warning("Unable to pivot spread history for %s vs %s", contract1, contract2)
        return None
    fname = f"spread_{contract1.replace('-', '')}_vs_{contract2.replace('-', '')}.png"
    output_path = output_dir / fname
    return generate_plot(pivoted, contract1, contract2, live_mark, output_path, cutoff_offset)


def _fmt_contract(dt: datetime) -> str:
    return dt.strftime("%b-%y")


def spread_pairs_from_front(front_label: str) -> List[Tuple[str, str]]:
    """Compute required spread pairs based on the front month."""

    front_dt = parse_contract_string(front_label)
    if front_dt is None:
        return []
    m = front_dt.month
    y = front_dt.year
    pairs: List[Tuple[str, str]] = []

    # Front vs last month of current season
    if m in (11, 12):
        pairs.append((_fmt_contract(front_dt), _fmt_contract(datetime(y + 1, 3, 1))))
    elif m in (1, 2):
        pairs.append((_fmt_contract(front_dt), _fmt_contract(datetime(y, 3, 1))))
    elif m == 3:
        pairs.append((_fmt_contract(front_dt), _fmt_contract(datetime(y, 10, 1))))
    else:
        pairs.append((_fmt_contract(front_dt), _fmt_contract(datetime(y, 10, 1))))

    # Mar vs Apr (use upcoming season if front is before Mar)
    mar_year = y if m >= 3 else y
    if m in (11, 12):
        mar_year = y + 1
    mar_apr = (_fmt_contract(datetime(mar_year, 3, 1)), _fmt_contract(datetime(mar_year, 4, 1)))
    pairs.append(mar_apr)

    # Apr vs Jan (always compare to next Jan; after April use current front vs next Jan)
    if m <= 4:
        pairs.append((_fmt_contract(datetime(y, 4, 1)), _fmt_contract(datetime(y + 1, 1, 1))))
    else:
        pairs.append((_fmt_contract(front_dt), _fmt_contract(datetime(y + 1, 1, 1))))

    # Nov vs Jan
    pairs.append((_fmt_contract(datetime(y, 11, 1)), _fmt_contract(datetime(y + 1, 1, 1))))

    # Nov vs Mar
    pairs.append((_fmt_contract(datetime(y, 11, 1)), _fmt_contract(datetime(y + 1, 3, 1))))

    # Ensure pure Python list of tuple strings
    normalized: List[Tuple[str, str]] = []
    for a, b in pairs:
        n1, n2 = normalize_contract_pair(str(a), str(b))
        normalized.append((n1, n2))
    return normalized
