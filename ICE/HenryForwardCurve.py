# minimal import safety for direct execution
import sys
from pathlib import Path
# repo root is one level up from /ICE
sys.path.append(str(Path(__file__).resolve().parents[1])) # Restored this path

# HenryForwardCurve.py
#
# Fetches a 24-month forward curve based on 'Settle' (for historical)
# and 'Last' (for today's live row).
# - Publishes Gas Day = Trade Date + 1
# - Clamps historical settles to Today's Gas Day
# - Appends a 'live' row for Tomorrow's Gas Day based on Today's Trade Date
#
# Output: ROOT / "INFO" / "HenryForwardCurve.csv"
#         ROOT / "INFO" / "HenryHub_Absolute_History.csv" (Derived)
#
# Requirements: icepython, pandas, python-dateutil; valid ICE entitlements

import os
from datetime import date, datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import icepython as ice 
from dateutil.relativedelta import relativedelta

from common.logs import get_file_logger
from common.pathing import ROOT

CONFIG = {
    "info_dir": ROOT / "INFO",
    "output_csv": ROOT / "INFO" / "HenryForwardCurve.csv",
}


def _resolve_path(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


# -----------------------
# CONFIG (static defaults; update-mode will override some at runtime)
# -----------------------

# Published Gas Day lower bound (first row in output for first full build)
PUBLISH_START_STATIC = pd.Timestamp("2006-01-01")

# Fetch buffer (covers holidays around New Year) used for first full build
FETCH_START_STATIC = date(2005, 12, 20)

INFO_DIR = _resolve_path(CONFIG["info_dir"])
OUTPUT_CSV = _resolve_path(CONFIG["output_csv"])
logger = get_file_logger(Path(__file__).stem if "__file__" in locals() else "HenryForwardCurve")

# Futures (NYMEX Henry via ICE)
BASE_SYMBOL = "HNG"
API_SUFFIX  = "-IUS"
NUM_FWD_MONTHS = 24            # FWD_00 .. FWD_23
EXTRA_BACKSTOP_MONTHS = 1      # pull M24 so month-end self-roll still yields 24
FUTURES_FIELD = "Settle"
LIVE_FIELD = "Last"            # Field to use for 'tomorrow's' live row
GRANULARITY   = "D"
SYMBOLS_PER_TIMESERIES_BATCH = 200

# Gas Day alignment + continuity
GAS_DAY_OFFSET = 1                  # GasDay = TradeDate + 1 day
INCLUDE_ALL_CALENDAR_DAYS = True    # make series continuous
FFILL_LIMIT_DAYS = 3                # Limited to 3 days

# Update-window (how many Gas Days to rebuild)
UPDATE_DAYS = int(os.getenv("HFC_UPDATE_DAYS", "30")) # Set to 30

MONTH_CODES = {1:"F",2:"G",3:"H",4:"J",5:"K",6:"M",7:"N",8:"Q",9:"U",10:"V",11:"X",12:"Z"}

# -----------------------
# HELPERS
# -----------------------

def ensure_info():
    INFO_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"INFO directory ok: {INFO_DIR}")

def hng_symbol_for_month(dt: date) -> str:
    mcode = MONTH_CODES[dt.month]
    yy = str(dt.year)[-2:]
    return f"{BASE_SYMBOL} {mcode}{yy}{API_SUFFIX}"

def front_label_for_month(d: date) -> str:
    return d.strftime("%b-%Y")

def forward_contract_months_from(trade_day: date, n: int) -> List[date]:
    """Calendar front = month AFTER trade_day's calendar month."""
    front = (trade_day.replace(day=1) + relativedelta(months=1))
    return [(front + relativedelta(months=i)) for i in range(n)]

def contract_months_needed(start_day: date, end_day: date, months_ahead: int) -> List[date]:
    earliest_front = (start_day.replace(day=1) + relativedelta(months=1))
    latest_front   = (end_day.replace(day=1) + relativedelta(months=1))
    last_needed    = latest_front + relativedelta(months=months_ahead - 1)
    months = []
    cur = earliest_front
    while cur <= last_needed:
        months.append(cur)
        cur += relativedelta(months=1)
    return months

def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def fetch_timeseries(symbols: List[str], field: str, start: date, end: date) -> pd.DataFrame:
    """Call ICE get_timeseries in batches; return columns like '<symbol>.<field>' indexed by TradeDate."""
    
    frames = []
    for chunk in batched(symbols, SYMBOLS_PER_TIMESERIES_BATCH):
        try:
            ts = ice.get_timeseries(chunk, [field], GRANULARITY, start.isoformat(), end.isoformat())
            if not ts or len(ts) < 2:
                continue
            df = pd.DataFrame(ts[1:], columns=ts[0])
            df.rename(columns={"Time": "TradeDate"}, inplace=True)
            df["TradeDate"] = pd.to_datetime(df["TradeDate"]).dt.normalize()
            frames.append(df.set_index("TradeDate"))
        except Exception as e:
            logger.error(f"Failed to fetch timeseries for symbols {chunk} with field {field}: {e}")
            
    if not frames:
        logger.warning(f"No data returned from ice.get_timeseries for field '{field}' between {start} and {end}")
        return pd.DataFrame(index=pd.Index([], name="TradeDate"))
        
    out = pd.concat(frames, axis=1).sort_index()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def symbol_to_month(col: str) -> date:
    """Maps a timeseries column name (e.g., 'HNG F15-IUS.Settle') to its contract month."""
    try:
        sym = col.split(".")[0]
        code = sym.replace(API_SUFFIX, "").split()[1]  # 'F15'
        mcode, yy = code[0], code[1:]
        mm = next(k for k, v in MONTH_CODES.items() if v == mcode)
        yyyy = 2000 + int(yy)
        return date(yyyy, mm, 1)
    except Exception as e:
        logger.error(f"Failed to parse symbol_to_month for column '{col}': {e}")
        return None 

def build_curve(publish_start: pd.Timestamp, fetch_start: date, fetch_end: date) -> pd.DataFrame:
    """Build curve rows from publish_start through last available Gas Day based on FUTURES_FIELD."""
    needed_months = contract_months_needed(fetch_start, fetch_end, NUM_FWD_MONTHS + EXTRA_BACKSTOP_MONTHS)
    if not needed_months:
        logger.warning(f"No contract months needed for fetch range {fetch_start} to {fetch_end}. Returning empty.")
        return pd.DataFrame(columns=["Date", "FrontMonth_Label"] + [f"FWD_{i:02d}" for i in range(NUM_FWD_MONTHS)])

    outright_symbols = [hng_symbol_for_month(m) for m in needed_months]
    logger.info(f"Building curve with '{FUTURES_FIELD}': {outright_symbols[0]} .. {outright_symbols[-1]}")

    fut_df = fetch_timeseries(outright_symbols, FUTURES_FIELD, fetch_start, fetch_end)

    if fut_df.empty:
        logger.warning("fetch_timeseries returned no data for settles.")
        out = pd.DataFrame({"Date": [publish_start], "FrontMonth_Label": [np.nan]})
        for i in range(NUM_FWD_MONTHS):
            out[f"FWD_{i:02d}"] = np.nan
        return out

    col_month_map: Dict[str, date] = {c: symbol_to_month(c) for c in fut_df.columns}
    col_month_map = {c: m for c, m in col_month_map.items() if m is not None} 

    rows = []
    for td in fut_df.index:  # dtype: datetime64[ns], normalized
        
        months_25 = forward_contract_months_from(td.date(), NUM_FWD_MONTHS + EXTRA_BACKSTOP_MONTHS)

        curve_24 = []
        for m in months_25[:NUM_FWD_MONTHS]: 
            col = next((c for c, mm in col_month_map.items() if mm == m), None)
            curve_24.append(fut_df.loc[td, col] if col and col in fut_df.columns else np.nan)

        front_label = front_label_for_month(months_25[0])

        gas_day = td + pd.Timedelta(days=GAS_DAY_OFFSET)
        row = {"Date": gas_day, "FrontMonth_Label": front_label}
        for i, v in enumerate(curve_24):
            row[f"FWD_{i:02d}"] = v
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("Date")

    if INCLUDE_ALL_CALENDAR_DAYS and not out.empty:
        out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
        start_gas_ts = publish_start
        
        today_ts = pd.Timestamp(date.today())
        end_gas_ts = min(out["Date"].max(), today_ts)

        if end_gas_ts < start_gas_ts:
            logger.warning(f"Clamped end_gas_ts ({end_gas_ts}) is before start_gas_ts ({start_gas_ts}). Resulting index will be empty.")
            full_idx = pd.Index([], dtype='datetime64[ns]')
        else:
            full_idx = pd.date_range(start_gas_ts, end_gas_ts, freq="D")

        if not full_idx.empty:
            out = (
                out.set_index("Date")
                    .groupby(level=0).last()
                    .reindex(full_idx)
                    .rename_axis("Date")
                    .reset_index()
            )
        else:
            logger.warning("Full index is empty, reindex step skipped.")
            out = out.reindex(columns=["Date", "FrontMonth_Label"] + [f"FWD_{i:02d}" for i in range(NUM_FWD_MONTHS)]).iloc[0:0]


        fwd_cols = [c for c in out.columns if c.startswith("FWD_")]
        if not out.empty:
            out[fwd_cols] = out[fwd_cols].ffill(limit=FFILL_LIMIT_DAYS)
            out["FrontMonth_Label"] = out["FrontMonth_Label"].ffill(limit=FFILL_LIMIT_DAYS)

    fwd_cols = [f"FWD_{i:02d}" for i in range(NUM_FWD_MONTHS)]
    out = out[["Date", "FrontMonth_Label"] + fwd_cols]
    return out

def fetch_and_append_live_row(out_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches the 'LIVE_FIELD' for today's TradeDate and appends it to the DataFrame
    as *tomorrow's* Gas Day.
    """
    logger.info(f"Attempting to fetch live '{LIVE_FIELD}' row for tomorrow's Gas Day.")
    today = date.today()
    today_ts = pd.Timestamp(today)
    tomorrows_gas_day = today_ts + pd.Timedelta(days=GAS_DAY_OFFSET)

    if not out_df.empty and tomorrows_gas_day in out_df["Date"].values:
        logger.warning(f"Live row for Gas Day {tomorrows_gas_day.date()} already exists. Skipping append.")
        return out_df

    months_25 = forward_contract_months_from(today, NUM_FWD_MONTHS + EXTRA_BACKSTOP_MONTHS)
    outright_symbols = [hng_symbol_for_month(m) for m in months_25]
    
    fut_df = fetch_timeseries(outright_symbols, LIVE_FIELD, start=today, end=today)

    if fut_df.empty or today_ts not in fut_df.index:
        logger.warning(f"No live '{LIVE_FIELD}' data returned for TradeDate {today}. Cannot append live row.")
        return out_df

    logger.info(f"Successfully fetched '{LIVE_FIELD}' data for TradeDate {today}.")
    col_month_map: Dict[str, date] = {c: symbol_to_month(c) for c in fut_df.columns}
    col_month_map = {c: m for c, m in col_month_map.items() if m is not None} 
    
    curve_24 = []
    for m in months_25[:NUM_FWD_MONTHS]: 
        col = next((c for c, mm in col_month_map.items() if mm == m), None)
        curve_24.append(fut_df.loc[today_ts, col] if col and col in fut_df.columns else np.nan)

    front_label = front_label_for_month(months_25[0])
    
    live_row = {"Date": tomorrows_gas_day, "FrontMonth_Label": front_label}
    for i, v in enumerate(curve_24):
        live_row[f"FWD_{i:02d}"] = v
    
    try:
        live_row_df = pd.DataFrame([live_row])
    except Exception as e:
        logger.error(f"Failed to create DataFrame for live row: {e}")
        return out_df
    
    try:
        out_df_with_live = pd.concat([out_df, live_row_df], ignore_index=True)
        logger.info(f"Successfully appended live row for Gas Day {tomorrows_gas_day.date()}.")
        return out_df_with_live
    except Exception as e:
        logger.error(f"Failed to concatenate live row to DataFrame: {e}")
        return out_df

# --- ABSOLUTE CURVE GENERATION ---
def generate_absolute_curve():
    """
    Reads INFO/HenryForwardCurve.csv (Rolling) and writes INFO/HenryHub_Absolute_History.csv (Absolute).
    """
    input_csv = _resolve_path(CONFIG["output_csv"])
    output_csv = input_csv.parent / "HenryHub_Absolute_History.csv"
    
    if not input_csv.exists():
        logger.warning(f"Skipping absolute curve generation: {input_csv} not found.")
        return

    logger.info(f"Generating Absolute Curve from: {input_csv}...")
    
    try:
        df = pd.read_csv(input_csv)
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception as e:
        logger.error(f"Error reading rolling curve for transformation: {e}")
        return

    # Incremental Update Check:
    # If output_csv exists, load it and find the last date.
    # Only transform rows from input_csv that are NEWER than that date.
    # However, transforming rows is fast. For robustness against corrections, 
    # we will regenerate the tail (last 90 days) or full build if small.
    # Given the pivot complexity, a full rebuild from the source dataframe is safest and cleaner
    # unless the file is massive (>1GB). At ~4000 rows (10 years), full rebuild is sub-second.
    
    long_data = []
    
    for idx, row in df.iterrows():
        trade_date = row['Date']
        
        if 'FrontMonth_Label' not in row or pd.isna(row['FrontMonth_Label']):
            continue
            
        try:
            fm_str = row['FrontMonth_Label']
            fm_date = datetime.strptime(fm_str, '%b-%Y')
        except Exception:
            continue 
            
        for i in range(24):
            col_name = f"FWD_{i:02d}"
            if col_name in row and pd.notna(row[col_name]):
                contract_date = fm_date + relativedelta(months=i)
                contract_str = contract_date.strftime('%Y-%m-%d')
                
                long_data.append({
                    'TradeDate': trade_date,
                    'ContractDate': contract_str, 
                    'Price': row[col_name]
                })
    
    if not long_data:
        logger.warning("No valid absolute data points generated.")
        return

    df_long = pd.DataFrame(long_data)
    df_pivot = df_long.pivot(index='TradeDate', columns='ContractDate', values='Price')
    df_pivot = df_pivot.sort_index()
    df_pivot = df_pivot.reindex(sorted(df_pivot.columns), axis=1)
    
    df_pivot.to_csv(output_csv)
    logger.info(f"Success: Absolute Curve saved to {output_csv}")


# -----------------------
# MAIN
# -----------------------

def main():
    ensure_info()

    today_dt = date.today()
    first_run = not OUTPUT_CSV.exists()

    if first_run:
        logger.info("Mode: FULL BUILD (no existing CSV).")
        publish_start = PUBLISH_START_STATIC
        fetch_start    = FETCH_START_STATIC
        fetch_end      = today_dt

        out = build_curve(publish_start=publish_start, fetch_start=fetch_start, fetch_end=fetch_end)

    else:
        # --- Update mode ---
        logger.info("Mode: UPDATE (existing CSV found).")
        try:
            existing = pd.read_csv(OUTPUT_CSV, parse_dates=["Date"])
        except Exception as e:
            logger.error(f"Failed to read existing CSV: {e}. Switching to FULL BUILD.")
            existing = pd.DataFrame() # Force empty

        if existing.empty:
            logger.info("Existing CSV is empty or failed to read; falling back to FULL BUILD.")
            publish_start = PUBLISH_START_STATIC
            fetch_start    = FETCH_START_STATIC
            fetch_end      = today_dt
            out = build_curve(publish_start=publish_start, fetch_start=fetch_start, fetch_end=fetch_end)
        else:
            existing["Date"] = existing["Date"].dt.normalize()
            last_date = existing["Date"].max()
            
            # Rebuild tail logic
            keep_until = (last_date - pd.Timedelta(days=UPDATE_DAYS - 1)).normalize()
            keep_until = max(keep_until, PUBLISH_START_STATIC) 

            logger.info(f"Existing last Gas Day: {last_date.date()} | Rebuilding tail from (inclusive): {keep_until.date()}")

            prefix = existing[existing["Date"] < keep_until].copy()

            publish_start = max(PUBLISH_START_STATIC, keep_until)
            
            # Widen fetch cushion
            trade_start = (publish_start - pd.Timedelta(days=GAS_DAY_OFFSET + 10)).date()
            fetch_start = max(trade_start, FETCH_START_STATIC)
            fetch_end   = today_dt

            logger.info(f"Building tail from publish_start={publish_start.date()}, fetch_start={fetch_start}, fetch_end={fetch_end}")

            tail = build_curve(publish_start=publish_start, fetch_start=fetch_start, fetch_end=fetch_end)

            # Concatenate
            out = pd.concat([prefix, tail], ignore_index=True)
            out = out.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    # --- Append live row ---
    out = fetch_and_append_live_row(out)
    
    # Final check
    fwd_cols = [f"FWD_{i:02d}" for i in range(NUM_FWD_MONTHS)]
    out.dropna(subset=fwd_cols, how='all', inplace=True)

    # Save Rolling Curve
    out.to_csv(OUTPUT_CSV, index=False, date_format="%Y-%m-%d")
    logger.info(f"SUCCESS: wrote {len(out):,} rows to {OUTPUT_CSV}")

    # --- GENERATE ABSOLUTE CURVE ---
    generate_absolute_curve()


if __name__ == "__main__":
    main()