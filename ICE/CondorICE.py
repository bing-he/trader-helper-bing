"""
Generates a comparative chart of a 'Spread vs Spread' structure from the ICE API.

This script prompts the user for two pairs of contract months (Spread 1 and Spread 2).
Example: Mar-26/Apr-26 vs Nov-26/Mar-27.

It calculates the differential between these two spreads:
    Value = (Spread 1) - (Spread 2)
          = (Leg1 - Leg2) - (Leg3 - Leg4)

It then overlays this current structure against its historical analogs (10+ years),
normalized by days to expiry.
"""

import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import icepython as ice
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# ==============================================================================
#   CONFIGURATION & CONSTANTS
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "Outputs"

YEARS_OF_HISTORY = 10
BASE_SYMBOL = 'HNG'
API_SUFFIX = '-IUS'

MONTH_CODES = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
}
MONTH_NAMES_TO_NUM = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

# ==============================================================================
#   HELPER FUNCTIONS
# ==============================================================================

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")

def parse_contract_string(contract_str: str) -> Optional[datetime]:
    try:
        month_str, year_str = contract_str.strip().split('-')
        month = MONTH_NAMES_TO_NUM[month_str.title()]
        year = int(f"20{year_str}")
        return datetime(year, month, 1)
    except Exception:
        logging.error(f"Invalid format '{contract_str}'. Use 'Mon-YY'.")
        return None

def get_symbol_from_date(date_obj: datetime) -> str:
    """Converts a datetime object into an ICE symbol (e.g., HNG F25-IUS)."""
    code = MONTH_CODES[date_obj.month]
    yr = str(date_obj.year)[-2:]
    return f"{BASE_SYMBOL} {code}{yr}{API_SUFFIX}"

def get_spread_symbol(date1: datetime, date2: datetime) -> str:
    """Constructs a spread ticker string (e.g., HNG F25:HNG G25-IUS)."""
    leg1 = f"{BASE_SYMBOL} {MONTH_CODES[date1.month]}{str(date1.year)[-2:]}"
    leg2 = f"{BASE_SYMBOL}{MONTH_CODES[date2.month]}{str(date2.year)[-2:]}" # Note: ICE spread syntax often tightens the second leg
    return f"{leg1}:{leg2}{API_SUFFIX}"

# ==============================================================================
#   CORE DATA FUNCTIONS
# ==============================================================================

def fetch_historical_structure(legs_dates: List[datetime], start_date: str, end_date: str) -> Optional[pd.Series]:
    """
    Fetches history for 4 legs and calculates (L1-L2) - (L3-L4).
    """
    symbols = [get_symbol_from_date(d) for d in legs_dates]
    
    # logging.info(f"Fetching history for: {symbols}...")
    
    try:
        # Fetch all 4 outrights at once
        ts_data = ice.get_timeseries(symbols, ['Settle'], 'D', start_date, end_date)
        if not ts_data or len(ts_data) < 2:
            return None

        # Clean Header
        header = [h.replace('.Settle', '') for h in ts_data[0]]
        df = pd.DataFrame(ts_data[1:], columns=header)
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.set_index('Time').apply(pd.to_numeric, errors='coerce')

        # Ensure we have all columns
        missing_cols = [s for s in symbols if s not in df.columns]
        if missing_cols:
            logging.warning(f"Missing data for {missing_cols}")
            return None
        
        # Calculate the Structure: (Leg1 - Leg2) - (Leg3 - Leg4)
        # Note: If user inputs 2 legs, we handle that, but here we assume 4 for the comparison.
        val_spread1 = df[symbols[0]] - df[symbols[1]]
        val_spread2 = df[symbols[2]] - df[symbols[3]]
        
        structure_value = val_spread1 - val_spread2
        return structure_value.dropna()

    except Exception as e:
        logging.error(f"Hist fetch error: {e}")
        return None

def fetch_live_structure(legs_dates: List[datetime]) -> Optional[float]:
    """
    Fetches live quotes for Spread 1 and Spread 2, then calculates the diff.
    """
    # Construct Spread Symbols
    spread1_sym = get_spread_symbol(legs_dates[0], legs_dates[1])
    spread2_sym = get_spread_symbol(legs_dates[2], legs_dates[3])
    
    logging.info(f"Fetching live quotes for:\n  1) {spread1_sym}\n  2) {spread2_sym}")

    try:
        quote_data = ice.get_quotes([spread1_sym, spread2_sym], ['bid', 'Ask'])
        if not quote_data or len(quote_data) < 3:
            logging.warning("Live quotes unavailable.")
            return None

        # Helper to extract mid from row
        def get_mid(row):
            b = pd.to_numeric(row[1], errors='coerce')
            a = pd.to_numeric(row[2], errors='coerce')
            if pd.notna(b) and pd.notna(a): return (b + a) / 2
            return None

        mid1 = get_mid(quote_data[1]) # Row 1 is Spread 1
        mid2 = get_mid(quote_data[2]) # Row 2 is Spread 2

        if mid1 is not None and mid2 is not None:
            diff = mid1 - mid2
            logging.info(f"Live Mids: {mid1:.4f} vs {mid2:.4f} -> Diff: {diff:.4f}")
            return diff
        return None

    except Exception as e:
        logging.error(f"Live quote error: {e}")
        return None

# ==============================================================================
#   PLOTTING
# ==============================================================================

def generate_plot(df: pd.DataFrame, title_dates: List[str], live_mark: Optional[float]):
    logging.info("Generating comparison chart...")
    fig, ax = plt.subplots(figsize=(18, 10))
    
    latest_year = df.columns.max()
    colors = plt.cm.tab10.colors
    last_valid_idx = df[latest_year].last_valid_index()

    for i, year in enumerate(sorted(df.columns, reverse=True)):
        series = df[year]
        is_latest = (year == latest_year)
        
        # Determine Label
        label = year
        if is_latest and live_mark is not None:
            label = f"{year} (Live: {live_mark:.3f})"
        elif is_latest and last_valid_idx:
             curr_val = series.loc[last_valid_idx]
             label = f"{year} (Last: {curr_val:.3f})"
        
        ax.plot(
            series.index, series,
            label=label,
            color='red' if is_latest else colors[i % len(colors)],
            linewidth=3.5 if is_latest else 1.5,
            alpha=1.0 if is_latest else 0.6,
            zorder=10 if is_latest else 5
        )

    # Titles
    s1_name = f"{title_dates[0]}/{title_dates[1]}"
    s2_name = f"{title_dates[2]}/{title_dates[3]}"
    ax.set_title(f'Spread Comparison Structure\n({s1_name})  MINUS  ({s2_name})', fontsize=18, weight='bold')
    
    ax.set_xlabel('Days to Expiry (of first leg)', fontsize=12)
    ax.set_ylabel('Structure Value ($)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlim(df.index.min(), 0)

    # Dynamic X-Axis (approximate months)
    # We just use day counts, but could map back to months if needed
    
    plt.tight_layout()
    
    # Save
    out_name = f"Compare_{title_dates[0]}_{title_dates[2]}.png"
    out_path = OUTPUT_DIR / out_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    logging.info(f"Chart saved: {out_path}")

# ==============================================================================
#   MAIN
# ==============================================================================

def main():
    setup_logging()
    logging.info("=== Spread Comparison Tool ===")
    
    # 1. Inputs
    # We default to the user's request for prompt ease, or use input()
    print("Enter the 4 contract months.")
    print("Structure: (Leg1 - Leg2) - (Leg3 - Leg4)")
    
    # For testing, you can hardcode or use input()
    l1 = input("Leg 1 (e.g. Mar-26): ") or "Mar-26"
    l2 = input("Leg 2 (e.g. Apr-26): ") or "Apr-26"
    l3 = input("Leg 3 (e.g. Nov-26): ") or "Nov-26"
    l4 = input("Leg 4 (e.g. Mar-27): ") or "Mar-27"
    
    input_strs = [l1, l2, l3, l4]
    base_dates = [parse_contract_string(x) for x in input_strs]
    
    if any(d is None for d in base_dates): return

    # 2. Calculate Offsets (To preserve structure shape historically)
    # We calculate the delta of Legs 2,3,4 relative to Leg 1
    offsets = [relativedelta(d, base_dates[0]) for d in base_dates]

    # 3. Live Quote
    live_val = fetch_live_structure(base_dates)

    # 4. Historical Loop
    hist_data = {}
    
    logging.info(f"Processing {YEARS_OF_HISTORY} years of history...")
    
    for i in range(YEARS_OF_HISTORY + 1):
        # Shift the ANCHOR (Leg 1) back by i years
        anchor_date = base_dates[0] - relativedelta(years=i)
        year_label = str(anchor_date.year)
        
        # Reconstruct the other 3 legs based on original offsets
        # This preserves the gap (e.g. Nov-26 to Mar-27 is 4 months)
        curr_dates = [anchor_date + off for off in offsets]
        
        # Define time window for this specific year
        # We fetch 1 year of data ending 5 days before the first leg expires
        expiry_approx = curr_dates[0]
        end_str = (expiry_approx - relativedelta(days=5)).strftime('%Y-%m-%d')
        start_str = (expiry_approx - relativedelta(years=1)).strftime('%Y-%m-%d')
        
        series = fetch_historical_structure(curr_dates, start_str, end_str)
        
        if series is not None and not series.empty:
            # Normalize Index: Days until Expiry of Leg 1
            days_to_expiry = (series.index - expiry_approx).days
            series.index = days_to_expiry
            hist_data[year_label] = series

    # 5. Charting
    if hist_data:
        df_final = pd.DataFrame(hist_data).sort_index()
    
        # Mask future data for the current year
        current_year_col = str(base_dates[0].year)
        if current_year_col in df_final.columns:
            # Find today's offset
            today_offset = (datetime.now() - base_dates[0]).days
            df_final.loc[df_final.index > today_offset, current_year_col] = np.nan

        generate_plot(df_final, input_strs, live_val)
    else:
        logging.error("No historical data found.")

if __name__ == "__main__":
    main()