"""
Generates a comparative chart of historical NYMEX spreads from the ICE API.

This script is an interactive tool that prompts the user for two contract months
(e.g., 'Aug-25' and 'Sep-25'). It then performs two main actions:
1.  Fetches a live bid/ask for the current spread between the two contracts.
2.  Fetches historical daily settlement data for the same spread over the past
    10+ years.

Finally, it generates and saves a high-quality PNG chart that overlays the
current year's spread against all historical years, normalized by the number
of days until contract expiry.
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
#  CONFIGURATION & CONSTANTS
# ==============================================================================

# --- File & Directory Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "Outputs"

# --- Chart & Data Configuration ---
YEARS_OF_HISTORY = 10
BASE_SYMBOL = 'HNG'
API_SUFFIX = '-IUS'

# --- Futures Month Codes ---
MONTH_CODES = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
}
MONTH_NAMES_TO_NUM = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

# ==============================================================================
#  SETUP & HELPER FUNCTIONS
# ==============================================================================

def setup_logging():
    """Configures a basic logger to show timestamped messages."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def parse_contract_string(contract_str: str) -> Optional[datetime]:
    """Parses a 'Mon-YY' string (e.g., 'Aug-25') into a datetime object."""
    try:
        month_str, year_str = contract_str.strip().split('-')
        month = MONTH_NAMES_TO_NUM[month_str.title()]
        year = int(f"20{year_str}")
        return datetime(year, month, 1)
    except (ValueError, KeyError):
        logging.error(f"Invalid contract format '{contract_str}'. Please use 'Mon-YY'.")
        return None

# ==============================================================================
#  CORE PIPELINE FUNCTIONS
# ==============================================================================

def fetch_historical_data(symbols: List[str], start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Fetches historical daily settlement prices for a list of symbols."""
    logging.info(f"Fetching historical data for {symbols} from {start_date} to {end_date}...")
    try:
        ts_data = ice.get_timeseries(symbols, ['Settle'], 'D', start_date, end_date)
        if not ts_data or len(ts_data) < 2:
            logging.warning("No data returned from API for this period.")
            return None

        header = [h.replace('.Settle', '') for h in ts_data[0]]
        df = pd.DataFrame(ts_data[1:], columns=header)
        df['Time'] = pd.to_datetime(df['Time'])
        df = df.set_index('Time').drop_duplicates()
        return df.apply(pd.to_numeric, errors='coerce')
    except Exception as e:
        logging.error(f"An error occurred during the historical API call: {e}")
        return None

def fetch_live_quote(contract1: str, contract2: str) -> Optional[float]:
    """Fetches the live bid/ask for a spread and returns the mid-point (mark)."""
    logging.info(f"Fetching live quote for the {contract1} vs {contract2} spread...")
    date1 = parse_contract_string(contract1)
    date2 = parse_contract_string(contract2)
    if not date1 or not date2: return None

    leg1 = f"{BASE_SYMBOL} {MONTH_CODES[date1.month]}{str(date1.year)[-2:]}"
    leg2 = f"{BASE_SYMBOL}{MONTH_CODES[date2.month]}{str(date2.year)[-2:]}"
    spread_symbol = f"{leg1}:{leg2}{API_SUFFIX}"
    
    try:
        quote_data = ice.get_quotes([spread_symbol], ['bid', 'Ask'])
        if not quote_data or len(quote_data) < 2 or '<NotEnt>' in quote_data[1]:
            logging.warning(f"Could not retrieve live quote for symbol: {spread_symbol}")
            return None

        bid = pd.to_numeric(quote_data[1][1], errors='coerce')
        ask = pd.to_numeric(quote_data[1][2], errors='coerce')
        
        if pd.notna(bid) and pd.notna(ask):
            mark_price = (bid + ask) / 2
            logging.info(f"Live Spread Mark found: {mark_price:.4f} (Bid: {bid:.4f}, Ask: {ask:.4f})")
            return mark_price
        else:
            logging.warning("Live quote returned invalid bid/ask values.")
            return None
    except Exception as e:
        logging.error(f"An error occurred during the live quote API call: {e}")
        return None

def generate_plot(df: pd.DataFrame, contract1: str, contract2: str, live_mark: Optional[float]):
    """Generates and saves the historical spread comparison chart."""
    logging.info("Generating plot...")
    fig, ax = plt.subplots(figsize=(18, 10))
    
    latest_year = df.columns.max()
    colors = plt.cm.tab10.colors
    
    # Find the last valid data point for the current year to label historical values
    last_valid_day_current_year = df[latest_year].last_valid_index()

    for i, year in enumerate(sorted(df.columns)):
        series = df[year]
        label = year
        
        # Create dynamic labels for the legend
        if year == latest_year and live_mark is not None:
            label = f"{year} (Live Mark: {live_mark:.3f})"
        elif last_valid_day_current_year is not None:
            historical_value = df.loc[last_valid_day_current_year, year]
            if pd.notna(historical_value):
                label = f"{year} ({historical_value:.3f})"
        
        # Style the latest year's line to make it stand out
        is_latest_year = (year == latest_year)
        ax.plot(
            series.index, series, label=label,
            color='red' if is_latest_year else colors[i % len(colors)],
            linewidth=3.0 if is_latest_year else 1.5,
            zorder=10 if is_latest_year else 5,
            alpha=1.0 if is_latest_year else 0.8
        )

    # --- Chart Formatting ---
    ax.set_title(f'Historical Spread Analysis: {contract1} vs {contract2}', fontsize=20, pad=20, weight='bold')
    ax.set_ylabel('Spread Value (USD/MMBtu)', fontsize=14)
    ax.set_xlabel('Calendar Month', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.6)
    ax.legend(title='Contract Year (Value at Current Date)', fontsize=11, loc='upper left')
    
    # --- Dynamic X-axis Ticks ---
    # Create ticks based on calendar months relative to expiry
    chart_end_date = parse_contract_string(contract1)
    tick_locations, tick_labels = [], []
    for month_offset in range(13, -1, -1):
        tick_date = (chart_end_date - relativedelta(months=month_offset)).replace(day=1)
        # Convert the date to "days from expiry" to place the tick correctly
        tick_location = (tick_date - chart_end_date).days
        if df.index.min() <= tick_location <= df.index.max():
            tick_locations.append(tick_location)
            tick_labels.append(tick_date.strftime('%b'))

    ax.set_xticks(tick_locations)
    ax.set_xticklabels(tick_labels, rotation=0, fontsize=10)
    ax.set_xlim(df.index.min(), 0) # Set x-limit from start to expiry (day 0)
    
    fig.tight_layout(pad=2.0)
    
    # --- Save the Figure ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_name = f"spread_analysis_{contract1}_vs_{contract2}.png"
    output_path = OUTPUT_DIR / file_name
    plt.savefig(output_path)
    logging.info(f"SUCCESS: Chart saved to:\n{output_path}")

# ==============================================================================
#  ORCHESTRATION
# ==============================================================================

def main():
    """Orchestrates the entire historical spread analysis and charting process."""
    setup_logging()
    logging.info("========= Starting Historical NYMEX Spread Charter =========")
    
    try:
        contract1 = input("Enter the first contract month (e.g., Aug-25): ")
        contract2 = input("Enter the second contract month (e.g., Sep-25): ")
        if not contract1 or not contract2:
            logging.critical("Both contract months must be specified. Exiting.")
            return

        live_mark = fetch_live_quote(contract1, contract2)
        
        # --- Data Fetching Loop ---
        base_date1 = parse_contract_string(contract1)
        base_date2 = parse_contract_string(contract2)
        if not base_date1 or not base_date2: return

        historical_spreads = {}
        for i in range(YEARS_OF_HISTORY + 1):
            year_label = str(base_date1.year - i)
            
            # Calculate dates and symbols for this historical iteration
            d1 = base_date1 - relativedelta(years=i)
            d2 = base_date2 - relativedelta(years=i)
            symbol1 = f"{BASE_SYMBOL} {MONTH_CODES[d1.month]}{str(d1.year)[-2:]}{API_SUFFIX}"
            symbol2 = f"{BASE_SYMBOL} {MONTH_CODES[d2.month]}{str(d2.year)[-2:]}{API_SUFFIX}"
            
            end_date = d1 - relativedelta(days=5)
            start_date = end_date - relativedelta(years=1)
            
            df_hist = fetch_historical_data([symbol1, symbol2], start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if df_hist is not None and not df_hist.empty and symbol1 in df_hist and symbol2 in df_hist:
                spread = df_hist[symbol1] - df_hist[symbol2]
                # Normalize the index to be "days from expiry"
                spread.index = (spread.index - end_date).days
                historical_spreads[year_label] = spread

        if not historical_spreads:
            logging.error("Could not retrieve any historical data. Halting.")
            return

        # --- Data Processing & Charting ---
        combined_df = pd.DataFrame(historical_spreads).sort_index()
        # Forward-fill data to create continuous lines, but then remove data for the
        # current year that is in the future to avoid a misleading flat line.
        last_valid_idx = combined_df[str(base_date1.year)].last_valid_index()
        combined_df.ffill(inplace=True)
        if last_valid_idx is not None:
            combined_df.loc[combined_df.index > last_valid_idx, str(base_date1.year)] = np.nan
            
        generate_plot(combined_df, contract1, contract2, live_mark)

    except Exception:
        logging.critical(traceback.format_exc())
    finally:
        logging.info("================== Process Finished ==================")

if __name__ == "__main__":
    main()  