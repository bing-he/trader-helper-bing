"""
Fetches, calculates, and plots historical NYMEX Henry Hub forward curves.

This script builds and compares the NYMEX forward curve for several historical
dates (today, 7 days ago, 14 days ago, etc.). It employs a sophisticated
method to construct the full 19-month curve for each date:

1.  For the current day's curve, it fetches a live anchor price for the prompt
    month and live spread marks for the next 12 liquid months. If live data
    is unavailable, it falls back to the most recent settlement prices.
2.  For historical curves, it fetches the daily settlement prices for all
    outright contracts.
3.  For the illiquid tail of the curve (months 14-19), it uses the shape of the
    previous day's settlement curve to extrapolate the remaining points.

The final output is a CSV data table and two PNG charts comparing the curves
in both absolute price and relative (normalized) terms.
"""

import logging
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import icepython as ice
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# ==============================================================================
#  CONFIGURATION & CONSTANTS
# ==============================================================================

# --- File & Directory Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "Outputs"

# --- API & Data Configuration ---
BASE_SYMBOL = 'HNG'
API_SUFFIX = '-IUS'
NUM_FORWARD_MONTHS = 19
NUM_LIQUID_MONTHS = 13
HISTORICAL_LOOKBACK_DAYS = [0, 7, 14, 28, 63]  # 0 represents today

# --- Futures Month Codes ---
MONTH_CODES = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
}

# ==============================================================================
#  SETUP & HELPER FUNCTIONS
# ==============================================================================

def setup_logging():
    """Configures a basic logger to show timestamped messages."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

def get_forward_contracts(start_date: datetime, num_months: int) -> List[datetime]:
    """Generates a list of datetime objects for each forward contract month."""
    # The prompt contract is always for the month after the current calendar month.
    prompt_month_start = (start_date.replace(day=1) + relativedelta(months=1))
    return [(prompt_month_start + relativedelta(months=i)) for i in range(num_months)]

# ==============================================================================
#  CORE DATA FETCHING & PROCESSING
# ==============================================================================

def fetch_live_curve(contract_dates: List[datetime]) -> pd.Series:
    """Builds today's forward curve using a live anchor price and spread marks."""
    logging.info("Fetching LIVE data to build today's curve...")
    outright_symbols = [f"{BASE_SYMBOL} {MONTH_CODES[d.month]}{str(d.year)[-2:]}{API_SUFFIX}" for d in contract_dates]
    prompt_symbol = outright_symbols[0]
    
    forward_prices = {}
    anchor_price = np.nan

    try:
        # 1. Get a robust anchor price for the prompt month
        quotes = ice.get_quotes([prompt_symbol], ['last', 'bid', 'ask'])
        if quotes and len(quotes) > 1:
            last, bid, ask = pd.to_numeric(quotes[1][1:], errors='coerce')
            anchor_price = last if pd.notna(last) and last > 0 else (bid + ask) / 2
        
        if pd.notna(anchor_price):
            logging.info(f"Live anchor price for {prompt_symbol}: {anchor_price:.4f}")
        else:
            # Fallback to previous day's settle if live price is invalid
            logging.warning(f"Could not determine a valid live anchor price for {prompt_symbol}. Falling back to previous settle.")
            prev_day_curve = fetch_historical_curve(datetime.now() - timedelta(days=1), contract_dates)
            anchor_price = prev_day_curve.get(contract_dates[0])
            if pd.notna(anchor_price):
                logging.info(f"Using previous settle as anchor price: {anchor_price:.4f}")
            else:
                raise ValueError("Fallback to previous settle also failed.")

        forward_prices[contract_dates[0]] = anchor_price

        # 2. Fetch live spreads and build the liquid part of the curve
        prompt_code = prompt_symbol.replace(API_SUFFIX, '')
        # --- FIX: Correctly format the second leg of the spread symbol by removing the space ---
        spread_symbols = [f"{prompt_code}:{s.replace(API_SUFFIX, '').replace(' ', '')}{API_SUFFIX}" for s in outright_symbols[1:NUM_LIQUID_MONTHS]]
        
        logging.info(f"Fetching live marks for {len(spread_symbols)} calendar spreads...")
        spread_quotes = ice.get_quotes(spread_symbols, ['bid', 'Ask'])
        
        priced_spreads_count = 0
        if spread_quotes and len(spread_quotes) > 1:
            spread_map = {row[0]: pd.to_numeric(row[1:], errors='coerce') for row in spread_quotes[1:]}
            for i, spread_symbol in enumerate(spread_symbols):
                prices = spread_map.get(spread_symbol)
                if prices is not None and len(prices) == 2 and pd.notna(prices).all():
                    spread_mark = (prices[0] + prices[1]) / 2
                    forward_prices[contract_dates[i + 1]] = anchor_price - spread_mark
                    priced_spreads_count += 1
        
        logging.info(f"Successfully priced {priced_spreads_count}/{len(spread_symbols)} liquid spreads.")

        # If spreads failed, fall back to historical settlements for the whole curve
        if priced_spreads_count < NUM_LIQUID_MONTHS - 2: # Heuristic for failure
            logging.warning("Live spread fetch was incomplete. Falling back to previous day's full settlement curve for shape.")
            prev_day_curve = fetch_historical_curve(datetime.now() - timedelta(days=1), contract_dates)
            if not prev_day_curve.empty:
                return prev_day_curve

        return pd.Series(forward_prices).sort_index()

    except Exception as e:
        logging.error(f"Failed to fetch live curve: {e}")
        return pd.Series(dtype='float64')


def fetch_historical_curve(as_of_date: datetime, contract_dates: List[datetime]) -> pd.Series:
    """Builds a historical forward curve using settlement prices for a specific past date."""
    logging.info(f"Fetching HISTORICAL settlement data for {as_of_date:%Y-%m-%d}...")
    outright_symbols = [f"{BASE_SYMBOL} {MONTH_CODES[d.month]}{str(d.year)[-2:]}{API_SUFFIX}" for d in contract_dates]
    
    try:
        # Widen lookback to account for weekends/holidays
        start_date = as_of_date - timedelta(days=7)
        ts_data = ice.get_timeseries(outright_symbols, ['Settle'], 'D', start_date.strftime('%Y-%m-%d'), as_of_date.strftime('%Y-%m-%d'))
        if not ts_data or len(ts_data) < 2: raise ValueError("No historical data returned.")

        df = pd.DataFrame(ts_data[1:], columns=ts_data[0])
        df['Time'] = pd.to_datetime(df['Time']).dt.date
        
        # Get the latest row on or before the target date
        settles_for_date = df[df['Time'] <= as_of_date.date()].iloc[-1]
        
        forward_prices = {
            dt: pd.to_numeric(settles_for_date.get(f"{symbol}.Settle"), errors='coerce')
            for dt, symbol in zip(contract_dates, outright_symbols)
        }
        return pd.Series(forward_prices).sort_index()
    except Exception as e:
        logging.error(f"Failed to fetch historical curve for {as_of_date:%Y-%m-%d}: {e}")
        return pd.Series(dtype='float64')

def calculate_illiquid_tail(curve: pd.Series, as_of_date: datetime, contract_dates: List[datetime]) -> pd.Series:
    """Extrapolates the illiquid end of the curve using the previous day's settlement shape."""
    # Check if the liquid part of the curve is mostly complete before trying to extrapolate
    if curve.count() < NUM_LIQUID_MONTHS - 2:
        logging.warning("Liquid part of the curve is too sparse to calculate illiquid tail. Skipping.")
        return curve

    if len(curve) >= len(contract_dates):
        return curve # Curve is already complete

    logging.info("Calculating illiquid tail (months 14-19)...")
    
    try:
        # Fetch settlements for the day before the curve date
        prev_day = as_of_date - timedelta(days=1)
        prev_day_curve = fetch_historical_curve(prev_day, contract_dates)
        if prev_day_curve.empty: raise ValueError("Could not get previous day's settlements.")

        # Extrapolate forward using the carry from the previous day's curve
        for i in range(len(curve) - 1, NUM_FORWARD_MONTHS - 1):
            if contract_dates[i+1] not in curve.index:
                last_price = curve.get(contract_dates[i])
                prev_day_carry = prev_day_curve.get(contract_dates[i+1]) - prev_day_curve.get(contract_dates[i])
                if pd.notna(last_price) and pd.notna(prev_day_carry):
                    curve[contract_dates[i+1]] = last_price + prev_day_carry
        return curve.sort_index()
    except Exception as e:
        logging.warning(f"Could not calculate illiquid tail: {e}")
        return curve

# ==============================================================================
#  PLOTTING
# ==============================================================================

def plot_curves(df: pd.DataFrame, title: str, ylabel: str, filename: str, normalize: bool = False):
    """Generic plotting function for creating and saving the charts."""
    if df.empty:
        logging.warning(f"Skipping plot '{filename}' due to empty data.")
        return

    fig, ax = plt.subplots(figsize=(18, 10))
    plot_df = df.copy()
    if normalize:
        # Normalize each curve to its own prompt month price
        plot_df = plot_df.apply(lambda x: (x / x.iloc[0]) * 100 if pd.notna(x.iloc[0]) and x.iloc[0] != 0 else x)

    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_df.columns)))
    for i, col in enumerate(plot_df.columns):
        ax.plot(plot_df.index, plot_df[col], label=col, color=colors[i], linewidth=2.5, marker='o', markersize=4, alpha=0.8)

    ax.set_title(title, fontsize=20, pad=20, weight='bold')
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel('Contract Month', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.6)
    ax.legend(title='As of Date', fontsize=11)
    
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%y'))
    plt.xticks(rotation=45, ha='right')

    fig.tight_layout(pad=2.0)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path)
    logging.info(f"SUCCESS: Chart saved to {output_path}")
    plt.close(fig)

# ==============================================================================
#  ORCHESTRATION
# ==============================================================================

def main():
    """Orchestrates the entire NYMEX forward curve analysis process."""
    setup_logging()
    logging.info("========= Starting NYMEX Forward Curve Analysis =========")
    
    try:
        all_curves = {}
        today = datetime.now()
        contract_dates = get_forward_contracts(today, NUM_FORWARD_MONTHS)
        logging.info(f"Analyzing curve from {contract_dates[0]:%b-%Y} to {contract_dates[-1]:%b-%Y}")

        for days_ago in sorted(HISTORICAL_LOOKBACK_DAYS):
            target_date = today - timedelta(days=days_ago)
            date_str = target_date.strftime('%Y-%m-%d')
            
            if days_ago == 0:
                curve = fetch_live_curve(contract_dates)
            else:
                curve = fetch_historical_curve(target_date, contract_dates)
            
            if not curve.empty:
                full_curve = calculate_illiquid_tail(curve, target_date, contract_dates)
                all_curves[date_str] = full_curve
            
            time.sleep(1) # Be respectful to the API

        if not all_curves:
            logging.critical("No data could be retrieved for any date. Exiting.")
            return

        # --- Create and Save Data Table ---
        final_df = pd.DataFrame(all_curves).sort_index(axis=1, ascending=False)
        final_df.index.name = "Contract"
        final_df.to_csv(OUTPUT_DIR / 'NYMEX_Curve_Table.csv', float_format='%.4f')
        logging.info(f"SUCCESS: Data table saved to {OUTPUT_DIR / 'NYMEX_Curve_Table.csv'}")

        # --- Generate Plots ---
        plot_df = final_df.reindex(sorted(final_df.columns, reverse=True), axis=1)
        plot_curves(plot_df, 'NYMEX Henry Hub Forward Curve Comparison', 'Price ($/MMBtu)', 'NYMEX_Curve_Levels.png', normalize=False)
        plot_curves(plot_df, 'NYMEX Henry Hub Relative Value (Normalized to Prompt Month)', 'Relative Price (Prompt Month = 100)', 'NYMEX_Curve_Relative_Value.png', normalize=True)

    except Exception:
        logging.critical("An unexpected error occurred in the main process.")
        logging.critical(traceback.format_exc())
    finally:
        logging.info("================== Process Finished ==================")

if __name__ == "__main__":
    main()
