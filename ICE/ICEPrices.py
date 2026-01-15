"""
Fetches a time series of price data for a user-specified symbol and date
range from the ICE API.

This script is an interactive tool that prompts the user for a symbol, a start
date, and an end date. It then fetches the specified data field, processes the
results by adding a 'Flow Date', and saves the clean time series to a
dynamically named CSV file in the INFO directory.
"""

import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import icepython as ice
import pandas as pd

# ==============================================================================
#  CONFIGURATION & CONSTANTS
# ==============================================================================

# --- File & Directory Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "Outputs" # Changed from INFO to a local Outputs folder

# --- API & Data Configuration ---
# These can be modified if the standard API details change.
SYMBOL_SUFFIX = ' D1-IPG'
TARGET_FIELD = 'VWAP Close'

# ==============================================================================
#  SETUP & HELPER FUNCTIONS
# ==============================================================================

def setup_logging():
    """Configures a basic logger to show timestamped messages."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def get_user_input() -> Optional[tuple[str, str, str]]:
    """
    Prompts the user for the symbol and date range and validates the input.

    Returns:
        A tuple containing (symbol, start_date_str, end_date_str), or None
        if the input is invalid.
    """
    symbol = input("Enter the symbol to fetch (e.g., YJT): ").strip().upper()
    if not symbol:
        logging.error("Symbol cannot be empty.")
        return None

    start_date_str = input("Enter the start date (YYYY-MM-DD): ").strip()
    end_date_str = input("Enter the end date (YYYY-MM-DD): ").strip()

    try:
        # Validate that the dates can be parsed correctly
        datetime.strptime(start_date_str, '%Y-%m-%d')
        datetime.strptime(end_date_str, '%Y-%m-%d')
        return symbol, start_date_str, end_date_str
    except ValueError:
        logging.error("Invalid date format. Please use YYYY-MM-DD.")
        return None

# ==============================================================================
#  CORE PIPELINE FUNCTIONS
# ==============================================================================

def fetch_ice_timeseries(symbol: str, field: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetches a time series from the ICE API for a given symbol and field.

    Args:
        symbol: The full API symbol to query.
        field: The data field to retrieve (e.g., 'VWAP Close').
        start_date: The start date in 'YYYY-MM-DD' format.
        end_date: The end date in 'YYYY-MM-DD' format.

    Returns:
        A DataFrame with the time series data, or None on failure.
    """
    logging.info(f"Fetching '{field}' for symbol '{symbol}' from {start_date} to {end_date}...")
    try:
        ts_data = ice.get_timeseries(
            symbols=[symbol],
            fields=[field],
            granularity='D',
            start_date=start_date,
            end_date=end_date
        )
        if not ts_data or len(ts_data) < 2:
            logging.warning("No data returned from the API for the specified symbol and date range.")
            return None

        df = pd.DataFrame(ts_data[1:], columns=ts_data[0])
        logging.info(f"Successfully fetched {len(df)} data points from API.")
        return df

    except Exception as e:
        logging.error(f"An error occurred during the ICE API call: {e}")
        return None

def process_data(df: pd.DataFrame, full_symbol: str, field: str) -> pd.DataFrame:
    """
    Cleans and transforms the raw data from the API.

    Args:
        df: The raw DataFrame returned by the API call.
        full_symbol: The full symbol name used in the query, for column renaming.
        field: The data field that was retrieved.

    Returns:
        A processed DataFrame with clean column names and a 'Flow Date'.
    """
    logging.info("Processing raw data...")
    price_column_name = f"{full_symbol}.{field}"
    
    df.rename(columns={'Time': 'Trade Date', price_column_name: 'Price'}, inplace=True)
    
    df['Trade Date'] = pd.to_datetime(df['Trade Date'])
    df['Flow Date'] = (df['Trade Date'] + timedelta(days=1))
    
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    final_df = df[['Flow Date', 'Price', 'Trade Date']].dropna(subset=['Price'])
    
    logging.info("Data processing complete.")
    return final_df.sort_values(by='Flow Date', ascending=False)

# ==============================================================================
#  ORCHESTRATION
# ==============================================================================

def main():
    """Orchestrates the fetching, processing, and saving of ICE price data."""
    setup_logging()
    logging.info("========= Starting Interactive ICE Price Fetcher =========")

    try:
        # --- 1. Get User Input ---
        user_input = get_user_input()
        if not user_input:
            logging.critical("Halting due to invalid user input.")
            return
        
        symbol, start_date_str, end_date_str = user_input
        
        # --- 2. Fetch Data ---
        full_symbol = f"{symbol}{SYMBOL_SUFFIX}"
        raw_df = fetch_ice_timeseries(full_symbol, TARGET_FIELD, start_date_str, end_date_str)
        
        if raw_df is None or raw_df.empty:
            logging.warning("Halting execution: No data was fetched from the API.")
            return

        # --- 3. Process Data ---
        final_df = process_data(raw_df, full_symbol, TARGET_FIELD)
        
        # --- 4. Save Data ---
        if not final_df.empty:
            # Create a dynamic output filename
            output_filename = f"ICE_{symbol}_{start_date_str}_to_{end_date_str}.csv"
            output_path = OUTPUT_DIR / output_filename

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            final_df.to_csv(output_path, index=False, date_format='%Y-%m-%d')
            logging.info(f"SUCCESS: Data saved to:\n{output_path}")
            logging.info(f"--- Latest Prices ---\n{final_df.head().to_string(index=False)}")
        else:
            logging.warning("No valid data remained after processing. No file was saved.")

    except Exception:
        logging.critical("An unexpected error occurred in the main process.")
        logging.critical(traceback.format_exc())
    finally:
        logging.info("================== Process Finished ==================")

if __name__ == "__main__":
    main()