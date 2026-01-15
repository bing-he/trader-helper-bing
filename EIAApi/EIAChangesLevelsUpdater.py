"""
Fetches and processes weekly natural gas storage data from the EIA API.

This script performs two main tasks:
1.  Fetches the latest weekly storage inventory levels for the U.S. Lower 48
    and its sub-regions, performing an incremental update on the local CSV file.
2.  Calculates the week-over-week change for each series based on the levels.

The final outputs are two CSV files: 'EIAtotals.csv' (for levels) and
'EIAchanges.csv' (for weekly changes), with canonical column names.
"""

import logging
import os
import time
import traceback
from pathlib import Path
from typing import Optional, Dict

import pandas as pd
import requests
...
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

# --- PATHING REFACTOR ---
# 1. Add the project root (TraderHelper) to the system path
# 2. This allows Python to find the 'common' module
import sys
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# --- END REFACTOR ---

from common.pathing import ROOT # <-- This import will now work

# ==============================================================================


# ==============================================================================
#  CONFIGURATION & CONSTANTS
# ==============================================================================

# --- File & Directory Paths ---
SCRIPT_DIR = ROOT / "EIAApi"
INFO_DIR = ROOT / "INFO"
TOTALS_OUTPUT_PATH = INFO_DIR / "EIAtotals.csv"
CHANGES_OUTPUT_PATH = INFO_DIR / "EIAchanges.csv"

# --- EIA API Configuration ---
API_BASE_URL = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
API_REQUEST_DELAY_SEC = 1

# Map API series IDs to canonical column names
WEEKLY_STORAGE_SERIES = {
    "NW2_EPG0_SWO_R48_BCF": "Lower48",
    "NW2_EPG0_SWO_R31_BCF": "East",
    "NW2_EPG0_SWO_R32_BCF": "Midwest",
    "NW2_EPG0_SSO_R33_BCF": "SouthCentral_Salt",
    "NW2_EPG0_SNO_R33_BCF": "SouthCentral_NonSalt",
    "NW2_EPG0_SWO_R33_BCF": "SouthCentral",
    "NW2_EPG0_SWO_R34_BCF": "Mountain",
    "NW2_EPG0_SWO_R35_BCF": "Pacific"
}

# --- Data Fetching Configuration ---
DEFAULT_START_DATE = "2010-01-01"
INCREMENTAL_REFRESH_WEEKS = 15

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

def get_api_key() -> Optional[str]:
    """Loads the EIA API key from a .env file in the script's directory."""
    logging.info("Loading EIA API key from .env file...")
    dotenv_path = SCRIPT_DIR / '.env'
    if not dotenv_path.exists():
        logging.critical(f".env file not found at {dotenv_path}")
        return None
    
    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        logging.critical("EIA_API_KEY not found in .env file.")
        return None
    
    logging.info("Successfully loaded EIA API key.")
    return api_key

# ==============================================================================
#  CORE PIPELINE FUNCTIONS
# ==============================================================================

def determine_fetch_start_date(file_path: Path) -> pd.Timestamp:
    """Determines the start date for the API fetch based on existing data."""
    try:
        df = pd.read_csv(file_path, index_col="Period", parse_dates=True)
        if not df.empty and isinstance(df.index, pd.DatetimeIndex):
            latest_date = df.index.max()
            start_date = latest_date - relativedelta(weeks=INCREMENTAL_REFRESH_WEEKS - 1)
            logging.info(f"Existing file found. Refreshing data since {start_date:%Y-%m-%d}.")
            return start_date
    except FileNotFoundError:
        logging.info(f"No existing file found. Fetching all data since {DEFAULT_START_DATE}.")
    except Exception as e:
        logging.warning(f"Could not read existing file: {e}. Performing full fetch.")
    
    return pd.to_datetime(DEFAULT_START_DATE)

def fetch_series_data(api_key: str, series_id: str, start_date: pd.Timestamp) -> Optional[pd.DataFrame]:
    """Fetches weekly data for a single series ID from the EIA API."""
    logging.info(f"Fetching data for Series ID: {series_id}...")
    params = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][0]": series_id,
        "start": start_date.strftime('%Y-%m-%d'),
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": "5000"
    }
    try:
        response = requests.get(API_BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json().get('response', {}).get('data', [])
        if not data:
            logging.warning(f"No new data returned for series {series_id}.")
            return None
        
        df = pd.DataFrame(data)
        df = df[['period', 'value']].rename(columns={'period': 'Period'})
        df['Period'] = pd.to_datetime(df['Period'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df.set_index('Period')

    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed for series {series_id}: {e}")
        return None

def fetch_all_series(api_key: str, start_date: pd.Timestamp) -> pd.DataFrame:
    """Fetches all defined EIA series and combines them into a single DataFrame."""
    all_series_dfs = {}
    for series_id, column_name in WEEKLY_STORAGE_SERIES.items():
        series_df = fetch_series_data(api_key, series_id, start_date)
        if series_df is not None:
            all_series_dfs[column_name] = series_df['value']
        time.sleep(API_REQUEST_DELAY_SEC)

    if not all_series_dfs:
        return pd.DataFrame()

    return pd.DataFrame(all_series_dfs)

def combine_and_clean_data(new_df: pd.DataFrame, existing_file_path: Path) -> pd.DataFrame:
    """Combines new data with historical data and cleans the result."""
    if new_df.empty:
        logging.warning("No new data was fetched.")
        try:
            return pd.read_csv(existing_file_path, index_col="Period", parse_dates=True)
        except FileNotFoundError:
            return pd.DataFrame()
            
    try:
        existing_df = pd.read_csv(existing_file_path, index_col="Period", parse_dates=True)
        # Update existing data with any new values and add new rows
        combined_df = existing_df.copy()
        combined_df.update(new_df)
        new_rows = new_df[~new_df.index.isin(existing_df.index)]
        final_df = pd.concat([combined_df, new_rows])
    except FileNotFoundError:
        final_df = new_df

    # --- Data Type and Integrity Checks ---
    # Ensure index is a DatetimeIndex of dates (not datetime)
    final_df.index = final_df.index.normalize()

    # Verify all dates are Fridays
    if not final_df.index.weekday.isin([4]).all():
        logging.warning("Non-Friday dates found in the index. This is unexpected.")
    
    # Convert all data columns to float
    for col in final_df.columns:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

    # Clean up and sort
    if not final_df.index.is_unique:
        final_df = final_df[~final_df.index.duplicated(keep='last')]
    final_df.sort_index(inplace=True)
    return final_df

def calculate_weekly_changes(totals_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the week-over-week change for each series."""
    logging.info("Calculating weekly changes...")
    changes_df = totals_df.diff()
    changes_df.columns = [f"{col}_Change" for col in changes_df.columns]
    return changes_df.dropna(axis=0, how='all')

def save_dataframe(df: pd.DataFrame, file_path: Path):
    """Saves a DataFrame to a CSV file."""
    if df.empty:
        logging.warning(f"DataFrame for {file_path.name} is empty. Nothing to save.")
        return
    
    try:
        df_to_save = df.copy()
        df_to_save.index.name = "Period"
        df_to_save.to_csv(file_path, float_format='%.0f')
        logging.info(f"SUCCESS: Data saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save {file_path.name}: {e}")

# ==============================================================================
#  ORCHESTRATION
# ==============================================================================

def main():
    """Orchestrates the entire EIA data update process."""
    setup_logging()
    logging.info("========= Starting EIA Weekly Storage Update Process =========")
    
    api_key = get_api_key()
    if not api_key:
        logging.critical("Halting process: EIA API key is required.")
        return

    try:
        INFO_DIR.mkdir(parents=True, exist_ok=True)
        
        # --- Process for Totals (Levels) ---
        start_date = determine_fetch_start_date(TOTALS_OUTPUT_PATH)
        new_totals_data = fetch_all_series(api_key, start_date)
        final_totals_df = combine_and_clean_data(new_totals_data, TOTALS_OUTPUT_PATH)
        save_dataframe(final_totals_df, TOTALS_OUTPUT_PATH)

        # --- Process for Changes ---
        if not final_totals_df.empty:
            final_changes_df = calculate_weekly_changes(final_totals_df)
            save_dataframe(final_changes_df, CHANGES_OUTPUT_PATH)

    except Exception:
        logging.critical("An unexpected error occurred in the main process.")
        logging.critical(traceback.format_exc())
    finally:
        logging.info("================== Process Finished ==================")

if __name__ == "__main__":
    main()
