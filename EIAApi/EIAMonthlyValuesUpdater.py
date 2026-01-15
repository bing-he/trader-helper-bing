"""
Fetches and processes monthly fundamental data from the EIA API.

This script retrieves key monthly U.S. natural gas data series (like production,
consumption, and trade flows), converts them from monthly totals in MMcf to an
average daily rate in Bcf, and saves the result. It performs an incremental
update on the local CSV file, refreshing the last several months of data to
account for any revisions from the EIA.
"""

import calendar
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
OUTPUT_PATH = INFO_DIR / "EIAFundamentalMonthlydayAvg.csv"

# --- EIA API Configuration ---
API_BASE_URL_TEMPLATE = "https://api.eia.gov/v2/{category_path}/data/"
API_REQUEST_DELAY_SEC = 1
TARGET_SERIES = {
    "N9070US2": "Prod", "N9102CN2": "CadIMP", "N3045US2": "Power Burn",
    "N3035US2": "Industrial", "N3010US2": "_Temp_ResidentialCons",
    "N3020US2": "_Temp_CommercialCons", "N9132MX2": "MexExp", "N9133US2": "LNGExp"
}
SERIES_CATEGORY_PATHS = {
    "N9070US2": "natural-gas/prod/sum", "N9102CN2": "natural-gas/move/impc",
    "N3045US2": "natural-gas/cons/sum", "N3035US2": "natural-gas/cons/sum",
    "N3010US2": "natural-gas/cons/sum", "N3020US2": "natural-gas/cons/sum",
    "N9132MX2": "natural-gas/move/expc", "N9133US2": "natural-gas/move/expc"
}

# --- Data Fetching & Processing Configuration ---
DEFAULT_START_DATE = "2015-01"
INCREMENTAL_REFRESH_MONTHS = 12
FINAL_COLUMN_ORDER = [
    "Prod", "CadIMP", "Power Burn", "Industrial", "ResCom", "MexExp", "LNGExp"
]

# ==============================================================================
#  SETUP & HELPER FUNCTIONS
# ==============================================================================

def setup_logging():
    """Configures a basic logger to show timestamped messages."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

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
            start_date = latest_date - relativedelta(months=INCREMENTAL_REFRESH_MONTHS - 1)
            logging.info(f"Existing file found. Refreshing data since {start_date:%Y-%m}.")
            return start_date
    except FileNotFoundError:
        logging.info(f"No existing file found. Fetching all data since {DEFAULT_START_DATE}.")
    except Exception as e:
        logging.warning(f"Could not read existing file: {e}. Performing full fetch.")
    
    return pd.to_datetime(DEFAULT_START_DATE)

def fetch_and_process_series(api_key: str, series_id: str, start_date: pd.Timestamp) -> Optional[pd.DataFrame]:
    """Fetches a single monthly series and converts it to a daily average in Bcf."""
    category_path = SERIES_CATEGORY_PATHS.get(series_id)
    if not category_path:
        logging.error(f"No category path defined for series ID {series_id}. Skipping.")
        return None

    logging.info(f"Fetching data for Series ID: {series_id}...")
    api_url = API_BASE_URL_TEMPLATE.format(category_path=category_path)
    params = {
        "api_key": api_key, "frequency": "monthly", "data[0]": "value",
        "facets[series][0]": series_id, "start": start_date.strftime('%Y-%m'),
        "sort[0][column]": "period", "sort[0][direction]": "asc", "length": "5000"
    }
    try:
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json().get('response', {}).get('data', [])
        if not data:
            logging.warning(f"No new data returned for series {series_id}.")
            return None
        
        df = pd.DataFrame(data)
        df['period'] = pd.to_datetime(df['period'], format='%Y-%m')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        # --- Core Business Logic: Convert Monthly MMcf to Daily Average Bcf ---
        def to_daily_avg_bcf(row):
            days_in_month = calendar.monthrange(row.period.year, row.period.month)[1]
            return (row.value / days_in_month) / 1000 if days_in_month > 0 else None
        
        df['daily_avg_bcf'] = df.apply(to_daily_avg_bcf, axis=1)
        
        return df[['period', 'daily_avg_bcf']].set_index('period')

    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed for series {series_id}: {e}")
        return None

def fetch_all_series(api_key: str, start_date: pd.Timestamp) -> pd.DataFrame:
    """Fetches all defined EIA series and combines them into a single DataFrame."""
    all_series_dfs = []
    for series_id, column_name in TARGET_SERIES.items():
        series_df = fetch_and_process_series(api_key, series_id, start_date)
        if series_df is not None:
            series_df.rename(columns={'daily_avg_bcf': column_name}, inplace=True)
            all_series_dfs.append(series_df)
        time.sleep(API_REQUEST_DELAY_SEC)

    if not all_series_dfs:
        return pd.DataFrame()
    return pd.concat(all_series_dfs, axis=1, join='outer')

def process_final_dataframe(new_df: pd.DataFrame, existing_file_path: Path) -> pd.DataFrame:
    """Combines new and old data, calculates derived columns, and finalizes the DataFrame."""
    try:
        existing_df = pd.read_csv(existing_file_path, index_col="Period", parse_dates=True)
        # Update existing data with new values and append rows not previously present
        combined_df = existing_df.copy()
        combined_df.update(new_df)
        new_rows = new_df[~new_df.index.isin(existing_df.index)]
        final_df = pd.concat([combined_df, new_rows])
    except FileNotFoundError:
        final_df = new_df

    # Calculate ResCom from its temporary components
    res_col, com_col = "_Temp_ResidentialCons", "_Temp_CommercialCons"
    if res_col in final_df.columns and com_col in final_df.columns:
        final_df['ResCom'] = final_df[res_col].fillna(0) + final_df[com_col].fillna(0)
    
    # Clean up, sort, and reorder columns
    final_df.drop(columns=[res_col, com_col], errors='ignore', inplace=True)
    final_df = final_df.reindex(columns=FINAL_COLUMN_ORDER)
    final_df.sort_index(inplace=True)
    return final_df

def save_dataframe(df: pd.DataFrame, file_path: Path):
    """Saves the final DataFrame to a CSV file."""
    if df.empty:
        logging.warning("Final DataFrame is empty. Nothing to save.")
        return
    
    try:
        df_to_save = df.copy()
        df_to_save.index.name = "Period"
        df_to_save.to_csv(file_path, float_format='%.4f', date_format='%Y-%m')
        logging.info(f"SUCCESS: Data saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save data: {e}")

# ==============================================================================
#  ORCHESTRATION
# ==============================================================================

def main():
    """Orchestrates the entire EIA monthly data update process."""
    setup_logging()
    logging.info("========= Starting EIA Monthly Fundamentals Update Process =========")
    
    api_key = get_api_key()
    if not api_key:
        logging.critical("Halting process: EIA API key is required.")
        return

    try:
        INFO_DIR.mkdir(parents=True, exist_ok=True)
        start_date = determine_fetch_start_date(OUTPUT_PATH)
        new_data = fetch_all_series(api_key, start_date)
        
        if new_data.empty:
            logging.warning("No new data was fetched for any series. Exiting.")
            return

        final_df = process_final_dataframe(new_data, OUTPUT_PATH)
        # CORRECTED LINE: Added the missing 'OUTPUT_PATH' argument
        save_dataframe(final_df, OUTPUT_PATH)

    except Exception:
        logging.critical("An unexpected error occurred in the main process.")
        logging.critical(traceback.format_exc())
    finally:
        logging.info("================== Process Finished ==================")

if __name__ == "__main__":
    main()
