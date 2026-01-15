"""
Fetches and maintains a historical price table for Platts symbols.

Key Features:
    - Reads symbols from 'PriceAdmin.csv' (Column K: PlattsCodePlatts).
    - Fetches 'Close/Value' for each symbol using the S&P Global Platts API.
    - Renames columns to 'Market Component' (Column A) for readability.
    - NO FILTERING: Captures every available date and price from the API.
    - Merges results into a single CSV.

Dependencies:
    - requests
    - pandas
    - python-dotenv
"""

import logging
import os
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import pandas as pd
import requests
from dotenv import load_dotenv

# ==============================================================================
#  CONFIGURATION & CONSTANTS
# ==============================================================================

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # Goes up two levels to TraderHelper

# Input: .../TraderHelper/INFO/PriceAdmin.csv
INPUT_DIR = PROJECT_ROOT / "INFO"
INPUT_FILENAME = "PriceAdmin.csv"
INPUT_FILE_PATH = INPUT_DIR / INPUT_FILENAME

# Output: .../TraderHelper/ICE/Outputs/PlattsPriceFD.csv (As requested)
OUTPUT_DIR = PROJECT_ROOT / "ICE" / "Outputs"
OUTPUT_FILENAME = "PlattsPriceFD.csv"
OUTPUT_FILE_PATH = OUTPUT_DIR / OUTPUT_FILENAME

# --- Data Settings ---
# Column Definitions (0-based index)
COL_MARKET_NAME = 0            # Column A: Market Component (Header Name)
COL_PLATTS_CODE = 10           # Column K: PlattsCodePlatts (API Symbol)

# --- API Settings ---
AUTH_URL = "https://api.ci.spglobal.com/auth/api"
HISTORY_URL = "https://api.ci.spglobal.com/market-data/v3/value/history/symbol"
BATES_FILTER = "U"  # Filter for used/settlement prices

# Load Environment Variables
load_dotenv()
PLATTS_USERNAME = os.getenv("PLATTS_USERNAME")
PLATTS_PASSWORD = os.getenv("PLATTS_PASSWORD")

# ==============================================================================
#  SETUP & HELPER FUNCTIONS
# ==============================================================================

def setup_logging():
    """Configures the logging format."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def get_access_token() -> Optional[str]:
    """Authenticates with the Platts API and retrieves an access token."""
    if not PLATTS_USERNAME or not PLATTS_PASSWORD:
        logging.critical("PLATTS_USERNAME or PLATTS_PASSWORD not set in environment.")
        return None

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = {"username": PLATTS_USERNAME, "password": PLATTS_PASSWORD}
    
    try:
        response = requests.post(AUTH_URL, headers=headers, data=payload, timeout=30)
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.exceptions.RequestException as e:
        logging.critical(f"Authentication failed: {e}")
        return None

def load_symbol_map(file_path: Path) -> Dict[str, str]:
    """
    Reads the CSV and creates a mapping from Platts Code to Market Name.
    """
    if not file_path.exists():
        logging.critical(f"Input file not found: {file_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(file_path)
        
        if len(df.columns) <= max(COL_MARKET_NAME, COL_PLATTS_CODE):
            logging.error("Input CSV does not have enough columns.")
            return {}

        # Extract relevant columns
        subset = df.iloc[:, [COL_PLATTS_CODE, COL_MARKET_NAME]].dropna()
        
        # Clean data
        subset.iloc[:, 0] = subset.iloc[:, 0].astype(str).str.strip().str.upper()
        subset.iloc[:, 1] = subset.iloc[:, 1].astype(str).str.strip()

        symbol_map = dict(zip(subset.iloc[:, 0], subset.iloc[:, 1]))
        
        logging.info(f"Loaded {len(symbol_map)} symbols from PriceAdmin.")
        return symbol_map

    except Exception as e:
        logging.error(f"Failed to read symbol map: {e}")
        return {}

def determine_fetch_range() -> Tuple[str, str, Optional[pd.DataFrame]]:
    """Determines start/end dates based on existing file."""
    end_date = date.today().strftime('%Y-%m-%d')
    
    if not OUTPUT_FILE_PATH.exists():
        logging.info("Output file not found. Starting fresh initialization.")
        return prompt_for_start_date(), end_date, None
    
    try:
        logging.info(f"Existing file found at {OUTPUT_FILE_PATH}. Analyzing...")
        df = pd.read_csv(OUTPUT_FILE_PATH, index_col='Date', parse_dates=True)
        
        if df.shape[1] < 2: 
            logging.warning("File appears corrupted or empty. Starting fresh.")
            return prompt_for_start_date(), end_date, None

        unique_dates = sorted(df.index.unique())
        
        if len(unique_dates) <= 5:
            logging.warning("Insufficient history. Refetching all.")
            start_date = unique_dates[0].strftime('%Y-%m-%d')
            return start_date, end_date, None

        # Rollback 5 days logic
        cutoff_date = unique_dates[-5]
        truncated_df = df[df.index < cutoff_date].copy()
        start_date = cutoff_date.strftime('%Y-%m-%d')
        
        logging.info(f"Rolling back to {start_date}. Retaining {len(truncated_df)} rows.")
        return start_date, end_date, truncated_df

    except Exception as e:
        logging.error(f"Error reading existing file: {e}. Starting fresh.")
        return prompt_for_start_date(), end_date, None

def prompt_for_start_date() -> str:
    while True:
        start_input = input("Enter Start Date (YYYY-MM-DD): ").strip()
        try:
            datetime.strptime(start_input, '%Y-%m-%d')
            return start_input
        except ValueError:
            logging.error("Invalid format. Use YYYY-MM-DD.")

# ==============================================================================
#  CORE PIPELINE
# ==============================================================================

def fetch_single_symbol(token: str, symbol_code: str, market_name: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    Fetches data for a single symbol using Platts API.
    Handles pagination to retrieve full history.
    """
    # Uses 'bate' (singular) with colon, and Capitalized Param Keys.
    filter_string = (
        f'symbol IN ("{symbol_code}") AND bate:"{BATES_FILTER}" '
        f'AND assessDate>="{start}" AND assessDate<="{end}"'
    )
    
    all_records = []
    page = 1
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }

    while True:
        # Platts API uses 1-based indexing for Page
        params = {
            "PageSize": 1000,
            "Page": page,
            "Filter": filter_string
        }

        try:
            response = requests.get(HISTORY_URL, headers=headers, params=params, timeout=30)
            
            if response.status_code == 404:
                # 404 on the first page means no data exists
                if page == 1:
                    print(f" [SKIP] {symbol_code} ({market_name}): No data found (404).")
                break
            
            # If we get an error on a subsequent page, we stop but keep what we have
            if response.status_code != 200:
                error_text = response.text
                print(f" [FAIL] {symbol_code}: API Error {response.status_code} - Page {page} - Response: {error_text}")
                break

            data = response.json()
            results = data.get("results", [])
            
            # If no results on first page, empty. If no results on subsequent page, we are done.
            if not results or not results[0].get('data'):
                if page == 1:
                    print(f" [SKIP] {symbol_code} ({market_name}): Empty results.")
                break

            # Parse Data from this page
            page_records = []
            for r in results[0]['data']:
                page_records.append({
                    "Date": r.get("assessDate"),
                    market_name: pd.to_numeric(r.get("value"), errors='coerce')
                })
            
            if not page_records:
                break
                
            all_records.extend(page_records)
            
            # Pagination Check:
            # If the number of records returned is less than the requested PageSize,
            # we have reached the last page.
            if len(page_records) < 1000:
                break
                
            page += 1
            # Small sleep between pages to be nice
            time.sleep(0.05)

        except Exception as e:
            print(f" [ERROR] {symbol_code}: Exception on page {page} -> {e}")
            break

    if not all_records:
        return None

    df = pd.DataFrame(all_records)

    # Standardize Date
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    df.set_index('Date', inplace=True)
    
    return df[[market_name]]

def fetch_batch_data(token: str, symbol_map: Dict[str, str], start: str, end: str) -> pd.DataFrame:
    """Iterates through symbols and collects data."""
    collected_frames = []
    total = len(symbol_map)
    sorted_items = sorted(symbol_map.items())

    print(f"\n--- Fetching Data: {start} to {end} ---")
    
    for i, (code, name) in enumerate(sorted_items, 1):
        print(f"[{i}/{total}] {code:<8} -> {name[:30]:<30}", end='\r')
        
        df = fetch_single_symbol(token, code, name, start, end)
        
        if df is not None:
            collected_frames.append(df)
            
        # Increased sleep to avoid rate limiting (429) or server-side issues
        time.sleep(0.2)

    print("\nFetch complete.")
    
    if not collected_frames:
        return pd.DataFrame()

    return pd.concat(collected_frames, axis=1)

def main():
    setup_logging()
    
    # 1. Auth
    token = get_access_token()
    if not token:
        return

    # 2. Load Map
    symbol_map = load_symbol_map(INPUT_FILE_PATH)
    if not symbol_map:
        return

    # 3. Setup
    start_date, end_date, existing_df = determine_fetch_range()

    # 4. Fetch
    new_df = fetch_batch_data(token, symbol_map, start_date, end_date)

    if new_df.empty:
        logging.warning("No new data fetched.")
        if existing_df is None:
            logging.error("No data available to save.")
            return

    # 5. Merge
    if existing_df is not None and not existing_df.empty:
        logging.info("Merging new data with existing history...")
        final_df = pd.concat([existing_df, new_df])
        # Group by index and take the last value to handle overlaps/updates
        final_df = final_df.groupby(final_df.index).last()
    else:
        final_df = new_df

    # 6. Save (No filtering!)
    if not final_df.empty:
        final_df.sort_index(ascending=False, inplace=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            final_df.to_csv(OUTPUT_FILE_PATH, na_rep='')
            logging.info(f"SUCCESS: Updated table saved to:\n{OUTPUT_FILE_PATH}")
        except Exception as e:
            logging.critical(f"Failed to save file: {e}")
    else:
        logging.warning("Resulting dataset is empty.")

if __name__ == "__main__":
    main()