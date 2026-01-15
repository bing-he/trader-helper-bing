"""
Fetches and maintains historical price tables for ICE symbols.

Key Features:
    - Reads symbols from 'PriceAdmin.csv'.
    - Runs two separate workflows:
        1. Daily Codes (Column C) -> Outputs to ICEPriceTD.csv
        2. GDA Codes (Column R)   -> Outputs to ICEPriceGDA.csv
    - Fetches 'VWAP Close' for each symbol.
    - Renames columns to 'Market Component' (Column A).
    - Filters:
        1. Removes weekends (Sat/Sun).
        2. Validates against Benchmark 'XGF' (if the code exists in the specific map).

Dependencies:
    - icepython
    - pandas
"""

import logging
import sys
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import icepython as ice
import pandas as pd

# ==============================================================================
#  CONFIGURATION & CONSTANTS
# ==============================================================================

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Input: .../TraderHelper/INFO/PriceAdmin.csv
INPUT_DIR = PROJECT_ROOT / "INFO"
INPUT_FILENAME = "PriceAdmin.csv"
INPUT_FILE_PATH = INPUT_DIR / INPUT_FILENAME

# Output Directory
OUTPUT_DIR = SCRIPT_DIR / "Outputs"

# Output Filenames
OUTPUT_FILENAME_TD = "ICEPriceTD.csv"
OUTPUT_FILENAME_GDA = "ICEPriceGDA.csv"

# --- Data Settings ---
# Column Definitions (0-based index)
COL_MARKET_NAME = 0            # Column A: Market Component (Header Name)
COL_DAILY_CODE = 2             # Column C: Daily Code (API Symbol)
COL_GDA_CODE = 17              # Column R: GDA Code (API Symbol)

TARGET_FIELD = 'VWAP Close'
SYMBOL_SUFFIX = ' D1-IPG'

# Benchmark Symbol for Data Validation
# If this symbol is NaN/Empty for a date, drop the row.
# Note: If this code is not found in the specific column (C or R), filtering is skipped.
BENCHMARK_SYMBOL_CODE = 'XGF' 

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

def load_symbol_map(file_path: Path, code_col_index: int) -> Dict[str, str]:
    """
    Reads the CSV and creates a mapping from the specified Code Column to Market Name.
    
    Args:
        file_path: Path to CSV.
        code_col_index: The 0-based index of the column containing the API codes.
    
    Returns:
        Dict: { 'Code': 'Market Component Name', ... }
    """
    if not file_path.exists():
        logging.critical(f"Input file not found: {file_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(file_path)
        
        # Ensure columns exist by index
        max_idx = max(COL_MARKET_NAME, code_col_index)
        if len(df.columns) <= max_idx:
            logging.error(f"Input CSV does not have enough columns (Needed index {max_idx}).")
            return {}

        # Extract relevant columns
        # We drop rows where either the Code or the Name is missing
        subset = df.iloc[:, [code_col_index, COL_MARKET_NAME]].dropna()
        
        # Clean data: Strip whitespace, ensure codes are Upper case
        subset.iloc[:, 0] = subset.iloc[:, 0].astype(str).str.strip().str.upper()
        subset.iloc[:, 1] = subset.iloc[:, 1].astype(str).str.strip()

        # Create dictionary: Key=Code, Value=Name
        symbol_map = dict(zip(subset.iloc[:, 0], subset.iloc[:, 1]))
        
        logging.info(f"Loaded {len(symbol_map)} symbols from Column index {code_col_index}.")
        return symbol_map

    except Exception as e:
        logging.error(f"Failed to read symbol map: {e}")
        return {}

def determine_fetch_range(output_path: Path) -> Tuple[str, str, Optional[pd.DataFrame]]:
    """Determines start/end dates based on the specific output file."""
    end_date = date.today().strftime('%Y-%m-%d')
    
    if not output_path.exists():
        logging.info(f"File {output_path.name} not found. Starting fresh.")
        return prompt_for_start_date(), end_date, None
    
    try:
        logging.info(f"Analyzing existing file: {output_path.name}")
        df = pd.read_csv(output_path, index_col='Trade Date', parse_dates=True)
        
        if df.shape[1] < 1: 
            logging.warning("File appears corrupted or empty. Starting fresh.")
            return prompt_for_start_date(), end_date, None

        unique_dates = sorted(df.index.unique())
        
        if len(unique_dates) <= 5:
            logging.warning("Insufficient history. Refetching all.")
            start_date = unique_dates[0].strftime('%Y-%m-%d') if unique_dates else prompt_for_start_date()
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
    # Basic validation to prevent endless loops if input is piped
    attempts = 0
    while attempts < 3:
        start_input = input("Enter Start Date (YYYY-MM-DD): ").strip()
        try:
            datetime.strptime(start_input, '%Y-%m-%d')
            return start_input
        except ValueError:
            logging.error("Invalid format. Use YYYY-MM-DD.")
            attempts += 1
    
    logging.error("Too many failed attempts. Defaulting to today.")
    return date.today().strftime('%Y-%m-%d')

# ==============================================================================
#  CORE PIPELINE
# ==============================================================================

def fetch_single_symbol(symbol_code: str, market_name: str, field: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Fetches data for a single symbol and renames the column to the Market Name."""
    full_symbol = f"{symbol_code}{SYMBOL_SUFFIX}"
    
    try:
        ts_data = ice.get_timeseries(
            symbols=[full_symbol],
            fields=[field],
            granularity='D',
            start_date=start,
            end_date=end
        )

        if not ts_data:
            # Silent skip or minimal logging to avoid spam
            return None
            
        if len(ts_data) > 0 and 'Error' in str(ts_data[0]):
             print(f" [FAIL] {symbol_code}: API Error -> {ts_data}")
             return None

        if len(ts_data) < 2:
            return None

        # Construct DataFrame
        df = pd.DataFrame(list(ts_data[1:]), columns=list(ts_data[0]))
        
        # Standardize Date
        df.rename(columns={'Time': 'Trade Date'}, inplace=True)
        df['Trade Date'] = pd.to_datetime(df['Trade Date'])
        df.set_index('Trade Date', inplace=True)
        
        # Identify Price Column
        cols = [c for c in df.columns if c != 'Trade Date']
        if not cols:
            return None
            
        price_col = cols[0]
        
        # Rename to the MARKET NAME (Column A)
        df.rename(columns={price_col: market_name}, inplace=True)
        
        # Numeric Conversion
        df[market_name] = pd.to_numeric(df[market_name], errors='coerce')
        
        return df[[market_name]]

    except Exception as e:
        print(f" [ERROR] {symbol_code}: Exception -> {e}")
        return None

def fetch_batch_data(symbol_map: Dict[str, str], field: str, start: str, end: str) -> pd.DataFrame:
    """Iterates through symbols and collects data."""
    collected_frames = []
    total = len(symbol_map)
    sorted_items = sorted(symbol_map.items()) 

    print(f"\n--- Fetching Data: {start} to {end} ---")
    
    for i, (code, name) in enumerate(sorted_items, 1):
        print(f"[{i}/{total}] {code:<5} -> {name[:30]:<30}", end='\r')
        
        df = fetch_single_symbol(code, name, field, start, end)
        
        if df is not None:
            collected_frames.append(df)

    print("\nFetch complete.")
    
    if not collected_frames:
        return pd.DataFrame()

    return pd.concat(collected_frames, axis=1)

def filter_weekends(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows where the Trade Date is a Saturday or Sunday."""
    logging.info("Filtering out weekend dates (Sat/Sun)...")
    initial_rows = len(df)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
        
    filtered_df = df[df.index.dayofweek < 5]
    dropped_rows = initial_rows - len(filtered_df)
    
    if dropped_rows > 0:
        logging.info(f"Dropped {dropped_rows} weekend trade dates.")
    
    return filtered_df

def apply_benchmark_filter(df: pd.DataFrame, symbol_map: Dict[str, str]) -> pd.DataFrame:
    """Removes rows where the benchmark symbol is empty."""
    benchmark_code = BENCHMARK_SYMBOL_CODE
    
    # Verify benchmark exists in our map
    if benchmark_code not in symbol_map:
        logging.warning(f"Benchmark code '{benchmark_code}' not found in current map. Skipping validation.")
        return df

    benchmark_name = symbol_map[benchmark_code]
    
    if benchmark_name not in df.columns:
        logging.warning(f"Benchmark '{benchmark_name}' ({benchmark_code}) has no data in this fetch.")
        return df

    logging.info(f"Filtering based on benchmark: {benchmark_name} ({benchmark_code})")
    
    initial_rows = len(df)
    filtered_df = df.dropna(subset=[benchmark_name])
    dropped_rows = initial_rows - len(filtered_df)
    
    if dropped_rows > 0:
        logging.info(f"Dropped {dropped_rows} dates missing benchmark data.")
    
    return filtered_df

def run_processing_pipeline(symbol_map: Dict[str, str], output_path: Path):
    """
    Executes the full pipeline for a given set of symbols and output path.
    """
    # 1. Setup Range
    start_date, end_date, existing_df = determine_fetch_range(output_path)

    # 2. Fetch
    new_df = fetch_batch_data(symbol_map, TARGET_FIELD, start_date, end_date)

    if new_df.empty:
        logging.warning("No new data fetched from API.")
        if existing_df is None:
            logging.error("No data available to save.")
            return

    # 3. Merge
    if existing_df is not None and not existing_df.empty:
        logging.info("Merging new data with existing history...")
        final_df = pd.concat([existing_df, new_df])
        # Deduplicate indices, keeping the last (newest) fetch
        final_df = final_df.groupby(final_df.index).last()
    else:
        final_df = new_df

    # 4. Filter Weekends
    final_df = filter_weekends(final_df)

    # 5. Filter by Benchmark
    final_df = apply_benchmark_filter(final_df, symbol_map)

    # 6. Save
    if not final_df.empty:
        final_df.sort_index(ascending=False, inplace=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            final_df.to_csv(output_path, na_rep='')
            logging.info(f"SUCCESS: Saved to {output_path.name}")
        except Exception as e:
            logging.critical(f"Failed to save file: {e}")
    else:
        logging.warning("Resulting dataset is empty after filtering. Nothing saved.")

# ==============================================================================
#  MAIN EXECUTION
# ==============================================================================

def main():
    setup_logging()
    
    # ---------------------------------------------------------
    # WORKFLOW 1: DAILY CODES (Column C) -> ICEPriceTD.csv
    # ---------------------------------------------------------
    logging.info("="*60)
    logging.info(f"Starting Workflow 1: Daily Codes (Col {COL_DAILY_CODE})")
    logging.info("="*60)
    
    symbol_map_td = load_symbol_map(INPUT_FILE_PATH, COL_DAILY_CODE)
    
    if symbol_map_td:
        output_td = OUTPUT_DIR / OUTPUT_FILENAME_TD
        run_processing_pipeline(symbol_map_td, output_td)
    else:
        logging.error("Skipping Workflow 1 due to map loading failure.")

    # ---------------------------------------------------------
    # WORKFLOW 2: GDA CODES (Column R) -> ICEPriceGDA.csv
    # ---------------------------------------------------------
    print("\n") # Visual spacer
    logging.info("="*60)
    logging.info(f"Starting Workflow 2: GDA Codes (Col {COL_GDA_CODE})")
    logging.info("="*60)
    
    symbol_map_gda = load_symbol_map(INPUT_FILE_PATH, COL_GDA_CODE)
    
    if symbol_map_gda:
        output_gda = OUTPUT_DIR / OUTPUT_FILENAME_GDA
        run_processing_pipeline(symbol_map_gda, output_gda)
    else:
        logging.error("Skipping Workflow 2 due to map loading failure.")

if __name__ == "__main__":
    main()