"""
Fetches and calculates forward marks for the upcoming month from the ICE API.

This script reads a list of markets from 'PriceAdmin.csv', dynamically
constructs symbols for both 'Financial Basis' and 'Monthly Basis' contracts for
the next calendar month, and fetches live price data (bid, ask, settle, last).

It then applies a complex set of rules to consolidate these prices into a single
'Final Mark' for each market and saves the detailed results to a CSV file.
"""

import logging
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import icepython as ice
import numpy as np
import pandas as pd

# ==============================================================================
#  CONFIGURATION & CONSTANTS
# ==============================================================================

# --- File & Directory Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
INFO_DIR = SCRIPT_DIR.parent / "INFO"
ADMIN_CSV_PATH = INFO_DIR / "PriceAdmin.csv"
OUTPUT_CSV_PATH = INFO_DIR / "ForwardMarks.csv"

# --- API Configuration ---
FIN_BASIS_SUFFIX = "-IUS"
MONTHLY_BASIS_SUFFIX = "-IPG"
API_FIELDS_TO_REQUEST = ['bid', 'Ask', 'recset', 'last']

# --- Futures Month Codes ---
FUTURES_MONTH_CODES = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
}

# --- Final Output Column Order ---
FINAL_COLUMN_ORDER = [
    'Market Component', 'Final Mark', 'Consolidated Bid', 'Consolidated Ask',
    'Valid Last Mark', 'Valid Settle Mark', 'Fin Bid', 'Fin Ask', 'Fin Prev. Settle',
    'Fin Last', 'Fin Symbol', 'Monthly Bid', 'Monthly Ask', 'Monthly Prev. Settle',
    'Monthly Last', 'Monthly Symbol'
]

# ==============================================================================
#  SETUP & HELPER FUNCTIONS
# ==============================================================================

def setup_logging():
    """Configures a basic logger to show timestamped messages."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# ==============================================================================
#  CORE PIPELINE FUNCTIONS
# ==============================================================================

def generate_symbols(df: pd.DataFrame) -> Tuple[Dict[str, List[str]], List[str]]:
    """Generates the API symbols for the upcoming month based on the admin data."""
    logging.info("Generating API symbols for the next calendar month...")
    today = datetime.now()
    # Add 32 days to safely jump to the next month, then reset to the 1st.
    next_month = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
    
    year_code = str(next_month.year)[-2:]
    month_code = FUTURES_MONTH_CODES.get(next_month.month)
    if not month_code:
        raise ValueError(f"Could not determine futures code for month: {next_month.month}")

    logging.info(f"Targeting upcoming month: {next_month:%B %Y} (Symbol Suffix: {month_code}{year_code})")

    market_to_symbols: Dict[str, List[str]] = {}
    all_symbols = []
    for _, row in df.iterrows():
        market = row['Market Component']
        symbols_for_market = []
        if pd.notna(row['Fin Basis']):
            symbols_for_market.append(f"{row['Fin Basis']} {month_code}{year_code}{FIN_BASIS_SUFFIX}")
        if pd.notna(row['Monthly Basis']):
            symbols_for_market.append(f"{row['Monthly Basis']} {month_code}{year_code}{MONTHLY_BASIS_SUFFIX}")
        
        if symbols_for_market:
            market_to_symbols[market] = symbols_for_market
            all_symbols.extend(symbols_for_market)
    
    unique_symbols = sorted(list(set(all_symbols)))
    logging.info(f"Generated {len(unique_symbols)} unique symbols to query.")
    return market_to_symbols, unique_symbols

def fetch_ice_quotes(symbols: List[str]) -> Optional[Dict[str, List]]:
    """Fetches live quote data from the ICE API for a list of symbols."""
    if not symbols:
        logging.warning("No symbols were generated to fetch.")
        return None

    logging.info(f"Fetching live prices for {len(symbols)} symbols from ICE API...")
    try:
        quote_data = ice.get_quotes(symbols, API_FIELDS_TO_REQUEST)
        if not quote_data or len(quote_data) < 2:
            logging.warning("No data returned from the API for any symbols.")
            return None
        
        price_results = {row[0]: row[1:] for row in quote_data[1:]}
        logging.info(f"Successfully received data for {len(price_results)} symbols.")
        return price_results
    except Exception as e:
        logging.error(f"An error occurred during the ICE API call: {e}")
        return None

def process_api_results(price_results: Dict[str, list], market_to_symbols: Dict[str, list]) -> pd.DataFrame:
    """Processes the raw API data and formats it into a wide-format DataFrame."""
    logging.info("Processing and formatting API results into a wide-format table...")
    processed_data = {}
    for market, symbols in market_to_symbols.items():
        market_data = {}
        for symbol in symbols:
            prices = price_results.get(symbol)
            if prices:
                prefix = "Fin" if symbol.endswith(FIN_BASIS_SUFFIX) else "Monthly"
                market_data[f'{prefix} Bid'] = pd.to_numeric(prices[0], errors='coerce')
                market_data[f'{prefix} Ask'] = pd.to_numeric(prices[1], errors='coerce')
                market_data[f'{prefix} Prev. Settle'] = pd.to_numeric(prices[2], errors='coerce')
                market_data[f'{prefix} Last'] = pd.to_numeric(prices[3], errors='coerce')
                market_data[f'{prefix} Symbol'] = symbol
        if market_data:
            processed_data[market] = market_data

    df = pd.DataFrame.from_dict(processed_data, orient='index').reset_index().rename(columns={'index': 'Market Component'})
    price_cols = [col for col in df.columns if any(price in col for price in ['Bid', 'Ask', 'Settle', 'Last'])]
    return df.dropna(subset=price_cols, how='all')

def calculate_final_marks(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the specific business logic to calculate a consolidated final mark."""
    logging.info("Calculating consolidated bids, asks, and final marks...")
    
    # --- 1. Consolidate Bid ---
    # The consolidated bid is the higher of the two bids, unless the Fin Bid
    # is invalid (higher than the Monthly Ask), in which case we must use the Monthly Bid.
    higher_bid = df[['Fin Bid', 'Monthly Bid']].max(axis=1)
    is_fin_bid_invalid = (df['Fin Bid'] > df['Monthly Ask'])
    df['Consolidated Bid'] = np.where(
        (higher_bid == df['Fin Bid']) & is_fin_bid_invalid,
        df['Monthly Bid'],
        higher_bid
    )

    # --- 2. Consolidate Ask ---
    # The consolidated ask is the lower of the two asks, unless the Fin Ask
    # is invalid (lower than the Monthly Bid), in which case we must use the Monthly Ask.
    lower_ask = df[['Fin Ask', 'Monthly Ask']].min(axis=1)
    is_fin_ask_invalid = (df['Fin Ask'] < df['Monthly Bid'])
    df['Consolidated Ask'] = np.where(
        (lower_ask == df['Fin Ask']) & is_fin_ask_invalid,
        df['Monthly Ask'],
        lower_ask
    )

    # --- 3. Validate Last and Settle Prices ---
    # A valid Last/Settle must fall between the consolidated bid and ask.
    avg_last = df[['Fin Last', 'Monthly Last']].mean(axis=1)
    is_last_invalid = (avg_last < df['Consolidated Bid']) | (avg_last > df['Consolidated Ask'])
    df['Valid Last Mark'] = np.where(is_last_invalid, np.nan, avg_last)

    avg_settle = df[['Fin Prev. Settle', 'Monthly Prev. Settle']].mean(axis=1)
    is_settle_invalid = (avg_settle < df['Consolidated Bid']) | (avg_settle > df['Consolidated Ask'])
    df['Valid Settle Mark'] = np.where(is_settle_invalid, np.nan, avg_settle)

    # --- 4. Calculate Final Mark ---
    # The Final Mark is the average of all available, valid price components for that market.
    marks_to_average = df[['Consolidated Bid', 'Consolidated Ask', 'Valid Last Mark', 'Valid Settle Mark']]
    df['Final Mark'] = marks_to_average.mean(axis=1, skipna=True)
    
    logging.info("Mark calculation complete.")
    return df

# ==============================================================================
#  ORCHESTRATION
# ==============================================================================

def main():
    """Orchestrates the entire forward marks fetching and calculation process."""
    setup_logging()
    logging.info("========= Starting ICE Forward Marks Update Process =========")

    try:
        logging.info(f"Loading market data from {ADMIN_CSV_PATH.name}...")
        admin_df = pd.read_csv(ADMIN_CSV_PATH)
        
        market_to_symbols, symbols_to_query = generate_symbols(admin_df)
        
        price_data = fetch_ice_quotes(symbols_to_query)
        if not price_data:
            logging.warning("Halting execution: No data returned from API.")
            return

        wide_df = process_api_results(price_data, market_to_symbols)
        if wide_df.empty:
            logging.warning("Halting execution: No valid price data to process.")
            return

        final_df = calculate_final_marks(wide_df)
        
        if not final_df.empty:
            INFO_DIR.mkdir(parents=True, exist_ok=True)
            final_df = final_df.reindex(columns=FINAL_COLUMN_ORDER)
            final_df.to_csv(OUTPUT_CSV_PATH, index=False, float_format='%.4f')
            logging.info(f"SUCCESS: Price table with {len(final_df)} entries saved to:\n{OUTPUT_CSV_PATH}")

    except FileNotFoundError:
        logging.critical(f"FATAL ERROR: The input file was not found at '{ADMIN_CSV_PATH}'")
    except Exception:
        logging.critical("An unexpected error occurred in the main process.")
        logging.critical(traceback.format_exc())
    finally:
        logging.info("================== Process Finished ==================")

if __name__ == "__main__":
    main()