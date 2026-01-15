"""
Fetches and saves live Balance of Month ("Balmo") prices from the ICE API.

This script reads a list of markets from 'PriceAdmin.csv', constructs
the appropriate Balmo symbols (e.g., 'ALS B0-IUS'), and fetches live
bid, ask, and last prices.

It calculates a 'Mark' based on the following logic:
  1. Extended Logic:
     - If Bid and Ask spread is <= $0.20, AND
     - Last price is present and within [Bid - 0.05, Ask + 0.05],
     - Then Mark = (Bid + Ask + Last) / 3.
  2. Fallback Logic:
     - If Bid and Ask spread is <= $0.05 (and above failed),
     - Then Mark = (Bid + Ask) / 2.
  3. Otherwise, the Mark is left blank (NaN).

The final results are saved to 'balmo.csv' in the INFO directory.
"""

import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

# Assume icepython is available in the environment
try:
    import icepython as ice
except ImportError:
    logging.critical("FATAL: The 'icepython' library is not installed.")
    # Mock class for testing
    class MockIce:
        """Mock icepython for testing without the real API."""
        def get_quotes(self, symbols: List[str], fields: List[str]
                       ) -> List[List[Any]]:
            """Returns mock data."""
            logging.warning("Using MOCK icepython library.")
            headers = ["Symbol"] + fields
            # Format: [Symbol, Bid, Ask, Last]
            data = [
                headers,
                ["ALS B0-IUS", "6.30", "6.70", "6.50"],    # Wide spread (0.40) -> NaN
                ["XYZ B0-IUS", "2.50", "2.52", "2.51"],    # Narrow spread (0.02) -> (2.50+2.52+2.51)/3
                ["ABC B0-IUS", "3.00", None, "3.05"],      # Missing Ask -> NaN
                ["UCS B0-IUS", "3.25", "3.35", "3.30"],    # Spread 0.10, Last in range -> (3.25+3.35+3.30)/3
            ]
            # Filter mock data for requested symbols
            return [headers] + [row for row in data[1:] if row[0] in symbols]

    ice = MockIce()


# ==============================================================================
#  CONFIGURATION & CONSTANTS
# ==============================================================================

# --- File & Directory Paths ---
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

INFO_DIR = SCRIPT_DIR.parent / "INFO"
ADMIN_CSV_PATH = INFO_DIR / "PriceAdmin.csv"
OUTPUT_CSV_PATH = INFO_DIR / "balmo.csv"

# --- API Configuration ---
API_SUFFIX = " B0-IUS"
# Added 'Last' to fetch list
FIELDS_TO_FETCH = ['bid', 'Ask', 'Last']

# --- Mark Logic Configuration ---
MARK_SPREAD_THRESHOLD_STRICT = 0.20 
MARK_SPREAD_THRESHOLD_WIDE = 0.20
LAST_TOLERANCE = 0.05

# ==============================================================================
#  SETUP
# ==============================================================================


def setup_logging():
    """Configures a basic logger to show timestamped messages."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# ==============================================================================
#  CORE FUNCTIONS
# ==============================================================================


def load_admin_data(file_path: Path) -> pd.DataFrame:
    """Loads market data from the PriceAdmin CSV."""
    logging.info(f"Loading market data from {file_path.name}...")
    if not file_path.exists():
        raise FileNotFoundError(
            f"FATAL ERROR: The input file was not found at '{file_path}'"
        )

    admin_df = pd.read_csv(file_path)
    admin_df.dropna(subset=['Balmo'], inplace=True)
    return admin_df[['Market Component', 'Balmo']].drop_duplicates()


def generate_symbols(admin_df: pd.DataFrame, suffix: str) -> Dict[str, str]:
    """Generates API symbols from the admin DataFrame."""
    logging.info("Generating Balmo symbols...")
    market_to_symbol = {
        row['Market Component']: f"{row['Balmo']}{suffix}"
        for _, row in admin_df.iterrows()
    }
    logging.info(f"Generated {len(market_to_symbol)} symbols.")
    return market_to_symbol


def fetch_ice_quotes(symbols: List[str],
                     fields: List[str]) -> Optional[Dict[str, List[Any]]]:
    """Fetches live quote data from the ICE API."""
    if not symbols:
        logging.warning("No symbols provided to fetch.")
        return None

    logging.info(f"Fetching live prices for {len(symbols)} Balmo symbols...")
    try:
        quote_data = ice.get_quotes(symbols, fields)

        if not quote_data or len(quote_data) < 2:
            logging.warning("No data returned from the API for any symbols.")
            return None

        # quote_data[0] is headers. Rows are [Symbol, Field1, Field2, Field3]
        price_results = {row[0]: row[1:] for row in quote_data[1:]}

        logging.info(
            f"Successfully received data for {len(price_results)} symbols."
        )
        return price_results
    except Exception as e:
        logging.error(f"An error occurred during the ICE API call: {e}")
        logging.error(traceback.format_exc())
        return None


def calculate_mark(bid: float, ask: float, last: float) -> float:
    """
    Applies the specific mark logic.
    
    Logic:
    1. If Spread <= 0.20 AND Last is within [Bid-0.05, Ask+0.05]:
       Mark = Average(Bid, Ask, Last)
    2. Else If Spread <= 0.05:
       Mark = Average(Bid, Ask)
    3. Else:
       NaN
    """
    if pd.isna(bid) or pd.isna(ask):
        return np.nan

    spread = ask - bid
    
    # 1. Extended Logic with Last
    if spread <= MARK_SPREAD_THRESHOLD_WIDE and not pd.isna(last):
        # Last must be >= Bid - 0.05 AND <= Ask + 0.05
        if (bid - LAST_TOLERANCE) <= last <= (ask + LAST_TOLERANCE):
            return (bid + ask + last) / 3

    # 2. Strict Fallback Logic (Bid/Ask only)
    if spread <= MARK_SPREAD_THRESHOLD_STRICT:
        return (bid + ask) / 2

    return np.nan


def process_price_data(
    price_results: Dict[str, List[Any]],
    market_to_symbol_map: Dict[str, str]
) -> pd.DataFrame:
    """Processes raw API data and applies the marking logic."""
    logging.info("Processing and formatting API results...")
    output_data = []

    for market_name, symbol in market_to_symbol_map.items():
        prices = price_results.get(symbol)

        if prices:
            # Expecting [Bid, Ask, Last]
            bid = pd.to_numeric(prices[0], errors='coerce')
            ask = pd.to_numeric(prices[1], errors='coerce')
            
            # Handle case where Last might not be returned if API fails partial
            last = np.nan
            if len(prices) > 2:
                last = pd.to_numeric(prices[2], errors='coerce')

            # Apply mark logic
            mark = calculate_mark(bid, ask, last)

            output_data.append({
                'Market Component': market_name,
                'Mark': mark,
                'Bid': bid,
                'Ask': ask,
                'Last': last,
                'Symbol': symbol
            })
        else:
            output_data.append({
                'Market Component': market_name,
                'Mark': np.nan,
                'Bid': np.nan,
                'Ask': np.nan,
                'Last': np.nan,
                'Symbol': symbol
            })

    if not output_data:
        logging.warning("No data was processed.")
        return pd.DataFrame()

    final_df = pd.DataFrame(output_data)
    logging.info(f"Formatted {len(final_df)} markets.")
    return final_df


def save_results(df: pd.DataFrame, output_path: Path):
    """Saves the processed DataFrame to a CSV file."""
    if df.empty:
        logging.warning("DataFrame is empty. No file will be saved.")
        return

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Reorder columns for readability
        cols = ['Market Component', 'Mark', 'Bid', 'Ask', 'Last', 'Symbol']
        # Only use cols that exist
        cols = [c for c in cols if c in df.columns]
        
        df.to_csv(output_path, index=False, columns=cols, float_format='%.4f')
        logging.info(f"SUCCESS: Price table saved to:\n{output_path}")

    except IOError as e:
        logging.error(f"IOError saving file to {output_path}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during save: {e}")


# ==============================================================================
#  ORCHESTRATION
# ==============================================================================


def main():
    """Orchestrates the entire Balmo price fetching process."""
    setup_logging()
    logging.info("========= Starting ICE Balmo Price Fetcher =========")

    try:
        # 1. Load and prepare data
        admin_data = load_admin_data(ADMIN_CSV_PATH)
        market_to_symbol = generate_symbols(admin_data, API_SUFFIX)

        # 2. Fetch data from API
        price_data = fetch_ice_quotes(
            list(market_to_symbol.values()),
            FIELDS_TO_FETCH
        )

        # 3. Process and save results
        if price_data:
            final_df = process_price_data(
                price_data,
                market_to_symbol
            )
            save_results(final_df, OUTPUT_CSV_PATH)
        else:
            logging.warning("No price data was fetched. Halting process.")

    except FileNotFoundError as e:
        logging.critical(str(e))
    except Exception as e:
        logging.critical("An unexpected error occurred in the main process.")
        logging.critical(traceback.format_exc())
    finally:
        logging.info("================== Process Finished ==================")


if __name__ == "__main__":
    main()