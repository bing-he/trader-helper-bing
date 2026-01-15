"""
ICE Seasonal Strip Price Fetcher.

This script fetches live bid, ask, and previous settle prices for Summer (Apr-Oct)
and Winter (Nov-Mar) strips from the ICE API. It dynamically calculates the 
upcoming strip symbols based on the current date, performs a batched API 
request for efficiency, and saves the results into separate CSV files.

Complies with PEP8, PEP257, and SOLID principles.
"""

import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import icepython as ice
import pandas as pd

# ==============================================================================
#  CONFIGURATION & PATH RESOLUTION
# ==============================================================================

# Dynamic path resolution to avoid hardcoding
SCRIPT_DIR = Path(__file__).resolve().parent
# Assuming standard structure: root/ICE/script.py and root/INFO/
BASE_DIR = SCRIPT_DIR.parent
INFO_DIR = BASE_DIR / "INFO"
ADMIN_CSV_PATH = INFO_DIR / "PriceAdmin.csv"

# Output filenames
WINTER_OUTPUT = INFO_DIR / "ICE_Winter_Strip_Prices.csv"
SUMMER_OUTPUT = INFO_DIR / "ICE_Summer_Strip_Prices.csv"

# API Settings
API_SUFFIX = "-IUS"
API_FIELDS = ["bid", "Ask", "recset"]


# ==============================================================================
#  LOGGING SETUP
# ==============================================================================

def setup_logging() -> None:
    """Configure timestamped logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ==============================================================================
#  CORE LOGIC CLASSES
# ==============================================================================

class StripGenerator:
    """Generates API symbols for seasonal strips based on current date logic."""

    def __init__(self, today: Optional[datetime] = None):
        """Initialize with a specific date for testing or default to now."""
        self.today = today or datetime.now()

    def get_winter_symbols(self, admin_df: pd.DataFrame) -> Dict[str, str]:
        """
        Calculate Winter Strip (Nov-Mar) symbols.
        
        Logic: If Nov/Dec, fetch next winter. Otherwise, fetch current/upcoming winter.
        """
        start_year = self.today.year if self.today.month < 11 else self.today.year + 1
        end_year = start_year + 1
        
        start_yy = str(start_year)[-2:]
        end_yy = str(end_year)[-2:]

        return {
            row["Market Component"]: f"{row['Fin Basis']} X{start_yy}SR:{row['Fin Basis']}H{end_yy}{API_SUFFIX}"
            for _, row in admin_df.iterrows()
        }

    def get_summer_symbols(self, admin_df: pd.DataFrame) -> Dict[str, str]:
        """
        Calculate Summer Strip (Apr-Oct) symbols.
        
        Logic: Roll date is April 1st.
        """
        strip_year = self.today.year if self.today.month < 4 else self.today.year + 1
        strip_yy = str(strip_year)[-2:]

        return {
            row["Market Component"]: f"{row['Fin Basis']} J{strip_yy}SR:{row['Fin Basis']}V{strip_yy}{API_SUFFIX}"
            for _, row in admin_df.iterrows()
        }


class IceDataOrchestrator:
    """Handles API communication and data formatting."""

    @staticmethod
    def fetch_batch_quotes(symbols: List[str]) -> Dict[str, List]:
        """
        Fetch quotes from ICE API in a single batched request.
        
        Args:
            symbols: List of unique ICE symbols.
            
        Returns:
            Dictionary mapping symbols to [bid, ask, settle].
        """
        if not symbols:
            return {}

        logging.info(f"Requesting {len(symbols)} symbols from ICE API...")
        try:
            quote_data = ice.get_quotes(symbols, API_FIELDS)
            if not quote_data or len(quote_data) < 2:
                return {}
            
            # Map symbol to the data row (skipping headers in index 0)
            return {row[0]: row[1:] for row in quote_data[1:]}
        except Exception as e:
            logging.error(f"ICE API Error: {e}")
            return {}

    @staticmethod
    def format_results(
        price_data: Dict[str, List], 
        market_map: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Convert raw API results into a structured DataFrame.
        
        Args:
            price_data: Dictionary of prices from the API.
            market_map: Mapping of Market Component names to API symbols.
        """
        processed_rows = []
        for market_name, symbol in market_map.items():
            prices = price_data.get(symbol)
            if not prices:
                continue

            bid = pd.to_numeric(prices[0], errors="coerce")
            ask = pd.to_numeric(prices[1], errors="coerce")
            settle = pd.to_numeric(prices[2], errors="coerce")

            if any(pd.notna(x) for x in [bid, ask, settle]):
                processed_rows.append({
                    "Market Component": market_name,
                    "Bid": bid,
                    "Ask": ask,
                    "Prev. Settle": settle,
                    "Symbol": symbol
                })

        return pd.DataFrame(processed_rows)


# ==============================================================================
#  MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution flow for fetching and saving seasonal strips."""
    setup_logging()
    logging.info("Starting Combined ICE Seasonal Strip Fetcher")

    try:
        # 1. Load Admin Data
        if not ADMIN_CSV_PATH.exists():
            logging.error(f"Admin file not found: {ADMIN_CSV_PATH}")
            return

        admin_df = pd.read_csv(ADMIN_CSV_PATH).dropna(subset=["Fin Basis"])
        
        # 2. Generate Symbols for both seasons
        generator = StripGenerator()
        winter_map = generator.get_winter_symbols(admin_df)
        summer_map = generator.get_summer_symbols(admin_df)

        # 3. Batched API Request
        all_unique_symbols = list(set(list(winter_map.values()) + list(summer_map.values())))
        raw_price_data = IceDataOrchestrator.fetch_batch_quotes(all_unique_symbols)

        if not raw_price_data:
            logging.warning("No price data retrieved from ICE API.")
            return

        # 4. Process and Save
        orchestrator = IceDataOrchestrator()
        INFO_DIR.mkdir(parents=True, exist_ok=True)

        # Process Winter
        winter_df = orchestrator.format_results(raw_price_data, winter_map)
        if not winter_df.empty:
            winter_df.to_csv(WINTER_OUTPUT, index=False, float_format="%.4f")
            logging.info(f"Winter prices saved to {WINTER_OUTPUT.name}")

        # Process Summer
        summer_df = orchestrator.format_results(raw_price_data, summer_map)
        if not summer_df.empty:
            summer_df.to_csv(SUMMER_OUTPUT, index=False, float_format="%.4f")
            logging.info(f"Summer prices saved to {SUMMER_OUTPUT.name}")

        logging.info("Process completed successfully.")

    except Exception:
        logging.critical("An unhandled exception occurred.")
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()