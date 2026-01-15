import sys
import os
from pathlib import Path
from time import sleep
from datetime import datetime

# --- 1. PATH FIX (Necessary for imports to work) ---
# Current file: .../TraderHelper/Platts/Prices/UpdateHistoricalFOM.py
# We need to reach: .../TraderHelper/ to import 'common'
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent.parent
sys.path.append(str(project_root))

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

# Now these imports will work
from common.logs import get_file_logger
from common.pathing import ROOT

CONFIG = {
    "info_dir": ROOT / "INFO",
    "priceadmin": ROOT / "INFO" / "PriceAdmin.csv",
    "historical_fom": ROOT / "INFO" / "HistoricalFOM.csv",
}

logger = get_file_logger(Path(__file__).stem)


def _resolve_path(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate

AUTH_URL = "https://api.ci.spglobal.com/auth/api"
HISTORY_URL = "https://api.ci.spglobal.com/market-data/v3/value/history/symbol"

def load_credentials() -> tuple[str | None, str | None]:
    # Updated: Look for .env in the Platts folder specifically
    env_path = ROOT / "Platts" / ".env"
    print(f"Loading .env from: {env_path}") # Debug visibility
    load_dotenv(env_path)
    return os.getenv("PLATTS_USERNAME"), os.getenv("PLATTS_PASSWORD")


# --- Auth ---
def get_access_token(username, password):
    """Requests and returns an access token from the Platts API."""
    logger.info("Requesting access token...")
    print("Requesting access token...") # Updated for visibility
    try:
        res = requests.post(
            AUTH_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={"username": username, "password": password},
            timeout=20
        )
        res.raise_for_status()
        token = res.json().get("access_token")
        logger.info("Token acquired.")
        print("Token acquired.") # Updated for visibility
        return token
    except Exception as e:
        logger.error("Auth error: %s", e)
        print(f"Auth error: {e}") # Updated for visibility
        return None

# --- Platts API Call ---
def fetch_iferc_monthly_prices(token, symbol, start_date):
    """Fetches historical price data for a given symbol from a start date."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json"
    }
    filter_str = (
        f'symbol IN ("{symbol}") AND bate IN ("u", "c") '
        f'AND assessDate >= "{start_date}"'
    )
    params = {
        "Filter": filter_str,
        "PageSize": 1000,
        "Sort": "assessDate:asc"
    }
    try:
        res = requests.get(HISTORY_URL, headers=headers, params=params)
        res.raise_for_status()
        results = res.json().get("results", [])
        if not results or not results[0].get("data"):
            return []
        raw_data = results[0]["data"]
        return [
            {
                "Date": r.get("assessDate"),
                "Symbol": symbol,
                "Price": pd.to_numeric(r.get("value"), errors="coerce")
            }
            for r in raw_data
        ]
    except Exception as e:
        logger.error("Error fetching %s: %s", symbol, e)
        print(f"Error fetching {symbol}: {e}")
        return []

# --- Main Execution ---
def main():
    """Main function to orchestrate the data fetching and processing."""
    print("--- Starting Script ---") # Updated for visibility
    paths = {k: _resolve_path(v) for k, v in CONFIG.items()}
    info_dir = paths["info_dir"]
    priceadmin_path = paths["priceadmin"]
    historical_path = paths["historical_fom"]
    info_dir.mkdir(parents=True, exist_ok=True)

    username, password = load_credentials()
    if not username or not password:
        logger.error("PLATTS_USERNAME/PLATTS_PASSWORD not set; update %s or export env vars.", ROOT / ".env")
        print("Error: Credentials not set in .env") # Updated for visibility
        return

    fetch_start_date = "2015-01-01"
    existing_df = pd.DataFrame()

    # Check for existing file and determine update strategy
    if historical_path.exists():
        logger.info("Found existing file: %s", historical_path)
        print(f"Found existing file: {historical_path}") # Updated for visibility
        try:
            existing_df = pd.read_csv(historical_path)
            
            if not existing_df.empty:
                existing_df['Date'] = pd.to_datetime(existing_df['settlement_year'].astype(str) + '-' + existing_df['settlement_month'], format='%Y-%B')
                last_date = existing_df['Date'].max()
                fetch_start_date_dt = (last_date - relativedelta(months=2)).replace(day=1)
                fetch_start_date = fetch_start_date_dt.strftime('%Y-%m-%d')

                logger.info(
                    "Last record found on: %s. Removing local data from %s onwards for a fresh update.",
                    last_date.strftime('%B %Y'),
                    fetch_start_date_dt.strftime('%B %Y'),
                )
                print(f"Refreshing data from {fetch_start_date}...") # Updated for visibility

                existing_df = existing_df[existing_df['Date'] < fetch_start_date_dt].copy()
                existing_df.drop(columns=['Date'], inplace=True)

        except Exception as e:
            logger.warning("Could not process existing file; starting a full re-fetch. Error: %s", e)
            print("Error processing existing file. Starting full re-fetch.") # Updated for visibility
            fetch_start_date = "2015-01-01"
            existing_df = pd.DataFrame()
    else:
        logger.info("No existing data file found at %s. Performing a full fetch from 2015 onwards.", historical_path)
        print("No existing file. Starting full fetch.") # Updated for visibility

    token = get_access_token(username, password)
    if not token:
        return
    
    try:
        priceadmin = pd.read_csv(priceadmin_path, dtype=str)
    except FileNotFoundError:
        logger.error("CRITICAL ERROR: Input file 'PriceAdmin.csv' not found at: %s", priceadmin_path)
        print(f"CRITICAL ERROR: 'PriceAdmin.csv' missing at {priceadmin_path}") # Updated for visibility
        return

    code_map = priceadmin[['Market Component', 'PlattsForwardCode']].dropna()
    code_map = code_map[code_map['PlattsForwardCode'].str.strip() != '']

    all_new_data = []
    henry_prices = []

    print(f"Processing {len(code_map)} symbols...") # Updated for visibility
    for _, row in code_map.iterrows():
        mc, symbol = row['Market Component'], row['PlattsForwardCode']
        logger.info("Fetching: %s (%s) from %s", mc, symbol, fetch_start_date)
        # print(f"Fetching: {mc}...") # Optional: uncomment if you want line-by-line updates
        data = fetch_iferc_monthly_prices(token, symbol, start_date=fetch_start_date)
        for d in data:
            d['market_component'] = mc
        all_new_data.extend(data)
        if mc.lower() == "henry":
            henry_prices.extend(data)
        sleep(0.5)

    if not all_new_data:
        logger.warning("No new data was fetched from the API. File remains unchanged.")
        print("No new data fetched. File unchanged.") # Updated for visibility
        if not existing_df.empty:
            existing_df.to_csv(historical_path, index=False)
        return

    # Process newly fetched data
    new_df = pd.DataFrame(all_new_data)
    new_df['Date'] = pd.to_datetime(new_df['Date'])
    new_df = new_df[new_df['Date'].dt.day == 1]
    new_df = new_df.sort_values(['market_component', 'Date']).reset_index(drop=True)
    new_df.rename(columns={"Price": "settlement"}, inplace=True)

    henry_df = pd.DataFrame(henry_prices)

    # Check if the DataFrame is not empty before processing it
    if not henry_df.empty:
        henry_df['Date'] = pd.to_datetime(henry_df['Date'])
        henry_df = henry_df[henry_df['Date'].dt.day == 1]
        henry_df = henry_df.rename(columns={"Price": "Henry_Price"})[['Date', 'Henry_Price']]
    else:
        logger.warning("No new Henry Hub prices were fetched. Basis will not be calculated for new data.")
        print("Warning: No Henry Hub prices fetched.") # Updated for visibility
        henry_df = pd.DataFrame(columns=['Date', 'Henry_Price'])

    final_new_df = pd.merge(new_df, henry_df, on='Date', how='left')
    final_new_df['settlement_basis'] = final_new_df['settlement'] - final_new_df['Henry_Price']
    final_new_df['settlement_month'] = final_new_df['Date'].dt.strftime('%B')
    final_new_df['settlement_year'] = final_new_df['Date'].dt.year
    final_new_df = final_new_df[['settlement_month', 'settlement_year', 'market_component', 'settlement', 'settlement_basis']]

    # Combine old and new data
    combined_df = pd.concat([existing_df, final_new_df], ignore_index=True)
    combined_df.sort_values(['market_component', 'settlement_year', 'settlement_month'], inplace=True)

    combined_df.to_csv(historical_path, index=False)
    logger.info("Successfully wrote %s rows to file.", len(combined_df))
    logger.info("Output path: %s", historical_path)
    print(f"Success! Wrote {len(combined_df)} rows to {historical_path}") # Updated for visibility

if __name__ == "__main__":
    main()