import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from common.logs import get_file_logger
from common.pathing import ROOT

CONFIG = {
    "price_admin": ROOT / "INFO" / "PriceAdmin.csv",
}
logger = get_file_logger(Path(__file__).stem)


def _resolve_path(path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate

# API Endpoints
AUTH_URL = "https://api.ci.spglobal.com/auth/api"
REFERENCE_DATA_SEARCH_URL = "https://api.ci.spglobal.com/market-data/reference-data/v3/search"

# Symbol Fetching Configuration
SYMBOL_API_Q_KEYWORD = "Natural Gas" 
FIELDS_TO_RETRIEVE = ["symbol", "description"]
PAGE_SIZE = 1000 
SUBSCRIBED_ONLY = True 

# Local Filtering Criteria
LOCAL_DESCRIPTION_FILTER_ENDSWITH = "fdt com" # Case-insensitive
PRELIM_FILTER_EXCLUDE_KEYWORD = "prelim" # Case-insensitive

def load_credentials() -> tuple[str | None, str | None]:
    load_dotenv(ROOT / ".env")
    return os.getenv("PLATTS_USERNAME"), os.getenv("PLATTS_PASSWORD")


def get_access_token(username, password):
    """Authenticates and retrieves an access token."""
    if not username or not password:
        logger.error("Username or Password not found. Set PLATTS_USERNAME/PLATTS_PASSWORD in %s", ROOT / ".env")
        return None
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = {"username": username, "password": password}
    logger.info("Attempting to obtain access token...")
    try:
        response = requests.post(AUTH_URL, headers=headers, data=payload, timeout=30)
        response.raise_for_status()  
        token_data = response.json()
        logger.info("Successfully obtained access token.")
        return token_data.get("access_token")
    except requests.exceptions.HTTPError as http_err:
        logger.error("Http Error during authentication: %s", http_err)
        if http_err.response is not None:
            logger.error("Response status: %s", http_err.response.status_code)
            logger.error("Response content: %s", http_err.response.content.decode())
    except Exception as e:
        logger.error("An unexpected error occurred during authentication: %s", e)
    return None

def fetch_and_filter_symbols(token):
    """Fetches and filters symbols based on predefined criteria."""
    all_symbols_data = []
    page = 1
    total_pages = 1 
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    logger.info("Starting API symbol search with keyword (q): '%s'.", SYMBOL_API_Q_KEYWORD)
    
    while page <= total_pages:
        params = {
            "q": SYMBOL_API_Q_KEYWORD,
            "Field": ",".join(FIELDS_TO_RETRIEVE), 
            "PageSize": PAGE_SIZE, 
            "Page": page, 
            "subscribed_only": str(SUBSCRIBED_ONLY).lower()
        }
        try:
            # print(f"Fetching page {page} of symbols...") # Verbose logging
            response = requests.get(REFERENCE_DATA_SEARCH_URL, headers=headers, params=params, timeout=60) 
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])
            if results:
                all_symbols_data.extend(results)
            
            metadata = data.get("metadata", {})
            current_total_pages = metadata.get('totalPages', metadata.get('total_pages'))
            if current_total_pages is not None:
                total_pages = current_total_pages
            
            if page >= total_pages or not results: 
                break
            page += 1
            time.sleep(0.2) 
        except requests.exceptions.HTTPError as http_err:
            logger.error("HTTP Error during API symbol search (page %s): %s", page, http_err)
            return None
        except Exception as e:
            logger.error("An unexpected error during API symbol search (page %s): %s", page, e)
            return None

    logger.info("Total symbols data fetched from API: %s", len(all_symbols_data))

    if all_symbols_data:
        df_raw_symbols = pd.DataFrame(all_symbols_data)
        
        if 'description' in df_raw_symbols.columns:
            df_raw_symbols['description'] = df_raw_symbols['description'].astype(str)
            
            df_filtered = df_raw_symbols[
                df_raw_symbols['description'].str.lower().str.endswith(LOCAL_DESCRIPTION_FILTER_ENDSWITH, na=False)
            ]
            
            df_filtered = df_filtered[
                ~df_filtered['description'].str.lower().str.contains(PRELIM_FILTER_EXCLUDE_KEYWORD, na=False)
            ]
            logger.info("Filtered to %s relevant symbols.", len(df_filtered))
            return df_filtered
        else:
            logger.warning("'description' column not found. Cannot apply filters.")
            return pd.DataFrame()
    return pd.DataFrame()

def main():
    """Main function to drive the PriceAdmin update script."""
    logger.info("--- Starting PriceAdminPLATTS Update Script ---")
    price_admin_path = _resolve_path(CONFIG["price_admin"])
    price_admin_path.parent.mkdir(parents=True, exist_ok=True)

    username, password = load_credentials()
    access_token = get_access_token(username, password)
    if not access_token:
        logger.error("Could not get access token. Aborting.")
        return

    # 1. Fetch all relevant symbols from Platts API
    platts_symbols_df = fetch_and_filter_symbols(access_token)
    if platts_symbols_df is None or platts_symbols_df.empty:
        logger.error("No symbols were fetched from the API. Aborting.")
        return

    # 2. Load the PriceAdmin.csv file, specifying dtypes to prevent warnings
    try:
        # Define the data types for columns that might mix strings and NaNs
        dtype_spec = {
            'PlattsCodePlatts': 'object',
            'PlattsCodeIce': 'object'
        }
        price_admin_df = pd.read_csv(price_admin_path, dtype=dtype_spec)
        logger.info("Successfully loaded '%s' with %s rows.", price_admin_path.name, len(price_admin_df))
    except FileNotFoundError:
        logger.error("PriceAdmin.csv not found at path: %s", price_admin_path)
        return
    except Exception as e:
        logger.error("Could not read PriceAdmin.csv. Reason: %s", e)
        return

    # 3. Separate Platts symbols into ICE and non-ICE
    platts_symbols_df['description_lower'] = platts_symbols_df['description'].str.lower()
    ice_symbols_df = platts_symbols_df[platts_symbols_df['description_lower'].str.startswith('ice ')].copy()
    non_ice_symbols_df = platts_symbols_df[~platts_symbols_df['description_lower'].str.startswith('ice ')].copy()
    
    # Create mapping dictionaries for efficient lookups
    ice_desc_map = pd.Series(ice_symbols_df.symbol.values, index=ice_symbols_df.description).to_dict()
    non_ice_desc_map = pd.Series(non_ice_symbols_df.symbol.values, index=non_ice_symbols_df.description).to_dict()

    # 4. Update missing codes in existing rows
    # For non-ICE (Column L -> K)
    codes_to_add_non_ice = price_admin_df['Platts Market'].map(non_ice_desc_map)
    update_mask_non_ice = price_admin_df['PlattsCodePlatts'].isnull() & codes_to_add_non_ice.notnull()
    price_admin_df.loc[update_mask_non_ice, 'PlattsCodePlatts'] = codes_to_add_non_ice[update_mask_non_ice]
    logger.info("Updated %s missing codes in Column K (PlattsCodePlatts).", update_mask_non_ice.sum())

    # For ICE (Column N -> M)
    codes_to_add_ice = price_admin_df['Platt Market w/ ICE in the name'].map(ice_desc_map)
    update_mask_ice = price_admin_df['PlattsCodeIce'].isnull() & codes_to_add_ice.notnull()
    price_admin_df.loc[update_mask_ice, 'PlattsCodeIce'] = codes_to_add_ice[update_mask_ice]
    logger.info("Updated %s missing codes in Column M (PlattsCodeIce).", update_mask_ice.sum())

    # 5. Find and append new symbols
    # New non-ICE symbols
    existing_non_ice = set(price_admin_df['Platts Market'].dropna())
    new_non_ice_df = non_ice_symbols_df[~non_ice_symbols_df['description'].isin(existing_non_ice)]
    
    # New ICE symbols
    existing_ice = set(price_admin_df['Platt Market w/ ICE in the name'].dropna())
    new_ice_df = ice_symbols_df[~ice_symbols_df['description'].isin(existing_ice)]

    logger.info("Found %s new non-ICE symbols to add.", len(new_non_ice_df))
    logger.info("Found %s new ICE symbols to add.", len(new_ice_df))

    if not new_non_ice_df.empty or not new_ice_df.empty:
        # Create a spacer of 5 empty rows
        spacer_df = pd.DataFrame(np.nan, index=range(5), columns=price_admin_df.columns)
        
        # Prepare new non-ICE rows
        new_non_ice_to_append = pd.DataFrame({
            'Platts Market': new_non_ice_df['description'],
            'PlattsCodePlatts': new_non_ice_df['symbol']
        })

        # Prepare new ICE rows
        new_ice_to_append = pd.DataFrame({
            'Platt Market w/ ICE in the name': new_ice_df['description'],
            'PlattsCodeIce': new_ice_df['symbol']
        })

        # Concatenate everything
        price_admin_df = pd.concat([
            price_admin_df, 
            spacer_df, 
            new_non_ice_to_append,
            spacer_df,
            new_ice_to_append
        ], ignore_index=True)

    # 6. Save the updated DataFrame back to CSV
    try:
        price_admin_df.to_csv(price_admin_path, index=False)
        logger.info("Successfully updated and saved '%s'.", price_admin_path.name)
    except Exception as e:
        logger.error("Could not save the updated file. Reason: %s", e)

    logger.info("--- Script finished. ---")

if __name__ == "__main__":
    main()
