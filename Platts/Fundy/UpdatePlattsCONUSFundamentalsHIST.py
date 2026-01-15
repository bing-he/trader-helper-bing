# update_gdmf_fundamentals.py

import os
import requests
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv

# --- Config ---
PLATTS_AUTH_API_URL = "https://api.ci.spglobal.com/auth/api"
PLATTS_NEWS_API_URL = "https://api.ci.spglobal.com/news-insights"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(SCRIPT_DIR, '..', '.env')
HIST_CSV_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'INFO', 'PlattsCONUSFundamentalsHIST.csv')

# --- Auth ---
def get_access_token(username, password):
    """Authenticates with the Platts API to get an access token."""
    payload = {"username": username, "password": password}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(PLATTS_AUTH_API_URL, data=payload, headers=headers)
    response.raise_for_status()
    return response.json().get("access_token")

# --- Find latest content ID ---
def find_latest_package(token):
    """Finds the most recent 'Gas Daily Market Fundamentals Data' package ID."""
    url = f"{PLATTS_NEWS_API_URL}/v1/search/packages"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    params = {
        "field": "publication",
        "filter": 'publication:"Gas Daily Market Fundamentals Data"',
        "PageSize": 1
    }
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    results = response.json().get("results", [])
    if results:
        return results[0]["id"]
    return None

# --- Download Excel content (in memory) ---
def download_excel_content(token, content_id):
    """Downloads the Excel file content for a given content ID."""
    url = f"{PLATTS_NEWS_API_URL}/v1/content/{content_id}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/octet-stream"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return BytesIO(response.content)

# --- Update Historical CSV ---
def update_historical_csv(excel_io, hist_csv_path):
    """
    Reads new data, renames columns, merges with historical data,
    and saves the updated CSV.
    """
    # Define the mapping from old column names to new column names.
    # This makes it easy to see all changes in one place.
    COLUMN_RENAME_MAP = {
        "GasDate": "Date",
        "LNG Sendout": "LNGSendout",
        "Net Imports from Canada": "CADImp",
        "NetExports to Mexico": "MexExp",
        "Total Demand": "TotalDemand",
        "LNG Feedgas": "LNGFeedgas"
    }

    # 1. Process the newly downloaded Excel data
    print("Reading 'US SupplyDemand' tab from downloaded Excel content...")
    new_df = pd.read_excel(excel_io, sheet_name="US SupplyDemand")
    new_df.columns = new_df.columns.map(str) # Ensure column names are strings
    
    # The first column from the source has no name, so we name it 'GasDate' first.
    new_df.rename(columns={new_df.columns[0]: "GasDate"}, inplace=True)
    
    # Now, apply the new, standardized names from our map.
    new_df.rename(columns=COLUMN_RENAME_MAP, inplace=True)
    
    # Convert the 'Date' column to datetime objects.
    new_df["Date"] = pd.to_datetime(new_df["Date"])

    # 2. Process the existing historical data from the CSV
    print(f"Reading historical CSV: {hist_csv_path}")
    hist_df = pd.read_csv(hist_csv_path)
    
    # Also apply the new names to the historical data for consistency.
    # This ensures that even if the CSV has old headers, it will be updated.
    hist_df.rename(columns=COLUMN_RENAME_MAP, inplace=True)

    # Clean up historical data: handle potential git conflicts or corrupted date entries.
    # We use the NEW column name 'Date' here.
    hist_df["Date"] = pd.to_datetime(hist_df["Date"], errors='coerce')
    hist_df.dropna(subset=['Date'], inplace=True) # Drop rows where date conversion failed

    # 3. Combine, de-duplicate, and sort
    # This works because both DataFrames now have the same column names.
    combined = pd.concat([hist_df, new_df], ignore_index=True)
    
    # Use the NEW column name 'Date' for de-duplicating and sorting.
    combined.drop_duplicates(subset=["Date"], keep="last", inplace=True)
    combined.sort_values("Date", inplace=True)

    # 4. Save the result
    combined.to_csv(hist_csv_path, index=False)
    print(f"Updated CSV saved with new headers to: {hist_csv_path}")

# --- Main ---
if __name__ == "__main__":
    print(f"Loading .env from: {ENV_PATH}")
    load_dotenv(ENV_PATH)

    username = os.getenv("PLATTS_USERNAME")
    password = os.getenv("PLATTS_PASSWORD")
    if not username or not password:
        raise ValueError("Missing PLATTS_USERNAME or PLATTS_PASSWORD in .env")

    try:
        print("Authenticating to Platts...")
        token = get_access_token(username, password)

        print("Searching for latest GDMF package...")
        content_id = find_latest_package(token)
        if not content_id:
            raise RuntimeError("Could not find latest GDMF content package.")

        print(f"Downloading Excel content for content ID: {content_id}")
        excel_io = download_excel_content(token, content_id)

        print("Merging with historical data...")
        update_historical_csv(excel_io, HIST_CSV_PATH)

        print("✅ Done.")

    except requests.exceptions.RequestException as e:
        print(f"A network error occurred: {e}")
    except (ValueError, RuntimeError, KeyError) as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")