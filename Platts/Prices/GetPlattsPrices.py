import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from your .env file
load_dotenv()
PLATTS_USERNAME = os.getenv("PLATTS_USERNAME")
PLATTS_PASSWORD = os.getenv("PLATTS_PASSWORD")

# API Endpoints from your provided files
AUTH_URL = "https://api.ci.spglobal.com/auth/api"
HISTORY_URL = "https://api.ci.spglobal.com/market-data/v3/value/history/symbol"

# --- Request Parameters ---
TARGET_SYMBOL = "IGBBL21"
DAYS_TO_FETCH = 20
# Per your PlattsHistoricalPricesCSVChart.py, "U" is used for Index/Settlement prices
BATES_FILTER = "U" 

def get_access_token(username, password):
    """Authenticates with the Platts API and retrieves an access token."""
    if not username or not password:
        print("ERROR: PLATTS_USERNAME or PLATTS_PASSWORD not set in the .env file.")
        return None
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    payload = {"username": username, "password": password}
    print("Attempting to get Platts access token...")
    try:
        response = requests.post(AUTH_URL, headers=headers, data=payload, timeout=30)
        response.raise_for_status()  
        token_data = response.json()
        print("Successfully obtained access token.")
        return token_data.get("access_token")
    except requests.exceptions.HTTPError as http_err:
        print(f"Http Error during authentication: {http_err}")
    except Exception as e:
        print(f"An unexpected error occurred during authentication: {e}")
    return None

def fetch_platts_prices(token, symbol, start_date, end_date):
    """Fetches historical prices for a single symbol from the Platts API."""
    if not token:
        print("Cannot fetch prices without an access token.")
        return []

    print(f"\nFetching data for symbol '{symbol}' from {start_date} to {end_date}...")
    
    # Construct the filter string to get the specific symbol, bate, and date range
    filter_str = (
        f'symbol IN ("{symbol}") AND bate:"{BATES_FILTER}" '
        f'AND assessDate>="{start_date}" AND assessDate<="{end_date}"'
    )

    params = {"Filter": filter_str, "PageSize": 1000} # Page size is ample for 15 days
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    
    price_data = []
    try:
        response = requests.get(HISTORY_URL, headers=headers, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        # The API nests the price data in a structure: results -> data
        results = data.get("results", [])
        if results and results[0].get('data'):
            for record in results[0]['data']:
                price_data.append({
                    "Date": record.get("assessDate"),
                    "Symbol": symbol,
                    "Settlement Price": pd.to_numeric(record.get("value"), errors='coerce')
                })
        else:
            print("No data found in the API response for the given criteria.")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the Platts API request: {e}")
    
    return price_data

def main():
    """Main function to get the token and fetch recent prices."""
    access_token = get_access_token(PLATTS_USERNAME, PLATTS_PASSWORD)
    if not access_token:
        return

    # Calculate the date range for the last 15 days
    end_date_str = datetime.now().strftime("%Y-%m-%d")
    start_date_str = (datetime.now() - timedelta(days=DAYS_TO_FETCH)).strftime("%Y-%m-%d")
    
    # Fetch the price data
    prices = fetch_platts_prices(access_token, TARGET_SYMBOL, start_date_str, end_date_str)
    
    if prices:
        # Convert to a pandas DataFrame for nice printing
        df = pd.DataFrame(prices)
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df = df.sort_values(by='Date', ascending=False).reset_index(drop=True)
        
        print("\n--- Recent Henry Hub Settlement Prices ---")
        # Using to_string() ensures the entire DataFrame is printed
        print(df.to_string())
    else:
        print("\nCould not retrieve any price data.")

if __name__ == "__main__":
    main()