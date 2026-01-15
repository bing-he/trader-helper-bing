# GridStatLoadAndForecast.py
#
# Description:
# This script fetches both historical and forecast daily load for all configured ISOs
# from the GridStatus.io API. It calculates the total energy (MWh) for each day
# for both data types and saves them to separate CSV files.
#
# The script intelligently updates the historical file by replacing the last 5 days
# of data. The forecast file is overwritten on each run. The forecast fetch
# begins on the day immediately following the last available historical day.
#
# Instructions:
# 1. Ensure a .env file with your GRIDSTATUS_API_KEY is in the same directory.
# 2. Run the script: python GridStatLoadAndForecast.py

import pandas as pd
import os
from dotenv import load_dotenv
from gridstatusio import GridStatusClient
from datetime import datetime, timedelta
from pathlib import Path

from common.pathing import ROOT

# --- Configuration ---

# The target directory for the output CSV files.
OUTPUT_DIRECTORY = ROOT / "INFO"

# The date to start fetching from if no historical file exists.
DEFAULT_HISTORY_START_DATE = "2025-07-01"

# --- ISO Configurations ---

# Configuration for HISTORICAL load datasets for each ISO.
ISO_HISTORICAL_CONFIG = {
    "PJM": {
        "dataset_id": "pjm_standardized_5_min", "time_column": "interval_start_utc",
        "value_column": "load.load", "interval_minutes": 5, "timezone": "America/New_York"
    },
    "SPP": {
        "dataset_id": "spp_load", "time_column": "interval_start_utc",
        "value_column": "load", "interval_minutes": 60, "timezone": "America/Chicago"
    },
    "NYISO": {
        "dataset_id": "nyiso_load", "time_column": "interval_start_utc",
        "value_column": "load", "interval_minutes": 60, "timezone": "America/New_York"
    },
    "MISO": {
        "dataset_id": "miso_load", "time_column": "interval_start_utc",
        "value_column": "load", "interval_minutes": 60, "timezone": "America/Chicago"
    },
    "ISONE": {
        "dataset_id": "isone_load", "time_column": "interval_start_utc",
        "value_column": "load", "interval_minutes": 60, "timezone": "America/New_York"
    },
    "IESO": {
        "dataset_id": "ieso_load", "time_column": "interval_start_utc",
        "value_column": "market_total_load", "interval_minutes": 60, "timezone": "America/Toronto"
    },
    "ERCOT": {
        "dataset_id": "ercot_standardized_hourly", "time_column": "interval_start_utc",
        "value_column": "load.load", "interval_minutes": 60, "timezone": "America/Chicago"
    },
    "CAISO": {
        "dataset_id": "caiso_load", "time_column": "interval_start_utc",
        "value_column": "load", "interval_minutes": 60, "timezone": "America/Los_Angeles"
    }
}

# Configuration for FORECAST load datasets for each ISO.
# *** CHANGE: Updated dataset_id and value_column names to match new GridStatus.io API standards. ***
ISO_FORECAST_CONFIG = {
    "PJM": {
        "dataset_id": "pjm_load_forecast", "time_column": "interval_start_utc",
        "value_column": "load_forecast", "interval_minutes": 60, "timezone": "America/New_York"
    },
    "SPP": {
        "dataset_id": "spp_load_forecast", "time_column": "interval_start_utc",
        "value_column": "load_forecast", "interval_minutes": 60, "timezone": "America/Chicago"
    },
    "NYISO": {
        "dataset_id": "nyiso_load_forecast", "time_column": "interval_start_utc",
        "value_column": "load_forecast", "interval_minutes": 60, "timezone": "America/New_York"
    },
    "MISO": {
        "dataset_id": "miso_load_forecast", "time_column": "interval_start_utc",
        "value_column": "load_forecast", "interval_minutes": 60, "timezone": "America/Chicago"
    },
    "ISONE": {
        "dataset_id": "isone_load_forecast", "time_column": "interval_start_utc",
        "value_column": "load_forecast", "interval_minutes": 60, "timezone": "America/New_York"
    },
    "IESO": {
        "dataset_id": "ieso_load_forecast", "time_column": "interval_start_utc",
        "value_column": "load_forecast", "interval_minutes": 60, "timezone": "America/Toronto"
    },
    "ERCOT": {
        "dataset_id": "ercot_load_forecast", "time_column": "interval_start_utc",
        "value_column": "load_forecast", "interval_minutes": 60, "timezone": "America/Chicago"
    },
    "CAISO": {
        "dataset_id": "caiso_load_forecast", "time_column": "interval_start_utc",
        "value_column": "load_forecast", "interval_minutes": 60, "timezone": "America/Los_Angeles"
    }
}


def initialize_client() -> GridStatusClient | None:
    """Loads the API key and initializes the GridStatusClient."""
    print("--- Initializing ---")
    load_dotenv()
    api_key = os.environ.get("GRIDSTATUS_API_KEY")
    if not api_key:
        print("❌ ERROR: GRIDSTATUS_API_KEY not found in environment variables.")
        return None
    print("✅ API Key retrieved.")
    try:
        client = GridStatusClient(api_key=api_key)
        print("✅ GridStatusClient initialized successfully.")
        return client
    except Exception as e:
        print(f"❌ ERROR: Could not initialize GridStatusClient: {e}")
        return None


def fetch_data(client: GridStatusClient, dataset_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches data for a given dataset and date range."""
    print(f"\n[Fetching] Data from '{dataset_id}'...")
    try:
        df = client.get_dataset(dataset=dataset_id, start=start_date, end=end_date)
        if df.empty:
            print(f"   ℹ️  No data returned for the period {start_date} to {end_date}.")
        else:
            print(f"   ✅ Successfully fetched {len(df)} rows.")
        return df
    except Exception as e:
        print(f"   ❌ ERROR while fetching data: {e}")
        return pd.DataFrame()


def process_data(df: pd.DataFrame, iso: str, config: dict, item_suffix: str) -> pd.DataFrame:
    """Validates, calculates, and formats total daily data (historical or forecast)."""
    if df.empty:
        return pd.DataFrame()

    print(f"[Processing] {item_suffix} data for {iso}...")
    time_col, value_col = config["time_column"], config["value_column"]
    interval_min, market_tz = config["interval_minutes"], config["timezone"]

    if time_col not in df.columns or value_col not in df.columns:
        print(f"   ❌ ERROR: Expected columns not found for {iso}.")
        print(f"   Required: '{time_col}', '{value_col}'. Found: {df.columns.tolist()}")
        return pd.DataFrame()

    df[time_col] = pd.to_datetime(df[time_col])
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df.dropna(subset=[time_col, value_col], inplace=True)

    df['market_time'] = df[time_col].dt.tz_convert(market_tz)
    df['market_date'] = df['market_time'].dt.date

    expected_intervals = (24 * 60) // interval_min
    daily_counts = df.groupby('market_date')[time_col].count()
    full_days_index = daily_counts[daily_counts >= expected_intervals].index
    
    if len(daily_counts) > len(full_days_index):
        partial_count = len(daily_counts) - len(full_days_index)
        print(f"   ⚠️  Discarded {partial_count} partial day(s) for {iso} due to incomplete data.")

    validated_df = df[df['market_date'].isin(full_days_index)].copy()
    if validated_df.empty:
        print(f"   ℹ️  No full days of data found for {iso} after validation.")
        return pd.DataFrame()

    mwh_col = 'total_mwh'
    validated_df[mwh_col] = validated_df[value_col] * (interval_min / 60)
    daily_total_df = validated_df.groupby('market_date')[mwh_col].sum().reset_index()
    
    daily_total_df["Item"] = f"{iso} - {item_suffix}"
    daily_total_df.rename(columns={'market_date': "Date", mwh_col: "Value"}, inplace=True)
    
    print(f"   ✅ Data for {iso} processed successfully.")
    return daily_total_df[["Date", "Item", "Value"]]


def main():
    """Main execution function to run historical and forecast load retrieval."""
    client = initialize_client()
    if not client: return

    # =================================================================
    # --- 1. PROCESS HISTORICAL DATA ---
    # =================================================================
    print("\n" + "="*25 + "\n--- Processing Historical Load Data ---\n" + "="*25)
    hist_filename = "GridStatLoadHist.csv"
    hist_path = OUTPUT_DIRECTORY / hist_filename
    existing_hist_df = pd.DataFrame()
    
    if hist_path.exists():
        print(f"\n--- Found existing file: {hist_path} ---")
        existing_hist_df = pd.read_csv(hist_path)
        hist_start_date = (datetime.now() - timedelta(days=5)).date()
        print(f"Updating data from {hist_start_date} onwards.")
    else:
        print("\n--- No existing file found. Building full history. ---")
        hist_start_date = datetime.strptime(DEFAULT_HISTORY_START_DATE, "%Y-%m-%d").date()
        print(f"Fetching all data from {hist_start_date} onwards.")
        
    hist_end_date = datetime.now().date()
    all_hist_results = []
    for iso, config in ISO_HISTORICAL_CONFIG.items():
        raw_df = fetch_data(client, config['dataset_id'], str(hist_start_date), str(hist_end_date))
        results_df = process_data(raw_df, iso, config, "TotalLoad")
        if not results_df.empty: all_hist_results.append(results_df)

    if not all_hist_results:
        print("\n--- No new historical data fetched. ---")
        combined_hist_df = existing_hist_df
    else:
        new_hist_df = pd.concat(all_hist_results, ignore_index=True)
        new_hist_df["Date"] = pd.to_datetime(new_hist_df["Date"]).dt.date
        if not existing_hist_df.empty:
            existing_hist_df["Date"] = pd.to_datetime(existing_hist_df["Date"]).dt.date
            existing_hist_df = existing_hist_df[existing_hist_df["Date"] < hist_start_date]
        combined_hist_df = pd.concat([existing_hist_df, new_hist_df], ignore_index=True)
        combined_hist_df.drop_duplicates(subset=['Date', 'Item'], keep='last', inplace=True)
        combined_hist_df.sort_values(by=["Item", "Date"], inplace=True)
        print(f"\n[Saving] Historical data to {hist_path}...")
        try:
            OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
            combined_hist_df.to_csv(hist_path, index=False, date_format='%Y-%m-%d')
            print(f"   ✅ Successfully saved historical data.")
        except Exception as e: print(f"   ❌ ERROR saving file: {e}")

    # =================================================================
    # --- 2. PROCESS FORECAST DATA ---
    # =================================================================
    print("\n" + "="*25 + "\n--- Processing Forecast Load Data ---\n" + "="*25)
    if combined_hist_df.empty:
        print("\n❌ Cannot run forecast process: historical data is empty.")
        return

    fcst_start_date = pd.to_datetime(combined_hist_df['Date']).max().date() + timedelta(days=1)
    fcst_end_date = fcst_start_date + timedelta(days=14)
    print(f"Fetching forecast data from {fcst_start_date} to {fcst_end_date}")

    all_fcst_results = []
    for iso, config in ISO_FORECAST_CONFIG.items():
        raw_df = fetch_data(client, config['dataset_id'], str(fcst_start_date), str(fcst_end_date))
        results_df = process_data(raw_df, iso, config, "ForecastLoad")
        if not results_df.empty: all_fcst_results.append(results_df)

    if not all_fcst_results:
        print("\n--- No new forecast data fetched. ---")
    else:
        fcst_df = pd.concat(all_fcst_results, ignore_index=True)
        fcst_df.sort_values(by=["Item", "Date"], inplace=True)
        fcst_filename = "GridStatLoadForecast.csv"
        fcst_path = OUTPUT_DIRECTORY / fcst_filename
        print(f"\n[Saving] Forecast data to {fcst_path}...")
        try:
            OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
            fcst_df.to_csv(fcst_path, index=False, date_format='%Y-%m-%d')
            print(f"   ✅ Successfully saved forecast data.")
        except Exception as e: print(f"   ❌ ERROR saving file: {e}")


if __name__ == "__main__":
    main()
    print("\n--- Script Finished ---")
