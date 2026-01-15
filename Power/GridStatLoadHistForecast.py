"""
GridStatLoadHistForecast.py
... existing docstring ...
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import pytz
from dotenv import load_dotenv
from gridstatusio import GridStatusClient

# --- Master Configuration ---
# Make paths relational (non-hardcoded) for portability
# Get the directory where this script is located (e.g., .../Project/scripts)
SCRIPT_DIR = Path(__file__).resolve().parent
# Get the project's root directory (e.g., .../Project)
PROJECT_ROOT = SCRIPT_DIR.parent
# Define the output directory relative to the project root
OUTPUT_DIRECTORY = PROJECT_ROOT / "INFO"

HIST_FILE = OUTPUT_DIRECTORY / "GridStatLoadHist.csv"
FCST_FILE = OUTPUT_DIRECTORY / "GridStatLoadForecast.csv"

# Global settings
FORECAST_DAYS = 14  # Number of days to fetch for the forecast
UPDATE_WINDOW_DAYS = 20  # Days to re-fetch when updating existing hist file
# Use a consistent "master" timezone for defining "today"
MASTER_TIMEZONE = "America/New_York"
EARLIEST_START_DATE = "2015-01-01"  # Earliest start date of all ISOs

# --- ISO CONFIGURATION DICTIONARY ---
# This dictionary drives all logic for each ISO.
#
# Parameters:
#   hist_dataset: (str) The dataset ID for historical data.
#   fcst_dataset: (Optional[str]) The dataset ID for forecast data.
#   is_pivot: (bool) True if the historical data is narrow and needs pivoting.
#   resample_hist: (bool) True if historical data is 5-min and needs 1-hr avg.
#   market_timezone: (str) The IANA timezone for the ISO.
#   start_date: (str) The earliest available data date (YYYY-MM-DD).
#   prefix: (str) The prefix for final column names (e.g., "PJM").
#   hist_col_map: (Optional[dict]) Maps historical columns to forecast names.
#   fcst_col_map: (Optional[dict]) Maps forecast columns to historical names.
#   final_columns: (List[str]) The master list of columns to process.
#
ISO_CONFIG: Dict[str, Dict[str, Any]] = {
    "PJM": {
        "hist_dataset": "pjm_load",
        "fcst_dataset": "pjm_load_forecast_hourly",
        "is_pivot": False,
        "resample_hist": True,
        "market_timezone": "America/New_York",
        "start_date": "2023-02-20",
        "prefix": "PJM",
        "hist_col_map": {
            "load": "load_forecast",
            "pjm_rto": "rto_combined",
            "pjm_southern_region": "southern_region",
            "pjm_mid_atlantic_region": "mid_atlantic_region",
            "pjm_western_region": "western_region",
            "ae": "ae_midatl",
            "aps": "ap",
            "bc": "bge_midatl",
            "dom": "dominion",
            "dpl": "dpl_midatl",
            "duq": "duquesne",
            "jc": "jcpl_midatl",
            "me": "meted_midatl",
            "pe": "peco_midatl",
            "pn": "penelec_midatl",
            "pep": "pepco_midatl",
            "pl": "ppl_midatl",
            "ps": "pseg_midatl",
            "reco": "reco_midatl",
            "ug": "ugi_midatl",
        },
        "fcst_col_map": None,
        "final_columns": [
            'interval_start_utc', 'interval_end_utc', 'publish_time_utc',
            'load_forecast', 'ae_midatl', 'aep', 'ap', 'atsi', 'bge_midatl',
            'comed', 'dayton', 'deok', 'dominion', 'dpl_midatl', 'duquesne',
            'ekpc', 'jcpl_midatl', 'meted_midatl', 'mid_atlantic_region',
            'peco_midatl', 'penelec_midatl', 'pepco_midatl', 'ppl_midatl',
            'pseg_midatl', 'reco_midatl', 'rto_combined', 'southern_region',
            'ugi_midatl', 'western_region',
        ],
    },
    "ERCOT": {
        "hist_dataset": "ercot_load_by_forecast_zone",
        "fcst_dataset": "ercot_load_forecast_by_forecast_zone",
        "is_pivot": False,
        "resample_hist": False,
        "market_timezone": "America/Chicago",
        "start_date": "2017-06-29",
        "prefix": "ERCOT",
        "hist_col_map": {"total": "system_total"},
        "fcst_col_map": None,
        "final_columns": [
            'interval_start_utc', 'interval_end_utc', 'publish_time_utc',
            'north', 'south', 'west', 'houston', 'system_total',
        ],
    },
    "ISONE": {
        "hist_dataset": "isone_load",
        "fcst_dataset": "isone_load_forecast_hourly",
        "is_pivot": False,
        "resample_hist": True,
        "market_timezone": "America/New_York",
        "start_date": "2020-11-30",
        "prefix": "ISONE",
        "hist_col_map": {"load": "load_forecast"},
        "fcst_col_map": None,
        "final_columns": [
            'interval_start_utc', 'interval_end_utc', 'publish_time_utc',
            'load_forecast',
        ],
    },
    "SPP": {
        "hist_dataset": "spp_load",
        "fcst_dataset": "spp_load_forecast_mid_term",
        "is_pivot": False,
        "resample_hist": True,
        "market_timezone": "America/Chicago",
        "start_date": "2015-01-01",
        "prefix": "SPP",
        "hist_col_map": {"load": "mtlf"},
        "fcst_col_map": None,
        "final_columns": [
            'interval_start_utc', 'interval_end_utc', 'publish_time_utc',
            'mtlf',
        ],
    },
    "MISO": {
        "hist_dataset": "miso_zonal_load_hourly",
        "fcst_dataset": "miso_load_forecast_mid_term",
        "is_pivot": False,
        "resample_hist": False,
        "market_timezone": "America/Chicago",
        "start_date": "2015-01-01",
        "prefix": "MISO",
        "hist_col_map": None,
        "fcst_col_map": {
            "miso_mtlf": "miso",
            "lrz1_mtlf": "lrz1",
            "lrz2_7_mtlf": "lrz2_7",
            "lrz3_5_mtlf": "lrz3_5",
            "lrz4_mtlf": "lrz4",
            "lrz6_mtlf": "lrz6",
            "lrz8_9_10_mtlf": "lrz8_9_10",
        },
        "final_columns": [
            'interval_start_utc', 'interval_end_utc', 'publish_time_utc',
            'lrz1', 'lrz2_7', 'lrz3_5', 'lrz4', 'lrz6', 'lrz8_9_10', 'miso',
        ],
    },
    "NYISO": {
        "hist_dataset": "nyiso_load",
        "fcst_dataset": "nyiso_zonal_load_forecast_hourly",
        "is_pivot": False,
        "resample_hist": True,
        "market_timezone": "America/New_York",
        "start_date": "2015-01-01",
        "prefix": "NYISO",
        "hist_col_map": {'load': 'nyiso'},
        "fcst_col_map": None,
        "final_columns": [
            'interval_start_utc', 'interval_end_utc', 'publish_time_utc',
            'nyiso', 'capitl', 'centrl', 'dunwod', 'genese', 'hud_vl',
            'longil', 'mhk_vl', 'millwd', 'nyc', 'north', 'west',
        ],
    },
    "CAISO": {
        "hist_dataset": "caiso_load_hourly",
        "fcst_dataset": None,  # CAISO only has historical
        "is_pivot": True,
        "resample_hist": False,
        "market_timezone": "America/Los_Angeles",
        "start_date": "2015-01-01",
        "prefix": "CAISO",
        "hist_col_map": None,
        "fcst_col_map": None,
        "final_columns": [
            'BCHA', 'BPAT', 'CA ISO-TAC', 'EPE', 'IPCO', 'LADWP',
            'MWD-TAC', 'NEVP', 'NWMT', 'PACE', 'PACW', 'PGE',
            'PGE-TAC', 'PNM', 'PSEI', 'SCE-TAC', 'SCL', 'SDGE-TAC',
            'SRP', 'TEPC', 'TIDC', 'TPWR', 'VEA-TAC', 'WALC',
            'WALCAEPCO', 'WALCDSW',
        ],
    },
}
# --- End of Configuration ---


def initialize_client() -> GridStatusClient | None:
    """
    Initializes the GridStatusClient using the API key from .env file.

    Returns:
        GridStatusClient | None: An initialized client or None if setup fails.
    """
    print("--- Initializing ---")
    load_dotenv()
    api_key = os.environ.get("GRIDSTATUS_API_KEY")
    if not api_key:
        print("❌ ERROR: GRIDSTATUS_API_KEY not found.")
        return None
    try:
        client = GridStatusClient(api_key=api_key)
        print("✅ GridStatusClient initialized successfully.")
        return client
    except Exception as e:
        print(f"❌ ERROR: Could not initialize GridStatusClient: {e}")
        return None


def fetch_historical_data(
    client: GridStatusClient,
    config: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetches and processes historical data for a single ISO.

    Handles resampling and pivoting logic based on the config.

    Args:
        client: The initialized GridStatusClient.
        config: The configuration dictionary for the ISO.
        start_date: The start date for the data pull (YYYY-MM-DD).
        end_date: The end date for the data pull (YYYY-MM-DD).

    Returns:
        pd.DataFrame: An hourly, aligned DataFrame of historical data.
    """
    iso_name = config["prefix"]
    print(
        f"\n[Fetching] Historical '{config['hist_dataset']}' for {iso_name} "
        f"from {start_date} to {end_date}"
    )

    api_params = {
        "dataset": config["hist_dataset"],
        "start": start_date,
        "end": end_date,
        "timezone": "market",
    }

    if config["resample_hist"]:
        print("   Resampling to 1-hour average...")
        api_params["resample"] = "1 hour"
        api_params["resample_function"] = "mean"

    try:
        df = client.get_dataset(**api_params)
        print(f"   ✅ Successfully fetched {len(df)} rows.")

        if df.empty:
            return pd.DataFrame()

        # 1. Handle Pivoting (CAISO)
        if config["is_pivot"]:
            df = df.pivot(
                index='interval_start_utc',
                columns='tac_area_name',
                values='load',
            )
            # Ensure all expected columns exist, adding any missing ones as NaN
            for area in config["final_columns"]:
                if area not in df.columns:
                    df[area] = pd.NA
        
        # 2. Handle Column Renaming
        if config["hist_col_map"]:
            df.rename(columns=config["hist_col_map"], inplace=True)

        # 3. Filter to keep only the columns that exist in the final list
        final_cols = [
            col for col in config["final_columns"] if col in df.columns
        ]

        # Add 'interval_start_utc' if it was the index
        if (
            'interval_start_utc' not in final_cols
            and df.index.name == 'interval_start_utc'
        ):
            df.reset_index(inplace=True)
            final_cols.insert(0, 'interval_start_utc')

        return df[final_cols]

    except Exception as e:
        print(f"   ❌ ERROR while fetching historical data for {iso_name}: {e}")
        return pd.DataFrame()


def fetch_forecast_data(
    client: GridStatusClient,
    config: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetches the most recent forecast data for a single ISO.

    Args:
        client: The initialized GridStatusClient.
        config: The configuration dictionary for the ISO.
        start_date: The start date for the data pull (YYYY-MM-DD).
        end_date: The end date for the data pull (YYYY-MM-DD).

    Returns:
        pd.DataFrame: An hourly DataFrame of the latest forecast data.
    """
    iso_name = config["prefix"]
    print(
        f"\n[Fetching] Latest '{config['fcst_dataset']}' for {iso_name} "
        f"from {start_date} to {end_date}"
    )
    try:
        df = client.get_dataset(
            dataset=config["fcst_dataset"],
            start=start_date,
            end=end_date,
            publish_time="latest_before:0 hours",
            timezone="market",
        )
        print(f"   ✅ Successfully fetched {len(df)} forecast rows.")

        if df.empty:
            return pd.DataFrame()

        # 1. Handle Column Renaming (MISO)
        if config["fcst_col_map"]:
            df.rename(columns=config["fcst_col_map"], inplace=True)

        # 2. Filter to keep only the relevant columns
        final_cols = [
            col for col in config["final_columns"] if col in df.columns
        ]
        return df[final_cols]

    except Exception as e:
        print(f"   ❌ ERROR while fetching forecast data for {iso_name}: {e}")
        return pd.DataFrame()


def summarize_daily_mwh(
    df: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Aggregates a wide hourly DataFrame into a daily MWh summary.

    - Sums hourly MW values to get daily MWh.
    - Renames columns to 'ISO.[component]' format.
    - Only includes days with 23 or more hours of data.

    Args:
        df: The hourly DataFrame to summarize.
        config: The configuration dictionary for the ISO.

    Returns:
        pd.DataFrame: A summarized DataFrame with daily MWh totals.
    """
    iso_name = config["prefix"]
    if df.empty:
        print(f"   ⚠️  Cannot summarize {iso_name}: input DataFrame is empty.")
        return pd.DataFrame()

    df_copy = df.copy()

    # Ensure interval_start_utc is a datetime object
    df_copy['interval_start_utc'] = pd.to_datetime(
        df_copy['interval_start_utc']
    )

    # Set index to the datetime for resampling
    df_copy.set_index('interval_start_utc', inplace=True)

    # Explicitly convert the index to the market timezone
    market_tz = config["market_timezone"]
    df_copy = df_copy.tz_convert(market_tz)

    # Identify component columns (all columns in config except time)
    component_cols = [
        col
        for col in config["final_columns"]
        if col in df_copy.columns
        and col
        not in ['interval_start_utc', 'interval_end_utc', 'publish_time_utc']
    ]

    # Reindex to a full hourly index to correctly identify missing hours
    min_date = df_copy.index.min().floor('D')
    max_date = df_copy.index.max().ceil('D')

    if pd.isna(min_date) or pd.isna(max_date):
        print(f"   ⚠️  Cannot summarize {iso_name}: No valid date range.")
        return pd.DataFrame()

    expected_index = pd.date_range(
        start=min_date, end=max_date, freq='h', tz=market_tz
    )
    df_reindexed = df_copy.reindex(expected_index)

    # Resample to Daily ('D').
    # min_count=23 ensures only "fully accounted for" days are summed.
    # This handles DST days and drops partial forecast days.
    daily_df = df_reindexed[component_cols].resample('D').sum(min_count=23)

    # Drop any rows that are all NaN (i.e., incomplete days)
    daily_df.dropna(how='all', inplace=True)

    # Rename columns to 'ISO.[component]'
    new_columns = {col: f"{iso_name}.{col}" for col in daily_df.columns}
    daily_df.rename(columns=new_columns, inplace=True)

    # Format index as Date
    daily_df.index = daily_df.index.date
    daily_df.index.name = 'Date'
    return daily_df


def get_hist_start_info(
    today_market: datetime.date
) -> (datetime.date, pd.DataFrame):
    """
    Determines the correct historical start date based on update logic.

    Args:
        today_market: Today's date in the master timezone.

    Returns:
        (datetime.date, pd.DataFrame): A tuple of the
        effective_start_date for API calls and the old_data_to_keep.
    """
    old_data_to_keep = pd.DataFrame()
    if HIST_FILE.exists():
        print(f"   Found existing file: {HIST_FILE}")
        print(f"   Updating data for the last {UPDATE_WINDOW_DAYS} days.")
        effective_start_date = today_market - timedelta(days=UPDATE_WINDOW_DAYS)
        try:
            old_data = pd.read_csv(
                HIST_FILE, index_col='Date', parse_dates=True
            )
            
            # --- ADD THIS LINE ---
            # Convert the index from Timestamp to simple date objects
            # to match the format from summarize_daily_mwh
            old_data.index = old_data.index.date
            
            # Filter old data to exclude the part we are updating
            # This comparison is now correctly date-to-date
            old_data_to_keep = old_data[
                old_data.index < effective_start_date
            ]
        except Exception as e:
            print(f"   ⚠️  Could not read old file, rebuilding from scratch. {e}")
            effective_start_date = datetime.strptime(
                EARLIEST_START_DATE, "%Y-%m-%d"
            ).date()
    else:
        print("   No existing file found. Building full history.")
        effective_start_date = datetime.strptime(
            EARLIEST_START_DATE, "%Y-%m-%d"
        ).date()
    
    return effective_start_date, old_data_to_keep


def main():
    """
    Main execution function to orchestrate the data processing pipeline
    for all configured ISOs.
    """
    client = initialize_client()
    if not client:
        return

    # Ensure output directory exists
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    # --- 1. Define Date Ranges ---
    market_tz = pytz.timezone(MASTER_TIMEZONE)
    today_market = datetime.now(market_tz).date()

    fcst_end_date = today_market + timedelta(days=FORECAST_DAYS)
    fcst_end_str = fcst_end_date.strftime("%Y-%m-%d")

    # --- 2. Process Historical Data ---
    print("\n" + "=" * 25 + "\n--- Processing Historical Data ---\n" + "=" * 25)
    
    # Determine start date based on 20-day update rule
    effective_hist_start, old_data_to_keep = get_hist_start_info(today_market)
    
    all_new_hist_dfs = [old_data_to_keep]
    all_fcst_dfs = []

    # --- 3. Main ISO Processing Loop ---
    for iso_name, config in ISO_CONFIG.items():
        print("\n" + "-" * 20 + f" Processing {iso_name} " + "-" * 20)
        
        # Define ISO-specific historical start date
        # (It's the later of the global start date or the ISO's own start)
        iso_start_date = datetime.strptime(
            config["start_date"], "%Y-%m-%d"
        ).date()
        
        hist_start_date_iso = max(effective_hist_start, iso_start_date)
        hist_start_str = hist_start_date_iso.strftime("%Y-%m-%d")
        
        # Fetch historical data up to "today"
        # Add 1 day to end to get all of today's intervals for stitching
        hist_end_str = (today_market + timedelta(days=1)).strftime("%Y-%m-%d")

        hourly_hist_df = fetch_historical_data(
            client, config, hist_start_str, hist_end_str
        )
        
        # --- 4. Process Forecast Data (if applicable) ---
        if config["fcst_dataset"]:
            fcst_start_str = today_market.strftime("%Y-%m-%d")

            # Filter historical data for "today's actuals"
            # We already fetched it, just need to filter
            today_actuals_df = pd.DataFrame()
            if not hourly_hist_df.empty:
                hourly_hist_df['interval_start_utc'] = pd.to_datetime(
                    hourly_hist_df['interval_start_utc']
                )
                today_actuals_df = hourly_hist_df[
                    hourly_hist_df['interval_start_utc'].dt.date == today_market
                ].copy()

            # Fetch the latest forecast (from today onwards)
            future_forecast_df = fetch_forecast_data(
                client, config, fcst_start_str, fcst_end_str
            )

            # Stitching Logic
            if today_actuals_df.empty and future_forecast_df.empty:
                print(
                    f"\n❌ ERROR ({iso_name}): Cannot create forecast. "
                    "Missing actuals AND forecast data for today."
                )
                final_hourly_fcst_df = pd.DataFrame()
            elif today_actuals_df.empty:
                print(
                    f"\n⚠️  WARNING ({iso_name}): Missing today's actuals. "
                    "Forecast file based on forecast data only."
                )
                final_hourly_fcst_df = future_forecast_df
            elif future_forecast_df.empty:
                print(
                    f"\n⚠️  WARNING ({iso_name}): Missing today's forecast. "
                    "Forecast file based on actuals data only."
                )
                final_hourly_fcst_df = today_actuals_df
            else:
                current_time_utc = datetime.now(pytz.UTC)
                today_actuals_df['interval_start_utc'] = pd.to_datetime(
                    today_actuals_df['interval_start_utc']
                ).dt.tz_convert('UTC')
                future_forecast_df['interval_start_utc'] = pd.to_datetime(
                    future_forecast_df['interval_start_utc']
                ).dt.tz_convert('UTC')

                actuals = today_actuals_df[
                    today_actuals_df['interval_start_utc'] < current_time_utc
                ]
                forecasts = future_forecast_df[
                    future_forecast_df['interval_start_utc'] >= current_time_utc
                ]
                
                print(
                    f"\n[Stitching {iso_name}] Combining {len(actuals)} "
                    f"rows of actuals with {len(forecasts)} rows of forecast."
                )

                final_hourly_fcst_df = pd.concat(
                    [actuals, forecasts], ignore_index=True
                )
                final_hourly_fcst_df.sort_values(
                    by="interval_start_utc", inplace=True
                )
                final_hourly_fcst_df.drop_duplicates(
                    subset=['interval_start_utc'], keep='first', inplace=True
                )

            # Summarize and append forecast
            daily_fcst_df = summarize_daily_mwh(final_hourly_fcst_df, config)
            all_fcst_dfs.append(daily_fcst_df)

        # --- 5. Process Historical Data ---
        # Filter historical data to be *before* today for the hist file
        if not hourly_hist_df.empty:
            hourly_hist_df['interval_start_utc'] = pd.to_datetime(
                hourly_hist_df['interval_start_utc']
            )
            hist_only_df = hourly_hist_df[
                hourly_hist_df['interval_start_utc'].dt.date < today_market
            ].copy()
            
            daily_hist_df = summarize_daily_mwh(hist_only_df, config)
            all_new_hist_dfs.append(daily_hist_df)
    
    # --- 6. Final Consolidation and Save ---
    print("\n" + "=" * 25 + "\n--- Consolidating All ISOs ---\n" + "=" * 25)

    # Consolidate Historical File
    if len(all_new_hist_dfs) > 0:
        # Use axis=1 to join dataframes on their 'Date' index
        final_hist_df = pd.concat(all_new_hist_dfs, axis=1)
        final_hist_df.sort_index(inplace=True)
        # Drop duplicates from the overlap (keeping new data)
        final_hist_df = final_hist_df[
            ~final_hist_df.index.duplicated(keep='last')
        ]
        
        final_hist_df.to_csv(HIST_FILE, date_format='%Y-%m-%d')
        print(f"✅ Master historical data saved to {HIST_FILE}")
    else:
        print("--- No new historical data to save. ---")

    # Consolidate Forecast File
    if len(all_fcst_dfs) > 0:
        final_fcst_df = pd.concat(all_fcst_dfs, axis=1)
        final_fcst_df.sort_index(inplace=True)
        
        final_fcst_df.to_csv(FCST_FILE, date_format='%Y-%m-%d')
        print(f"✅ Master forecast data saved to {FCST_FILE}")
    else:
        print("--- No new forecast data to save. ---")


if __name__ == "__main__":
    main()
    print("\n--- Master Script Finished ---")