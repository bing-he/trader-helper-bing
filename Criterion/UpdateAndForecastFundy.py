"""
Fetches, processes, and forecasts fundamental energy data series.
Input:
Using:
    1. Criterion/CriterionInfo/database_tables_list.csv (master mapping file)
Output: 
    1. INFO/Fundy.csv (historical data)
    2. INFO/FundyForecast.csv (forecast data)

This script manages data from a single master mapping file.
It performs two main tasks:
1.  Updates a historical/actuals CSV file with the latest data from the database,
    performing an incremental update to preserve history. This file IS patched
    with forecast data *up to and including the current day* (to fix missing/zero actuals).
2.  Creates a full forecast CSV file by fetching all available forecast data.

The forecast data is cleaned *before* analysis to:
    1.  Truncate the data window to (Current Month + 2 Months).
    2.  Backfill 'GOM - Prod' data from EOM values.
    3.  Truncate each item *after* its last valid (non-zero) data point.

The actuals data is cleaned *before* merging to:
    1.  Trim leading historical zeros from items (since 2015-01-01).
"""

import logging
import os
import traceback
from pathlib import Path
from typing import Optional, Dict, Set
from datetime import timedelta, datetime

import numpy as np # Import numpy for nan handling
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, Engine
from common.pathing import ROOT

# ==============================================================================
#  CONFIGURATION & CONSTANTS
# ==============================================================================

# --- File & Directory Paths ---
SCRIPT_DIR = ROOT / "Criterion"
INFO_DIR = ROOT / "INFO"
MAPPING_DIR = SCRIPT_DIR / "CriterionInfo"

# --- Master File Configuration ---
MASTER_MAPPING_FILE = "database_tables_list.csv"
MASTER_ACTUALS_OUTPUT = "Fundy.csv"  # Output for historical data
MASTER_FORECAST_OUTPUT = "FundyForecast.csv" # Output for forecast data

# --- Data Configuration ---
START_DATE_FULL_FETCH = pd.to_datetime("2015-01-01")
INCREMENTAL_LOOKBACK_DAYS = 60
DATE_FORMAT = "%Y-%m-%d" # Using standard date format

# --- Seam Analysis Configuration ---
ANOMALY_JUMP_THRESHOLD_PERCENT = 0.50
NEAR_ZERO_THRESHOLD = 0.001
STALE_ZERO_EXEMPT_LIST = ['CONUS - LNGimp', 'CONUS - MexExp', 'CONUS - CADimp']

# --- Database Details ---
DB_HOST = "dda.criterionrsch.com"
DB_PORT = "443"
DB_NAME = "production"

# --- Region Normalization Mapping ---
REGION_MAP = {
    "Conus": "Lower48", "Lower 48": "Lower48", "South Central": "SouthCentral",
    "Rockies": "Mountain", "West/California": "Pacific", "West": "Pacific",
    "Midwest": "Midwest", "East": "East", "Northeast": "Northeast",
    "SouthEast": "Southeast"
}

# ==============================================================================
#  SETUP & HELPER FUNCTIONS
# ==============================================================================

def setup_logging():
    """Configures a basic logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def get_db_engine() -> Optional[Engine]:
    """Loads credentials and returns a SQLAlchemy engine."""
    logging.info("Connecting to the database...")
    try:
        load_dotenv(dotenv_path=SCRIPT_DIR / ".env", override=True)
        db_user, db_password = os.getenv("DB_USER"), os.getenv("DB_PASSWORD")
        if not db_user or not db_password:
            raise ValueError("DB credentials not found in environment.")
        
        conn_url = f"postgresql://{db_user}:{db_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(conn_url, connect_args={"sslmode": "require"})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logging.info("Database connection confirmed.")
        return engine
    except Exception as e:
        logging.critical(f"Database connection failed: {e}")
        return None

def fetch_single_ticker(engine: Engine, ticker: str) -> pd.DataFrame:
    """Fetches and prepares data for a single ticker."""
    if pd.isna(ticker) or not str(ticker).strip():
        return pd.DataFrame()

    query = text("SELECT DISTINCT * FROM data_series.fin_json_to_excel_tickers(:ticker)")
    try:
        df = pd.read_sql(query, engine, params={"ticker": str(ticker)})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df.dropna(subset=['date', 'value'], inplace=True)
        df.rename(columns={'date': 'Date', 'value': 'value'}, inplace=True)
        return df
    except Exception as e:
        logging.error(f"Error fetching ticker '{ticker}': {e}")
        return pd.DataFrame()

def get_region_for_item(item_name: str, region_map: Dict[str, str]) -> str:
    """Assigns a region based on item name, using a map and fallback rules."""
    if item_name in region_map:
        return region_map[item_name]
    
    if item_name == "GOM - Prod":
        return "SouthCentral"
    
    item_lower = str(item_name).lower()
    rules = {
        "Northeast": ["northeast"], "Midwest": ["midwest"],
        "SouthCentral": ["southcentral"], "SouthEast": ["southeast"],
        "Rockies": ["rockies"], "West": ["west"], "Lower48": ["conus", "l48", "us"],
    }
    for region, keywords in rules.items():
        if any(item_lower.startswith(keyword) for keyword in keywords):
            return region
    return "Unknown"

def load_existing_data(file_path: Path) -> pd.DataFrame:
    """Loads and prepares the existing CSV data file."""
    try:
        logging.info(f"Loading existing data from {file_path.name}...")
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.rename(columns={"Item": "item", "Value": "value", "Region": "region"}, inplace=True)
        df.dropna(subset=['Date', 'item', 'value'], inplace=True)
        return df
    except FileNotFoundError:
        logging.warning(f"{file_path.name} not found. Will create it from scratch.")
        return pd.DataFrame(columns=['Date', 'item', 'value', 'region'])

# ==============================================================================
#  CORE PIPELINE
# ==============================================================================

def fetch_all_base_series(
    engine: Engine, item_to_ticker_map: Dict[str, str], mode: str
) -> pd.DataFrame:
    """
    Fetches all data for a given ticker map (Mode: ACTUALS or FORECAST).
    """
    logging.info(f"--- Starting FULL data fetch (Mode: {mode}) ---")
    all_items_data = []
    
    for item_name, ticker in item_to_ticker_map.items():
        logging.info(f"Fetching item: '{item_name}' (Ticker: {ticker})")
        new_data = fetch_single_ticker(engine, ticker)

        if new_data.empty:
            logging.warning(f"No data returned for item '{item_name}'")
            continue
        
        new_data['item'] = item_name
        all_items_data.append(new_data)
    
    if not all_items_data:
        logging.error(f"No data fetched for {mode}. Aborting.")
        return pd.DataFrame()

    full_df = pd.concat(all_items_data, ignore_index=True)
    full_df['region'] = full_df['item'].apply(get_region_for_item, region_map={})
    logging.info(f"--- Completed FULL data fetch (Mode: {mode}) ---")
    return full_df


# --- Data Cleaning Functions ---

def clean_actuals_data(actuals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the historical actuals dataframe by trimming leading zeros.
    """
    logging.info("Cleaning actuals data: Trimming leading historical zeros...")
    
    cleaned_dfs = []
    all_items = actuals_df['item'].unique()
    
    for item in all_items:
        item_df = actuals_df[actuals_df['item'] == item].sort_values('Date')
        
        if item in STALE_ZERO_EXEMPT_LIST:
            cleaned_dfs.append(item_df)
            continue
            
        item_df_filtered = item_df[item_df['Date'] >= START_DATE_FULL_FETCH]
        
        if item_df_filtered.empty:
            continue
            
        first_valid_record = item_df_filtered[
            item_df_filtered['value'].abs() > NEAR_ZERO_THRESHOLD
        ]
        
        if first_valid_record.empty:
            logging.info(f"Item '{item}' has no non-zero data since 2015. Dropping.")
            continue
        
        first_valid_date = first_valid_record.iloc[0]['Date']
        item_df_trimmed = item_df[item_df['Date'] >= first_valid_date]
        cleaned_dfs.append(item_df_trimmed)

    if not cleaned_dfs:
        logging.warning("Actuals data cleaning resulted in an empty dataframe.")
        return pd.DataFrame()
        
    logging.info("Finished cleaning actuals data.")
    return pd.concat(cleaned_dfs, ignore_index=True)


def _truncate_forecast_window(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Truncates forecast data to current month + 2 months."""
    logging.info("Truncating forecast data to current month + 2 months.")
    
    today = pd.Timestamp.now()
    end_of_current_month = today + pd.offsets.MonthEnd(0)
    cutoff_date = (end_of_current_month + pd.offsets.MonthEnd(2)).normalize()
    
    logging.info(f"Forecast window cutoff date calculated as: {cutoff_date.strftime(DATE_FORMAT)}")
    
    original_rows = len(forecast_df)
    forecast_df_truncated = forecast_df[forecast_df['Date'] <= cutoff_date].copy()
    new_rows = len(forecast_df_truncated)
    
    if original_rows > new_rows:
        logging.info(f"Removed {original_rows - new_rows} forecast records beyond cutoff date.")
        
    return forecast_df_truncated

def _backfill_gom_prod(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Backfills 'GOM - Prod' from EOM values to full month."""
    logging.info("Backfilling 'GOM - Prod' forecast data...")
    item_name = 'GOM - Prod'
    
    if item_name not in forecast_df['item'].values:
        logging.info("'GOM - Prod' not in forecast data, skipping backfill.")
        return forecast_df

    gom_prod_df = forecast_df[forecast_df['item'] == item_name].copy()
    other_df = forecast_df[forecast_df['item'] != item_name]
    
    if gom_prod_df.empty:
        return forecast_df
    
    region = gom_prod_df['region'].iloc[0]
    gom_prod_df = gom_prod_df.set_index('Date').sort_index()
    
    full_date_range = pd.date_range(
        start=gom_prod_df.index.min(), 
        end=gom_prod_df.index.max(), 
        freq='D'
    )
    
    gom_prod_reindexed = gom_prod_df.reindex(full_date_range)
    
    gom_prod_reindexed['value'] = gom_prod_reindexed.groupby(
        pd.Grouper(freq='ME')
    )['value'].transform(lambda x: x.ffill().bfill())
    
    gom_prod_reindexed['item'] = item_name
    gom_prod_reindexed['region'] = region
    gom_prod_reindexed.dropna(subset=['value'], inplace=True)
    gom_prod_reindexed.reset_index(inplace=True)
    gom_prod_reindexed.rename(columns={'index': 'Date'}, inplace=True)

    logging.info(f"Backfilled 'GOM - Prod' from {len(gom_prod_df)} records to {len(gom_prod_reindexed)} records.")
    
    return pd.concat([other_df, gom_prod_reindexed], ignore_index=True)

def _truncate_stale_forecasts(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Trims stale data from each forecast item by finding its
    last valid (non-zero) date and cutting off everything after.
    """
    logging.info("Cleaning forecast data: Truncating stale (near-zero) items...")
    
    cleaned_dfs = []
    all_items = forecast_df['item'].unique()
    
    for item in all_items:
        item_df = forecast_df[forecast_df['item'] == item].sort_values('Date')
        
        if item in STALE_ZERO_EXEMPT_LIST:
            cleaned_dfs.append(item_df)
            continue
            
        valid_rows = item_df[item_df['value'].abs() > NEAR_ZERO_THRESHOLD]
        
        if valid_rows.empty:
            logging.info(f"Forecast item '{item}' has no non-zero data. Dropping.")
            continue
        
        last_valid_date = valid_rows['Date'].max()
        item_df_trimmed = item_df[item_df['Date'] <= last_valid_date]
        cleaned_dfs.append(item_df_trimmed)

    if not cleaned_dfs:
        logging.warning("Forecast data cleaning resulted in an empty dataframe.")
        return pd.DataFrame()
        
    logging.info("Finished truncating stale forecasts.")
    return pd.concat(cleaned_dfs, ignore_index=True)

def clean_forecast_data(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a series of cleaning steps to the raw forecast dataframe.
    """
    logging.info("--- Starting Forecast Data Cleaning Process ---")
    
    if forecast_df.empty:
        logging.warning("Forecast dataframe is empty. Skipping cleaning.")
        return forecast_df
    
    forecast_df_cleaned = _truncate_forecast_window(forecast_df)
    forecast_df_cleaned = _backfill_gom_prod(forecast_df_cleaned)
    forecast_df_cleaned = _truncate_stale_forecasts(forecast_df_cleaned)
    
    logging.info("--- Forecast Data Cleaning Finished ---")
    return forecast_df_cleaned


def get_common_items(actuals_df: pd.DataFrame, forecast_df: pd.DataFrame) -> Set[str]:
    """Finds items present in both dataframes and logs differences."""
    logging.info("Analyzing common items for balance calculations...")
    
    actuals_items = set(actuals_df['item'].unique())
    forecast_items = set(forecast_df['item'].unique())
    
    in_actuals_not_forecast = actuals_items - forecast_items
    if in_actuals_not_forecast:
        logging.warning(
            "Items in Actuals but NOT Forecast. "
            f"Will be EXCLUDED from balances: {sorted(list(in_actuals_not_forecast))}"
        )
        
    in_forecast_not_actuals = forecast_items - actuals_items
    if in_forecast_not_actuals:
        logging.warning(
            "Items in Forecast but NOT Actuals. "
            f"Will be EXCLUDED from balances: {sorted(list(in_forecast_not_actuals))}"
        )
    
    common_items_safe_list = actuals_items.intersection(forecast_items)
    logging.info(
        f"Found {len(common_items_safe_list)} common items "
        "to be used in balance calculations."
    )
    return common_items_safe_list


def perform_seam_analysis(
    actuals_df: pd.DataFrame, forecast_df: pd.DataFrame, common_items: Set[str]
) -> pd.DataFrame:
    """
    Analyzes the seam between actuals and forecast data for gaps and jumps.
    Prioritizes non-zero actuals over forecast data.
    Returns a *trimmed* forecast_df with overlaps removed.
    """
    logging.info("--- Starting Seam Analysis (Gaps, Overlaps, Jumps) ---")
    
    forecast_df_trimmed = forecast_df.copy()
    items_to_drop_from_forecast = []

    for item in common_items:
        item_actuals = actuals_df[actuals_df['item'] == item].sort_values('Date')
        item_forecasts = forecast_df_trimmed[
            forecast_df_trimmed['item'] == item
        ].sort_values('Date')
        
        if item_actuals.empty or item_forecasts.empty:
            continue
            
        valid_actuals = item_actuals[item_actuals['value'].abs() > NEAR_ZERO_THRESHOLD]
        
        if valid_actuals.empty:
            logging.info(f"No valid (non-zero) actuals found for '{item}'. Using all forecasts.")
            last_valid_actual_date = pd.Timestamp.min 
        else:
            last_valid_actual_record = valid_actuals.iloc[-1]
            last_valid_actual_date = last_valid_actual_record['Date']
        
        first_forecast_record = item_forecasts.iloc[0]
        first_forecast_date = first_forecast_record['Date']
        
        expected_forecast_start = last_valid_actual_date + timedelta(days=1)
        
        if first_forecast_date > expected_forecast_start:
            gap_days = (first_forecast_date - expected_forecast_start).days
            logging.warning(
                f"GAP DETECTED: Item '{item}' has a {gap_days}-day gap. "
                f"(Last valid actual: {last_valid_actual_date.strftime(DATE_FORMAT)}, "
                f"Forecast starts: {first_forecast_date.strftime(DATE_FORMAT)})"
            )
        
        elif first_forecast_date <= last_valid_actual_date:
            overlap_days = (last_valid_actual_date - first_forecast_date).days + 1
            logging.info(
                f"OVERLAP DETECTED: Item '{item}' overlaps valid actuals by {overlap_days} days. "
                "Trimming forecast data to prioritize actuals."
            )
            
            indices_to_drop = forecast_df_trimmed[
                (forecast_df_trimmed['item'] == item) &
                (forecast_df_trimmed['Date'] <= last_valid_actual_date)
            ].index
            items_to_drop_from_forecast.extend(indices_to_drop)
            
        # 2. Check for Value Jumps
        post_trim_forecast_records = item_forecasts[
            item_forecasts['Date'] > last_valid_actual_date
        ]
        if post_trim_forecast_records.empty:
            logging.info(f"No forecast data remains for '{item}' after overlap trim.")
            continue
            
        first_valid_forecast_record = post_trim_forecast_records.iloc[0]
        
        if first_valid_forecast_record['Date'] == expected_forecast_start:
            if valid_actuals.empty:
                continue
                
            last_val_record = valid_actuals.iloc[-1]
            last_val = last_val_record['value']
            first_val = first_valid_forecast_record['value']
            
            if pd.isna(last_val) or pd.isna(first_val):
                continue
            
            try:
                if (abs(last_val) > NEAR_ZERO_THRESHOLD):
                    pct_jump = (first_val - last_val) / last_val
                    if abs(pct_jump) > ANOMALY_JUMP_THRESHOLD_PERCENT:
                        logging.warning(
                            f"JUMP DETECTED: Item '{item}' jumped {pct_jump:,.1%} "
                            f"at the seam. (From {last_val:,.2f} to {first_val:,.2f})"
                        )
                elif (abs(first_val) > NEAR_ZERO_THRESHOLD):
                     logging.warning(
                        f"JUMP DETECTED: Item '{item}' jumped from near-zero to {first_val:,.2f} "
                        "at the seam."
                    )
            except Exception:
                pass

    if items_to_drop_from_forecast:
        forecast_df_trimmed.drop(items_to_drop_from_forecast, inplace=True)
        logging.info(
            f"Removed {len(items_to_drop_from_forecast)} overlapping forecast records."
        )

    logging.info("--- Seam Analysis Finished ---")
    return forecast_df_trimmed


def merge_incremental_data(
    existing_df: pd.DataFrame,
    new_full_actuals_df: pd.DataFrame,
    all_mapped_items: Set[str]
) -> pd.DataFrame:
    """
    Combines existing data with new data using the incremental lookback logic.
    """
    logging.info("Starting incremental merge for Actuals data...")
    
    if existing_df.empty:
        logging.info("No existing actuals file. Using full fetched data.")
        return new_full_actuals_df # new_full_actuals_df has already been cleaned

    max_date = existing_df['Date'].max()
    if pd.isna(max_date):
        logging.warning("Existing actuals file has no valid dates. Using full fetch.")
        return new_full_actuals_df

    cutoff_date = max_date - timedelta(days=INCREMENTAL_LOOKBACK_DAYS)
    
    logging.info(
        f"Existing data found up to {max_date.strftime(DATE_FORMAT)}. "
        f"Performing incremental update from {cutoff_date.strftime(DATE_FORMAT)}."
    )
    
    old_data_to_keep = existing_df[existing_df['Date'] < cutoff_date]
    new_data_for_update = new_full_actuals_df[
        new_full_actuals_df['Date'] >= cutoff_date
    ]
    unmapped_items_df = existing_df[~existing_df['item'].isin(all_mapped_items)]
    
    if not unmapped_items_df.empty:
        logging.info(
            f"Keeping {unmapped_items_df['item'].nunique()} "
            "items from existing file that are no longer in the mapping."
        )
    
    final_actuals_df = pd.concat(
        [old_data_to_keep, new_data_for_update, unmapped_items_df],
        ignore_index=True
    )
    
    logging.info("Incremental merge finished.")
    return final_actuals_df


def patch_actuals_with_forecast(
    final_actuals_df: pd.DataFrame,
    trimmed_forecast_df: pd.DataFrame,
    common_items: Set[str]
) -> pd.DataFrame:
    """
    Patches the final actuals dataframe with recent forecast data
    where actuals are missing or zero, *up to the current date*.
    """
    logging.info("Patching missing recent actuals with forecast data...")
    
    # --- BUG FIX ---
    # Get today's date, normalized (to midnight)
    today = pd.Timestamp.now().normalize()
    # --- END BUG FIX ---
    
    patched_dfs = []
    all_items = final_actuals_df['item'].unique()
    
    for item in all_items:
        if item not in common_items:
            patched_dfs.append(final_actuals_df[final_actuals_df['item'] == item])
            continue
            
        item_actuals = final_actuals_df[
            final_actuals_df['item'] == item
        ].sort_values('Date')
        
        valid_actuals = item_actuals[item_actuals['value'].abs() > NEAR_ZERO_THRESHOLD]
        
        if valid_actuals.empty:
            last_valid_actual_date = pd.Timestamp.min
        else:
            last_valid_actual_date = valid_actuals['Date'].max()
            
        # --- BUG FIX ---
        # Get forecast data *after* this date but *only up to today*.
        forecast_patch_data = trimmed_forecast_df[
            (trimmed_forecast_df['item'] == item) &
            (trimmed_forecast_df['Date'] > last_valid_actual_date) &
            (trimmed_forecast_df['Date'] <= today) # This ensures no future dates
        ]
        # --- END BUG FIX ---
        
        patched_item_df = pd.concat([item_actuals, forecast_patch_data], ignore_index=True)
        patched_item_df.drop_duplicates(subset=['Date', 'item'], keep='last', inplace=True)
        
        patched_dfs.append(patched_item_df)
        
    logging.info("Finished patching actuals.")
    return pd.concat(patched_dfs, ignore_index=True)


def calculate_derived_series(
    base_df: pd.DataFrame, common_items_safe_list: Set[str]
) -> pd.DataFrame:
    """
    Calculates derived and aggregate series using ONLY the common_items_safe_list.
    """
    if base_df.empty or 'value' not in base_df.columns:
        return pd.DataFrame()

    logging.info("Calculating derived balance series...")
    
    df_pivot = base_df.pivot_table(
        index='Date', columns='item', values='value'
    ) # .fillna(0) is removed

    def get_col(col_name):
        if col_name in df_pivot and col_name in common_items_safe_list:
            return df_pivot[col_name]
        else:
            if col_name in df_pivot:
                logging.debug(
                    f"Item '{col_name}' excluded from balance (not in safe list)."
                )
            return pd.Series(np.nan, index=df_pivot.index) # Return NaN, not 0

    df_pivot['West - Ind'] = get_col('West[PNW] - Ind').add(get_col('West[CA] - Ind'), fill_value=0)
    df_pivot['West - ResCom'] = get_col('West[PNW] - ResCom').add(get_col('West[CA] - ResCom'), fill_value=0)
    df_pivot['West - Power'] = get_col('West[PNW] - Power').add(get_col('West[CA] - Power'), fill_value=0)

    for region in ["Northeast", "Midwest", "SouthCentral", "SouthEast", "Rockies", "West"]:
        prod = get_col(f"{region} - Prod").fillna(0)
        ind = get_col(f"{region} - Ind").fillna(0)
        rescom = get_col(f"{region} - ResCom").fillna(0)
        power = get_col(f"{region} - Power").fillna(0)
        
        balance = prod - (ind + rescom + power)
        
        all_zero = (prod.abs() < NEAR_ZERO_THRESHOLD) & \
                   (ind.abs() < NEAR_ZERO_THRESHOLD) & \
                   (rescom.abs() < NEAR_ZERO_THRESHOLD) & \
                   (power.abs() < NEAR_ZERO_THRESHOLD)
        
        # --- ROBUSTNESS FIX (to avoid FutureWarning) ---
        df_pivot.loc[all_zero, f"{region} - Balance"] = np.nan
        df_pivot.loc[~all_zero, f"{region} - Balance"] = balance
        # --- END FIX ---

    conus_supply = get_col('CONUS - Prod').fillna(0)
    conus_demand_items = [
        'CONUS - Ind', 'CONUS - ResCom', 'CONUS - Power', 'CONUS - L&P',
        "CONUS - P'loss", 'CONUS - LNGexp'
    ]
    conus_demand = sum(get_col(c).fillna(0) for c in conus_demand_items)
    
    conus_balance = conus_supply - conus_demand
    
    all_demand_zero = (conus_demand.abs() < NEAR_ZERO_THRESHOLD)
    all_zero = (conus_supply.abs() < NEAR_ZERO_THRESHOLD) & all_demand_zero
    
    # --- ROBUSTNESS FIX (to avoid FutureWarning) ---
    df_pivot.loc[all_zero, 'CONUS - Balance'] = np.nan
    df_pivot.loc[~all_zero, 'CONUS - Balance'] = conus_balance
    # --- END FIX ---
    
    logging.info("Finished calculating derived balances.")
    return df_pivot


def finalize_and_save(
    derived_df: pd.DataFrame, base_df: pd.DataFrame, output_path: Path
):
    """Unpivots the final data, applies formatting, and saves to CSV."""
    if derived_df.empty:
        logging.warning(f"No derived data to save for {output_path.name}.")
        return

    logging.info(f"Finalizing and formatting data for {output_path.name}...")
    final_df_long = derived_df.reset_index().melt(
        id_vars='Date', var_name='item', value_name='value'
    )
    
    region_map = base_df.dropna(
        subset=['region']
    ).set_index('item')['region'].to_dict()
    
    final_df_long['region'] = final_df_long['item'].apply(
        get_region_for_item, region_map=region_map
    )
    
    final_df_long.dropna(subset=['value'], inplace=True)
    
    final_df_long.rename(columns={
        "item": "Item", "value": "Value", "region": "Region"
    }, inplace=True)
    
    final_df_long["Region"] = final_df_long["Region"].str.strip().replace(REGION_MAP)
    final_df_long["Date"] = pd.to_datetime(final_df_long["Date"]).dt.date
    final_df_long["Value"] = pd.to_numeric(final_df_long["Value"], errors='coerce').astype("float")
    final_df_long["Item"] = final_df_long["Item"].astype("string")
    final_df_long["Region"] = final_df_long["Region"].astype("string")
    
    try:
        logging.info(f"Saving final data to {output_path}...")
        final_df_long.sort_values(by=['Item', 'Date'], inplace=True)
        final_df_long.drop_duplicates(subset=['Item', 'Date'], keep='last', inplace=True)
        final_df_long.to_csv(output_path, index=False, date_format=DATE_FORMAT)
        logging.info(f"SUCCESS: {output_path.name} saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save final CSV file '{output_path.name}': {e}")

# ==============================================================================
#  ORCHESTRATION (MAIN)
# ==============================================================================

def main():
    """Main entry point for the script."""
    setup_logging()
    logging.info("========= Starting Fundy Data Update Process =========")

    engine = get_db_engine()
    if not engine:
        return

    try:
        INFO_DIR.mkdir(exist_ok=True)
        
        # 1. Load Master Mapping File
        logging.info(f"Loading master mapping file: {MASTER_MAPPING_FILE}")
        mapping_path = MAPPING_DIR / MASTER_MAPPING_FILE
        if not mapping_path.exists():
            logging.critical(f"Mapping file not found: {mapping_path}. Aborting.")
            return
            
        mapping_df = pd.read_csv(mapping_path)
        
        if 'History Ticker' not in mapping_df.columns or 'Forecast Ticker' not in mapping_df.columns:
            logging.critical(
                "Mapping file must contain 'History Ticker' and 'Forecast Ticker'. Aborting."
            )
            return

        actuals_ticker_map = mapping_df.set_index('Item')['History Ticker'].dropna().to_dict()
        forecast_ticker_map = mapping_df.set_index('Item')['Forecast Ticker'].dropna().to_dict()
        all_mapped_items = set(mapping_df['Item'].unique())

        # 2. Fetch ALL base data for both modes first
        actuals_base_df = fetch_all_base_series(engine, actuals_ticker_map, "ACTUALS")
        forecast_base_df = fetch_all_base_series(engine, forecast_ticker_map, "FORECAST")

        # 3. Clean both datasets
        actuals_base_df = clean_actuals_data(actuals_base_df)
        forecast_base_df = clean_forecast_data(forecast_base_df)

        if actuals_base_df.empty:
             logging.warning("Actuals base dataset is empty after cleaning. Proceeding with forecast only.")
        if forecast_base_df.empty:
            logging.warning("Forecast base dataset is empty after cleaning. Cannot create forecast file.")

        # 4. Find Common Items
        common_items_safe_list = get_common_items(actuals_base_df, forecast_base_df)

        # 5. Perform Seam Analysis
        trimmed_forecast_base_df = perform_seam_analysis(
            actuals_base_df, forecast_base_df, common_items_safe_list
        )
        
        # 6. Process ACTUALS file (Incremental)
        logging.info("--- Starting ACTUALS File Processing ---")
        actuals_output_path = INFO_DIR / MASTER_ACTUALS_OUTPUT
        existing_actuals_df = load_existing_data(actuals_output_path)
        
        final_actuals_df = merge_incremental_data(
            existing_actuals_df, actuals_base_df, all_mapped_items
        )
        
        final_actuals_df = patch_actuals_with_forecast(
            final_actuals_df, trimmed_forecast_base_df, common_items_safe_list
        )
        
        derived_actuals_df = calculate_derived_series(
            final_actuals_df, common_items_safe_list
        )
        
        finalize_and_save(
            derived_actuals_df, final_actuals_df, actuals_output_path
        )
        
        # 7. Process FORECAST file (Full Replace)
        logging.info("--- Starting FORECAST File Processing ---")
        forecast_output_path = INFO_DIR / MASTER_FORECAST_OUTPUT
        
        if trimmed_forecast_base_df.empty:
            logging.warning("No forecast data to process. Skipping forecast file creation.")
        else:
            derived_forecast_df = calculate_derived_series(
                trimmed_forecast_base_df, common_items_safe_list
            )
            
            finalize_and_save(
                derived_forecast_df, trimmed_forecast_base_df, forecast_output_path
            )

    except Exception:
        logging.critical("A critical error occurred in the main process.")
        logging.critical(traceback.format_exc())
    finally:
        if engine:
            engine.dispose()
            logging.info("Database connection closed.")
        logging.info("================== Process Finished ==================")

if __name__ == "__main__":
    main()
