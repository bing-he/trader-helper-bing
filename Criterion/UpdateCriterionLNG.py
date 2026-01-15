"""
Fetches, processes, and saves historical and forecast LNG feed gas data.

This script connects to a database to retrieve two types of LNG data:
1.  Historical daily feed gas data for various LNG terminals.
2.  Forward-looking feed gas forecasts for the same terminals.

It processes each dataset, performing an incremental update for historical data
and a full overwrite for forecast data, saving the results to separate CSV files.
"""

import logging
import os
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Union

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
HIST_OUTPUT_PATH = INFO_DIR / "CriterionLNGHist.csv"
FORECAST_OUTPUT_PATH = INFO_DIR / "CriterionLNGForecast.csv"

# --- Data Configuration ---
START_DATE_FULL_FETCH = pd.to_datetime("2015-01-01")
INCREMENTAL_LOOKBACK_DAYS = 60
FORECAST_WINDOW_DAYS = 60
DATE_FORMAT = "%Y-%m-%d"

# --- Database Details ---
DB_HOST = "dda.criterionrsch.com"
DB_PORT = "443"
DB_NAME = "production"

# --- Tickers and Mappings ---
HISTORICAL_TICKERS = [
    "PLAG.LNGEXP.SUM.CALCP.A",
    "PLAG.LNGEXP.SUM.CAMER.A",
    "PLAG.LNGEXP.SUM.CCL.A",
    "PLAG.LNGEXP.SUM.COVE.A",
    "PLAG.LNGEXP.SUM.ELBA.A",
    "PLAG.LNGEXP.SUM.FLNG.A",
    "PLAG.LNGEXP.SUM.GP.A",
    "PLAG.LNGEXP.SUM.PLQ.A",
    "PLAG.LNGEXP.SUM.SPL.A",
]

FORECAST_TICKERS_MAP = {
    "PLAG.LNGEXP.CALCP.MAINTCAP.F": "Calcasieu Pass LNG Feed Gas",
    "PLAG.LNGEXP.CAMER.MAINTCAP.F": "Cameron LNG Feed Gas",
    "PLAG.LNGEXP.CCL.MAINTCAP.F": "Corpus Christi LNG Feed Gas",
    "PLAG.LNGEXP.COVE.MAINTCAP.F": "Cove Point LNG Feed Gas",
    "PLAG.LNGEXP.ELBA.MAINTCAP.F": "Elba Island LNG Feed Gas",
    "PLAG.LNGEXP.FLNG.MAINTCAP.F": "Freeport LNG Feed Gas",
    "PLAG.LNGEXP.GP.MAINTCAP.F": "Golden Pass LNG Feed Gas",
    "PLAG.LNGEXP.PLQ.MAINTCAP.F": "Plaquemines LNG Feed Gas",
    "PLAG.LNGEXP.SPL.MAINTCAP.F": "Sabine Pass LNG Feed Gas",     
}

# --- State Name to Code Mapping ---
STATE_NAME_TO_CODE = {
    "Louisiana": "LA",
    "Texas": "TX",
    "Georgia": "GA",
    "Maryland": "MD",
    "Unknown": None,
}

# ==============================================================================
#  SETUP & HELPER FUNCTIONS
# ==============================================================================


def setup_logging():
    """Configures a basic logger for console output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_db_engine() -> Optional[Engine]:
    """Loads credentials and creates a SQLAlchemy engine."""
    logging.info("Attempting to connect to the database...")
    try:
        load_dotenv(dotenv_path=SCRIPT_DIR / ".env", override=True)
        db_user, db_password = os.getenv("DB_USER"), os.getenv("DB_PASSWORD")
        if not db_user or not db_password:
            raise ValueError("DB credentials not found in environment.")

        conn_url = f"postgresql://{db_user}:{db_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(conn_url, connect_args={"sslmode": "require"})
        # sanity check
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logging.info("Database engine created and connection confirmed.")
        return engine
    except Exception as e:
        logging.critical(f"Database connection failed: {e}")
        return None


def fetch_data_from_db(engine: Engine, tickers: List[str]) -> pd.DataFrame:
    """Fetches and prepares data for a list of tickers from the database."""
    if not tickers:
        logging.warning("No tickers provided for database query.")
        return pd.DataFrame()

    query = text("SELECT DISTINCT * FROM data_series.fin_json_to_excel_tickers(:tickers)")
    try:
        df = pd.read_sql(query, engine, params={"tickers": ",".join(tickers)})
        logging.info(f"Fetched {len(df)} rows from the database.")
        return df
    except Exception as e:
        logging.error(f"Database query failed: {e}")
        return pd.DataFrame()


# ==============================================================================
#  CORE PIPELINE FUNCTIONS
# ==============================================================================


def process_data(raw_df: pd.DataFrame, description_source: Union[str, Dict[str, str]]) -> pd.DataFrame:
    """
    Processes raw data by assigning item descriptions and standardizing columns.
    Defensive against column name variants and categorical dtypes.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=["Date", "Item", "Value", "State"])

    # Normalize common column name variants up front
    rename_map = {
        "series_desc": "series_description",
        "date": "Date",
        "value": "Value",
        "values": "Value",  # handle plural from some sources
        "state_name": "State",
        "state": "State",
        "ticker": "ticker",
    }
    raw_df = raw_df.rename(columns=rename_map)

    # Source of Item/description
    if description_source == "DB":
        if "series_description" not in raw_df.columns:
            logging.error(
                "Description source set to 'DB' but 'series_description' column not found."
            )
            return pd.DataFrame(columns=["Date", "Item", "Value", "State"])
        raw_df["Item"] = raw_df["series_description"]
    else:
        if "ticker" not in raw_df.columns:
            logging.error("Ticker column not found for mapping descriptions.")
            return pd.DataFrame(columns=["Date", "Item", "Value", "State"])
        # map descriptions, fallback to ticker if missing
        raw_df["Item"] = raw_df["ticker"].map(description_source).fillna(raw_df["ticker"])

    # Default State if missing
    if "State" not in raw_df.columns:
        raw_df["State"] = "Unknown"

    # --- TYPE COERCION (stable, consistent) ---
    # Dates: keep as pandas datetime64[ns]
    raw_df["Date"] = pd.to_datetime(raw_df["Date"], errors="coerce")

    # Numeric Value
    raw_df["Value"] = pd.to_numeric(raw_df["Value"], errors="coerce")

    # Strings (force out of 'category' if present)
    raw_df["Item"] = raw_df["Item"].astype("string")

    # Map State names to codes, then to pandas StringDtype (nullable)
    raw_df["State"] = raw_df["State"].map(STATE_NAME_TO_CODE).astype("string")

    out = raw_df[["Date", "Item", "Value", "State"]].copy()

    # Drop rows with NA Date or Item (cannot sort/dedupe reliably otherwise)
    out = out.dropna(subset=["Date", "Item"])

    return out


def _normalize_dtypes_inplace(df: pd.DataFrame) -> None:
    """
    In-place: ensure no unordered categoricals remain; enforce stable dtypes.
    """
    if df is None or df.empty:
        return
    # Convert any categoricals to string
    for col in df.select_dtypes(include=["category"]).columns:
        df[col] = df[col].astype("string")
    # Enforce stable types
    df["Item"] = df["Item"].astype("string")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Value should be numeric float
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    # State stays as nullable string
    if "State" in df.columns:
        df["State"] = df["State"].astype("string")


def save_data(df: pd.DataFrame, output_path: Path):
    """Saves the final DataFrame to a CSV file."""
    if df.empty:
        logging.warning(
            f"Final DataFrame is empty. No file will be saved to {output_path.name}."
        )
        return

    try:
        df_to_save = df.copy()
        _normalize_dtypes_inplace(df_to_save)

        # Sort and dedupe (mergesort is stable)
        df_to_save.sort_values(by=["Item", "Date"], inplace=True, kind="mergesort")
        df_to_save.drop_duplicates(subset=["Date", "Item"], keep="last", inplace=True)

        # Save with consistent date format
        df_to_save.to_csv(output_path, index=False, date_format=DATE_FORMAT)
        logging.info(f"SUCCESS: Data successfully saved to:\n{output_path}")
    except Exception as e:
        logging.error(
            "Failed to save data to %s: %s\nDtypes at failure:\n%s",
            output_path.name,
            e,
            df_to_save.dtypes.to_string(),
        )


# ==============================================================================
#  ORCHESTRATION
# ==============================================================================


def run_historical_pipeline(engine: Engine):
    """Orchestrates the entire historical data update process."""
    logging.info("========= Processing HISTORICAL LNG Data =========")

    start_date = START_DATE_FULL_FETCH
    existing_df: Optional[pd.DataFrame] = None

    try:
        existing_df = pd.read_csv(HIST_OUTPUT_PATH, parse_dates=["Date"])
        if not existing_df.empty:
            _normalize_dtypes_inplace(existing_df)
            max_date = existing_df["Date"].max()
            # Pull a lookback window to safely overwrite recent history
            start_date = max_date - timedelta(days=INCREMENTAL_LOOKBACK_DAYS)
            logging.info(f"Existing file found. Fetching new data since {start_date:%Y-%m-%d}.")
    except FileNotFoundError:
        logging.info(
            f"No existing file found. Performing full data fetch since {start_date:%Y-%m-%d}."
        )
    except Exception as e:
        logging.warning(f"Could not read existing file: {e}. Performing full fetch.")

    new_data_raw = fetch_data_from_db(engine, HISTORICAL_TICKERS)
    if new_data_raw.empty:
        logging.warning("No new historical data fetched. Skipping update.")
        return

    new_data = process_data(new_data_raw, "DB")
    # Filter to the refresh window (inclusive)
    new_data = new_data[new_data["Date"] >= pd.to_datetime(start_date)]

    if existing_df is not None and not existing_df.empty:
        # Keep old rows strictly before the start_date (we'll replace the window)
        old_data_to_keep = existing_df[existing_df["Date"] < pd.to_datetime(start_date)]
        final_df = pd.concat([old_data_to_keep, new_data], ignore_index=True)
        _normalize_dtypes_inplace(final_df)
        logging.info(f"Combined {len(old_data_to_keep)} old rows with {len(new_data)} new rows.")
    else:
        final_df = new_data
        _normalize_dtypes_inplace(final_df)

    # Helpful debug (can be removed later)
    logging.info("About to save HIST with dtypes:\n%s", final_df.dtypes.to_string())
    save_data(final_df, HIST_OUTPUT_PATH)


def run_forecast_pipeline(engine: Engine):
    """Orchestrates the entire forecast data update process."""
    logging.info("========= Processing FORECAST LNG Data =========")

    tickers = list(FORECAST_TICKERS_MAP.keys())
    new_data_raw = fetch_data_from_db(engine, tickers)
    if new_data_raw.empty:
        logging.warning("No new forecast data fetched. Skipping update.")
        return

    processed_df = process_data(new_data_raw, FORECAST_TICKERS_MAP)

    start_date = pd.to_datetime(datetime.now().date())
    end_date = start_date + timedelta(days=FORECAST_WINDOW_DAYS)

    mask = (processed_df["Date"] >= start_date) & (processed_df["Date"] <= end_date)
    final_df = processed_df.loc[mask].copy()
    _normalize_dtypes_inplace(final_df)

    logging.info(
        f"Filtered forecast data for window: {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}."
    )
    save_data(final_df, FORECAST_OUTPUT_PATH)


def main():
    """Main entry point for the script."""
    setup_logging()
    logging.info("========= Starting LNG Data Update Process =========")

    INFO_DIR.mkdir(parents=True, exist_ok=True)
    engine = get_db_engine()
    if not engine:
        logging.critical("Halting process: Database connection failed.")
        return

    try:
        run_historical_pipeline(engine)
        run_forecast_pipeline(engine)
    except Exception:
        logging.critical("An unexpected error occurred in the main process.")
        logging.critical(traceback.format_exc())
    finally:
        if engine:
            engine.dispose()
            logging.info("Database connection closed.")
        logging.info("================== Process Finished ==================")


if __name__ == "__main__":
    main()
