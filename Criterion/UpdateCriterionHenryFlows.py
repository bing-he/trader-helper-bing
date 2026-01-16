"""
Fetches and updates daily nomination data for Henry Hub locations.

This script identifies specific 'Henry' market component tickers from a mapping
file, finds their corresponding database IDs, and fetches their daily scheduled
and available flow data. It supports incremental updates by checking for an
existing output file and fetching only recent data if one is found.
Outputs:

INFO/CriterionHenryFlows.csv
for each day, how much natural gas was scheduled to flow through each Henry Hub pipeline connection versus how much capacity was available
"""
# --- allow running as a script while keeping package imports working ---
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]  # Criterion/.. -> repo root
    sys.path.insert(0, str(repo_root))
# ----------------------------------------------------------------------

###-------------------------------- Imports ---------------------------------###
import logging
import os
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Optional, List, Tuple

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
LOCS_FILE_PATH = INFO_DIR / "locs_list.csv"
OUTPUT_FILE_PATH = INFO_DIR / "CriterionHenryFlows.csv"

# --- Data Configuration ---
START_DATE_FULL_FETCH = pd.to_datetime("2015-01-01")
INCREMENTAL_LOOKBACK_DAYS = 45
DATE_FORMAT = "%Y-%m-%d"

# --- Database Details ---
DB_HOST = "dda.criterionrsch.com"
DB_PORT = 443
DB_NAME = "production"

# --- SQL Queries (Restored from original script to fix error) ---
METADATA_ID_QUERY = text("SELECT metadata_id FROM pipelines.metadata WHERE ticker IN :tickers")
FLOWS_QUERY = text("""
    SELECT
        noms.eff_gas_day,
        meta.loc_name,
        noms.scheduled_quantity,
        noms.operationally_available
    FROM pipelines.nomination_points AS noms
    JOIN pipelines.metadata AS meta ON noms.metadata_id = meta.metadata_id
    WHERE noms.metadata_id IN :ids AND noms.eff_gas_day >= :start_date
""")

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
    """Loads DB credentials and creates a SQLAlchemy engine."""
    logging.info("Connecting to the database...")
    try:
        dotenv_path = SCRIPT_DIR / ".env"
        load_dotenv(dotenv_path=dotenv_path, override=True)
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")

        if not all([db_user, db_password]):
            raise ValueError("DB credentials not found in .env file.")

        conn_url = f"postgresql://{db_user}:{db_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(conn_url, connect_args={"sslmode": "require"})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logging.info("Database connection successful.")
        return engine
    except Exception as e:
        logging.critical(f"Database connection failed: {e}")
        return None

def get_henry_metadata_ids(engine: Engine, mapping_path: Path) -> Optional[Tuple[int, ...]]:
    """Loads the location mapping file, filters for Henry tickers, and gets their DB IDs."""
    logging.info(f"Loading tickers from {mapping_path.name}...")
    try:
        locs_df = pd.read_csv(mapping_path)
        henry_locs = locs_df[locs_df['market_component'].str.strip().str.lower() == 'henry']
        henry_tickers = tuple(henry_locs['ticker'].dropna().unique())

        if not henry_tickers:
            logging.error(f"No tickers with market_component 'henry' found in {mapping_path.name}.")
            return None

        with engine.connect() as connection:
            ids_df = pd.read_sql_query(METADATA_ID_QUERY, connection, params={'tickers': henry_tickers})
        
        metadata_ids = tuple(ids_df['metadata_id'].unique())
        if not metadata_ids:
            logging.error("Could not find matching metadata_ids for the Henry tickers.")
            return None
        
        logging.info(f"Found {len(metadata_ids)} unique metadata_ids.")
        return metadata_ids
    except FileNotFoundError:
        logging.critical(f"Mapping file not found at: {mapping_path}")
        return None
    except Exception as e:
        logging.error(f"Failed to process mapping file or get metadata IDs: {e}")
        return None

def determine_fetch_start_date(output_path: Path) -> pd.Timestamp:
    """Determines the data fetch start date based on existing file content."""
    try:
        df = pd.read_csv(output_path, parse_dates=["Date"])
        if not df.empty:
            last_date = df["Date"].max()
            start_date = last_date - timedelta(days=INCREMENTAL_LOOKBACK_DAYS)
            logging.info(f"Incremental update. Last date is {last_date.strftime(DATE_FORMAT)}.")
            return start_date
    except (FileNotFoundError, KeyError):
        logging.info("Output file not found. Performing full data fetch.")
    except Exception as e:
        logging.warning(f"Could not read existing file: {e}. Performing full fetch.")
    
    return START_DATE_FULL_FETCH

def fetch_flow_data(engine: Engine, metadata_ids: Tuple[int, ...], start_date: pd.Timestamp) -> pd.DataFrame:
    """Fetches flow data for the given metadata IDs from the specified start date."""
    logging.info(f"Fetching flow data from {start_date.strftime(DATE_FORMAT)}...")
    try:
        with engine.connect() as connection:
            df = pd.read_sql_query(
                FLOWS_QUERY, connection, params={'ids': metadata_ids, 'start_date': start_date}
            )
        logging.info(f"Successfully fetched {len(df)} rows.")
        return df
    except Exception as e:
        logging.error(f"Failed to fetch flow data: {e}")
        return pd.DataFrame()

def process_and_combine_data(
    new_data: pd.DataFrame, existing_data_path: Path, fetch_start_date: pd.Timestamp
) -> pd.DataFrame:
    """Cleans, renames, and combines new data with existing historical data."""
    if new_data.empty:
        logging.warning("No new data to process.")
        try:
            return pd.read_csv(existing_data_path, parse_dates=["Date"])
        except FileNotFoundError:
            return pd.DataFrame()

    # --- RENAME, FORMAT, and SET TYPES ---
    new_data.rename(columns={
        "eff_gas_day": "Date",
        "scheduled_quantity": "Scheduled",
        "operationally_available": "OperationallyAvailable"
    }, inplace=True)

    new_data["Date"] = pd.to_datetime(new_data["Date"]).dt.date
    new_data["loc_name"] = new_data["loc_name"].astype("string")
    new_data["Scheduled"] = pd.to_numeric(new_data["Scheduled"], errors='coerce').astype("float")
    new_data["OperationallyAvailable"] = pd.to_numeric(new_data["OperationallyAvailable"], errors='coerce').astype("float")

    processed_new_data = new_data[["Date", "loc_name", "Scheduled", "OperationallyAvailable"]]

    # --- Combine with existing data for incremental updates ---
    try:
        existing_df = pd.read_csv(existing_data_path, parse_dates=["Date"])
        existing_df["Date"] = existing_df["Date"].dt.date
        
        old_data_to_keep = existing_df[existing_df["Date"] < fetch_start_date.date()]
        final_df = pd.concat([old_data_to_keep, processed_new_data], ignore_index=True)
        logging.info(f"Combined {len(old_data_to_keep)} old rows with {len(processed_new_data)} new rows.")
    except FileNotFoundError:
        final_df = processed_new_data
        logging.info("No existing data file found; using only new data.")
    
    final_df.sort_values(by=["loc_name", "Date"], inplace=True)
    final_df.drop_duplicates(subset=["loc_name", "Date"], keep="last", inplace=True)
    return final_df

def save_data(df: pd.DataFrame, output_path: Path):
    """Saves the final DataFrame to a CSV file."""
    if df.empty:
        logging.warning("Final DataFrame is empty. Nothing to save.")
        return
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, date_format=DATE_FORMAT)
        logging.info(f"SUCCESS: Data successfully saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save data to CSV: {e}")

# ==============================================================================
#  ORCHESTRATION
# ==============================================================================

def main():
    """Orchestrates the entire Henry Hub flows update process."""
    setup_logging()
    logging.info("========= Starting Henry Hub Flows Update Process ==========")
    
    engine = get_db_engine()
    if not engine:
        logging.critical("Halting process: Database connection failed.")
        return

    try:
        metadata_ids = get_henry_metadata_ids(engine, LOCS_FILE_PATH)
        if not metadata_ids:
            logging.critical("Halting process: Could not retrieve necessary metadata IDs.")
            return

        start_date = determine_fetch_start_date(OUTPUT_FILE_PATH)
        new_data = fetch_flow_data(engine, metadata_ids, start_date)
        final_df = process_and_combine_data(new_data, OUTPUT_FILE_PATH, start_date)
        save_data(final_df, OUTPUT_FILE_PATH)

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
