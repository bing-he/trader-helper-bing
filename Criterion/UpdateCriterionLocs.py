import pandas as pd
from pathlib import Path
import sys
import os
import logging
import traceback
from typing import Optional, List
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, Engine

# ==============================================================================
#  CONFIGURATION & CONSTANTS
# ==============================================================================

# --- File & Directory Paths ---
from common.pathing import ROOT
SCRIPT_DIR = ROOT / "Criterion"
INFO_DIR = ROOT / "INFO"
MASTER_CSV_PATH = INFO_DIR / "locs_list.csv"

# --- Database Details ---
DB_HOST = "dda.criterionrsch.com"
DB_PORT = 443
DB_NAME = "production"

# --- Column & Key Definitions ---
# Columns to fetch from the database
DB_COLUMNS = [
    "loc_name", "loc", "pipeline_name", "loc_zone", "category_short",
    "sub_category_desc", "sub_category_2_desc", "state_name", "county_name",
    "connecting_pipeline", "storage_name", "ticker"
]
# The final, complete order of columns for the output CSV
FINAL_COLUMN_ORDER = [
    "loc_name", "loc", "pipeline_name", "loc_zone", "category_short",
    "sub_category_desc", "sub_category_2_desc", "state_name", "county_name",
    "connecting_pipeline", "storage_name", "market_component", "ticker"
]
# The unique key used to identify a row
MERGE_KEY = 'loc'

# ==============================================================================
#  SETUP & HELPER FUNCTIONS
# ==============================================================================

def setup_logging():
    """Configures a basic logger to show timestamped messages."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def get_db_engine() -> Optional[Engine]:
    """Loads credentials from .env and creates a SQLAlchemy database engine."""
    logging.info("Attempting to connect to the Criterion database...")
    try:
        # Load credentials from a .env file located in the same directory as the script
        load_dotenv(dotenv_path=SCRIPT_DIR / ".env", override=True)
        db_user, db_password = os.getenv("DB_USER"), os.getenv("DB_PASSWORD")
        if not db_user or not db_password:
            raise ValueError("DB_USER and/or DB_PASSWORD not found in .env file.")
        
        conn_url = f"postgresql://{db_user}:{db_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(conn_url, connect_args={"sslmode": "require"})
        
        # Test the connection to ensure it's working before proceeding
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logging.info("Database connection confirmed.")
        return engine
    except Exception as e:
        logging.critical(f"Database connection failed: {e}")
        return None

def load_master_data(path: Path) -> pd.DataFrame:
    """Loads the local master CSV file and prepares it for the update."""
    try:
        logging.info(f"Loading master file from: {path.name}")
        df = pd.read_csv(path, dtype={MERGE_KEY: str})
        logging.info(f"Successfully loaded master file with {len(df)} records.")
        return df
    except FileNotFoundError:
        logging.warning(f"Master file '{path.name}' not found. A new one will be created.")
        return pd.DataFrame(columns=FINAL_COLUMN_ORDER)

def fetch_database_data(engine: Engine) -> Optional[pd.DataFrame]:
    """Fetches the live location data from the database."""
    logging.info("Fetching live location data from the database...")
    query = text(f"""
        SELECT DISTINCT {', '.join(DB_COLUMNS)}
        FROM pipelines.metadata WHERE country_name = 'United States'
    """)
    try:
        df = pd.read_sql(query, engine)
        logging.info(f"Successfully fetched {len(df)} unique US locations.")
        df[MERGE_KEY] = df[MERGE_KEY].astype(str)
        return df
    except Exception as e:
        logging.error(f"Failed to fetch data from database: {e}")
        return None

# ==============================================================================
#  ORCHESTRATION
# ==============================================================================

def main():
    """Orchestrates the entire location list synchronization process."""
    setup_logging()
    logging.info("========= Starting Location List Update Process =========")
    
    engine = get_db_engine()
    if not engine:
        logging.critical("Halting process: Cannot continue without a database connection.")
        return

    try:
        # Step 1: Load the local master data
        master_df = load_master_data(MASTER_CSV_PATH)
        
        # Step 2: Fetch the live source data from the database
        source_df = fetch_database_data(engine)
        
        if source_df is None or source_df.empty:
            logging.error("Source data from database is empty. Aborting update.")
            return

        # Step 3: Identify new rows
        if master_df.empty:
            new_rows_df = source_df
        else:
            existing_locs = set(master_df[MERGE_KEY].dropna())
            new_rows_df = source_df[~source_df[MERGE_KEY].isin(existing_locs)]

        # Step 4: Append new rows and save the result
        if not new_rows_df.empty:
            logging.info(f"Found {len(new_rows_df)} new locations to add.")
            
            # Combine the old data with the new rows
            updated_df = pd.concat([master_df, new_rows_df], ignore_index=True)
            
            # Reorder columns to the final, consistent schema and save
            updated_df = updated_df.reindex(columns=FINAL_COLUMN_ORDER)
            updated_df.sort_values(by=['pipeline_name', 'loc_name'], inplace=True)
            updated_df.to_csv(MASTER_CSV_PATH, index=False)
            
            logging.info(f"SUCCESS: Master file updated. Total records: {len(updated_df)}")
        else:
            logging.info("No new locations found. The master file is already up-to-date.")

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
