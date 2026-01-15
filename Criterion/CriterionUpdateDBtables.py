import logging
import os
import pandas as pd
from sqlalchemy import create_engine, text, Engine
from typing import Optional
from pathlib import Path
from common.pathing import ROOT

# --- Configuration & Constants ---
SCRIPT_DIR = ROOT / "Criterion"
INFO_DIR = ROOT / "Criterion" / "CriterionInfo"
# The script will now read from and write back to the SAME file.
DATA_FILE_PATH = INFO_DIR / "all_database_tickers_and_descriptions.csv"

# --- Database Details ---
DB_HOST = "dda.criterionrsch.com"
DB_PORT = 443
DB_NAME = "production"

# --- Reconciliation Settings ---
MERGE_KEY = 'ticker'
METADATA_QUERY = text("SELECT * FROM pipelines.metadata")

# --- Setup Functions ---
def setup_logging():
    """Configures a basic logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-m-d %H:%M:%S",
    )

def get_db_engine() -> Optional[Engine]:
    """Loads credentials and returns a SQLAlchemy engine."""
    logging.info("Attempting to connect to the database...")
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=SCRIPT_DIR / ".env", override=True)
        
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")

        if not db_user or not db_password:
            raise ValueError("DB_USER or DB_PASSWORD not found in .env file.")

        conn_url = f"postgresql://{db_user}:{db_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(
            conn_url, connect_args={"sslmode": "require", "connect_timeout": 10}
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logging.info("Database connection successful.")
        return engine
    except Exception as e:
        logging.critical(f"Database connection failed: {e}")
        return None

# --- Main Execution ---
def main():
    """Orchestrates the in-place update process."""
    setup_logging()
    logging.info("========= Starting In-Place File Update Process =========")

    engine = get_db_engine()
    if not engine:
        return

    try:
        # 1. Load Local Data and store its structure
        logging.info(f"Loading local data from {DATA_FILE_PATH.name}")
        local_df = pd.read_csv(DATA_FILE_PATH)
        # IMPORTANT: Remember the original columns and their order
        original_columns = local_df.columns.tolist()
        logging.info(f"Original file has {len(local_df)} rows.")

        # 2. Fetch Live Database Data
        logging.info("Fetching live metadata from the database...")
        db_df = pd.read_sql_query(METADATA_QUERY, engine)
        logging.info(f"Successfully fetched {len(db_df)} records from the database.")

        # 3. Merge the two datasets
        logging.info(f"Merging data using the '{MERGE_KEY}' column...")
        local_df[MERGE_KEY] = local_df[MERGE_KEY].astype(str)
        db_df[MERGE_KEY] = db_df[MERGE_KEY].astype(str)

        merged_df = pd.merge(
            local_df, db_df, on=MERGE_KEY, how='outer', suffixes=('_local', '_db')
        )
        
        # 4. Prioritize database values for any conflicting columns
        # For any column that was in both files (e.g., state_name), the merge
        # creates state_name_local and state_name_db. We prefer the newer db value.
        for col in original_columns:
            if f"{col}_db" in merged_df.columns:
                # Use the database value if it exists, otherwise keep the local value
                merged_df[col] = merged_df[f"{col}_db"].combine_first(merged_df[f"{col}_local"])

        # 5. Filter and Reorder to match the original file's structure
        logging.info("Restoring original column structure...")
        final_df = merged_df[original_columns]
        logging.info(f"Updated file will have {len(final_df)} rows.")

        # 6. Save the result back to the original file path
        logging.info(f"Saving updated data back to {DATA_FILE_PATH.name}...")
        final_df.to_csv(DATA_FILE_PATH, index=False)
        logging.info("SUCCESS: The file has been updated in-place.")

    except FileNotFoundError:
        logging.error(f"The file to update could not be found at {DATA_FILE_PATH}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        if engine:
            engine.dispose()
            logging.info("Database connection closed.")
        logging.info("================== Process Finished ==================")

if __name__ == "__main__":
    main()
