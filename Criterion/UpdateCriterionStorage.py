"""
Fetches and processes daily storage change data from a database.

This script calculates the daily net storage change for various facilities.
It connects to a production database, fetches nomination data, aggregates it,
and saves the result to a CSV file.

The script supports incremental updates. If an output file already exists, it
will only fetch the last 60 days of data to update the records. Otherwise,
it performs a full data fetch from a predefined start date.

Data: daily storage change per facility
Negative daily_storage_change often means net withdrawal (inventory going down) given your sign convention.
Positive daily_storage_change often means net injection.
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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, Engine
from common.pathing import ROOT

# --- Configuration & Constants ---

SCRIPT_DIR = ROOT / "Criterion"
OUTPUT_DIR = ROOT / "INFO"
OUTPUT_FILE = OUTPUT_DIR / "CriterionStorageChange.csv"

FULL_FETCH_START_DATE = "2015-01-01"
INCREMENTAL_UPDATE_DAYS = 60
DATE_FORMAT = "%Y-%m-%d"

# Renamed 'eff_gas_day' to 'Date' directly in the SQL query
STORAGE_QUERY = text("""
    SELECT
        meta.storage_name,
        noms.eff_gas_day AS "Date",
        SUM(noms.scheduled_quantity * meta.rec_del_sign) AS intermediate_daily_net_flow
    FROM
        pipelines.nomination_points AS noms
    JOIN
        pipelines.metadata AS meta ON meta.metadata_id = noms.metadata_id
    WHERE
        meta.storage_calc_flag = 'T'
        AND meta.category_short = 'Storage'
        AND meta.sub_category_desc = 'Daily Flows'
        AND (meta.ticker LIKE '%.7' OR meta.ticker LIKE '%.8')
        AND noms.eff_gas_day BETWEEN :start_date AND :end_date
    GROUP BY
        meta.storage_name,
        noms.eff_gas_day
""")

# --- Helper Functions ---

def setup_logging():
    """Configures a basic logger to show timestamped messages."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt=DATE_FORMAT
    )

def get_db_engine() -> Optional[Engine]:
    """
    Load database credentials and create a SQLAlchemy engine.
    This function is restored from your original script to fix the connection.
    """
    logging.info("Connecting to the database...")
    load_dotenv(dotenv_path=SCRIPT_DIR / ".env", override=True)
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")

    if not db_user or not db_password:
        logging.critical("Database credentials (DB_USER, DB_PASSWORD) not found in .env file.")
        return None

    database_url = (
        f"postgresql://{db_user}:{db_password}"
        f"@dda.criterionrsch.com:443/production"
    )
    try:
        engine = create_engine(
            database_url, connect_args={"sslmode": "require"}
        )
        engine.connect().close() # Test connection
        logging.info("Database connection successful.")
        return engine
    except Exception as e:
        logging.critical(f"Database connection failed: {e}")
        return None

def determine_fetch_start_date(file_path: Path) -> datetime:
    """Determines the start date for data fetching based on existing data."""
    try:
        # Changed to read 'Date' column after SQL rename
        df = pd.read_csv(file_path, parse_dates=["Date"])
        if not df.empty:
            max_date = df["Date"].max()
            start_date = max_date - timedelta(days=INCREMENTAL_UPDATE_DAYS)
            logging.info(f"Incremental update from {start_date.strftime(DATE_FORMAT)}")
            return start_date
    except FileNotFoundError:
        logging.info("Output file not found. Performing full data fetch.")
    except Exception as e:
        logging.warning(f"Could not read existing file: {e}. Performing full fetch.")

    return pd.to_datetime(FULL_FETCH_START_DATE)

def fetch_data(engine: Engine, start_date: datetime) -> pd.DataFrame:
    """Fetches storage data from the database."""
    today = datetime.now()
    logging.info(f"Fetching data from {start_date.strftime(DATE_FORMAT)} to {today.strftime(DATE_FORMAT)}.")
    try:
        with engine.connect() as connection:
            params = {
                "start_date": start_date.strftime(DATE_FORMAT),
                "end_date": today.strftime(DATE_FORMAT),
            }
            df = pd.read_sql_query(STORAGE_QUERY, connection, params=params)
            logging.info(f"Fetched {len(df)} rows from the database.")
            return df
    except Exception as e:
        logging.error(f"Failed to fetch data from database: {e}")
        return pd.DataFrame()

def process_and_combine_data(
    new_data: pd.DataFrame, existing_data_path: Path, fetch_start_date: datetime
) -> pd.DataFrame:
    """Processes new data and combines it with historical data."""
    if new_data.empty:
        logging.info("No new data to process.")
        try:
            return pd.read_csv(existing_data_path, parse_dates=["Date"])
        except FileNotFoundError:
            return pd.DataFrame()

    # --- Type Conversion ---
    new_data['daily_storage_change'] = (new_data["intermediate_daily_net_flow"] * -1).astype('float')
    new_data['Date'] = pd.to_datetime(new_data['Date']).dt.date
    new_data['storage_name'] = new_data['storage_name'].astype('string')
    
    processed_new_data = new_data[["storage_name", "Date", "daily_storage_change"]]

    try:
        existing_df = pd.read_csv(existing_data_path, parse_dates=["Date"])
        existing_df['Date'] = existing_df['Date'].dt.date
        
        old_data_to_keep = existing_df[existing_df["Date"] < fetch_start_date.date()]
        final_df = pd.concat([old_data_to_keep, processed_new_data], ignore_index=True)
        logging.info(f"Combined {len(old_data_to_keep)} old rows with {len(processed_new_data)} new rows.")
    except FileNotFoundError:
        final_df = processed_new_data
        logging.info("No existing data file found; using only new data.")
    
    final_df = final_df.sort_values(by=["storage_name", "Date"])
    final_df = final_df.drop_duplicates(subset=["storage_name", "Date"], keep="last")
    return final_df

def save_data_to_csv(df: pd.DataFrame, output_path: Path):
    """Saves a DataFrame to a CSV file."""
    if df.empty:
        logging.warning("Final DataFrame is empty. Nothing to save.")
        return
        
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, date_format=DATE_FORMAT)
        logging.info(f"Data successfully saved to:\n{output_path}")
    except Exception as e:
        logging.error(f"Failed to save data to CSV: {e}")

def main():
    """Main orchestration function for the ETL process."""
    setup_logging()
    logging.info("Starting storage data update process...")

    engine = get_db_engine()
    if not engine:
        return

    try:
        fetch_start_date = determine_fetch_start_date(OUTPUT_FILE)
        new_data_df = fetch_data(engine, fetch_start_date)
        final_df = process_and_combine_data(
            new_data_df, OUTPUT_FILE, fetch_start_date
        )
        save_data_to_csv(final_df, OUTPUT_FILE)
    finally:
        if engine:
            engine.dispose()
            logging.info("Database connection closed.")
    logging.info("Process finished.")

if __name__ == "__main__":
    main()
