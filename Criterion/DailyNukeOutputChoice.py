import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta
import ast  # To safely evaluate strings as lists
import os
import logging
from typing import Optional, Dict, List

# --- Database Libraries ---
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, Engine
from common.pathing import ROOT

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SCRIPT_DIR = ROOT / "Criterion"
CRITERION_INFO_DIR = SCRIPT_DIR / "CriterionInfo"
PLANT_MAPPING_PATH = CRITERION_INFO_DIR / "Plant_Group_Mapping.csv"
PAIRS_FILE_PATH = CRITERION_INFO_DIR / "NuclearPairs.csv"

# --- Database Details ---
DB_HOST = "dda.criterionrsch.com"
DB_PORT = 443
DB_NAME = "production"

# --- SQL Queries (as used in your original script) ---
HISTORICAL_DATA_QUERY = text("""
    SELECT
        dat.date,
        dat.reactor_name,
        dat.operational_percent::float / 100 * map.net_summer_capacity AS generation
    FROM power.nrc_raw_data AS dat
    LEFT JOIN power.nrc_mappings AS map ON map.nrc_reactor_name = dat.reactor_name
    WHERE dat.date >= :start_date AND dat.date <= :end_date AND dat.reactor_name = ANY(:units)
""")

FORECAST_DATA_QUERY = text(
    "SELECT date, ticker, value FROM data_series.fin_json_to_excel_tickers(:tickers) WHERE date >= :start_date AND date <= :end_date"
)

# ==============================================================================
# HELPER FUNCTIONS
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
    logging.info("Attempting to connect to the database...")
    try:
        dotenv_path = SCRIPT_DIR / ".env"
        if not dotenv_path.exists():
            raise FileNotFoundError(f"CRITICAL: .env file not found at {dotenv_path}. Please create it with your DB_USER and DB_PASSWORD.")

        load_dotenv(dotenv_path=dotenv_path, override=True)
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

def get_user_input(available_groups: List[str]):
    """Prompts the user for facility and date range."""
    print("--- Available Nuclear Facilities (Groups) ---")
    for group in sorted(available_groups):
        print(f"- {group}")
    print("-" * 41)

    selected_group = input("Enter the name of the nuclear facility to view: ")
    if selected_group not in available_groups:
        logging.error(f"Facility '{selected_group}' not found.")
        return None, None, None

    try:
        start_date_str = input("Enter a start date (YYYY-MM-DD): ")
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date_str = input("Enter an end date (YYYY-MM-DD): ")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        if end_date < start_date:
            logging.error("End date cannot be before the start date.")
            return None, None, None
        return selected_group, start_date, end_date
    except ValueError:
        logging.error("Invalid date format. Please use YYYY-MM-DD.")
        return None, None, None

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Orchestrates the entire data fetching and display process."""
    setup_logging()
    logging.info("========= Live Nuclear Generation Data Viewer =========")

    engine = get_db_engine()
    if not engine:
        return

    try:
        # --- 1. Load mapping files ---
        plant_map_df = pd.read_csv(PLANT_MAPPING_PATH)
        plant_map_df['Units'] = plant_map_df['Units'].apply(ast.literal_eval)
        group_to_units_dict = plant_map_df.set_index('Group')['Units'].to_dict()

        pairs_df = pd.read_csv(PAIRS_FILE_PATH)
        item_to_forecast_ticker = pairs_df.set_index('Item')['Forecast Ticker'].to_dict()

        # --- 2. Get user input ---
        available_groups = plant_map_df["Group"].unique().tolist()
        group, start_date, end_date = get_user_input(available_groups)
        if not group:
            return

        # --- 3. Determine units and tickers for the selected group ---
        units_in_group = group_to_units_dict.get(group, [])
        forecast_tickers = [item_to_forecast_ticker[u] for u in units_in_group if u in item_to_forecast_ticker]

        logging.info(f"Fetching data for '{group}' from {start_date} to {end_date}")
        logging.info(f"Associated Units: {units_in_group}")
        logging.info(f"Associated Tickers: {forecast_tickers}")

        all_data = []
        today = date.today()

        # --- 4. Fetch data directly from the database ---
        with engine.connect() as conn:
            # Fetch Historical Data for dates *before* today
            if start_date < today:
                hist_end_date = min(end_date, today - timedelta(days=1))
                if hist_end_date >= start_date:
                    logging.info(f"Querying for historical data from {start_date} to {hist_end_date}...")
                    hist_data = pd.read_sql(
                        HISTORICAL_DATA_QUERY, conn, params={"start_date": start_date, "end_date": hist_end_date, "units": units_in_group}
                    )
                    hist_data.rename(columns={"generation": "Value"}, inplace=True)
                    if not hist_data.empty:
                        all_data.append(hist_data[['date', 'Value']])
                    logging.info(f"Found {len(hist_data)} historical records.")

            # Fetch Forecast Data for dates from *today* onwards
            if end_date >= today and forecast_tickers:
                forecast_start_date = max(start_date, today)
                logging.info(f"Querying for forecast data from {forecast_start_date} to {end_date}...")
                all_forecast_data = []
                for ticker in forecast_tickers:
                    forecast_df_single = pd.read_sql(
                        FORECAST_DATA_QUERY, conn, params={"start_date": forecast_start_date, "end_date": end_date, "tickers": ticker}
                    )
                    if not forecast_df_single.empty:
                        all_forecast_data.append(forecast_df_single)
                
                if all_forecast_data:
                    forecast_data = pd.concat(all_forecast_data, ignore_index=True)
                    forecast_data.rename(columns={"value": "Value"}, inplace=True)
                    all_data.append(forecast_data[['date', 'Value']])
                    logging.info(f"Found a total of {len(forecast_data)} forecast records.")

        # --- 5. Process and display results ---
        if not all_data:
            print("\n--- No data found for the selected facility and date range. ---")
            return

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df['date']).dt.date

        # Aggregate data by day
        aggregated_df = combined_df.groupby('date')['Value'].sum().reset_index()

        print(f"\n--- Total Daily Generation for {group} ---")
        display_df = aggregated_df.sort_values(by="date")
        display_df['date'] = display_df['date'].apply(lambda d: d.strftime('%Y-%m-%d'))
        display_df['Value'] = display_df['Value'].map('{:,.2f}'.format)
        print(display_df.to_string(index=False))
        print("-" * 35)

    except Exception as e:
        logging.critical(f"An unexpected error occurred: {e}")
    finally:
        if engine:
            engine.dispose()
            logging.info("Database connection closed.")
        logging.info("================== Process Finished ==================")

if __name__ == "__main__":
    main()
