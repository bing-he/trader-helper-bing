"""
Fetches, processes, and saves historical and forecast nuclear generation data.

This script connects to a database to retrieve historical and forecast generation
data for all nuclear units. It aggregates the data from individual units to the
plant level, adds State and Region information, and saves the final results
to two separate CSV files.
Output: 
    CriterionNuclearHist.csv
    CriterionNuclearForecast.csv
"""
# --- allow running as a script while keeping package imports working ---
if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]  # Criterion/.. -> repo root
    sys.path.insert(0, str(repo_root))
# ----------------------------------------------------------------------

###-------------------------------- Imports ---------------------------------###
import ast
import logging
import os
import traceback
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, text
from common.pathing import ROOT

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
SCRIPT_DIR = ROOT / "Criterion"
INFO_DIR = ROOT / "INFO"
CRITERION_INFO_DIR = SCRIPT_DIR / "CriterionInfo"
PAIRS_FILE_PATH = CRITERION_INFO_DIR / "NuclearPairs.csv"
PLANT_GROUP_MAPPING_PATH = CRITERION_INFO_DIR / "Plant_Group_Mapping.csv"
HIST_OUTPUT_PATH = INFO_DIR / "CriterionNuclearHist.csv"
FORECAST_OUTPUT_PATH = INFO_DIR / "CriterionNuclearForecast.csv"

DB_HOST = "dda.criterionrsch.com"
DB_PORT = "443"
DB_NAME = "production"

# --- SQL Queries (Restored from your original script) ---
HISTORICAL_DATA_QUERY = text("""
    SELECT
        dat.date,
        dat.reactor_name,
        dat.operational_percent::float / 100 * map.net_summer_capacity AS summer_generation,
        dat.operational_percent::float / 100 * map.nameplate_capacity AS nameplate_generation,
        map.state AS "State",
        reg.eia_ng_regions AS "EIA Region"
    FROM power.nrc_raw_data AS dat
    LEFT JOIN power.nrc_mappings AS map ON map.nrc_reactor_name = dat.reactor_name
    LEFT JOIN pipelines.regions AS reg ON reg.state_abb = map.state
    WHERE dat.date >= :start_date
""")

FORECAST_DATA_QUERY = text(
    "SELECT date, ticker, value FROM data_series.fin_json_to_excel_tickers(:tickers) WHERE date >= :start_date"
)

LOCATION_MAPPING_QUERY = text("""
    SELECT DISTINCT
        map.nrc_reactor_name AS "Item",
        map.state AS "State",
        reg.eia_ng_regions AS "EIA Region"
    FROM power.nrc_mappings AS map
    LEFT JOIN pipelines.regions AS reg ON reg.state_abb = map.state
    WHERE map.nrc_reactor_name IS NOT NULL AND map.state IS NOT NULL
""")


# ==============================================================================
# SETUP & HELPER FUNCTIONS
# ==============================================================================

def setup_logging():
    """Configures a basic logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def get_db_engine() -> Optional[Engine]:
    """Loads database credentials and creates a SQLAlchemy engine."""
    logging.info("Attempting to connect to the database...")
    try:
        dotenv_path = SCRIPT_DIR / ".env"
        load_dotenv(dotenv_path=dotenv_path)
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")

        if not all([db_user, db_password]):
            logging.critical("Database credentials not found in .env file.")
            return None

        database_url = f"postgresql://{db_user}:{db_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(database_url, connect_args={"sslmode": "require"})
        
        with engine.connect():
            logging.info("Database connection successful.")
        return engine

    except Exception as e:
        logging.critical(f"Database connection failed: {e}")
        return None

def apply_formatting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies requested renaming and data type conversions.
    """
    # Rename 'EIA Region' to 'Region'
    if "EIA Region" in df.columns:
        df = df.rename(columns={"EIA Region": "Region"})

    # Set correct data types
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["Item"] = df["Item"].astype("string")
    df["Value"] = pd.to_numeric(df["Value"], errors='coerce').astype("float")
    if "Region" in df.columns:
        df["Region"] = df["Region"].astype("string")
        
    return df

# ==============================================================================
# DATA LOADING & MAPPING
# ==============================================================================

def load_mappings(engine: Engine) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Loads all necessary unit, plant, and ticker mappings."""
    logging.info("Loading mapping files and location data...")
    
    plant_map_df = pd.read_csv(PLANT_GROUP_MAPPING_PATH)
    unit_to_plant_map = {
        unit: row["Group"]
        for _, row in plant_map_df.iterrows()
        for unit in ast.literal_eval(row["Units"])
    }

    pairs_df = pd.read_csv(PAIRS_FILE_PATH)
    pairs_df.dropna(subset=['Forecast Ticker', 'Item'], inplace=True)
    forecast_ticker_map = pairs_df.set_index("Forecast Ticker")["Item"].to_dict()

    location_df = pd.read_sql(LOCATION_MAPPING_QUERY, engine)
    location_df['Plant'] = location_df['Item'].map(unit_to_plant_map)
    location_df.dropna(subset=['Plant'], inplace=True)
    
    plant_info = location_df[['Plant', 'State', 'EIA Region']].drop_duplicates().set_index('Plant')
    plant_to_state_map = plant_info['State'].to_dict()
    plant_to_region_map = plant_info['EIA Region'].to_dict()

    logging.info(f"Loaded {len(unit_to_plant_map)} unit-to-plant mappings.")
    return unit_to_plant_map, forecast_ticker_map, plant_to_state_map, plant_to_region_map

# ==============================================================================
# CORE PROCESSING LOGIC
# ==============================================================================

def process_historical_data(engine: Engine, unit_to_plant_map: Dict[str, str]) -> Optional[pd.DataFrame]:
    """Fetches and processes historical nuclear generation data."""
    logging.info("--- Processing Historical Data ---")
    try:
        df = pd.read_sql(
            HISTORICAL_DATA_QUERY,
            engine,
            params={"start_date": datetime(2015, 1, 1)},
            parse_dates=["date"],
        )
        logging.info(f"Fetched {len(df)} historical rows.")
    except Exception as e:
        logging.error(f"Historical database query failed: {e}")
        return None

    df.rename(columns={"date": "Date", "reactor_name": "Item"}, inplace=True)
    is_summer = df["Date"].dt.month.isin([4, 5, 6, 7, 8, 9, 10])
    df["Value"] = df["summer_generation"].where(is_summer, df["nameplate_generation"])
    df.dropna(subset=["Item", "Value", "State", "EIA Region"], inplace=True)

    df["Plant"] = df["Item"].map(unit_to_plant_map)
    df.dropna(subset=['Plant'], inplace=True)
    
    agg_df = df.groupby(["Date", "Plant", "State", "EIA Region"])["Value"].sum().reset_index()
    agg_df.rename(columns={"Plant": "Item"}, inplace=True)
    
    return agg_df[["Date", "Item", "Value", "State", "EIA Region"]]

def process_forecast_data(engine: Engine, ticker_map: Dict[str, str], unit_to_plant_map: Dict[str, str], plant_to_state_map: Dict[str, str], plant_to_region_map: Dict[str, str], start_date: date) -> Optional[pd.DataFrame]:
    """Fetches and processes forecast nuclear generation data."""
    logging.info(f"--- Processing Forecast Data from {start_date:%Y-%m-%d} ---")
    try:
        df = pd.read_sql(
            FORECAST_DATA_QUERY,
            engine,
            params={"tickers": ",".join(ticker_map.keys()), "start_date": start_date},
            parse_dates=["date"],
        )
        logging.info(f"Fetched {len(df)} forecast rows.")
    except Exception as e:
        logging.error(f"Forecast database query failed: {e}")
        return None

    df.rename(columns={"date": "Date", "value": "Value"}, inplace=True)
    df["Item"] = df["ticker"].map(ticker_map)
    df.dropna(subset=["Item", "Value"], inplace=True)

    df["Plant"] = df["Item"].map(unit_to_plant_map)
    df.dropna(subset=['Plant'], inplace=True)
    agg_df = df.groupby(["Date", "Plant"])["Value"].sum().reset_index()
    agg_df.rename(columns={"Plant": "Item"}, inplace=True)

    agg_df["State"] = agg_df["Item"].map(plant_to_state_map)
    agg_df["EIA Region"] = agg_df["Item"].map(plant_to_region_map)
    
    return agg_df[["Date", "Item", "Value", "State", "EIA Region"]]

# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

def main():
    """Orchestrates the entire nuclear data update process."""
    setup_logging()
    logging.info("========= Starting Nuclear Data Update Process =========")

    engine = get_db_engine()
    if not engine:
        return

    try:
        INFO_DIR.mkdir(exist_ok=True)
        unit_to_plant, forecast_map, plant_to_state, plant_to_region = load_mappings(engine)
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)

        # --- Process and Save Historical Data ---
        hist_df = process_historical_data(engine, unit_to_plant)
        if hist_df is not None and not hist_df.empty:
            hist_df_final = hist_df[hist_df["Date"].dt.date <= today].copy()
            if not hist_df_final.empty:
                hist_df_final = apply_formatting(hist_df_final)
                hist_df_final.sort_values(by=["Item", "Date"], inplace=True)
                hist_df_final.to_csv(HIST_OUTPUT_PATH, index=False, date_format="%Y-%m-%d")
                logging.info(f"SUCCESS: Historical data saved to {HIST_OUTPUT_PATH.name}")

        # --- Process and Save Forecast Data ---
        forecast_df = process_forecast_data(engine, forecast_map, unit_to_plant, plant_to_state, plant_to_region, start_date=tomorrow)
        if forecast_df is not None and not forecast_df.empty:
            forecast_df_final = forecast_df[forecast_df["Date"].dt.date > today].copy()
            if not forecast_df_final.empty:
                forecast_df_final = apply_formatting(forecast_df_final)
                forecast_df_final.sort_values(by=["Item", "Date"], inplace=True)
                forecast_df_final.to_csv(FORECAST_OUTPUT_PATH, index=False, date_format="%Y-%m-%d")
                logging.info(f"SUCCESS: Forecast data saved to {FORECAST_OUTPUT_PATH.name}")

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
