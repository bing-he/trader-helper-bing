#!/usr/bin/env python3
"""
Data Integration Pipeline for EIA-Related Sources

Combines multiple regional data inputs (Criterion Storage, Fundy,
Platts Power Fundy, Weather, Criterion Extra) into one wide-format table.

Core principles follow Python Enhancement Proposals (PEPs):
- PEP 8: Style conventions
- PEP 20: The Zen of Python
- PEP 257: Docstring conventions
- PEP 484/585: Typing and annotations
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Dict, List

import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data_wrangler.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration class for paths and mappings."""

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.info_dir = os.path.join(os.path.dirname(self.base_dir), "INFO")
        self.output_dir = os.path.join(self.base_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

        self.paths = {
            "criterion_storage": os.path.join(
                self.info_dir, "CriterionStorageChange.csv"
            ),
            "locs_list": os.path.join(self.info_dir, "locs_list.csv"),
            "fundy": os.path.join(self.info_dir, "Fundy.csv"),
            "platts": os.path.join(self.info_dir, "PlattsPowerFundy.csv"),
            "weather": os.path.join(self.info_dir, "WEATHER.csv"),
            "criterion_extra": os.path.join(self.info_dir, "CriterionExtra.csv"),
            "output": os.path.join(self.output_dir, "Combined_Wide_Data.csv"),
            "eia_changes_src": os.path.join(self.info_dir, "EIAchanges.csv"),
            "eia_changes_dst": os.path.join(self.output_dir, "EIAchanges.csv"),
            "prices": os.path.join(self.info_dir, "PRICES.csv"),
        }

        self.region_map = {
            "Alabama": "South Central",
            "Arkansas": "South Central",
            "Kansas": "South Central",
            "Louisiana": "South Central",
            "Mississippi": "South Central",
            "Oklahoma": "South Central",
            "Texas": "South Central",
            "Illinois": "Midwest",
            "Indiana": "Midwest",
            "Iowa": "Midwest",
            "Kentucky": "Midwest",
            "Michigan": "Midwest",
            "Minnesota": "Midwest",
            "Missouri": "Midwest",
            "Ohio": "Midwest",
            "Tennessee": "Midwest",
            "Wisconsin": "Midwest",
            "Connecticut": "East",
            "Delaware": "East",
            "Florida": "East",
            "Georgia": "East",
            "Maine": "East",
            "Maryland": "East",
            "Massachusetts": "East",
            "New Hampshire": "East",
            "New Jersey": "East",
            "New York": "East",
            "North Carolina": "East",
            "Pennsylvania": "East",
            "Rhode Island": "East",
            "South Carolina": "East",
            "Vermont": "East",
            "Virginia": "East",
            "West Virginia": "East",
            "Arizona": "Mountain",
            "Colorado": "Mountain",
            "Idaho": "Mountain",
            "Montana": "Mountain",
            "Nevada": "Mountain",
            "New Mexico": "Mountain",
            "North Dakota": "Mountain",
            "South Dakota": "Mountain",
            "Utah": "Mountain",
            "Wyoming": "Mountain",
            "California": "Pacific",
            "Oregon": "Pacific",
            "Washington": "Pacific",
        }

        self.fundy_region_map = {
            "SouthCentral": "South Central",
            "Northeast": "East",
            "SouthEast": "East",
            "Midwest": "Midwest",
            "Rockies": "Mountain",
            "West": "Pacific",
            "GOM": "CONUS",
            "CONUS": "CONUS",
        }

        self.platts_region_map = {
            "AESO": ["Mountain"],  # Alberta ----> Rockies adjacency
            "BPA": ["Pacific"],  # Bonneville ----> Pacific Northwest
            "CAISO": ["Pacific"],  # California ISO ----> Pacific
            "ERCOT": ["South Central"],  # Texas ----> South Central
            "IESO": ["Midwest"],  # Ontario power flows tie mainly to Midwest gas
            "ISONE": ["East"],  # New England ----> East
            "MISO": ["Midwest"],  # Core footprint = Midwest gas (drop SC overlap)
            "NYISO": ["East"],  # New York ----> East
            "PJM": ["East"],  # Dominant gas linkage = East
            "SPP": ["South Central"],  # Strongest tie = SC (Oklahoma/Kansas core)
        }

        self.weather_region_map = {
            # East
            "KATL": "East",  # Atlanta
            "KBOS": "East",  # Boston
            "KBUF": "East",  # Buffalo
            "KPIT": "East",  # Pittsburgh
            "KPHL": "East",  # Philadelphia
            "KDCA": "East",  # Washington National
            "KIAD": "East",  # Washington Dulles
            "KRIC": "East",  # Richmond
            "KRDU": "East",  # Raleigh–Durham
            "KJFK": "East",  # New York JFK
            "KTPA": "East",  # Tampa
            # Midwest
            "KORD": "Midwest",  # Chicago O'Hare
            "KDTW": "Midwest",  # Detroit
            "KMSP": "Midwest",  # Minneapolis
            "KSTL": "Midwest",  # St. Louis
            "KIND": "Midwest",  # Indianapolis
            # Mountain
            "KDEN": "Mountain",  # Denver
            "KPHX": "Mountain",  # Phoenix
            "KABQ": "Mountain",  # Albuquerque
            "KSLC": "Mountain",  # Salt Lake City
            "KBIS": "Mountain",  # Bismarck
            # Pacific
            "KLAX": "Pacific",  # Los Angeles
            "KSFO": "Pacific",  # San Francisco
            "KSEA": "Pacific",  # Seattle
            "KSMF": "Pacific",  # Sacramento (optional)
            "KPDX": "Pacific",  # Portland (optional)
            # South Central
            "KIAH": "South Central",  # Houston IAH
            "KDFW": "South Central",  # Dallas–Fort Worth
            "KSAT": "South Central",  # San Antonio
            "KMAF": "South Central",  # Midland
            "KMSY": "South Central",  # New Orleans
            "KLIT": "South Central",  # Little Rock
            "KOKC": "South Central",  # Oklahoma City
            "KBHM": "South Central",  # Birmingham
            "KJAN": "South Central",  # Jackson
        }

        self.crit_extra_map = {
            "CONUS - STORAGE": "CONUS",
            "Total Demand - California": "Pacific",
            "Total Demand - Lower 48": "CONUS",
            "Total Demand - Midwest": "Midwest",
            "Total Demand - Northeast": "East",
        }

        self.crit_extra_items = [
            "CONUS - STORAGE",
            "Total Demand - California",
            "Total Demand - Lower 48",
            "Total Demand - Midwest",
            "Total Demand - Northeast",
        ]

        # Load prices region map from JSON file
        prices_map_path = os.path.join(self.base_dir, "prices_region_map.json")
        with open(prices_map_path, "r") as f:
            self.prices_region_map = json.load(f)


config = Config()


# === 2. Helpers ===
def safe_read_csv(path: str, required_cols: List[str] | None = None) -> pd.DataFrame:
    """Safely load a CSV file, enforcing presence of required columns if given."""
    logger.info(f"Reading CSV file: {path}")
    if not os.path.exists(path):
        logger.error(f"Missing input file: {path}")
        raise FileNotFoundError(f"Missing input file: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    logger.debug(f"Loaded dataframe with shape: {df.shape}")
    if required_cols:
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Column {col} not found in {path}")
                raise ValueError(f"Column {col} not found in {path}")
    return df


# === 3. Criterion Storage ===
def process_criterion_storage(config: Config) -> pd.DataFrame:
    """Process Criterion Storage data."""
    logger.info("Processing Criterion Storage data")
    storage_df = safe_read_csv(
        config.paths["criterion_storage"], ["storage_name", "Date"]
    )
    locs_df = safe_read_csv(config.paths["locs_list"], ["storage_name", "state_name"])

    storage_df = storage_df.dropna(subset=["storage_name"])
    storage_df["Date"] = pd.to_datetime(storage_df["Date"])
    storage_df = storage_df[storage_df["Date"] >= "2018-01-01"]
    logger.debug(
        f"Criterion Storage dataframe shape after filtering: {storage_df.shape}"
    )

    new_headers: Dict[str, str] = {}
    for name in storage_df["storage_name"].unique():
        if name == "Dawn Storage Hub":
            new_headers[name] = f"East_CritStor_ON_{name}"
            continue
        loc_match = locs_df.loc[locs_df["storage_name"] == name, "state_name"]
        if not loc_match.empty:
            state = loc_match.mode()[0]
            region = config.region_map.get(state, "UnknownRegion")
            if region == "UnknownRegion":
                logger.warning(
                    f"Criterion Storage: Unknown region for {name} -> {state}"
                )
            new_headers[name] = f"{region}_CritStor_{state}_{name}"
        else:
            new_headers[name] = f"UnknownRegion_CritStor_{name}"
            logger.warning(f"Criterion Storage: No location info for {name}")

    criterion_wide = storage_df.pivot(
        index="Date", columns="storage_name", values="daily_storage_change"
    ).rename(columns=new_headers)
    logger.info(
        f"Criterion Storage processed, resulting dataframe shape: {criterion_wide.shape}"
    )
    return criterion_wide


# === 4. Fundy ===
def process_fundy(config: Config) -> pd.DataFrame:
    """Process Fundy data."""
    logger.info("Processing Fundy data")
    fundy_df = safe_read_csv(config.paths["fundy"], ["Item", "Date", "Value"])
    fundy_df["Date"] = pd.to_datetime(fundy_df["Date"])
    fundy_df = fundy_df[fundy_df["Date"] >= "2018-01-01"]
    logger.debug(f"Fundy dataframe shape after filtering: {fundy_df.shape}")

    def get_region(item: str) -> str:
        for prefix, region in config.fundy_region_map.items():
            if item.startswith(prefix):
                return region
        return "UnknownRegion"

    fundy_headers = {}
    for item in fundy_df["Item"].unique():
        region = get_region(item)
        if region == "UnknownRegion":
            logger.warning(f"Fundy: Unknown region for {item}")
        fundy_headers[item] = f"{region}_Fundy_{item}"

    fundy_wide = fundy_df.pivot(index="Date", columns="Item", values="Value").rename(
        columns=fundy_headers
    )
    logger.info(f"Fundy processed, resulting dataframe shape: {fundy_wide.shape}")
    return fundy_wide


# === 5. Platts Power Fundy ===
def process_platts(config: Config) -> pd.DataFrame:
    """Process Platts Power Fundy data."""
    logger.info("Processing Platts Power Fundy data")
    platts_df = safe_read_csv(config.paths["platts"], ["Item", "Date", "Value"])
    platts_df["Date"] = pd.to_datetime(platts_df["Date"])
    platts_df = platts_df[platts_df["Date"] >= "2018-01-01"]
    logger.debug(f"Platts dataframe shape after filtering: {platts_df.shape}")

    platts_wide = platts_df.pivot(index="Date", columns="Item", values="Value")

    platts_cols = []
    for col in platts_wide.columns:
        iso = col.split(" - ")[0].upper()
        regions = config.platts_region_map.get(iso)
        if regions:
            for region in regions:
                platts_cols.append(platts_wide[col].rename(f"{region}_PwrFndy_{col}"))
        else:
            logger.warning(f"Platts Power: Unknown ISO for {col}")

    platts_final = pd.concat(platts_cols, axis=1)
    logger.info(
        f"Platts Power Fundy processed, resulting dataframe shape: {platts_final.shape}"
    )
    return platts_final


# === 6. Weather ===
def process_weather(config: Config) -> pd.DataFrame:
    """Process Weather data."""
    logger.info("Processing Weather data")
    weather_df = safe_read_csv(config.paths["weather"], ["City Symbol", "Date"])
    weather_df["Date"] = pd.to_datetime(weather_df["Date"])
    weather_df = weather_df[weather_df["Date"] >= "2018-01-01"]
    logger.debug(f"Weather dataframe shape after filtering: {weather_df.shape}")

    drop_cols = ["City", "State", "Country", "City Title"]
    weather_df = weather_df.drop(
        columns=[c for c in drop_cols if c in weather_df.columns]
    )

    id_vars = ["Date", "City Symbol"]
    val_vars = [c for c in weather_df.columns if c not in id_vars]
    weather_long = weather_df.melt(
        id_vars=id_vars, value_vars=val_vars, var_name="Measurement", value_name="Value"
    )
    weather_long["full_item"] = (
        weather_long["City Symbol"]
        + "_"
        + weather_long["Measurement"].str.replace(" ", "_")
    )
    weather_wide = weather_long.pivot(index="Date", columns="full_item", values="Value")

    weather_headers = {}
    for col in weather_wide.columns:
        symbol = col.split("_")[0]
        region = config.weather_region_map.get(symbol, "UnknownRegion")
        if region == "UnknownRegion":
            logger.warning(f"Weather: Unknown region for {col}")
        weather_headers[col] = f"{region}_Weather_{col}"

    weather_wide = weather_wide.rename(columns=weather_headers)
    logger.info(f"Weather processed, resulting dataframe shape: {weather_wide.shape}")
    return weather_wide


# === 7. Criterion Extra ===
def process_criterion_extra(config: Config) -> pd.DataFrame:
    """Process Criterion Extra data."""
    logger.info("Processing Criterion Extra data")
    crit_extra = safe_read_csv(
        config.paths["criterion_extra"], ["Item", "Date", "Value"]
    )
    crit_extra = crit_extra[crit_extra["Item"].isin(config.crit_extra_items)]
    crit_extra["Date"] = pd.to_datetime(crit_extra["Date"])
    crit_extra = crit_extra[crit_extra["Date"] >= "2018-01-01"]
    logger.debug(f"Criterion Extra dataframe shape after filtering: {crit_extra.shape}")

    crit_extra_wide = crit_extra.pivot(index="Date", columns="Item", values="Value")

    crit_headers = {}
    for col in crit_extra_wide.columns:
        region = config.crit_extra_map.get(col, "UnknownRegion")
        if region == "UnknownRegion":
            logger.warning(f"Criterion Extra: Unknown region for {col}")
        crit_headers[col] = f"{region}_CritExtra_{col}"

    crit_extra_wide = crit_extra_wide.rename(columns=crit_headers)
    logger.info(
        f"Criterion Extra processed, resulting dataframe shape: {crit_extra_wide.shape}"
    )
    return crit_extra_wide


# === 5b. PRICES ===
def process_prices(config: Config) -> pd.DataFrame:
    """Process daily hub prices into wide format with regional tags."""
    logger.info("Processing PRICES data")
    prices_df = safe_read_csv(config.paths["prices"])
    if "Date" not in prices_df.columns:
        raise ValueError("PRICES.csv missing 'Date' column")

    prices_df["Date"] = pd.to_datetime(prices_df["Date"])
    prices_df = prices_df[prices_df["Date"] >= "2018-01-01"]

    rename_map = {}
    for col in prices_df.columns:
        if col == "Date":
            continue
        hub = col.strip()
        region = config.prices_region_map.get(hub)
        if region:
            rename_map[hub] = f"{region}_Price_{hub.replace(' ', '').replace('-', '')}"
        else:
            logger.warning(f"PRICES: No region mapping for hub {hub}")
            rename_map[hub] = (
                f"UnknownRegion_Price_{hub.replace(' ', '').replace('-', '')}"
            )

    prices_df = prices_df.rename(columns=rename_map)
    prices_df = prices_df.set_index("Date")
    logger.info(f"PRICES processed, shape: {prices_df.shape}")
    return prices_df


# === 8. Merge ===
def merge_dataframes(
    criterion_wide: pd.DataFrame,
    fundy_wide: pd.DataFrame,
    platts_final: pd.DataFrame,
    weather_wide: pd.DataFrame,
    crit_extra_wide: pd.DataFrame,
    prices_wide: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    """Merge all processed dataframes."""
    logger.info("Merging all dataframes")
    combined = criterion_wide.join(
        [fundy_wide, platts_final, weather_wide, crit_extra_wide, prices_wide],
        how="outer",
    )
    combined.index.name = "date"
    logger.info(f"Combined dataframe shape: {combined.shape}")
    return combined


def save_combined_data(combined: pd.DataFrame, config: Config) -> None:
    """Save the combined dataframe to CSV."""
    logger.info(f"Saving combined data to {config.paths['output']}")
    combined.to_csv(config.paths["output"])
    logger.info("Combined table saved successfully")


def copy_eia_changes(config: Config) -> None:
    """Copy EIAchanges.csv if it exists."""
    src = config.paths["eia_changes_src"]
    dst = config.paths["eia_changes_dst"]
    if os.path.exists(src):
        logger.info(f"Copying EIAchanges.csv from {src} to {dst}")
        shutil.copy(src, dst)
        logger.info("EIAchanges.csv copied successfully")
    else:
        logger.warning("EIAchanges.csv not found in INFO directory")


# === 9. Main Function ===
def main() -> None:
    """Main data processing pipeline."""
    logger.info("Starting Data Integration Pipeline")
    try:
        criterion_wide = process_criterion_storage(config)
        fundy_wide = process_fundy(config)
        platts_final = process_platts(config)
        weather_wide = process_weather(config)
        crit_extra_wide = process_criterion_extra(config)
        prices_wide = process_prices(config)

        combined = merge_dataframes(
            criterion_wide,
            fundy_wide,
            platts_final,
            weather_wide,
            crit_extra_wide,
            prices_wide,
            config,
        )

        save_combined_data(combined, config)
        copy_eia_changes(config)

        logger.info("Data Integration Pipeline completed successfully")
        logger.info(f"Combined table saved at: {config.paths['output']}")
        logger.debug(f"First few rows of combined data:\n{combined.head()}")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
