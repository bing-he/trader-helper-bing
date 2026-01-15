# minimal import safety for direct execution (repo root is one level up from /EIAGuesser)
import sys
from pathlib import Path
import os

# Add repo root (one level up) to sys.path to allow importing 'common'
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from common.pathing import ROOT

CONFIG = {
    "script_dir": ROOT / "EIAGuesser",
    "info_dir": ROOT / "INFO",
    "output_dir": ROOT / "EIAGuesser" / "output",
}

script_dir = CONFIG["script_dir"]
info_dir = CONFIG["info_dir"]
output_dir = CONFIG["output_dir"]
output_dir.mkdir(parents=True, exist_ok=True)

# Define the full paths to the input files and the future output file.
criterion_storage_path = info_dir / "CriterionStorageChange.csv"
locs_list_path = info_dir / "locs_list.csv"
fundy_path = info_dir / "Fundy.csv"
platts_path = info_dir / "PlattsPowerFundy.csv"
weather_path = info_dir / "WEATHER.csv"
output_file_path = os.path.join(
    output_dir, "Combined_Wide_Data.csv"
)  # Updated output filename

# List to keep track of items that couldn't be mapped to a region.
unknown_region_items = []

# --- 2. Load and Process Criterion Data ---
try:
    storage_df = pd.read_csv(criterion_storage_path)
    locs_df = pd.read_csv(locs_list_path)
except FileNotFoundError as e:
    print(
        f"Error loading Criterion data: {e}. Please ensure the input files are in the INFO directory."
    )
    exit()

storage_df.columns = storage_df.columns.str.strip()
locs_df.columns = locs_df.columns.str.strip()
storage_df.dropna(subset=["storage_name"], inplace=True)
storage_df["Date"] = pd.to_datetime(storage_df["Date"])
storage_df = storage_df[storage_df["Date"] >= "2018-01-01"]

eia_region_map = {
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

new_column_headers = {}
unique_storage_names = storage_df["storage_name"].unique()
for name in unique_storage_names:
    if name == "Dawn Storage Hub":
        new_header = f"East_CritStor_ON_{name}"
        new_column_headers[name] = new_header
        continue

    matching_locs = locs_df[locs_df["storage_name"] == name]
    if not matching_locs.empty:
        state_mode = matching_locs["state_name"].mode()
        if not state_mode.empty:
            state_name = state_mode[0]
            eia_region = eia_region_map.get(state_name, "UnknownRegion")
            if eia_region == "UnknownRegion":
                unknown_region_items.append(
                    f"Criterion Storage (Region Not Mapped for State): {name} -> {state_name}"
                )
            new_header = f"{eia_region}_CritStor_{state_name}_{name}"
            new_column_headers[name] = new_header
        else:
            new_column_headers[name] = f"UnknownRegion_CritStor_UnknownState_{name}"
            unknown_region_items.append(f"Criterion Storage (Unknown State): {name}")
    else:
        new_column_headers[name] = f"UnknownRegion_CritStor_NoLocInfo_{name}"
        unknown_region_items.append(f"Criterion Storage (No Loc Info): {name}")

criterion_wide_df = storage_df.pivot(
    index="Date", columns="storage_name", values="daily_storage_change"
)
criterion_wide_df.rename(columns=new_column_headers, inplace=True)


# --- 3. Load and Process Fundy Data ---
try:
    fundy_df = pd.read_csv(fundy_path)
except FileNotFoundError as e:
    print(
        f"Error loading Fundy.csv: {e}. Please ensure the file is in the INFO directory."
    )
    exit()

fundy_df.columns = fundy_df.columns.str.strip()
fundy_df.dropna(subset=["Item"], inplace=True)
fundy_df["Date"] = pd.to_datetime(fundy_df["Date"])
fundy_df = fundy_df[fundy_df["Date"] >= "2018-01-01"]

fundy_region_map = {
    "SouthCentral": "South Central",
    "Northeast": "East",
    "SouthEast": "East",
    "Midwest": "Midwest",
    "Rockies": "Mountain",
    "West": "Pacific",
    "GOM": "CONUS",
    "CONUS": "CONUS",
}


def get_fundy_region(item_name):
    for prefix, region in fundy_region_map.items():
        if item_name.startswith(prefix):
            return region
    return "UnknownRegion"


fundy_new_headers = {}
for item in fundy_df["Item"].unique():
    region = get_fundy_region(item)
    if region == "UnknownRegion":
        unknown_region_items.append(f"Fundy: {item}")
    fundy_new_headers[item] = f"{region}_Fundy_{item}"

fundy_wide_df = fundy_df.pivot(index="Date", columns="Item", values="Value")
fundy_wide_df.rename(columns=fundy_new_headers, inplace=True)


# --- 4. Load and Process Platts Power Fundy Data ---
try:
    platts_df = pd.read_csv(platts_path)
except FileNotFoundError as e:
    print(
        f"Error loading PlattsPowerFundy.csv: {e}. Please ensure the file is in the INFO directory."
    )
    exit()

platts_df.columns = platts_df.columns.str.strip()
platts_df.dropna(subset=["Item"], inplace=True)
platts_df["Date"] = pd.to_datetime(platts_df["Date"])
platts_df = platts_df[platts_df["Date"] >= "2018-01-01"]

platts_wide_df = platts_df.pivot(index="Date", columns="Item", values="Value")

platts_region_map = {
    "AESO": ["Mountain"],
    "BPA": ["Pacific"],
    "CAISO": ["Pacific"],
    "ERCOT": ["South Central"],
    "IESO": ["Midwest", "East"],
    "ISONE": ["East"],
    "MISO": ["Midwest", "South Central"],
    "NYISO": ["East"],
    "PJM": ["East", "Midwest"],
    "SPP": ["South Central", "Mountain"],
}

platts_columns_list = []
for item_name in platts_wide_df.columns:
    iso = item_name.split(" - ")[0]
    regions = platts_region_map.get(iso.upper())
    if regions:
        for region in regions:
            new_header = f"{region}_PwrFndy_{item_name}"
            new_column = platts_wide_df[item_name].rename(new_header)
            platts_columns_list.append(new_column)
    else:
        unknown_region_items.append(f"Platts Power: {item_name}")

final_platts_df = pd.concat(platts_columns_list, axis=1)


# --- 5. Load and Process Weather Data ---
try:
    weather_df = pd.read_csv(weather_path)
except FileNotFoundError as e:
    print(
        f"Error loading WEATHER.csv: {e}. Please ensure the file is in the INFO directory."
    )
    exit()

weather_df.columns = weather_df.columns.str.strip()
weather_df.rename(columns={"City Symbol": "City_Symbol"}, inplace=True)
weather_df.dropna(subset=["City_Symbol", "Date"], inplace=True)
weather_df["Date"] = pd.to_datetime(weather_df["Date"])
weather_df = weather_df[weather_df["Date"] >= "2018-01-01"]

# CORRECTED: Explicitly drop all known non-numeric descriptive columns.
# This prevents text like 'Atlanta' and 'City Title' from getting into the final data.
cols_to_drop = ["City", "State", "Country", "City Title"]
weather_df.drop(
    columns=[col for col in cols_to_drop if col in weather_df.columns], inplace=True
)

# Melt the dataframe to convert it from semi-wide to long format.
id_vars = ["Date", "City_Symbol"]
value_vars = [col for col in weather_df.columns if col not in id_vars]
weather_long_df = weather_df.melt(
    id_vars=id_vars, value_vars=value_vars, var_name="Measurement", value_name="Value"
)

weather_long_df["full_item"] = (
    weather_long_df["City_Symbol"]
    + "_"
    + weather_long_df["Measurement"].str.replace(" ", "_")
)
weather_wide_df = weather_long_df.pivot(
    index="Date", columns="full_item", values="Value"
)

weather_region_map = {
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

weather_new_headers = {}
for item_name in weather_wide_df.columns:
    city_symbol = item_name.split("_")[0]
    region = weather_region_map.get(city_symbol, "UnknownRegion")
    if region == "UnknownRegion":
        unknown_region_items.append(f"Weather: {item_name}")
    weather_new_headers[item_name] = f"{region}_Weather_{item_name}"

weather_wide_df.rename(columns=weather_new_headers, inplace=True)


# --- 6. Merge All DataFrames (CriterionExtra removed) ---
combined_df = criterion_wide_df.join(fundy_wide_df, how="outer")
combined_df = combined_df.join(final_platts_df, how="outer")
combined_df = combined_df.join(weather_wide_df, how="outer")
final_df = combined_df


# --- 7. Print Unknown Region Items ---
if unknown_region_items:
    print("\n" + "-" * 20 + " Items Assigned to Unknown Region " + "-" * 20)
    for item in sorted(list(set(unknown_region_items))):  # Sort and remove duplicates
        print(item)
    print("-" * (40 + len(" Items Assigned to Unknown Region ")))


# --- 8. Save the Output ---
final_df.index.name = "date"
final_df.to_csv(output_file_path)

print(f"\nSuccessfully created combined table at: {output_file_path}")
print("\nFirst 5 rows of the new combined table:")
print(final_df.head())
