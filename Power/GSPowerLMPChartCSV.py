# grid_explorer_app.py

import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv
from gridstatusio import GridStatusClient
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from common.pathing import ROOT

OUTPUT_DIR = ROOT / "Power" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Configuration for supported ISOs ---
ISO_CONFIGS = {
    "CAISO": {
        "dataset_id": "caiso_lmp_day_ahead_hourly",
        "location_types_for_listing": ["DLAP", "Trading Hub"],
        "price_column": "lmp",
        "market": "day_ahead",
        "interval": "hourly",
        "refine_hubs_suffix": None
    },
    "ERCOT": {
        "dataset_id": "ercot_spp_day_ahead_hourly",
        "location_types_for_listing": ["Trading Hub", "Load Zone"],
        "price_column": "spp",
        "market": "day_ahead",
        "interval": "hourly",
        "refine_hubs_suffix": None
    },
    "ISONE": {
        "dataset_id": "isone_lmp_day_ahead_hourly",
        "location_types_for_listing": ["HUB", "EXT. NODE", "LOAD ZONE"],
        "price_column": "lmp",
        "market": "day_ahead",
        "interval": "hourly",
        "refine_hubs_suffix": None
    },
    "MISO": {
        "dataset_id": "miso_lmp_day_ahead_hourly",
        "location_types_for_listing": ["Hub"],
        "price_column": "lmp",
        "market": "day_ahead",
        "interval": "hourly",
        "refine_hubs_suffix": ".HUB"
    },
    "NYISO": {
        "dataset_id": "nyiso_lmp_day_ahead_hourly",
        "location_types_for_listing": ["Zone"],
        "price_column": "lmp",
        "market": "day_ahead",
        "interval": "hourly",
        "refine_hubs_suffix": None
    },
    "PJM": {
        "dataset_id": "pjm_lmp_day_ahead_hourly",
        "location_types_for_listing": ["ZONE", "HUB"],
        "price_column": "lmp",
        "market": "day_ahead",
        "interval": "hourly",
        "refine_hubs_suffix": None
    },
    "SPP": {
        "dataset_id": "spp_lmp_day_ahead_hourly",
        "location_types_for_listing": ["Interface", "Hub"],
        "price_column": "lmp",
        "market": "day_ahead",
        "interval": "hourly",
        "refine_hubs_suffix": None
    }
}

# --- Function 1: fetch_lmp_data ---
def fetch_lmp_data(
    client: "GridStatusClient",
    iso: str,
    market: str,
    interval: str,
    start_date_str: str,
    end_date_str: str = None,
    target_location_types: list[str] = None,
    target_locations: list[str] = None,
    limit: int = None,
    verbose: bool = True,
    override_dataset_id: str = None
) -> pd.DataFrame:
    if verbose:
        print(f"DEBUG fetch_lmp_data: ENTERING FUNCTION.")
        print(f"DEBUG fetch_lmp_data: Received override_dataset_id = '{override_dataset_id}'")
        print(f"DEBUG fetch_lmp_data: Received iso = '{iso}', market = '{market}', interval = '{interval}'")

    if override_dataset_id and override_dataset_id.strip():
        dataset_id = override_dataset_id.strip()
        if verbose:
            print(f"Using override_dataset_ID: '{dataset_id}'")
    else:
        dataset_id = f"{iso.lower()}_lmp_{market.lower()}_{interval.lower()}"
        if verbose:
            print(f"Constructed dataset ID (as override_dataset_id was None/empty): '{dataset_id}'")

    try:
        start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    except ValueError:
        if verbose:
            print(f"Error: Invalid start_date_str format: {start_date_str}. Please use YYYY-MM-DD.")
        return pd.DataFrame()

    if end_date_str is None:
        end_dt = start_dt + timedelta(days=1)
        actual_end_date_str = end_dt.strftime("%Y-%m-%d")
        if verbose:
            print(f"No end_date_str provided, API will fetch data for the full day of {start_date_str} (API end: {actual_end_date_str}).")
    else:
        try:
            end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
            if end_dt <= start_dt:
                if verbose:
                    print(f"Error: end_date_str ({end_date_str}) for API must be after start_date_str ({start_date_str}).")
                return pd.DataFrame()
            actual_end_date_str = end_date_str
        except ValueError:
            if verbose:
                print(f"Error: Invalid end_date_str format: {end_date_str}. Please use YYYY-MM-DD.")
            return pd.DataFrame()

    api_filter_column = None
    api_filter_value = None
    api_filter_operator = None

    if target_locations:
        api_filter_column = "location"
        api_filter_value = target_locations if isinstance(target_locations, list) else [target_locations]
        api_filter_operator = "in" if isinstance(target_locations, list) and len(target_locations) > 0 else "="
    elif target_location_types:
        api_filter_column = "location_type"
        api_filter_value = target_location_types if isinstance(target_location_types, list) else [target_location_types]
        api_filter_operator = "in" if isinstance(target_location_types, list) and len(target_location_types) > 0 else "="

    if verbose and api_filter_column:
        print(f"API Filter: column='{api_filter_column}', operator='{api_filter_operator}', value(s)='{api_filter_value}'")

    df_result = pd.DataFrame()
    try:
        if verbose:
            print(f"Fetching data for '{dataset_id}' from {start_date_str} (inclusive) to {actual_end_date_str} (exclusive)...")

        data_response = client.get_dataset(
            dataset=dataset_id,
            start=start_date_str,
            end=actual_end_date_str,
            filter_column=api_filter_column,
            filter_value=api_filter_value,
            filter_operator=api_filter_operator,
            limit=limit
        )

        if isinstance(data_response, pd.DataFrame):
            df_result = data_response
        elif isinstance(data_response, dict) and "data" in data_response:
            df_result = pd.DataFrame(data_response["data"])
        elif isinstance(data_response, list):
            df_result = pd.DataFrame(data_response)
        else:
            if verbose:
                print(f"Unexpected data format received from API: {type(data_response)}")

        if df_result.empty and verbose:
            print("No data retrieved for the given parameters.")
        elif verbose and not df_result.empty:
            print(f"Successfully fetched and processed {len(df_result)} rows.")
    except Exception as e:
        if verbose:
            print(f"Error fetching or processing data for '{dataset_id}': {e}")
    return df_result

# --- Function 2: get_iso_filtered_locations ---
def get_iso_filtered_locations(
    client: "GridStatusClient",
    iso: str,
    market: str,
    interval: str,
    target_location_types: list[str],
    lookback_days: int = 30,
    limit_per_type: int = 2000,
    verbose: bool = True,
    dataset_id_for_locations: str = None
) -> list[str]:
    if verbose:
        print(f"DEBUG get_iso_filtered_locations: Using dataset_id_for_locations = '{dataset_id_for_locations}'")
        print(f"Fetching unique {iso} locations for types {target_location_types} from the last {lookback_days} days...")

    end_lookback_dt = datetime.now(timezone.utc)
    start_lookback_dt = end_lookback_dt - timedelta(days=lookback_days)
    start_lookback_str = start_lookback_dt.strftime("%Y-%m-%d")
    api_end_lookback_str = (end_lookback_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    all_locations = set()
    for loc_type in target_location_types:
        if verbose:
            print(f"\nFetching locations for ISO: {iso}, Type: {loc_type} using dataset: {dataset_id_for_locations or 'default constructed'}...")

        df_type_locations = fetch_lmp_data(
            client=client,
            iso=iso,
            market=market,
            interval=interval,
            start_date_str=start_lookback_str,
            end_date_str=api_end_lookback_str,
            target_location_types=[loc_type],
            limit=limit_per_type,
            verbose=verbose,
            override_dataset_id=dataset_id_for_locations
        )
        if not df_type_locations.empty and "location" in df_type_locations.columns:
            unique_for_type = df_type_locations["location"].unique()
            all_locations.update(unique_for_type)
            if verbose:
                print(f"Found {len(unique_for_type)} unique locations for type '{loc_type}'.")
        elif verbose:
            print(f"No locations found or 'location' column missing for ISO '{iso}', type '{loc_type}'.")

    if not all_locations:
        if verbose:
            print(f"\nNo locations found for ISO '{iso}' matching types {target_location_types} in the lookback period using dataset {dataset_id_for_locations or 'default'}.")
        return []

    sorted_locations = sorted(list(all_locations))
    if verbose:
        print(f"\nTotal unique locations found for ISO '{iso}', types {target_location_types}: {len(sorted_locations)}")
    return sorted_locations

# --- Function 3: interactive_iso_price_explorer_module ---
def interactive_iso_price_explorer_module(client: "GridStatusClient"):
    print("--- Interactive ISO Price Explorer ---")
    module_verbose = True # Set to False to reduce debug prints from helper functions

    # 1. Get ISO from user
    print("\nAvailable ISOs:")
    iso_options = list(ISO_CONFIGS.keys())
    for i, iso_name in enumerate(iso_options):
        print(f"{i+1}. {iso_name}")

    selected_iso = None
    while True:
        try:
            iso_choice_input = input("Enter the number or name of the ISO: ")
            try:
                iso_idx = int(iso_choice_input) -1
                if 0 <= iso_idx < len(iso_options):
                    selected_iso = iso_options[iso_idx]
                    break
            except ValueError:
                iso_choice_upper = iso_choice_input.upper()
                if iso_choice_upper in iso_options: # Allow typing full name
                    selected_iso = iso_choice_upper
                    break
            print("Invalid ISO selection. Please choose from the list.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return
    print(f"You selected ISO: {selected_iso}")

    config = ISO_CONFIGS[selected_iso]
    dataset_id = config["dataset_id"]
    location_types_for_listing = config["location_types_for_listing"]
    price_column = config["price_column"]
    market = config["market"]
    interval = config["interval"]
    refine_suffix = config.get("refine_hubs_suffix")

    if module_verbose:
        print(f"\nDEBUG: Calling get_iso_filtered_locations for {selected_iso} with dataset_id_for_locations = '{dataset_id}' and types = {location_types_for_listing}")

    initial_locations = get_iso_filtered_locations(
        client=client,
        iso=selected_iso,
        market=market,
        interval=interval,
        target_location_types=location_types_for_listing,
        lookback_days=30,
        verbose=module_verbose,
        dataset_id_for_locations=dataset_id
    )

    available_locations_for_user = initial_locations
    if selected_iso == "MISO" and refine_suffix:
        refined = [loc for loc in initial_locations if isinstance(loc, str) and loc.endswith(refine_suffix)]
        if refined:
            if module_verbose:
                print(f"Refining MISO Hub list: {len(initial_locations)} total found, {len(refined)} end with '{refine_suffix}'.")
            available_locations_for_user = refined
        elif module_verbose:
            print(f"No MISO Hubs ending with '{refine_suffix}' found. Presenting all initially found.")

    if not available_locations_for_user:
        print(f"Could not retrieve a list of {selected_iso} locations for types {location_types_for_listing} using dataset {dataset_id}. Exiting.")
        return

    print(f"\nAvailable {selected_iso} Locations ({', '.join(location_types_for_listing)} from '{dataset_id}'):")
    for i, loc_name in enumerate(available_locations_for_user):
        print(f"{i+1}. {loc_name}")
    print("-" * 30)

    selected_location_name = None
    while True:
        try:
            loc_choice_input = input(f"Enter the number or full name of the {selected_iso} location: ")
            try:
                loc_idx = int(loc_choice_input) - 1
                if 0 <= loc_idx < len(available_locations_for_user):
                    selected_location_name = available_locations_for_user[loc_idx]
                    break
            except ValueError:
                if loc_choice_input in available_locations_for_user: # Allow typing full name
                    selected_location_name = loc_choice_input
                    break
            print("Invalid location. Please choose from the list.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return
    print(f"You selected location: {selected_location_name}")

    while True:
        try:
            start_date_input = input("Enter the start date (YYYY-MM-DD): ")
            datetime.strptime(start_date_input, "%Y-%m-%d")
            break
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return

    while True:
        try:
            end_date_input = input("Enter the end date (YYYY-MM-DD, inclusive): ")
            end_dt_obj = datetime.strptime(end_date_input, "%Y-%m-%d")
            start_dt_obj = datetime.strptime(start_date_input, "%Y-%m-%d")
            if end_dt_obj < start_dt_obj:
                print("End date cannot be before the start date.")
            else:
                break
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return

    api_end_date_for_fetch = (datetime.strptime(end_date_input, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    output_choice = ""
    while output_choice not in ['1', '2', '3']:
        try:
            output_choice = input("\nChoose output:\n1. Chart Only\n2. CSV Download Only\n3. Chart and CSV Download\nEnter choice (1, 2, or 3): ")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return
        if output_choice not in ['1', '2', '3']:
            print("Invalid choice. Please enter 1, 2, or 3.")

    print(f"\nFetching {market.replace('_',' ').title()} {interval.title()} {price_column.upper()} data for {selected_iso} location '{selected_location_name}' from {start_date_input} to {end_date_input}...")

    price_df = fetch_lmp_data(
        client=client,
        iso=selected_iso,
        market=market,
        interval=interval,
        start_date_str=start_date_input,
        end_date_str=api_end_date_for_fetch,
        target_locations=[selected_location_name],
        verbose=module_verbose,
        override_dataset_id=dataset_id
    )

    if price_df.empty:
        print(f"No {price_column.upper()} data found for {selected_iso} location '{selected_location_name}' in the specified date range using dataset '{dataset_id}'.")
        return

    try:
        if price_column not in price_df.columns:
            print(f"Error: Price column '{price_column}' not found. Available columns: {price_df.columns.tolist()}")
            return
        price_df[price_column] = pd.to_numeric(price_df[price_column], errors='coerce')

        time_col = None
        if 'interval_start_utc' in price_df.columns:
            time_col = 'interval_start_utc'
        elif 'time' in price_df.columns:
            time_col = 'time'

        if not time_col:
            print("Error: Suitable time column ('interval_start_utc' or 'time') not found.")
            return

        price_df[time_col] = pd.to_datetime(price_df[time_col], errors='coerce')
        price_df = price_df.dropna(subset=[price_column, time_col])

        if price_df.empty:
            print("Data became empty after cleaning (e.g., price or time values were not valid).")
            return

        if output_choice in ['1', '3']:
            price_df_for_chart = price_df.set_index(time_col)
            daily_max_price = price_df_for_chart[price_column].resample('D').max().dropna()

            if daily_max_price.empty:
                print("No daily maximum price data to plot after resampling.")
            else:
                plt.figure(figsize=(15, 7))
                daily_max_price.plot(marker='o', linestyle='-')
                plt.title(f"Daily Maximum {market.replace('_',' ').title()} {interval.title()} {price_column.upper()} for {selected_iso} Location:\n{selected_location_name}\n({start_date_input} to {end_date_input})", fontsize=14)
                plt.xlabel("Date", fontsize=12)
                plt.ylabel(f"Maximum {price_column.upper()} ($/MWh)", fontsize=12)
                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.xticks(rotation=45, ha="right")
                
                ax = plt.gca()
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                num_days_plot = (datetime.strptime(end_date_input, "%Y-%m-%d") - datetime.strptime(start_date_input, "%Y-%m-%d")).days
                if num_days_plot <= 14:
                     ax.xaxis.set_major_locator(mdates.DayLocator())
                elif num_days_plot <= 70 :
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
                else:
                    ax.xaxis.set_major_locator(mdates.MonthLocator())
                plt.tight_layout()
                plt.show() # This will display the plot in a new window

        if output_choice in ['2', '3']:
            csv_filename = f"{selected_iso}_{selected_location_name.replace('.','_').replace('/','_')}_{price_column}_{start_date_input}_to_{end_date_input}.csv"
            # Ensure filename is valid for all OS - simple replacement for now
            csv_filename = "".join(c if c.isalnum() or c in ['_', '-', '.'] else '_' for c in csv_filename)

            save_path = OUTPUT_DIR / csv_filename
            price_df.to_csv(save_path, index=False)
            print(f"\nCSV file saved locally to: {save_path}")

    except Exception as e:
        print(f"An error occurred during data processing, charting, or download: {e}")
        import traceback
        traceback.print_exc()

# --- Main execution block ---
if __name__ == "__main__":
    # Load environment variables from .env file
    # Make sure your .env file is in the root of your TraderHelper project (ROOT/.env)
    # and contains a line like: GRIDSTATUS_API_KEY='your_key_here'
    # You also need to have `python-dotenv` installed in your .venv: pip install python-dotenv
    env_path = ROOT / ".env"
    load_dotenv(env_path)
    API_KEY = os.environ.get("GRIDSTATUS_API_KEY")

    if API_KEY:
        print("API Key loaded successfully from .env file.")
        client = GridStatusClient(api_key=API_KEY)
        if client:
            interactive_iso_price_explorer_module(client)
        else:
        print("Failed to initialize GridStatusClient.")
    else:
        print("Error: GRIDSTATUS_API_KEY not found in .env file or environment variables.")
        print(f"Please ensure you have a .env file in your project root ({env_path})")
        print("and it contains a line like: GRIDSTATUS_API_KEY='your_key_here'")
