# advanced_grid_explorer.py

import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv
from gridstatusio import GridStatusClient
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import traceback

from common.pathing import ROOT

OUTPUT_DIR = ROOT / "Power" / "gridstatus_outputs"
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

# --- Helper Function: fetch_price_data (modified from original fetch_lmp_data) ---
def fetch_price_data(
    client: "GridStatusClient",
    iso_name: str, # Added for context
    dataset_id: str,
    market: str,   # Used for general context if needed
    interval: str, # Used for general context if needed
    start_date_str: str,
    end_date_str: str, # This should be the API end date (exclusive)
    target_locations: list[str],
    limit: int = None,
    verbose: bool = True
) -> pd.DataFrame:
    if verbose:
        print(f"  Fetching for {iso_name} - Dataset: '{dataset_id}', Locations: {target_locations}, Period: {start_date_str} to {end_date_str}")

    api_filter_column = "location"
    api_filter_value = target_locations
    api_filter_operator = "in" if len(target_locations) > 1 else "="
    
    df_result = pd.DataFrame()
    try:
        data_response = client.get_dataset(
            dataset=dataset_id,
            start=start_date_str,
            end=end_date_str,
            filter_column=api_filter_column,
            filter_value=api_filter_value,
            filter_operator=api_filter_operator,
            limit=limit
        )
        if isinstance(data_response, pd.DataFrame):
            df_result = data_response
        elif verbose:
            print(f"  Unexpected data format for {iso_name}, {target_locations}: {type(data_response)}")

        if df_result.empty and verbose:
            print(f"  No data retrieved for {iso_name}, {target_locations}.")
        elif verbose and not df_result.empty:
            print(f"  Successfully fetched {len(df_result)} rows for {iso_name}, {target_locations}.")
            
    except Exception as e:
        if verbose:
            print(f"  Error fetching data for {iso_name}, {target_locations} using dataset '{dataset_id}': {e}")
            # traceback.print_exc() # Uncomment for full error details during debugging
    return df_result

# --- Helper Function: get_iso_filtered_locations (modified slightly) ---
def get_iso_locations(
    client: "GridStatusClient",
    iso_name: str,
    dataset_id: str, # Specific dataset for fetching locations
    location_types: list[str],
    market: str,   # For context if needed by fetch_price_data
    interval: str, # For context if needed by fetch_price_data
    lookback_days: int = 7, # Shorter lookback for faster location listing
    limit_per_type: int = 5000, # Increased limit if many locations exist
    verbose: bool = True
) -> list[str]:
    if verbose:
        print(f"  Getting available locations for {iso_name} (types: {location_types}) using dataset '{dataset_id}'...")

    end_lookback_dt = datetime.now(timezone.utc)
    start_lookback_dt = end_lookback_dt - timedelta(days=lookback_days)
    start_lookback_str = start_lookback_dt.strftime("%Y-%m-%d")
    api_end_lookback_str = (end_lookback_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    all_locations_for_iso = set()
    # Fetch all specified types at once if API supports it well, or loop if necessary
    # For simplicity here, we'll rely on the location_types filter in one go if fetch_price_data is adapted,
    # but the original notebook did it one type at a time.
    # Let's assume a single call for efficiency in location discovery.
    
    # We need a way to get locations without fetching full price data,
    # The client.list_locations(iso=iso_name) might be better if it exists and is reliable.
    # For now, using fetch_price_data with a short date range and location_type filter.
    
    temp_df_locations = pd.DataFrame()
    try:
        temp_data_response = client.get_dataset(
            dataset=dataset_id,
            start=start_lookback_str,
            end=api_end_lookback_str,
            filter_column="location_type",
            filter_value=location_types,
            filter_operator="in",
            limit=limit_per_type # Limit to avoid excessive data pull just for names
        )
        if isinstance(temp_data_response, pd.DataFrame):
            temp_df_locations = temp_data_response
    except Exception as e:
        if verbose:
            print(f"  Error fetching locations for {iso_name} type {location_types}: {e}")

    if not temp_df_locations.empty and "location" in temp_df_locations.columns:
        unique_for_types = temp_df_locations["location"].unique()
        all_locations_for_iso.update(unique_for_types)
        if verbose:
            print(f"  Found {len(unique_for_types)} unique locations for types '{location_types}'.")
    elif verbose:
        print(f"  No locations found or 'location' column missing for {iso_name}, types '{location_types}'.")

    if not all_locations_for_iso:
        if verbose:
            print(f"  No locations found for {iso_name} matching types {location_types}.")
        return []

    sorted_locations = sorted(list(all_locations_for_iso))
    if verbose:
        print(f"  Total unique locations for {iso_name}, types {location_types}: {len(sorted_locations)}")
    return sorted_locations


# --- Main Interactive Module ---
def multi_iso_location_price_explorer(client: "GridStatusClient"):
    print("--- Multi-ISO & Location Price Explorer ---")
    verbose_fetch = True # Set to False to reduce debug prints during actual data fetch

    # 1. Select Multiple ISOs
    print("\nAvailable ISOs:")
    iso_options = list(ISO_CONFIGS.keys())
    for i, iso_name_option in enumerate(iso_options):
        print(f"{i+1}. {iso_name_option}")

    selected_iso_names = []
    while True:
        try:
            iso_choices_str = input("Enter ISO numbers (comma-separated, e.g., 1,3) or 'all' or 'done': ").strip().lower()
            if not iso_choices_str or iso_choices_str == 'done':
                if not selected_iso_names:
                    print("No ISOs selected. Exiting.")
                    return
                break
            if iso_choices_str == 'all':
                selected_iso_names = iso_options
                print(f"Selected all ISOs: {', '.join(selected_iso_names)}")
                break

            chosen_indices = [int(x.strip()) - 1 for x in iso_choices_str.split(',')]
            temp_selected_isos = []
            valid_selection = True
            for idx in chosen_indices:
                if 0 <= idx < len(iso_options):
                    temp_selected_isos.append(iso_options[idx])
                else:
                    print(f"Invalid ISO number: {idx + 1}. Please choose from the list.")
                    valid_selection = False
                    break
            if valid_selection:
                selected_iso_names.extend(temp_selected_isos)
                selected_iso_names = sorted(list(set(selected_iso_names))) # Remove duplicates and sort
                print(f"Currently selected ISOs: {', '.join(selected_iso_names)}")
                cont = input("Add more ISOs? (y/n, default n): ").strip().lower()
                if cont != 'y':
                    break
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas (e.g., 1,3) or 'all' or 'done'.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return
    
    if not selected_iso_names:
        print("No ISOs were selected. Exiting.")
        return

    # 2. Select Multiple Locations for each selected ISO
    iso_location_selections = {} # Store as {'ISO_NAME': ['loc1', 'loc2'], ...}
    for iso_name in selected_iso_names:
        print(f"\n--- Configuring locations for ISO: {iso_name} ---")
        config = ISO_CONFIGS[iso_name]
        available_locations = get_iso_locations(
            client=client,
            iso_name=iso_name,
            dataset_id=config["dataset_id"],
            location_types=config["location_types_for_listing"],
            market=config["market"],
            interval=config["interval"],
            verbose=True # Verbose for location listing
        )

        if config.get("refine_hubs_suffix") and iso_name == "MISO": # Example MISO refinement
             available_locations = [loc for loc in available_locations if isinstance(loc, str) and loc.endswith(config["refine_hubs_suffix"])]
             print(f"  Refined MISO locations to those ending with '{config['refine_hubs_suffix']}'. Count: {len(available_locations)}")


        if not available_locations:
            print(f"No locations found for {iso_name}. Skipping this ISO.")
            continue

        print(f"\nAvailable locations for {iso_name}:")
        for i, loc_name_option in enumerate(available_locations):
            print(f"{i+1}. {loc_name_option}")
        
        selected_locations_for_iso = []
        while True:
            try:
                loc_choices_str = input(f"Enter location numbers for {iso_name} (comma-separated, e.g., 1,3), or 'all', or 'done': ").strip().lower()
                if not loc_choices_str or loc_choices_str == 'done':
                    break 
                if loc_choices_str == 'all':
                    selected_locations_for_iso = available_locations
                    print(f"  Selected all {len(available_locations)} locations for {iso_name}.")
                    break
                
                chosen_indices = [int(x.strip()) - 1 for x in loc_choices_str.split(',')]
                temp_selected_locs = []
                valid_loc_selection = True
                for idx in chosen_indices:
                    if 0 <= idx < len(available_locations):
                        temp_selected_locs.append(available_locations[idx])
                    else:
                        print(f"  Invalid location number: {idx + 1} for {iso_name}. Please choose from the list.")
                        valid_loc_selection = False
                        break
                if valid_loc_selection:
                    selected_locations_for_iso.extend(temp_selected_locs)
                    selected_locations_for_iso = sorted(list(set(selected_locations_for_iso)))
                    print(f"  Currently selected locations for {iso_name}: {', '.join(selected_locations_for_iso)}")
                    cont_loc = input(f"  Add more locations for {iso_name}? (y/n, default n): ").strip().lower()
                    if cont_loc != 'y':
                        break
            except ValueError:
                print(f"  Invalid input for {iso_name} locations. Please enter numbers separated by commas.")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                return
        
        if selected_locations_for_iso:
            iso_location_selections[iso_name] = selected_locations_for_iso

    if not iso_location_selections:
        print("No locations selected for any ISO. Exiting.")
        return

    # 3. Get Date Range
    while True:
        try:
            start_date_input = input("\nEnter the overall start date (YYYY-MM-DD): ")
            datetime.strptime(start_date_input, "%Y-%m-%d")
            break
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return

    while True:
        try:
            end_date_input = input("Enter the overall end date (YYYY-MM-DD, inclusive for data): ")
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
    
    # API end_date is exclusive, so add one day
    api_end_date_for_fetch = (datetime.strptime(end_date_input, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    # 4. Fetch Data for all selections
    print("\n--- Fetching Data ---")
    all_dataframes_list = []
    for iso_name, locations in iso_location_selections.items():
        if not locations:
            continue
        config = ISO_CONFIGS[iso_name]
        df_iso_loc = fetch_price_data(
            client=client,
            iso_name=iso_name,
            dataset_id=config["dataset_id"],
            market=config["market"],
            interval=config["interval"],
            start_date_str=start_date_input,
            end_date_str=api_end_date_for_fetch,
            target_locations=locations,
            verbose=verbose_fetch
        )
        if not df_iso_loc.empty:
            # Ensure 'iso' column exists for pivot later
            if 'iso' not in df_iso_loc.columns and 'balancing_authority' in df_iso_loc.columns:
                 df_iso_loc['iso'] = df_iso_loc['balancing_authority']
            elif 'iso' not in df_iso_loc.columns:
                 df_iso_loc['iso'] = iso_name # Add iso name if not present

            # Standardize price column
            price_col_original = config["price_column"]
            if price_col_original in df_iso_loc.columns and price_col_original != "price":
                df_iso_loc.rename(columns={price_col_original: "price"}, inplace=True)
            
            # Select and ensure essential columns for combination
            time_col = 'interval_start_utc' if 'interval_start_utc' in df_iso_loc.columns else 'time'
            if 'price' in df_iso_loc.columns and time_col in df_iso_loc.columns and 'location' in df_iso_loc.columns:
                 # Convert time column to datetime
                df_iso_loc[time_col] = pd.to_datetime(df_iso_loc[time_col], errors='coerce')
                df_iso_loc.dropna(subset=[time_col, 'price'], inplace=True) # Drop rows where essential data is missing
                
                # Create a unique series identifier for pivoting
                df_iso_loc['series_id'] = df_iso_loc['iso'].astype(str) + "_" + df_iso_loc['location'].astype(str)
                
                all_dataframes_list.append(df_iso_loc[[time_col, 'series_id', 'price']])
            else:
                print(f"  Skipping data for {iso_name} - {locations} due to missing essential columns (time, location, or price after standardization).")

    if not all_dataframes_list:
        print("No data successfully fetched for any selection. Exiting.")
        return

# 5. Combine, Pivot, and Save to CSV
    print("\n--- Processing and Saving Data to CSV ---")
    wide_df = pd.DataFrame() # Initialize wide_df to ensure it exists for plotting section even if try block fails early
    try:
        if not all_dataframes_list: # Check if the list of dataframes is empty
            print("WARNING_TRACE: all_dataframes_list is empty. No data to process for CSV/Plot.")
        else:
            long_combined_df = pd.concat(all_dataframes_list, ignore_index=True)
            time_col_name = 'interval_start_utc' if 'interval_start_utc' in long_combined_df.columns else 'time'

            print(f"Long combined DataFrame shape (before daily resampling): {long_combined_df.shape}")
            
            if not long_combined_df.empty:
                print(f"Unique series_id values: {long_combined_df['series_id'].nunique()}")

                long_combined_df[time_col_name] = pd.to_datetime(long_combined_df[time_col_name])
                long_combined_df.set_index(time_col_name, inplace=True)

                print("  Resampling to daily maximum prices for each location...")
                daily_max_series = long_combined_df.groupby('series_id')['price'].resample('D').max()
                daily_max_long_df = daily_max_series.reset_index()
                print(f"Long daily max DataFrame shape: {daily_max_long_df.shape}")

                if not daily_max_long_df.empty:
                    wide_df = daily_max_long_df.pivot_table( 
                        index=time_col_name,
                        columns='series_id',
                        values='price'
                    )
                    wide_df.sort_index(inplace=True)

                    # This is the critical section for CSV saving
                    if not wide_df.empty:
                        print("DEBUG_TRACE: === CSV SAVING BLOCK START ===")
                        print(f"DEBUG_TRACE: wide_df shape for CSV is: {wide_df.shape}")

                        safe_start_date = "".join(c if c.isalnum() or c == '-' else '_' for c in start_date_input)
                        safe_end_date = "".join(c if c.isalnum() or c == '-' else '_' for c in end_date_input)
                        csv_filename = f"combined_daily_max_prices_{safe_start_date}_to_{safe_end_date}.csv"
                        
                        save_path = OUTPUT_DIR / csv_filename
                        
                        print(f"DEBUG_TRACE: Full path for CSV is set to: {save_path}")
                        try:
                            print("DEBUG_TRACE: Calling wide_df.to_csv()...")
                            wide_df.to_csv(save_path)
                            print("DEBUG_TRACE: wide_df.to_csv() command FINISHED.") 
                            
                            if os.path.exists(save_path):
                                print(f"SUCCESS_TRACE: CSV file CREATED at: {save_path}")
                            else:
                                print(f"ERROR_TRACE: CSV file NOT CREATED at {save_path} (os.path.exists is false).")
                        except Exception as e_csv_specific:
                            print(f"ERROR_TRACE_CSV_SPECIFIC: Exception directly during CSV save: {e_csv_specific}")
                            traceback.print_exc()
                        print("DEBUG_TRACE: === CSV SAVING BLOCK END ===")
                    else:
                        print("WARNING_TRACE: Pivoted DataFrame (wide_df) was empty. CSV saving skipped.")
                else:
                    print("WARNING_TRACE: DataFrame (daily_max_long_df) is empty after resampling. CSV/Plotting skipped.")
            else:
                print("WARNING_TRACE: Combined DataFrame (long_combined_df) is empty. CSV/Plotting skipped.")
    
    except Exception as e_outer:
        print(f"ERROR_TRACE_OUTER: An error occurred in the main CSV/Processing try block: {e_outer}")
        traceback.print_exc()

    # 6. Plotting
    # This section should be at the same indentation level as the `print("\n--- Processing and Saving Data to CSV ---")`
    # and its corresponding `try...except` block above.
    print("\n--- Generating Plot ---")
    if not wide_df.empty: 
        num_series = len(wide_df.columns)
        if num_series == 0:
            print("No data columns in the wide DataFrame to plot.")
        else:
            print(f"Plotting {num_series} series (Daily Maximums)...")
            try: # Added try-except around plotting as well for robustness
                plt.figure(figsize=(18, 8)) 
                wide_df.plot(ax=plt.gca()) 
                
                plt.title(f"Daily Maximum Market Prices from {start_date_input} to {end_date_input}", fontsize=16)
                plt.xlabel("Date", fontsize=12)
                plt.ylabel("Daily Maximum Price ($/MWh)", fontsize=12)
                
                if num_series > 10:
                    plt.legend(title="ISO_Location", fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))
                else:
                    plt.legend(title="ISO_Location", fontsize='medium')

                plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                plt.xticks(rotation=45, ha="right")
                
                ax = plt.gca()
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                num_days_plot = (datetime.strptime(end_date_input, "%Y-%m-%d") - datetime.strptime(start_date_input, "%Y-%m-%d")).days
                if num_days_plot <= 0: num_days_plot = 1 

                if num_days_plot <= 14:
                        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, num_days_plot // 7)))
                elif num_days_plot <= 90 : 
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=max(1, num_days_plot // 30)))
                else: 
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, num_days_plot // 90)))

                plt.tight_layout(rect=[0, 0, 0.85 if num_series > 10 else 1, 1]) 
                plt.show()
            except Exception as e_plot:
                print(f"ERROR_TRACE_PLOT: An error occurred during plotting: {e_plot}")
                traceback.print_exc()
    else:
         print("Plotting skipped as there is no data in wide_df (e.g. wide_df is empty or was not created).")

# --- Main execution block ---
if __name__ == "__main__":
    env_path = ROOT / ".env"
    load_dotenv(env_path) 
    API_KEY = os.environ.get("GRIDSTATUS_API_KEY")

    if API_KEY:
        print("API Key loaded successfully.")
        try:
            client = GridStatusClient(api_key=API_KEY)
            if client:
                multi_iso_location_price_explorer(client) # This is the end of the function call
            else:
                print("Failed to initialize GridStatusClient (client object is None).")
        except Exception as e:
            print(f"Error initializing GridStatusClient: {e}")
            traceback.print_exc()
    else:
        print("Error: GRIDSTATUS_API_KEY not found in .env file or environment variables.")
        print(f"Please ensure you have a .env file in your project root (e.g., {env_path})")
        print("and it contains a line like: GRIDSTATUS_API_KEY='your_key_here'")
    
    print("\n--- Script Finished ---")
