import requests
import json
import os
import datetime 
from dotenv import load_dotenv 
from requests.auth import HTTPBasicAuth 
import pandas as pd 

# --- Configuration ---
PLVIEW_API_BASE_URL = "https://api.connect.ihsmarkit.com/cs/v1/plview"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFO_FOLDER = os.path.join(SCRIPT_DIR, '..', '..', 'INFO') 
CSV_FILE_PATH = os.path.join(INFO_FOLDER, "PlattsStorageChange.csv") # Corrected CSV filename

# --- PointLogic Views API Functions ---

def list_available_views(pl_username, pl_password):
    """Lists all available views from the PointLogic Views API."""
    endpoint = f"{PLVIEW_API_BASE_URL}/views"
    headers = {"Accept": "application/json"} 
    print(f"\nListing available PointLogic Views from: {endpoint}")
    try:
        response = requests.get(endpoint, auth=HTTPBasicAuth(pl_username, pl_password), headers=headers)
        response.raise_for_status()
        print("Successfully listed available views.")
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error listing views: {http_err}")
        print(f"Response status: {response.status_code}, content: {response.text}")
    except Exception as e:
        print(f"An error occurred listing views: {e}")
    return None

def get_view_metadata(pl_username, pl_password, view_name):
    """Retrieves metadata for a specific PointLogic view."""
    endpoint = f"{PLVIEW_API_BASE_URL}/views/{view_name}"
    headers = {"Accept": "application/json"}
    print(f"\nGetting metadata for view '{view_name}' from: {endpoint}")
    try:
        response = requests.get(endpoint, auth=HTTPBasicAuth(pl_username, pl_password), headers=headers)
        response.raise_for_status()
        print(f"Successfully retrieved metadata for view '{view_name}'.")
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error getting metadata for view '{view_name}': {http_err}")
        print(f"Response status: {response.status_code}, content: {response.text}")
    except Exception as e:
        print(f"An error occurred getting metadata for view '{view_name}': {e}")
    return None

def retrieve_view_data(pl_username, pl_password, view_name, params=None):
    """Retrieves data from a specific PointLogic view."""
    endpoint = f"{PLVIEW_API_BASE_URL}/retrieve/{view_name}"
    headers = {"Accept": "application/json"}
    print(f"\nRetrieving data for view '{view_name}' from: {endpoint}")
    if params:
        print(f"With parameters: {params}")
    try:
        response = requests.get(endpoint, auth=HTTPBasicAuth(pl_username, pl_password), headers=headers, params=params)
        response.raise_for_status()
        print(f"Successfully retrieved data for view '{view_name}'.")
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error retrieving data for view '{view_name}': {http_err}")
        print(f"Response status: {response.status_code}, content: {response.text}")
    except Exception as e:
        print(f"An error occurred retrieving data for view '{view_name}': {e}")
    return None

# --- Main Script ---
if __name__ == "__main__":
    print(f"Script directory: {SCRIPT_DIR}")
    ENV_FILE_PATH_FROM_SCRIPT = os.path.join(SCRIPT_DIR, '..', '.env') 
    print(f"Attempting to load .env file from: {ENV_FILE_PATH_FROM_SCRIPT}")

    if not load_dotenv(dotenv_path=ENV_FILE_PATH_FROM_SCRIPT):
        print(f"Warning: Could not find .env file at {ENV_FILE_PATH_FROM_SCRIPT}")
        exit("Exiting: Credentials .env file not found.")
    else:
        print(f"Successfully loaded .env file from {ENV_FILE_PATH_FROM_SCRIPT}")

    POINTLOGIC_PAT_USERNAME = os.getenv("PointLogic_USERNAME")
    POINTLOGIC_PAT_PASSWORD = os.getenv("PointLogic_PASSWORD")

    if not POINTLOGIC_PAT_USERNAME or not POINTLOGIC_PAT_PASSWORD:
        print("PointLogic_USERNAME or PointLogic_PASSWORD not found in .env file.")
        exit("Exiting: PointLogic PAT Credentials not loaded.")
    
    print(f"Using PointLogic PAT Username: {POINTLOGIC_PAT_USERNAME[:8]}...")
    print(f"Target INFO folder for CSV: {INFO_FOLDER}")
    print(f"Target CSV file path: {CSV_FILE_PATH}")

    if not os.path.exists(INFO_FOLDER):
        try:
            os.makedirs(INFO_FOLDER)
            print(f"Created directory: {INFO_FOLDER}")
        except OSError as e:
            print(f"Error creating directory {INFO_FOLDER}: {e}")
            exit(f"Exiting: Could not create INFO directory.")

    print("\n--- Step 1: Identify Target View and Columns (Pre-defined/Confirmed) ---")
    # Current target view is a sample view with limited data (approx 122 days).
    # To get more comprehensive data, you may need to:
    # 1. Run list_available_views() (commented out below) to find other view names.
    # 2. Update target_view_name with a more suitable view.
    # 3. Run get_view_metadata() for the new view to confirm column names.
    target_view_name = "us_samplestorage_facility" 
    facility_name_column = "name" 
    date_column = "flowdate"       
    volume_column = "volume"       
    
    # --- Optional: Step 1a - List all available views (run once to identify a better target_view_name if needed) ---
    # print("\n--- Optional Step 1a: List Available PointLogic Views ---")
    # available_views = list_available_views(POINTLOGIC_PAT_USERNAME, POINTLOGIC_PAT_PASSWORD)
    # if available_views:
    #     print("Available views:", json.dumps(available_views, indent=2))
    # else:
    #     print("Could not retrieve available views.")
    # exit() # Exit after listing views so you can inspect and update target_view_name

    # --- Optional: Step 1b - Get metadata for a chosen target_view_name (run if changing target_view_name) ---
    # chosen_view_for_metadata = "YOUR_NEW_TARGET_VIEW_NAME" # Replace with a view name from the list above
    # print(f"\n--- Optional Step 1b: Get Metadata for View: '{chosen_view_for_metadata}' ---")
    # view_metadata = get_view_metadata(POINTLOGIC_PAT_USERNAME, POINTLOGIC_PAT_PASSWORD, chosen_view_for_metadata)
    # if view_metadata and 'elements' in view_metadata:
    #     print(f"Metadata for '{chosen_view_for_metadata}':", json.dumps(view_metadata, indent=2))
    #     # Inspect 'elements' to confirm facility_name_column, date_column, volume_column for the new view
    # else:
    #     print(f"Could not get metadata for {chosen_view_for_metadata}")
    # exit() # Exit after getting metadata so you can update column names

    print(f"Using target view: '{target_view_name}'")
    print(f"Facility Name column in API response: '{facility_name_column}'")
    print(f"Date column in API response: '{date_column}'")
    print(f"Volume column in API response: '{volume_column}'")

    print(f"\n--- Step 2: Retrieve New Data from API for View: '{target_view_name}' ---")
    
    # For 'us_samplestorage_facility', date filtering caused issues. Retrieving all data from this sample view.
    # If using a different, more comprehensive view, you can re-enable date filtering:
    # end_date = datetime.date.today() 
    # start_date = end_date - datetime.timedelta(days=365*2) # Example: 2 years of data
    # start_date_str = start_date.strftime('%Y-%m-%d')
    # end_date_str = end_date.strftime('%Y-%m-%d')
    # api_filter_params = {
    #     "filter": f"{date_column}>=\"{start_date_str}\" AND {date_column}<=\"{end_date_str}\""
    # }
    # print(f"Attempting to retrieve data for {target_view_name} from {start_date_str} to {end_date_str}.")
    # all_view_data_from_api = retrieve_view_data(
    #     POINTLOGIC_PAT_USERNAME, 
    #     POINTLOGIC_PAT_PASSWORD, 
    #     target_view_name,
    #     params=api_filter_params 
    # )
    
    print(f"Attempting to retrieve all available data from the sample view: '{target_view_name}'.")
    all_view_data_from_api = retrieve_view_data(
        POINTLOGIC_PAT_USERNAME, 
        POINTLOGIC_PAT_PASSWORD, 
        target_view_name
        # No params for date filter for this specific sample view due to previous errors
    )

    newly_fetched_df = pd.DataFrame() 

    if all_view_data_from_api:
        data_for_df = []
        print(f"\nProcessing all facility data from API response using column '{facility_name_column}'...")
        for record in all_view_data_from_api:
            if isinstance(record, dict) and facility_name_column in record:
                current_facility_name_in_record = record.get(facility_name_column)
                
                if current_facility_name_in_record: 
                    storage_date = record.get(date_column)
                    storage_volume = record.get(volume_column)
                    
                    data_for_df.append({
                        "Facility Name": current_facility_name_in_record, 
                        "Date": storage_date,
                        "Volume": storage_volume
                    })
        
        if data_for_df:
            newly_fetched_df = pd.DataFrame(data_for_df)
            try:
                newly_fetched_df["Date"] = pd.to_datetime(newly_fetched_df["Date"], errors='coerce')
                newly_fetched_df.dropna(subset=['Date'], inplace=True) 
                newly_fetched_df["Volume"] = pd.to_numeric(newly_fetched_df["Volume"], errors='coerce')
                newly_fetched_df.sort_values(by=["Facility Name", "Date"], inplace=True)
                print(f"Successfully processed {len(newly_fetched_df)} records from API into DataFrame.")
                print("\n--- Sample of Newly Fetched DataFrame (first 5 rows) ---")
                print(newly_fetched_df.head())
            except Exception as e:
                print(f"Error processing newly fetched data into DataFrame: {e}")
                newly_fetched_df = pd.DataFrame() 
        else:
            print("No data processed from the API response (perhaps no records with facility names or filter yielded no results).")
    else:
        print(f"Could not retrieve data for view '{target_view_name}'. No new data to process.")

    print(f"\n--- Step 3: Handle CSV File: '{CSV_FILE_PATH}' ---")
    
    final_df_to_write = pd.DataFrame()

    if os.path.exists(CSV_FILE_PATH):
        print(f"Found existing CSV file: {CSV_FILE_PATH}")
        try:
            existing_df = pd.read_csv(CSV_FILE_PATH)
            print(f"Read {len(existing_df)} records from existing CSV.")
            existing_df["Date"] = pd.to_datetime(existing_df["Date"], errors='coerce')
            existing_df.dropna(subset=['Date'], inplace=True) 
            
            if not newly_fetched_df.empty:
                if not existing_df.empty:
                    max_date_in_existing = existing_df["Date"].max()
                    if pd.notna(max_date_in_existing):
                        cutoff_date = max_date_in_existing - datetime.timedelta(days=60)
                        print(f"Max date in existing CSV: {max_date_in_existing.strftime('%Y-%m-%d')}")
                        print(f"60-day cutoff date for existing data: {cutoff_date.strftime('%Y-%m-%d')}")
                        
                        older_existing_data = existing_df[existing_df["Date"] < cutoff_date]
                        print(f"Retaining {len(older_existing_data)} records from existing CSV (older than cutoff).")
                        
                        combined_df = pd.concat([older_existing_data, newly_fetched_df], ignore_index=True)
                        print(f"Combined DataFrame size before duplicate removal: {len(combined_df)}")
                    else: 
                        print("No valid max date in existing CSV, using all new data.")
                        combined_df = newly_fetched_df
                else: # existing_df is empty
                    combined_df = newly_fetched_df
                
                combined_df.sort_values(by=["Facility Name", "Date"], inplace=True)
                final_df_to_write = combined_df.drop_duplicates(subset=["Facility Name", "Date"], keep="last")
                print(f"Final DataFrame size after duplicate removal: {len(final_df_to_write)}")

            elif not existing_df.empty: 
                print("No new data fetched, using existing data.")
                final_df_to_write = existing_df
            else: 
                 print("No new data and existing CSV is empty or problematic. No data to write.")
                 
        except pd.errors.EmptyDataError:
            print(f"Existing CSV file {CSV_FILE_PATH} is empty. Will use new data if available.")
            final_df_to_write = newly_fetched_df
        except Exception as e:
            print(f"Error reading or processing existing CSV {CSV_FILE_PATH}: {e}")
            print("Proceeding to use only newly fetched data if available.")
            final_df_to_write = newly_fetched_df 
    
    elif not newly_fetched_df.empty: 
        print(f"CSV file {CSV_FILE_PATH} does not exist. Creating new file with fetched data.")
        final_df_to_write = newly_fetched_df
    else: 
        print("No new data fetched and no existing CSV file. Nothing to write.")

    if not final_df_to_write.empty:
        try:
            if 'Date' in final_df_to_write.columns and pd.api.types.is_datetime64_any_dtype(final_df_to_write['Date']):
                 final_df_to_write.loc[:, 'Date'] = final_df_to_write['Date'].dt.strftime('%Y-%m-%d')

            df_output = final_df_to_write[["Facility Name", "Date", "Volume"]]
            df_output.to_csv(CSV_FILE_PATH, index=False)
            print(f"Successfully wrote/updated data to {CSV_FILE_PATH}")
        except Exception as e:
            print(f"Error writing final DataFrame to CSV: {e}")
    else:
        print(f"No data to write to {CSV_FILE_PATH}.")

    print("\n--- Script finished ---")
