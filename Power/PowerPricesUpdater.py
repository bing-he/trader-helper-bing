import pandas as pd
import os
from dotenv import load_dotenv
from gridstatusio import GridStatusClient
from datetime import datetime, timezone, timedelta
from pathlib import Path

# --- Define Base Path ---
try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except NameError:
    BASE_DIR = Path(".").resolve()
    if not (BASE_DIR / "INFO").exists() and (BASE_DIR.parent / "INFO").exists():
        BASE_DIR = BASE_DIR.parent
    elif not (BASE_DIR / "INFO").exists() and (BASE_DIR.parent.parent / "INFO").exists():
        BASE_DIR = BASE_DIR.parent.parent

ENV_PATH = BASE_DIR / "Power" / ".env" 
INFO_DIR = BASE_DIR / "INFO"

load_dotenv(dotenv_path=ENV_PATH)
GRIDSTATUS_API_KEY = os.getenv("GRIDSTATUS_API_KEY")

# --- Filtered locations per ISO (UPDATED ISO-NE to ISONE) ---
FILTERED_LOCATIONS = {
    "CAISO": ['DLAP_PGAE-APND', 'DLAP_SCE-APND', 'DLAP_SDGE-APND', 'DLAP_VEA-APND', 'TH_LNODE10A', 'TH_LNODE12A', 'TH_LNODE13A', 'TH_LNODE14A', 'TH_LNODER1A', 'TH_LNODER2A', 'TH_LNODER4A', 'TH_LNODER5A', 'TH_LNODER9A', 'TH_NP15_GEN-APND', 'TH_NP15_GEN_OFFPEAK-APND', 'TH_SP15_GEN-APND', 'TH_SP15_GEN_OFFPEAK-APND', 'TH_ZP26_GEN-APND', 'TH_ZP26_GEN_OFFPEAK-APND'],
    "ERCOT": ["HB_PAN", "HB_SOUTH", "HB_WEST", "LZ_AEN", "LZ_CPS", "LZ_HOUSTON", "LZ_LCRA", "LZ_NORTH", "LZ_RAYBN", "LZ_SOUTH", "LZ_WEST"],
    "NYISO": ["HUD VL", "LONGIL", "MHK VL", "MILLWD", "N", "NORTH", "NPX", "O H", "PJM", "WEST"],
    "PJM": ["AECO", "AEP", "AEP GEN HUB", "AEP-DAYTON HUB", "APS", "ATSI", "ATSI GEN HUB", "BGE", "CHICAGO GEN HUB", "CHICAGO HUB", "COMED", "DAY", "DEOK", "DOM", "DOMINION HUB", "DPL", "DUQ", "EASTERN HUB", "EKPC", "JCPL", "METED", "N ILLINOIS HUB", "NEW JERSEY HUB", "OHIO HUB", "OVEC", "PECO", "PENELEC", "PEPCO", "PJM-RTO", "PPL", "PSEG", "RECO", "WEST INT HUB", "WESTERN HUB"],
    "SPP": ["SPPNORTH_HUB", "SPPSOUTH_HUB"],
    "MISO": ["ARKANSAS.HUB", "ILLINOIS.HUB", "INDIANA.HUB", "LOUISIANA.HUB", "MICHIGAN.HUB", "MINN.HUB", "MS.HUB", "TEXAS.HUB"],
    "ISONE": ['.H.INTERNAL_HUB', '.I.HQHIGATE120 2', '.I.HQ_P1_P2345 5', '.I.NRTHPORT138 5', '.I.ROSETON 345 1', '.I.SALBRYNB345 1', '.I.SHOREHAM138 99', '.Z.CONNECTICUT', '.Z.MAINE', '.Z.NEMASSBOST', '.Z.NEWHAMPSHIRE', '.Z.RHODEISLAND', '.Z.SEMASS', '.Z.VERMONT', '.Z.WCMASS'] # Changed key from ISO-NE
}

# --- Dataset map (UPDATED ISO-NE to ISONE) ---
ISO_DATASET_MAP = {
    "CAISO": "caiso_lmp_day_ahead_hourly",
    "ERCOT": "ercot_spp_day_ahead_hourly", 
    "PJM": "pjm_lmp_day_ahead_hourly",
    "MISO": "miso_lmp_day_ahead_hourly",
    "SPP": "spp_lmp_day_ahead_hourly",
    "NYISO": "nyiso_lmp_day_ahead_hourly",
    "ISONE": "isone_lmp_day_ahead_hourly" # Changed key from ISO-NE
}

def get_update_date_range(days_back=5): 
    today_utc = datetime.now(timezone.utc).date()
    return today_utc - timedelta(days=days_back), today_utc

def fetch_lmp_for_iso(client, iso, dataset_id, start_date, end_date, locations):
    # iso here will be "ISONE" when called for New England
    print(f"Fetching {iso} data from {start_date} to {end_date} for {len(locations)} locations...")
    try:
        df = client.get_dataset(
            dataset=dataset_id,
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(), 
            filter_column="location",
            filter_value=locations,
            filter_operator="in"
        )
        if df.empty:
            print(f"No data returned from gridstatus.io for {iso} in the specified range/locations.")
            return pd.DataFrame()

        df["ISO"] = iso # This correctly assigns the key, e.g., "ISONE"
        df["Date"] = pd.to_datetime(df["interval_start_utc"]).dt.date
        df["Location"] = df["location"].astype(str).str.upper().str.strip()

        if "lmp" in df.columns:
            df["LMP"] = pd.to_numeric(df["lmp"], errors="coerce")
        elif "spp" in df.columns: 
            df["LMP"] = pd.to_numeric(df["spp"], errors="coerce")
        else:
            print(f"No 'lmp' or 'spp' column found in {iso} response. Available columns: {df.columns.tolist()}")
            return pd.DataFrame()
        
        # --- DEBUGGING FOR LOCATION MATCHING ---
        if iso.upper() in ["CAISO", "ISONE"]: # Focus on these ISOs for debugging
            print(f"\n--- DEBUGGING {iso} LOCATION MATCHING (inside fetch_lmp_for_iso) ---")
            print(f"  Raw locations from gridstatus.io for {iso} (after initial fetch & before explicit filter_value in get_dataset took effect for this specific print):")
            raw_gs_locations = df['Location'].unique() # df here is already filtered by 'locations' in get_dataset
            print(f"    Unique locations in fetched data: {len(raw_gs_locations)}. Sample: {raw_gs_locations[:10] if len(raw_gs_locations) > 0 else 'None'}")
            
            print(f"  Target locations used in get_dataset filter_value for {iso}:")
            print(f"    Total target: {len(locations)}. Sample: {locations[:10] if len(locations) > 0 else 'None'}")
            
            # Since get_dataset already filters, df should only contain matched locations if successful
            if not df.empty:
                 print(f"  Data successfully fetched for the targeted locations for {iso}.")
            print(f"--- END DEBUGGING {iso} LOCATION MATCHING ---\n")


        daily_max = df.groupby(["ISO", "Date", "Location"])["LMP"].max().reset_index()
        daily_max.rename(columns={"LMP": "Max LMP"}, inplace=True) 
        return daily_max

    except Exception as e:
        print(f"Error fetching {iso} data from gridstatus.io: {e}")
        return pd.DataFrame()

def update_power_prices_csv():
    start_date, end_date = get_update_date_range() 
    
    if not GRIDSTATUS_API_KEY:
        print("❌ GRIDSTATUS_API_KEY not found in .env. Please set it to proceed.")
        return
    client = GridStatusClient(api_key=GRIDSTATUS_API_KEY)

    all_new_data = []
    # Iterate using the standardized keys (e.g., "ISONE")
    for iso_key_standardized, dataset_id in ISO_DATASET_MAP.items():
        locations = FILTERED_LOCATIONS.get(iso_key_standardized, [])
        if not locations:
            print(f"No filtered locations defined for {iso_key_standardized}, skipping.")
            continue
        
        df_new = fetch_lmp_for_iso(client, iso_key_standardized, dataset_id, start_date, end_date, locations)
        if not df_new.empty:
            all_new_data.append(df_new)

    if not all_new_data:
        print("No new data collected from any ISO for the update period.")
        return

    newly_fetched_combined_df = pd.concat(all_new_data, ignore_index=True)
    
    output_path = INFO_DIR / "PowerPrices.csv"
    final_columns = ["ISO", "Location", "Date", "Max LMP"] 
    final_combined_df = pd.DataFrame(columns=final_columns) 

    if output_path.exists():
        try:
            print(f"\n--- Loading existing data from {output_path} ---")
            existing_df = pd.read_csv(
                output_path,
                header=0, 
                skiprows=lambda x: x == 1, 
                dtype=str,  
                encoding="utf-8"
            )
            print(f"  Initial columns read: {existing_df.columns.tolist()}")
            
            expected_header_if_misaligned = ['ISO_orig', 'HeaderDate_orig', 'HeaderLocation_orig', 'HeaderMaxLMP_orig'] # Use temp names
            
            if len(existing_df.columns) == 4:
                existing_df.columns = expected_header_if_misaligned
                print(f"  Renamed columns for processing: {existing_df.columns.tolist()}")

                processed_existing_df = pd.DataFrame()
                processed_existing_df["ISO"] = existing_df["ISO_orig"].str.strip().str.upper()
                # Standardize ISO name from existing file: "ISO-NE" -> "ISONE"
                processed_existing_df["ISO"] = processed_existing_df["ISO"].replace({"ISO-NE": "ISONE"})
                
                processed_existing_df["Location"] = existing_df["HeaderDate_orig"].astype(str).str.strip().str.upper() 
                processed_existing_df["Date"] = pd.to_datetime(existing_df["HeaderLocation_orig"], format="%m/%d/%Y", errors="coerce") 
                processed_existing_df["Max LMP"] = pd.to_numeric(existing_df["HeaderMaxLMP_orig"], errors="coerce")
                
                print(f"  Head of existing_df (after assigning to standard names & ISO normalization):\n{processed_existing_df.head()}")
                
                initial_rows = len(processed_existing_df)
                processed_existing_df.dropna(subset=["Date", "Location", "ISO", "Max LMP"], how='any', inplace=True)
                if len(processed_existing_df) < initial_rows:
                     print(f"  Dropped {initial_rows - len(processed_existing_df)} rows from existing data due to NaT/NaN in critical fields.")
                
                if not processed_existing_df.empty:
                    processed_existing_df["Date"] = processed_existing_df["Date"].dt.date
                else:
                    print("  Existing DataFrame became empty after processing and NaT/NaN drop.")
                existing_df = processed_existing_df 
            else:
                print(f"  Warning: Existing CSV {output_path} does not have 4 columns as expected. Skipping merge with old data.")
                existing_df = pd.DataFrame(columns=final_columns)

            print("\n--- Debugging Merging Logic ---")
            # (Debugging prints remain the same)
            print(f"Newly fetched dates (sample): {newly_fetched_combined_df['Date'].unique()[:5] if not newly_fetched_combined_df.empty else 'N/A'}")
            print(f"Newly fetched locations (sample): {newly_fetched_combined_df['Location'].unique()[:5] if not newly_fetched_combined_df.empty else 'N/A'}")
            print(f"Newly fetched ISOs (sample): {newly_fetched_combined_df['ISO'].unique()[:5] if not newly_fetched_combined_df.empty else 'N/A'}")
            print(f"Existing dates (sample): {existing_df['Date'].unique()[:5] if not existing_df.empty else 'N/A'}")
            print(f"Existing locations (sample): {existing_df['Location'].unique()[:5] if not existing_df.empty else 'N/A'}")
            print(f"Existing ISOs (sample): {existing_df['ISO'].unique()[:5] if not existing_df.empty else 'N/A'}") # Check this for ISONE
            print(f"Newly fetched Date dtype: {newly_fetched_combined_df['Date'].dtype if not newly_fetched_combined_df.empty else 'N/A'}")
            print(f"Existing Date dtype: {existing_df['Date'].dtype if not existing_df.empty else 'N/A'}")
            print("-------------------------------\n")

            if not existing_df.empty:
                dates_to_replace = newly_fetched_combined_df["Date"].unique()
                locations_to_replace = newly_fetched_combined_df["Location"].unique()
                isos_to_replace = newly_fetched_combined_df["ISO"].unique()

                existing_df_filtered = existing_df[
                    ~((existing_df['ISO'].isin(isos_to_replace)) &
                      (existing_df['Date'].isin(dates_to_replace)) & 
                      (existing_df['Location'].isin(locations_to_replace)))
                ].copy()
                final_combined_df = pd.concat([existing_df_filtered, newly_fetched_combined_df], ignore_index=True)
            else: 
                final_combined_df = newly_fetched_combined_df
            print("Successfully merged new data with existing PowerPrices.csv")

        except Exception as e:
            print(f"Failed to load or merge with existing PowerPrices.csv. Creating new file from fetched data. Error: {e}")
            final_combined_df = newly_fetched_combined_df 
    else:
        print(f"PowerPrices.csv not found at {output_path}. Creating a new file from fetched data.")
        final_combined_df = newly_fetched_combined_df

    if final_combined_df.empty:
        print("No data (neither existing nor new) to save. Exiting.")
        return

    final_combined_df.drop_duplicates(subset=["ISO", "Date", "Location"], keep='last', inplace=True)
    final_combined_df.sort_values(by=["ISO", "Date", "Location"], inplace=True)
    
    if not final_combined_df.empty and 'Date' in final_combined_df.columns:
        final_combined_df['Date'] = pd.to_datetime(final_combined_df['Date'], errors='coerce').dt.date
        final_combined_df.dropna(subset=['Date'], inplace=True) 
        if not final_combined_df.empty:
             final_combined_df["Date"] = final_combined_df["Date"].apply(lambda x: x.strftime("%m/%d/%Y") if pd.notnull(x) else None)
        else:
            print("Warning: final_combined_df became empty after trying to coerce Date column before saving.")
            return
    else:
        print("Warning: 'Date' column missing or DataFrame empty before final date formatting.")
        return

    final_combined_df = final_combined_df[final_columns]

    try:
        final_combined_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"✅ PowerPrices.csv updated and saved to: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save PowerPrices.csv to {output_path}. Error: {e}")

if __name__ == "__main__":
    print("Starting GridStatusIO Power Price Update with filtered locations...")
    update_power_prices_csv()
    print("Done.")

