# minimal import safety for direct execution
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import pandas as pd
import os
from datetime import datetime
import glob

from common.pathing import ROOT

# --- New Function to Find the Latest File ---
def find_latest_mdfd_file(directory):
    """
    Scans a directory for files matching the 'MDFD_YYYYMMDD.xlsx' pattern
    and returns the full path to the one with the most recent date in its name.
    
    Args:
        directory (str): The path to the directory to search.
        
    Returns:
        str: The full path to the latest file, or None if no valid file is found.
    """
    search_pattern = os.path.join(directory, "MDFD_*.xlsx")
    candidate_files = glob.glob(search_pattern)
    
    latest_file = None
    latest_date = None
    
    if not candidate_files:
        return None
        
    for file_path in candidate_files:
        filename = os.path.basename(file_path)
        try:
            # Extract date part from filename like 'MDFD_20250530.xlsx'
            date_str = filename.split('_')[1].split('.')[0]
            current_date = datetime.strptime(date_str, "%Y%m%d")
            
            if latest_date is None or current_date > latest_date:
                latest_date = current_date
                latest_file = file_path
        except (IndexError, ValueError):
            # This handles filenames that match the pattern but have a malformed date
            print(f"Warning: Could not parse date from filename: {filename}")
            continue
            
    return latest_file

# --- Updated Configuration Section ---
script_directory = os.path.dirname(os.path.abspath(__file__))

# Dynamically find the latest MDFD file instead of hardcoding it
excel_file_path = find_latest_mdfd_file(script_directory)

sheets_to_exclude = [
    "MWDMF",
    "Weather",
    "ISO DA Prices",
    "ISO RT Prices",
    "Bilateral Indexes"
]

output_folder_path = str(ROOT / "INFO")
output_csv_name = "PlattsPowerFundy.csv"
output_csv_path = os.path.join(output_folder_path, output_csv_name)

generation_sheet_configs = {
    "CAISO Generation": {
        "skip_rows": 2, 
        "item_prefix": "CAISO - ",
        "value_columns": ['Thermal Power', 'Power Imports', 'Nuclear Power', 'Hydro', 
                          'Solar PV', 'Solar Thermal', 'Wind', 'Geothermal Steam', 
                          'Biomass', 'Biogas', 'Total Generation']
    },
    "ERCOT Supply": { 
        "skip_rows": 2,
        "item_prefix": "ERCOT - ",
        "value_columns": ['Coal', 'Solar', 'Natural Gas', 'Hydro', 'Nuclear', 'Wind', 'Other'],
        "calculate_total_from": ['Coal', 'Solar', 'Natural Gas', 'Hydro', 'Nuclear', 'Wind', 'Other'] 
    },
    "IESO Generation": {
        "skip_rows": 2,
        "item_prefix": "IESO - ",
        "value_columns": ['Natural Gas', 'Nuclear Power', 'Hydro', 'Wind', 'Biofuel', 'Total Generation']
    },
    "MISO Generation": {
        "skip_rows": 2,
        "item_prefix": "MISO - ",
        "value_columns": ['Coal', 'Natural Gas', 'Nuclear', 'Hydro', 'Wind', 'Other', 'Total Generation']
    },
    "ISONE Generation": { 
        "skip_rows": 3, 
        "item_prefix": "ISONE - ",
        "value_columns": ['Natural Gas', 'Nuclear Power', 'Hydro', 'Other', 
                          'Solar Thermal', 'Wind', 'Coal', 'Residual Oil', 'Total Generation']
    },
    "NYISO Generation": {
        "skip_rows": 2,
        "item_prefix": "NYISO - ",
        "value_columns": ['Coal or Oil', 'Dual Fuel', 'Natural Gas', 'Nuclear Power', 
                          'Hydro', 'Wind', 'Other Renewables', 'Total Generation']
    },
    "PJM Generation": {
        "skip_rows": 2,
        "item_prefix": "PJM - ",
        "value_columns": ['Coal', 'Natural Gas', 'Nuclear Power', 'Hydro', 'Wind', 
                          'Residual Oil', 'Renewables', 'Solar', 'Other', 'Total Generation']
    },
    "SPP Generation": {
        "skip_rows": 2,
        "item_prefix": "SPP - ",
        "value_columns": ['Coal', 'Natural Gas', 'Wind', 'Nuclear Power', 'Hydro', 'Diesel', 'Total Generation']
    }
}

# --- Main Script Logic (Processing parts remain the same) ---
print(f"Script is running from: {script_directory}")

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# This check is now the main gatekeeper for running the script
if excel_file_path is None:
    print(f"Error: No valid 'MDFD_YYYYMMDD.xlsx' file was found in '{script_directory}'.")
else:
    print(f"Found and selected latest Excel file: {os.path.basename(excel_file_path)}")
    all_processed_data_for_final_csv = [] 

    try:
        xls = pd.ExcelFile(excel_file_path)
        all_sheet_names_in_file = xls.sheet_names
        
        temp_display_list = ["Peak load by ISO"]
        for sheet_name in generation_sheet_configs.keys():
            if sheet_name in all_sheet_names_in_file and sheet_name not in sheets_to_exclude:
                temp_display_list.append(sheet_name)
        
        print("\nSheets that will eventually be processed:")
        for s_name in temp_display_list: print(f"- {s_name}")

        # --- 1. Process "Peak load by ISO" ---
        peak_load_sheet_name = "Peak load by ISO"
        if peak_load_sheet_name not in sheets_to_exclude and peak_load_sheet_name in all_sheet_names_in_file:
            print(f"\n--- Processing sheet: '{peak_load_sheet_name}' ---")
            # ... (Existing processing logic for Peak Load)
            df_peak_original = xls.parse(peak_load_sheet_name)
            if not df_peak_original.empty:
                peak_new_headers = df_peak_original.iloc[0]
                df_peak = df_peak_original[1:].copy()
                df_peak.columns = peak_new_headers
                df_peak.reset_index(drop=True, inplace=True)
                date_column_name_peak = peak_new_headers.iloc[0]
                df_peak.rename(columns={date_column_name_peak: 'Date'}, inplace=True)
                df_peak['Date'] = pd.to_datetime(df_peak['Date'])
                df_peak = df_peak[df_peak['Date'] >= '2015-01-01'].copy()
                if not df_peak.empty:
                    peak_iso_columns = [col for col in df_peak.columns if col != 'Date']
                    df_peak_melted = df_peak.melt(id_vars=['Date'], value_vars=peak_iso_columns, var_name='ISO_Name', value_name='Value')
                    df_peak_melted['Item'] = df_peak_melted['ISO_Name'] + ' - PeakLoad'
                    df_peak_final = df_peak_melted[['Date', 'Item', 'Value']].copy()
                    df_peak_final.dropna(subset=['Value'], inplace=True)
                    if not df_peak_final.empty:
                        all_processed_data_for_final_csv.append(df_peak_final)
                        print(f"Finished processing '{peak_load_sheet_name}'. {len(df_peak_final)} rows transformed.")
            else:
                print(f"Sheet '{peak_load_sheet_name}' was empty.")
        
        # --- Process All Configured Generation Sheets ---
        for sheet_name, config in generation_sheet_configs.items():
            if sheet_name in all_sheet_names_in_file and sheet_name not in sheets_to_exclude:
                print(f"\n--- Processing sheet: '{sheet_name}' ---")
                try:
                    # ... (Existing processing logic for generation sheets)
                    df_raw = xls.parse(sheet_name, header=config["skip_rows"])
                    if df_raw.empty: print(f"Sheet '{sheet_name}' is empty after loading with header={config['skip_rows']}."); continue
                    df_raw.columns = [col if not (isinstance(col, str) and ('Unnamed:' in col or col.strip() == '')) else f'DROP_COL_{i}' for i, col in enumerate(df_raw.columns)]
                    df_raw = df_raw[[col for col in df_raw.columns if not col.startswith('DROP_COL_')]]
                    if 'Date' not in df_raw.columns and not df_raw.columns.empty: df_raw.rename(columns={df_raw.columns[0]: 'Date'}, inplace=True)
                    if 'Date' not in df_raw.columns: print(f"'Date' column not found in '{sheet_name}'. Skipping."); continue
                    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
                    df_filtered = df_raw[df_raw['Date'] >= '2015-01-01'].copy()
                    if df_filtered.empty: print(f"No data for '{sheet_name}' after date filtering."); continue
                    current_value_columns = [col for col in config["value_columns"] if col in df_filtered.columns]
                    if sheet_name == "ERCOT Supply" and "calculate_total_from" in config:
                        cols_to_sum = [col for col in config["calculate_total_from"] if col in df_filtered.columns]
                        for col_to_sum in cols_to_sum: df_filtered[col_to_sum] = pd.to_numeric(df_filtered[col_to_sum], errors='coerce')
                        df_filtered['Total Generation'] = df_filtered[cols_to_sum].sum(axis=1)
                        if 'Total Generation' not in current_value_columns: current_value_columns.append('Total Generation')
                    if not current_value_columns: print(f"No value columns for '{sheet_name}'. Skipping melt."); continue
                    df_melted = df_filtered.melt(id_vars=['Date'], value_vars=current_value_columns, var_name='Source_Type', value_name='Value')
                    df_melted['Item'] = config["item_prefix"] + df_melted['Source_Type']
                    df_final = df_melted[['Date', 'Item', 'Value']].copy()
                    df_final.dropna(subset=['Value'], inplace=True)
                    if not df_final.empty:
                        all_processed_data_for_final_csv.append(df_final)
                        print(f"Finished processing '{sheet_name}'. {len(df_final)} rows transformed.")
                except Exception as e_sheet:
                    print(f"Error processing sheet '{sheet_name}': {e_sheet}")

        # --- Combine all newly processed data ---
        if not all_processed_data_for_final_csv:
            print("\nNo data was processed from any sheet. CSV will not be updated.")
        else:
            newly_processed_df = pd.concat(all_processed_data_for_final_csv, ignore_index=True)
            newly_processed_df['Date'] = pd.to_datetime(newly_processed_df['Date'])
            print(f"\nTotal new rows processed from Excel: {len(newly_processed_df)}")

            final_df_to_save = pd.DataFrame()

            if os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0:
                print(f"Existing CSV found at {output_csv_path}. Applying update logic.")
                try:
                    existing_df = pd.read_csv(output_csv_path)
                    existing_df['Date'] = pd.to_datetime(existing_df['Date'])
                    print(f"Read {len(existing_df)} rows from existing CSV.")

                    if not newly_processed_df.empty:
                        max_new_date = newly_processed_df['Date'].max()
                        cutoff_date_for_old_data = max_new_date - pd.Timedelta(days=365) 
                        print(f"Latest date in new data: {max_new_date.strftime('%Y-%m-%d')}")
                        print(f"Preserving existing data older than: {cutoff_date_for_old_data.strftime('%Y-%m-%d')}")
                        
                        preserved_old_df = existing_df[existing_df['Date'] < cutoff_date_for_old_data]
                        
                        final_df_to_save = pd.concat([preserved_old_df, newly_processed_df], ignore_index=True)
                        print(f"Combined {len(preserved_old_df)} preserved old rows with {len(newly_processed_df)} new rows.")
                    else:
                        final_df_to_save = existing_df
                        print("No new data to process, existing CSV data will be kept as is (after potential re-sort/de-dupe).")

                except pd.errors.EmptyDataError:
                    print(f"Existing CSV at {output_csv_path} is empty. It will be overwritten with new data.")
                    final_df_to_save = newly_processed_df
                except Exception as e_read_csv:
                    print(f"Error reading existing CSV {output_csv_path}: {e_read_csv}. Will overwrite with new data.")
                    final_df_to_save = newly_processed_df
            else:
                print(f"No existing CSV found or it is empty at {output_csv_path}. Creating new file with processed data.")
                final_df_to_save = newly_processed_df

            if not final_df_to_save.empty:
                final_df_to_save.drop_duplicates(subset=['Date', 'Item'], keep='last', inplace=True)
                final_df_to_save.sort_values(by=['Date', 'Item'], inplace=True)
                
                final_df_to_save.to_csv(output_csv_path, index=False)
                print(f"\nSuccessfully updated and saved data to: {output_csv_path}")
                print(f"Total rows in final CSV: {len(final_df_to_save)}")
            else:
                print("\nNo data (new or existing) to save. CSV not written.")

    except Exception as e:
        print(f"An overall error occurred: {e}")
        import traceback
        traceback.print_exc()
