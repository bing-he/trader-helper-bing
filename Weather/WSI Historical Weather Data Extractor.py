import requests
import pandas as pd
from io import StringIO
from datetime import datetime, timedelta
import questionary
import os
from pathlib import Path
from dotenv import load_dotenv
import time
import numpy as np # For NaN and calculations
from common.pathing import ROOT

# --- Configuration & Constants ---
# Load credentials from .env file
load_dotenv()
USERNAME = os.getenv("WSI_ACCOUNT_USERNAME") 
PROFILE = os.getenv("WSI_PROFILE_EMAIL")    
PASSWORD = os.getenv("WSI_PASSWORD")

# API Base URLs
BASE_SERVICE_URL = "https://www.wsitrader.com/Services/CSVDownloadService.svc"

# Output directory (repo-relative for portability)
OUTPUT_DIR_BASE = ROOT / "Weather"
OUTPUT_DIR = OUTPUT_DIR_BASE / "Outputs"

# Available attributes for user selection and their mapping to API/processing
AVAILABLE_ATTRIBUTES = {
    # Daily - Fetched or Calculated from Daily
    "Min Temp": {"type": "daily", "api_param": "Min Temp", "source": "api"},
    "Max Temp": {"type": "daily", "api_param": "Max Temp", "source": "api"},
    "Avg Temp": {"type": "daily", "api_param": "Avg Temp", "source": "api_or_calc"},
    "Daily HDD": {"type": "daily", "api_param": "HDD", "source": "calc", "base_col": "Avg Temp"},
    "Daily CDD": {"type": "daily", "api_param": "CDD", "source": "calc", "base_col": "Avg Temp"},
    
    # Hourly - Fetched from Hourly API and Aggregated
    "Avg Hourly Temp": {"type": "hourly", "api_param": "temperature", "agg": "mean", 
                        "expected_api_cols": ["temp (f)", "temperature", "temp"]},
    "Max Feels Like (Heat Index)": {"type": "hourly", "api_param": "heatindex", "agg": "max", 
                                    "calc_fallback_needs": ["Avg Hourly Temp", "Avg Relative Humidity"], 
                                    "expected_api_cols": ["heat index (f)", "heatindex"]},
    "Min Feels Like (Wind Chill)": {"type": "hourly", "api_param": "windChill", "agg": "min", 
                                    "expected_api_cols": ["wind chill (f)", "windchill"]},
    "Avg Surface Wind": {"type": "hourly", "api_param": "windSpeed", "agg": "mean", 
                         "expected_api_cols": ["wind speed (mph)", "windspeed"]},
    "Avg Dewpoint": {"type": "hourly", "api_param": "dewpoint", "agg": "mean", 
                     "expected_api_cols": ["dew point (f)", "dewpoint", "dewpt"]},
    "Max Dewpoint": {"type": "hourly", "api_param": "dewpoint", "agg": "max", # Uses same source data as Avg Dewpoint
                     "expected_api_cols": ["dew point (f)", "dewpoint", "dewpt"]}, 
    "Avg Relative Humidity": {"type": "hourly", "api_param": "relativeHumidity", "agg": "mean", 
                              "expected_api_cols": ["rh", "relativehumidity", "relative humidity (%)", "rhh"]}, # Added RHH
    "Avg Cloud Cover": {"type": "hourly", "api_param": "cloudCover", "agg": "mean", 
                        "expected_api_cols": ["cloud cover", "cloudcover", "cld cver", "cloudcvr", "cloud coveer"]}, # Added typo
    "Total Daily Precip from Hourly": {"type": "hourly", "api_param": "precipitation", "agg": "sum", 
                                       "expected_api_cols": ["precip (in)", "precipitation", "precip"]},
}

# --- Helper Functions ---
def validate_date(text):
    """Validates date string format."""
    try:
        datetime.strptime(text, "%Y-%m-%d")
        return True
    except ValueError:
        return "Please enter date in YYYY-MM-DD format."

def calculate_heat_index(T, RH):
    """
    Calculates Heat Index using NOAA's formula (Steadman's coefficients).
    T: Temperature in Fahrenheit
    RH: Relative Humidity in percent (0-100)
    Returns calculated Heat Index or T if conditions not met / HI < T.
    """
    if T is None or RH is None or pd.isna(T) or pd.isna(RH) or not isinstance(T, (int, float)) or not isinstance(RH, (int, float)):
        return np.nan # Return NaN if inputs are invalid
    if T < 80.0: 
        return T # Heat Index is typically not calculated or is same as T below 80F
    
    # Ensure RH is within a sensible range for the formula (0-100)
    RH = np.clip(RH, 0, 100)

    # NOAA's Heat Index Equation (Regression equation from Steadman)
    HI = (-42.379 +
          2.04901523 * T +
          10.14333127 * RH -
          0.22475541 * T * RH -
          6.83783e-3 * T * T -
          5.481717e-2 * RH * RH +
          1.22874e-3 * T * T * RH +
          8.5282e-4 * T * RH * RH -
          1.99e-6 * T * T * RH * RH)

    # Adjustments for specific conditions (from NOAA)
    if RH < 13 and (80.0 <= T <= 112.0):
        adjustment = ((13.0 - RH) / 4.0) * np.sqrt((17.0 - abs(T - 95.0)) / 17.0)
        HI -= adjustment
    elif RH > 85.0 and (80.0 <= T <= 87.0):
        adjustment = ((RH - 85.0) / 10.0) * ((87.0 - T) / 5.0)
        HI += adjustment
    
    # Per NWS, if HI < T, then HI is set to T.
    return HI if HI > T else T


def get_city_map():
    """Fetches city IDs and names from the API."""
    print("🔄 Fetching city list from WSI Trader...")
    url = f"{BASE_SERVICE_URL}/GetCityIds"
    params = {"Account": USERNAME, "Profile": PROFILE, "Password": PASSWORD}
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        df.columns = df.columns.str.strip()
        
        if "SiteId" not in df.columns or "Station Name" not in df.columns:
            print("❌ ERROR: 'SiteId' or 'Station Name' not found in GetCityIds response.")
            print(f"Available columns: {df.columns.tolist()}")
            return None

        df = df[["SiteId", "Station Name"]].dropna().drop_duplicates()
        
        exclude_keywords = ["ZONE", "REGION", "CONSUM", "PRODUCING", "MIDWEST", "PACIFIC",
                            "EAST", "WEST", "CENTRAL", "MOUNTAIN", "AREA", "SOUTH", "NORTH",
                            "ISO", "POOL", "SYSTEM", "HUB", "TOTAL", "AGGREGATE"]
        pattern = '|'.join(exclude_keywords)
        df_filtered = df[~df["Station Name"].str.upper().str.contains(pattern, na=False)]
        
        if df_filtered.empty and not df.empty : 
            print("⚠️ Warning: Filtering removed all cities. Using unfiltered list. Please check keywords if this is unexpected.")
            city_dict = dict(zip(df["Station Name"], df["SiteId"]))
        elif df_filtered.empty and df.empty:
             print("⚠️ Warning: No cities returned from API initially.")
             return None
        else:
            city_dict = dict(zip(df_filtered["Station Name"], df_filtered["SiteId"]))
        
        print(f"✅ Found {len(city_dict)} potential cities.")
        return city_dict
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR fetching city list: {e}")
        if hasattr(e, 'response') and e.response is not None: print(f"Response content: {e.response.text[:500]}")
        return None
    except pd.errors.EmptyDataError:
        print("❌ ERROR: No data returned from GetCityIds endpoint (empty CSV).")
        if 'response' in locals() and response is not None: print(f"Response content: {response.text[:500]}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred in get_city_map: {e}")
        if 'response' in locals() and response is not None: print(f"Response content: {response.text[:500]}")
        return None

def prompt_user_selections(city_map_with_ids):
    if not city_map_with_ids: return None
    selected_cities_names = questionary.checkbox("🏙️ Select cities:", choices=sorted(city_map_with_ids.keys()), validate=lambda x: True if len(x) > 0 else "Please select at least one city.").ask()
    if not selected_cities_names: print("No cities selected. Exiting."); return None
    selected_cities_list = [{"name": name, "id": city_map_with_ids[name]} for name in selected_cities_names]
    today_str = datetime.today().strftime("%Y-%m-%d")
    week_ago_str = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
    start_date_str = questionary.text("📅 Enter start date (YYYY-MM-DD):", default=week_ago_str, validate=validate_date).ask()
    end_date_str = questionary.text("📅 Enter end date (YYYY-MM-DD):", default=today_str, validate=validate_date).ask()
    if not start_date_str or not end_date_str: print("Start or end date not provided. Exiting."); return None
    if datetime.strptime(start_date_str, "%Y-%m-%d") > datetime.strptime(end_date_str, "%Y-%m-%d"): print("Start date cannot be after end date. Exiting."); return None
    if datetime.strptime(end_date_str, "%Y-%m-%d") > datetime.today(): end_date_str = today_str; print(f"End date was future, set to today: {today_str}")
    print("\nℹ️ Note: 'Avg Irradiance' is not directly available as a historical data point from this API's typical observations.")
    selected_attrs = questionary.checkbox("📊 Select weather attributes:", choices=list(AVAILABLE_ATTRIBUTES.keys()), validate=lambda x: True if len(x) > 0 else "Please select at least one attribute.").ask()
    if not selected_attrs: print("No attributes selected. Exiting."); return None
    return {"cities": selected_cities_list, "start_date": start_date_str, "end_date": end_date_str, "attributes": selected_attrs}

def find_actual_column_name(df_columns, expected_api_cols_list):
    """
    Finds the actual column name in the DataFrame that matches one of the expected API column names.
    Matching is case-insensitive and ignores spaces and units like (F), (mph), (%), and common variations.
    """
    df_cols_simplified = {
        col.lower().replace(" ", "").replace("(f)", "").replace("(mph)", "").replace("(%)", "").replace("utc", "").replace(".", ""): col 
        for col in df_columns
    }
    for expected in expected_api_cols_list:
        simplified_expected = expected.lower().replace(" ", "") # Keep expected simple for matching
        if simplified_expected in df_cols_simplified:
            return df_cols_simplified[simplified_expected]
    return None

def fetch_single_city_data(city_info, start_date_str, end_date_str, requested_attributes):
    city_id, city_name = city_info["id"], city_info["name"]
    print(f"\n🔄 Processing city: {city_name} ({city_id})")
    daily_df, hourly_aggregated_df = pd.DataFrame(), pd.DataFrame()

    # --- Daily Data ---
    daily_attrs_requested = [attr for attr in requested_attributes if AVAILABLE_ATTRIBUTES[attr]["type"] == "daily"]
    if any(AVAILABLE_ATTRIBUTES[attr]["source"] != "calc" for attr in daily_attrs_requested): # Fetch if any non-calculated daily attr is needed
        print(f"  Fetching daily data for {city_name}...")
        daily_params = {"Account": USERNAME, "Profile": PROFILE, "Password": PASSWORD, "CityIds[]": city_id,
                        "StartDate": datetime.strptime(start_date_str, "%Y-%m-%d").strftime("%m/%d/%Y"),
                        "EndDate": datetime.strptime(end_date_str, "%Y-%m-%d").strftime("%m/%d/%Y"),
                        "HistoricalProductId": "HISTORICAL_DAILY_OBSERVED",
                        "TempUnits": "F", "IsTemp": "true", "IsDaily": "true", "IsDisplayDates": "false"}
        try:
            response = requests.get(f"{BASE_SERVICE_URL}/GetHistoricalObservations", params=daily_params, timeout=60)
            response.raise_for_status()
            raw_csv_text = response.text
            if not raw_csv_text.strip() or "no data" in raw_csv_text.lower():
                print(f"  ⚠️ No daily API data for {city_name}.")
            else:
                temp_df = pd.read_csv(StringIO(raw_csv_text), header=2)
                if not temp_df.empty:
                    temp_df.columns = temp_df.columns.str.strip()
                    # More robust renaming based on expected patterns from API
                    rename_map = {}
                    for col in temp_df.columns:
                        col_s = str(col).strip().lower()
                        if col_s == "date": rename_map[col] = "Date"
                        elif col_s == "min (f)": rename_map[col] = "Min Temp"
                        elif col_s == "max (f)": rename_map[col] = "Max Temp"
                        elif col_s == "avg (f)": rename_map[col] = "Avg Temp"
                        # Precipitation is no longer in AVAILABLE_ATTRIBUTES for daily
                    temp_df = temp_df.rename(columns=rename_map)

                    if "Date" in temp_df.columns:
                        temp_df["Date"] = pd.to_datetime(temp_df["Date"], format='%d-%b-%Y', errors='coerce')
                        temp_df = temp_df.dropna(subset=["Date"])
                        # Calculate Avg Temp if not directly available but Min/Max are
                        if "Avg Temp" not in temp_df.columns and \
                           "Min Temp" in temp_df.columns and \
                           "Max Temp" in temp_df.columns:
                            min_T = pd.to_numeric(temp_df["Min Temp"], errors='coerce')
                            max_T = pd.to_numeric(temp_df["Max Temp"], errors='coerce')
                            temp_df["Avg Temp"] = (min_T + max_T) / 2
                        
                        # Select only columns that were successfully mapped or calculated if they were requested
                        cols_to_select_daily = ["Date"]
                        for attr_name in daily_attrs_requested:
                            if AVAILABLE_ATTRIBUTES[attr_name]["source"] != "calc" and \
                               AVAILABLE_ATTRIBUTES[attr_name]["api_param"] in temp_df.columns and \
                               AVAILABLE_ATTRIBUTES[attr_name]["api_param"] not in cols_to_select_daily:
                                cols_to_select_daily.append(AVAILABLE_ATTRIBUTES[attr_name]["api_param"])
                        
                        daily_df = temp_df[cols_to_select_daily].copy()
                        print(f"  ✅ Daily API data processed for {city_name}. Shape: {daily_df.shape}")
                    else: print(f"  ⚠️ 'Date' column not found/standardized in daily API data for {city_name}. Columns: {temp_df.columns.tolist()}")
                else: print(f"  ⚠️ Daily API DataFrame empty after read for {city_name}.")
        except Exception as e: print(f"  ❌ ERROR fetching/processing daily API data for {city_name}: {e}")

    # Calculate Daily HDD/CDD if requested and Avg Temp is available
    if "Avg Temp" in daily_df.columns and any(attr in ["Daily HDD", "Daily CDD"] for attr in requested_attributes):
        avg_temp_num = pd.to_numeric(daily_df["Avg Temp"], errors='coerce')
        if "Daily HDD" in requested_attributes: daily_df["Daily HDD"] = (65 - avg_temp_num).clip(lower=0).round(1)
        if "Daily CDD" in requested_attributes: daily_df["Daily CDD"] = (avg_temp_num - 65).clip(lower=0).round(1)
    elif any(attr in ["Daily HDD", "Daily CDD"] for attr in requested_attributes):
        print(f"  ⚠️ Cannot calculate HDD/CDD for {city_name} as 'Avg Temp' is unavailable from daily data.")


    # --- Hourly Data ---
    hourly_attrs_requested = [attr for attr in requested_attributes if AVAILABLE_ATTRIBUTES[attr]["type"] == "hourly"]
    api_datatypes_to_fetch = list(set(AVAILABLE_ATTRIBUTES[attr]["api_param"] for attr in hourly_attrs_requested))
    
    # Ensure dependencies for Heat Index calculation are included in the fetch list
    if "Max Feels Like (Heat Index)" in requested_attributes:
        if AVAILABLE_ATTRIBUTES["Avg Hourly Temp"]["api_param"] not in api_datatypes_to_fetch:
            api_datatypes_to_fetch.append(AVAILABLE_ATTRIBUTES["Avg Hourly Temp"]["api_param"])
        if AVAILABLE_ATTRIBUTES["Avg Relative Humidity"]["api_param"] not in api_datatypes_to_fetch:
            api_datatypes_to_fetch.append(AVAILABLE_ATTRIBUTES["Avg Relative Humidity"]["api_param"])
        api_datatypes_to_fetch = list(set(api_datatypes_to_fetch)) # Remove potential duplicates

    if api_datatypes_to_fetch:
        print(f"  Fetching hourly data ({', '.join(api_datatypes_to_fetch)}) for {city_name}...")
        hourly_params = {"Account": USERNAME, "Profile": PROFILE, "Password": PASSWORD, "CityIds[]": city_id,
                         "StartDate": datetime.strptime(start_date_str, "%Y-%m-%d").strftime("%m/%d/%Y"),
                         "EndDate": datetime.strptime(end_date_str, "%Y-%m-%d").strftime("%m/%d/%Y"),
                         "HistoricalProductId": "HISTORICAL_HOURLY_OBSERVED",
                         "TempUnits": "F", "DataTypes[]": api_datatypes_to_fetch, "timeutc": "true"}
        try:
            response = requests.get(f"{BASE_SERVICE_URL}/GetHistoricalObservations", params=hourly_params, timeout=120)
            response.raise_for_status()
            raw_csv_text_hourly = response.text
            if not raw_csv_text_hourly.strip() or "no data" in raw_csv_text_hourly.lower():
                print(f"  ⚠️ No hourly API data for {city_name}.")
            else:
                hourly_df_raw = pd.read_csv(StringIO(raw_csv_text_hourly), header=1) # header=1 to skip city name line
                hourly_df_raw.columns = hourly_df_raw.columns.str.strip()

                # Standardize Date and Time columns
                date_utc_col = find_actual_column_name(hourly_df_raw.columns, ["dateutc", "date"])
                hour_utc_col = find_actual_column_name(hourly_df_raw.columns, ["hourutc", "hour"])

                if date_utc_col and hour_utc_col:
                    hourly_df_raw["Timestamp"] = pd.to_datetime(
                        hourly_df_raw[date_utc_col] + " " + hourly_df_raw[hour_utc_col].astype(str), 
                        format='%m/%d/%Y %H', errors='coerce'
                    )
                    hourly_df_raw = hourly_df_raw.dropna(subset=["Timestamp"])
                    hourly_df_raw["Day"] = hourly_df_raw["Timestamp"].dt.normalize()
                    
                    # Attempt to calculate Heat Index if requested and not directly available
                    api_heatindex_direct_col = find_actual_column_name(hourly_df_raw.columns, AVAILABLE_ATTRIBUTES["Max Feels Like (Heat Index)"]["expected_api_cols"])
                    
                    if "Max Feels Like (Heat Index)" in requested_attributes and not api_heatindex_direct_col:
                        actual_temp_col_for_hi = find_actual_column_name(hourly_df_raw.columns, AVAILABLE_ATTRIBUTES["Avg Hourly Temp"]["expected_api_cols"])
                        actual_rh_col_for_hi = find_actual_column_name(hourly_df_raw.columns, AVAILABLE_ATTRIBUTES["Avg Relative Humidity"]["expected_api_cols"])
                        
                        if actual_temp_col_for_hi and actual_rh_col_for_hi:
                            print(f"  Calculating Heat Index for {city_name} as API did not provide it directly.")
                            T_hourly = pd.to_numeric(hourly_df_raw[actual_temp_col_for_hi], errors='coerce')
                            RH_hourly = pd.to_numeric(hourly_df_raw[actual_rh_col_for_hi], errors='coerce')
                            hourly_df_raw["calculated_heatindex"] = np.vectorize(calculate_heat_index)(T_hourly, RH_hourly)
                        else:
                            print(f"  ⚠️ Cannot calculate Heat Index: Required Temp ('{actual_temp_col_for_hi}') or RH ('{actual_rh_col_for_hi}') column not found in hourly response.")
                            print(f"     Available hourly columns for HI calc check: {hourly_df_raw.columns.tolist()}")


                    aggregations = {}
                    for attr_name in hourly_attrs_requested:
                        attr_details = AVAILABLE_ATTRIBUTES[attr_name]
                        agg_method = attr_details["agg"]
                        
                        # Prioritize calculated heat index if it exists and this is the attribute
                        if attr_name == "Max Feels Like (Heat Index)" and "calculated_heatindex" in hourly_df_raw.columns:
                            aggregations[attr_name] = pd.NamedAgg(column="calculated_heatindex", aggfunc=agg_method)
                            continue

                        # Otherwise, find the column based on expected_api_cols
                        expected_cols_for_attr = attr_details.get("expected_api_cols", [attr_details["api_param"]])
                        actual_col_name_in_df = find_actual_column_name(hourly_df_raw.columns, expected_cols_for_attr)
                        
                        if actual_col_name_in_df:
                            hourly_df_raw[actual_col_name_in_df] = pd.to_numeric(hourly_df_raw[actual_col_name_in_df], errors='coerce')
                            # Use attr_name as the key for NamedAgg to get the correct final column name
                            aggregations[attr_name] = pd.NamedAgg(column=actual_col_name_in_df, aggfunc=agg_method)
                        else:
                            # Only warn if it's not the Heat Index we attempted (and possibly failed) to calculate
                            if not (attr_name == "Max Feels Like (Heat Index)" and "calculated_heatindex" in hourly_df_raw.columns):
                                print(f"  ⚠️ Hourly data column for '{attr_name}' (expected: {', '.join(expected_cols_for_attr)}) not found. Avail: {hourly_df_raw.columns.tolist()}")
                    
                    if aggregations:
                        hourly_aggregated_df = hourly_df_raw.groupby("Day").agg(**aggregations).reset_index()
                        hourly_aggregated_df = hourly_aggregated_df.rename(columns={"Day": "Date"}) # Rename Day to Date for merging
                        print(f"  ✅ Hourly data processed for {city_name}. Shape: {hourly_aggregated_df.shape}")
                    else: print(f"  ⚠️ No valid hourly attributes to aggregate for {city_name}.")
                else: print(f"  ⚠️ '{date_utc_col or 'Date UTC'}' or '{hour_utc_col or 'Hour UTC'}' columns not found in hourly API data for {city_name}.")
        except Exception as e: print(f"  ❌ ERROR fetching/processing hourly API data for {city_name}: {e}")

    # --- Merge & Finalize ---
    merged_df = pd.DataFrame()
    if not daily_df.empty and "Date" in daily_df.columns:
        daily_df["Date"] = pd.to_datetime(daily_df["Date"], errors='coerce').dt.normalize()
        daily_df.dropna(subset=['Date'], inplace=True)
        merged_df = daily_df.copy()

    if not hourly_aggregated_df.empty and "Date" in hourly_aggregated_df.columns:
        hourly_aggregated_df["Date"] = pd.to_datetime(hourly_aggregated_df["Date"], errors='coerce').dt.normalize()
        hourly_aggregated_df.dropna(subset=['Date'], inplace=True)
        if not merged_df.empty:
            merged_df = pd.merge(merged_df, hourly_aggregated_df, on="Date", how="outer")
        else:
            merged_df = hourly_aggregated_df.copy()
    
    if merged_df.empty:
        print(f"  ⚠️ No data for {city_name} after processing daily/hourly.")
        return pd.DataFrame()

    merged_df.insert(0, "City", city_name)
    merged_df.insert(1, "SiteId", city_id)
    
    # Ensure all requested attributes are columns in the final DataFrame, filled with NaN if missing
    final_cols_ordered = ["City", "SiteId", "Date"]
    for attr in requested_attributes:
        if attr not in final_cols_ordered: final_cols_ordered.append(attr)
        if attr not in merged_df.columns: merged_df[attr] = np.nan # Add as NaN if not found
    
    # Select only the ordered columns that actually exist, plus any extras that might have been formed
    final_df_cols = [col for col in final_cols_ordered if col in merged_df.columns]
    for col in merged_df.columns: # Add any other columns not in the initial ordered list
        if col not in final_df_cols:
            final_df_cols.append(col)
            
    return merged_df[final_df_cols]


# --- Main Execution ---
if __name__ == "__main__":
    if not all([USERNAME, PROFILE, PASSWORD]):
        print("❌ Critical Error: WSI credentials not found in .env. Please check WSI_ACCOUNT_USERNAME, WSI_PROFILE_EMAIL, WSI_PASSWORD.")
        exit()

    print(f"ℹ️ Output directory set to: {OUTPUT_DIR.resolve()}")
    if not OUTPUT_DIR.exists():
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            print(f"📂 Created output directory: {OUTPUT_DIR.resolve()}")
        except Exception as e:
            print(f"❌ Error creating output directory {OUTPUT_DIR}: {e}")
            print("Please ensure the path is correct and you have write permissions.")
            exit()

    city_map = get_city_map()
    if not city_map: print("Could not retrieve city map. Exiting."); exit()
    selections = prompt_user_selections(city_map)
    if not selections: print("No valid selections made. Exiting."); exit()

    all_cities_data = []
    for city_info in selections["cities"]:
        city_df = fetch_single_city_data(city_info, selections["start_date"], selections["end_date"], selections["attributes"])
        if not city_df.empty: all_cities_data.append(city_df)
        # time.sleep(0.1) # Consider a small delay if making many calls

    if not all_cities_data:
        print("\n❌ No data retrieved for any of the selected cities.")
    else:
        final_combined_df = pd.concat(all_cities_data, ignore_index=True)
        if "Date" in final_combined_df.columns:
            final_combined_df = final_combined_df.sort_values(by=["City", "Date"])
        
        city_names_str = "_".join(c["name"].replace(" ", "").split("(")[0] for c in selections["cities"][:2]) 
        if len(selections["cities"]) > 2: city_names_str += "_etc"
        filename_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_filename = OUTPUT_DIR / f"WSI_Historical_Weather_{city_names_str}_{selections['start_date']}_to_{selections['end_date']}_{filename_ts}.csv"
        
        try:
            final_combined_df.to_csv(output_filename, index=False, date_format='%Y-%m-%d')
            print(f"\n✅ Successfully saved data to: {output_filename}")
            print(f"Total rows: {len(final_combined_df)}")
            if not final_combined_df.empty:
                print("\n🔎 Preview of the first 5 rows of combined data:")
                print(final_combined_df.head().to_string())
        except Exception as e:
            print(f"\n❌ Error saving data to CSV: {e}")
