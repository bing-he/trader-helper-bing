"""
================================================================================
UpdateWeatherandWeatherForecast.py
================================================================================
Description:
This script automates fetching, processing, and saving historical and forecast
weather data from the WSI Trader API. It maintains two key data files:
WEATHER.csv (historical observations) and WEATHERforecast.csv (future
predictions).

Key Features:
- Centralized Configuration: Manages all file paths, API details, and column
  definitions in a single configuration class.
- API Client Abstraction: A dedicated class handles all communication with the
  WSI Trader API, centralizing request logic, authentication, and error handling.
- Modular Processing: Core logic is broken into smaller, single-responsibility
  functions for clarity and testability.
- Smart Historical Updates: Checks existing data, removes the last 14 days
  to ensure data integrity, and fetches only the necessary new records.
- Efficient Normals Calculation: Calculates 10-year normals using vectorized
  pandas operations, avoiding slow row-by-row iteration.
- Clear Execution Flow: The main function orchestrates the process in clear,
  logical steps.

Execution Flow:
1. Initialize configuration and the API client.
2. Get the list of city symbols from PriceAdmin.csv.
3. Fetch full city titles for better labeling.
4. Update WEATHER.csv with the latest historical data.
5. Update WEATHERforecast.csv using the newly updated historical data for context.
"""

import os
import time
from datetime import datetime, timedelta
from io import StringIO
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# ============================================================================
# ONE-TIME BACKFILL CONFIGURATION
# ============================================================================
# Set to True to fetch all historical data from the start date defined in
# the Config class. This is for initial setup or adding new columns.
# WARNING: This will take a very long time. Set back to False for normal use.
PERFORM_FULL_BACKFILL = False
# ============================================================================


@dataclass
class Config:
    """Holds all configuration for the script."""
    # --- Credentials (will be loaded in __post_init__) ---
    wsi_username: str = None
    wsi_profile: str = None
    wsi_password: str = None

    # --- API Details ---
    base_service_url: str = "https://www.wsitrader.com/Services/CSVDownloadService.svc"
    api_timeout_seconds: int = 120
    hist_start_date: str = "2015-01-01"

    # --- File Paths (using relative paths) ---
    script_dir: str = os.path.dirname(os.path.abspath(__file__))
    base_project_dir: str = os.path.dirname(script_dir)
    info_dir: str = os.path.join(base_project_dir, "INFO")
    price_admin_path: str = os.path.join(info_dir, "PriceAdmin.csv")
    weather_csv_path: str = os.path.join(info_dir, "WEATHER.csv")
    weather_forecast_csv_path: str = os.path.join(info_dir, "WEATHERforecast.csv")

    # --- Column Definitions ---
    core_columns: list[str] = field(default_factory=lambda: [
        'City Symbol', 'City Title', 'Date', 'Min Temp', 'Max Temp',
        'Avg Temp', 'CDD', 'HDD'
    ])
    normal_columns: list[str] = field(default_factory=lambda: [
        '10yr Min Temp', '10yr Max Temp', '10yr Avg Temp', '10yr CDD', '10yr HDD'
    ])
    new_attributes: list[str] = field(default_factory=lambda: [
        'Max Dewpoint', 'Avg Cloud Cover', 'Max Surface Wind',
        'Min Feels Like', 'Max Feels Like', 'Daily Precip Amount'
    ])
    
    @property
    def final_column_order(self) -> list[str]:
        """Returns the enforced final column order for output files."""
        return self.core_columns + self.normal_columns + self.new_attributes

    def __post_init__(self):
        """Load credentials after initialization and create directories."""
        self.wsi_username = os.getenv("WSI_ACCOUNT_USERNAME")
        self.wsi_profile = os.getenv("WSI_PROFILE_EMAIL")
        self.wsi_password = os.getenv("WSI_PASSWORD")
        
        # Create the INFO directory if it doesn't exist.
        os.makedirs(self.info_dir, exist_ok=True)


class WSIClient:
    """A client to interact with the WSI Trader API."""

    def __init__(self, config: Config):
        self.config = config
        if not all([config.wsi_username, config.wsi_profile, config.wsi_password]):
            raise ValueError("Missing WSI credentials. Ensure .env file is in the script's directory and loaded correctly.")

    def _make_request(self, endpoint: str, params: dict) -> str | None:
        """Generic method to make a GET request and return the response text."""
        base_params = {
            "Account": self.config.wsi_username,
            "Profile": self.config.wsi_profile,
            "Password": self.config.wsi_password
        }
        url = f"{self.config.base_service_url}/{endpoint}"
        
        try:
            response = requests.get(
                url, params={**base_params, **params},
                timeout=self.config.api_timeout_seconds
            )
            response.raise_for_status()
            if "no data" in response.text.lower():
                return None
            return response.text
        except requests.RequestException as e:
            print(f"  - ❌ API Error for {url}: {e}")
            return None

    def get_city_ids(self) -> pd.DataFrame:
        """Fetches the list of all available cities and their IDs."""
        response_text = self._make_request("GetCityIds", {})
        if response_text:
            return pd.read_csv(StringIO(response_text), header=0)
        return pd.DataFrame()

    def get_historical_daily(self, city_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetches daily observed weather data."""
        params = {
            "TempUnits": "F", "HistoricalProductID": "HISTORICAL_DAILY_OBSERVED",
            "StartDate": start_date, "EndDate": end_date,
            "IsDisplayDates": "false", "IsTemp": "true", "CityIds[]": city_id
        }
        response_text = self._make_request("GetHistoricalObservations", params)
        if response_text:
            return pd.read_csv(
                StringIO(response_text), 
                skiprows=3, 
                header=None, 
                names=["Date", "Min Temp", "Max Temp", "AvgTemp", "Daily Precip Amount"]
            )
        return pd.DataFrame()

    def get_historical_hourly(self, city_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetches hourly observed weather data for detailed metrics."""
        params = {
            "HistoricalProductID": "HISTORICAL_HOURLY_OBSERVED",
            "DataTypes[]": ["dewpoint", "temperature", "cloudCover", "windSpeed", "heatIndex", "windChill"],
            "TempUnits": "F", "StartDate": start_date, "EndDate": end_date, "CityIds[]": city_id
        }
        response_text = self._make_request("GetHistoricalObservations", params)
        if response_text:
            return pd.read_csv(
                StringIO(response_text), 
                skiprows=2, 
                header=None, 
                names=["Date", "Hour", "Temperature", "Dewpoint", "WindChill", 
                       "HeatIndex", "WindSpeed", "CloudCover", "Precip_hourly"]
            )
        return pd.DataFrame()

    def get_forecast_hourly(self, city_id: str) -> pd.DataFrame:
        """Fetches 15-day hourly forecast data."""
        params = {"region": "NA", "SiteIds[]": city_id, "TempUnits": "F"}
        response_text = self._make_request("GetHourlyForecast", params)
        if response_text:
            return pd.read_csv(
                StringIO(response_text), 
                skiprows=2, 
                header=None, 
                names=["DateHour", "Temperature", "TempDiff", "TempNormal", "DewPoint", "CloudCover",
                       "FeelsLikeTemp", "FeelsLikeTempDiff", "Precip", "WindDirection", "WindSpeed", "GHI"]
            )
        return pd.DataFrame()

# --- Data Processing Functions ---

def get_city_symbols(file_path: str) -> list[str]:
    """Reads PriceAdmin.csv to get a unique list of city symbols."""
    print(f"\n--- Reading City List from {os.path.basename(file_path)} ---")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CRITICAL: {os.path.basename(file_path)} not found at {file_path}")
    
    df = pd.read_csv(file_path, usecols=[9], header=0)
    city_symbols = df.iloc[:, 0].dropna().unique().tolist()
    
    if not city_symbols:
        raise ValueError("CRITICAL: No city symbols found in Column J of PriceAdmin.csv.")
        
    print(f"SUCCESS: Found {len(city_symbols)} unique city symbols.")
    return city_symbols


def fetch_city_titles(client: WSIClient, city_symbols: list[str]) -> dict[str, str]:
    """Fetches full city names for the given symbols."""
    print("🔄 Fetching city titles...")
    city_ids_df = client.get_city_ids()
    if city_ids_df.empty:
        print("⚠️ Could not fetch city titles. Using symbols as fallback.")
        return {symbol: symbol for symbol in city_symbols}
    
    # Robustly handle the columns: take the first two, whatever they are.
    city_ids_df = city_ids_df.iloc[:, :2]
    city_ids_df.columns = ["SiteId", "Station Name"]
    
    city_ids_df["SiteId"] = city_ids_df["SiteId"].str.strip()
    
    title_map = city_ids_df[city_ids_df['SiteId'].isin(city_symbols)]
    city_title_map = pd.Series(title_map['Station Name'].values, index=title_map['SiteId']).to_dict()
    
    # Fill in any missing titles with the symbol itself
    for symbol in city_symbols:
        city_title_map.setdefault(symbol, symbol)
        
    print("✅ Successfully fetched city titles.")
    return city_title_map


def calculate_feels_like(temp: pd.Series, heat_index: pd.Series, wind_chill: pd.Series, wind_speed: pd.Series) -> pd.Series:
    """Determines the 'feels like' temperature based on conditions."""
    conditions = [(temp > 80), (temp < 50) & (wind_speed >= 3)]
    choices = [heat_index, wind_chill]
    return np.select(conditions, choices, default=temp)


def calculate_10yr_normals(df_to_update: pd.DataFrame, hist_context_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Calculates 10-year normals using vectorized pandas operations."""
    print("🔄 Calculating 10-year normals...")
    if hist_context_df.empty:
        print("⚠️ Historical context is empty. Skipping normals calculation.")
        return df_to_update.assign(**{col: np.nan for col in config.normal_columns})

    # Prepare historical data
    hist_df = hist_context_df.copy()
    hist_df['Date'] = pd.to_datetime(hist_df['Date'])
    hist_df['MonthDay'] = hist_df['Date'].dt.strftime('%m-%d')
    hist_df['Year'] = hist_df['Date'].dt.year

    source_cols = ['Min Temp', 'Max Temp', 'Avg Temp', 'CDD', 'HDD']
    hist_df = hist_df[['City Symbol', 'MonthDay', 'Year'] + source_cols].dropna()

    # Prepare target data
    target_df = df_to_update.copy()
    target_df['Date'] = pd.to_datetime(target_df['Date'])
    target_df['MonthDay'] = target_df['Date'].dt.strftime('%m-%d')
    target_df['Year'] = target_df['Date'].dt.year
    
    # Merge target dates with all matching historical calendar days
    merged = pd.merge(
        target_df[['City Symbol', 'Date', 'MonthDay', 'Year']],
        hist_df,
        on=['City Symbol', 'MonthDay'],
        suffixes=('_target', '_hist')
    )
    
    # Filter for the correct 10-year window for each target date
    valid_hist = merged[
        (merged['Year_hist'] >= merged['Year_target'] - 10) &
        (merged['Year_hist'] < merged['Year_target'])
    ]
    
    # Group by the original target records and calculate the mean
    normals = valid_hist.groupby(['City Symbol', 'Date'])[source_cols].mean().reset_index()
    normals.rename(columns=dict(zip(source_cols, config.normal_columns)), inplace=True)
    
    # Merge the calculated normals back into the original dataframe
    result_df = pd.merge(df_to_update, normals, on=['City Symbol', 'Date'], how='left')
    print("✅ 10-year normals calculation complete.")
    return result_df


def process_historical_city_data(city_id: str, client: WSIClient, start_date_api: str, end_date_api: str) -> pd.DataFrame:
    """Fetches, combines, and processes daily and hourly historical data for one city."""
    daily_df = client.get_historical_daily(city_id, start_date_api, end_date_api)
    if daily_df.empty:
        print(f"  - No daily data for {city_id}.")
        return pd.DataFrame()

    hourly_df = client.get_historical_hourly(city_id, start_date_api, end_date_api)
    
    # Process hourly data to get daily summaries
    if not hourly_df.empty:
        numeric_cols = ["Temperature", "Dewpoint", "WindChill", "HeatIndex", "WindSpeed"]
        for col in numeric_cols:
            hourly_df[col] = pd.to_numeric(hourly_df[col], errors='coerce')
        
        hourly_df["CloudCover"] = pd.to_numeric(hourly_df["CloudCover"].astype(str).str.replace("%", "", regex=False), errors='coerce')
        hourly_df["FeelsLike"] = calculate_feels_like(hourly_df["Temperature"], hourly_df["HeatIndex"], hourly_df["WindChill"], hourly_df["WindSpeed"])
        hourly_df["Date"] = pd.to_datetime(hourly_df["Date"], format='%m/%d/%Y').dt.date

        hourly_summary = hourly_df.groupby("Date").agg(
            Max_Dewpoint=('Dewpoint', 'max'),
            Avg_Cloud_Cover=('CloudCover', 'mean'),
            Max_Surface_Wind=('WindSpeed', 'max'),
            Min_Feels_Like=('FeelsLike', 'min'),
            Max_Feels_Like=('FeelsLike', 'max')
        ).reset_index()
        hourly_summary.columns = ["Date", "Max Dewpoint", "Avg Cloud Cover", "Max Surface Wind", "Min Feels Like", "Max Feels Like"]
    else:
        hourly_summary = pd.DataFrame(columns=["Date"])

    # Combine daily and hourly data
    daily_df["Date"] = pd.to_datetime(daily_df["Date"], format='%d-%b-%Y').dt.date
    if not hourly_summary.empty:
        combined_df = pd.merge(daily_df, hourly_summary, on="Date", how="left")
    else:
        combined_df = daily_df
    
    # Calculate derived columns
    combined_df['Avg Temp'] = (pd.to_numeric(combined_df['Min Temp'], errors='coerce') + pd.to_numeric(combined_df['Max Temp'], errors='coerce')) / 2
    combined_df['HDD'] = (65 - combined_df['Avg Temp']).clip(lower=0).round(2)
    combined_df['CDD'] = (combined_df['Avg Temp'] - 65).clip(lower=0).round(2)
    
    return combined_df

# --- Orchestration Functions ---

def update_historical_data(client: WSIClient, config: Config, city_symbols: list[str], city_titles: dict):
    """Orchestrates the update of the historical WEATHER.csv file."""
    print("\n" + "="*50)
    print("--- 1. Starting Historical Data Update (WEATHER.csv) ---")
    print("="*50)

    # 1. Determine the date range for fetching data
    fetch_start_date = datetime.strptime(config.hist_start_date, '%Y-%m-%d')
    existing_df = pd.DataFrame()
    
    if not PERFORM_FULL_BACKFILL and os.path.exists(config.weather_csv_path):
        print(f"Reading existing data from {os.path.basename(config.weather_csv_path)}...")
        existing_df = pd.read_csv(config.weather_csv_path, parse_dates=['Date'])
        if not existing_df.empty:
            last_date = existing_df['Date'].max()
            fetch_start_date = last_date - timedelta(days=14)
            print(f"Last record: {last_date.date()}. Rolling back 14 days. New fetch start: {fetch_start_date.date()}.")
            existing_df = existing_df[existing_df['Date'] < fetch_start_date] # Keep only data before the fetch window
    
    fetch_start_date_str = fetch_start_date.strftime('%Y-%m-%d')
    end_date_str = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    start_date_api = fetch_start_date.strftime("%m/%d/%Y")
    end_date_api = datetime.strptime(end_date_str, "%Y-%m-%d").strftime("%m/%d/%Y")
    
    # 2. Fetch and process data for all cities
    all_new_data = []
    for i, city_id in enumerate(city_symbols):
        print(f"\n🔄 ({i+1}/{len(city_symbols)}) Fetching historical data for: {city_titles.get(city_id, city_id)}")
        city_df = process_historical_city_data(city_id, client, start_date_api, end_date_api)
        if not city_df.empty:
            city_df.insert(0, "City Symbol", city_id)
            city_df.insert(1, "City Title", city_titles.get(city_id, city_id))
            all_new_data.append(city_df)
            print(f"  - ✅ Processed {len(city_df)} new records for {city_id}.")
        time.sleep(0.2) # Be courteous to the API

    if not all_new_data:
        print("\n⚠️ No new historical data fetched. Skipping update.")
        return

    # 3. Combine old and new data
    newly_fetched_df = pd.concat(all_new_data, ignore_index=True)
    final_historical_df = pd.concat([existing_df, newly_fetched_df], ignore_index=True)
    
    # FIX: Standardize dtypes immediately after combining to prevent errors.
    final_historical_df['City Symbol'] = final_historical_df['City Symbol'].astype(str)
    final_historical_df['Date'] = pd.to_datetime(final_historical_df['Date'])
    
    final_historical_df.drop_duplicates(subset=['City Symbol', 'Date'], keep='last', inplace=True)
    final_historical_df.sort_values(by=['City Symbol', 'Date'], inplace=True)

    # 4. Calculate 10-year normals and format for saving
    final_historical_df = calculate_10yr_normals(final_historical_df, final_historical_df, config)
    
    for col in config.final_column_order:
        if col not in final_historical_df.columns:
            final_historical_df[col] = np.nan
    
    final_historical_df['Date'] = pd.to_datetime(final_historical_df['Date']).dt.strftime('%Y-%m-%d')
    final_historical_df = final_historical_df[config.final_column_order]
    
    # 5. Save to CSV
    final_historical_df.to_csv(config.weather_csv_path, index=False)
    print(f"\n✅ Historical data saved to: {config.weather_csv_path}")
    print(f"   Total rows: {len(final_historical_df)}")


def update_forecast_data(client: WSIClient, config: Config, city_symbols: list[str], city_titles: dict):
    """Orchestrates the creation of the WEATHERforecast.csv file."""
    print("\n" + "="*50)
    print("--- 2. Starting Forecast Data Update (WEATHERforecast.csv) ---")
    print("="*50)

    try:
        historical_context_df = pd.read_csv(config.weather_csv_path)
    except FileNotFoundError:
        print(f"❌ CRITICAL: Cannot read {config.weather_csv_path} for context. Aborting forecast.")
        return
        
    all_forecast_data = []
    for i, city_id in enumerate(city_symbols):
        print(f"\n🔄 ({i+1}/{len(city_symbols)}) Fetching forecast for: {city_titles.get(city_id, city_id)}")
        forecast_df = client.get_forecast_hourly(city_id)
        if forecast_df.empty:
            print(f"  - No forecast data for {city_id}.")
            continue

        # Process and summarize forecast data
        forecast_df["Date"] = pd.to_datetime(forecast_df["DateHour"], format="mixed").dt.date
        numeric_cols = ["Temperature", "DewPoint", "CloudCover", "FeelsLikeTemp", "Precip", "WindSpeed"]
        for col in numeric_cols:
            forecast_df[col] = pd.to_numeric(forecast_df[col], errors="coerce")

        summary = forecast_df.groupby("Date").agg(
            Min_Temp=('Temperature', 'min'), 
            Max_Temp=('Temperature', 'max'),
            Max_Dewpoint=('DewPoint', 'max'), 
            Min_Feels_Like=('FeelsLikeTemp', 'min'),
            Max_Feels_Like=('FeelsLikeTemp', 'max'), 
            Max_Surface_Wind=('WindSpeed', 'max'),
            Daily_Precip_Amount=('Precip', 'sum'), 
            Avg_Cloud_Cover=('CloudCover', 'mean')
        ).reset_index()

        # Apply custom temperature adjustments
        temp_diff = summary['Max_Temp'] - summary['Min_Temp']
        summary['Max_Temp'] += np.where(temp_diff <= 18, 2 * temp_diff / 18, 2.0)
        summary['Min_Temp'] -= 1
        
        # FIX: Rename all aggregated columns to match the final output format.
        summary.rename(columns={
            'Min_Temp': 'Min Temp', 
            'Max_Temp': 'Max Temp',
            'Max_Dewpoint': 'Max Dewpoint',
            'Min_Feels_Like': 'Min Feels Like',
            'Max_Feels_Like': 'Max Feels Like',
            'Max_Surface_Wind': 'Max Surface Wind',
            'Daily_Precip_Amount': 'Daily Precip Amount',
            'Avg_Cloud_Cover': 'Avg Cloud Cover'
        }, inplace=True)

        summary['Avg Temp'] = (summary['Min Temp'] + summary['Max Temp']) / 2
        summary['HDD'] = (65 - summary['Avg Temp']).clip(lower=0).round(2)
        summary['CDD'] = (summary['Avg Temp'] - 65).clip(lower=0).round(2)

        summary.insert(0, "City Symbol", city_id)
        summary.insert(1, "City Title", city_titles.get(city_id, city_id))
        all_forecast_data.append(summary)
        print(f"  - ✅ Processed forecast for {city_id}.")
        time.sleep(0.2)

    if not all_forecast_data:
        print("\n⚠️ No forecast data fetched. Skipping update.")
        return

    # 2. Combine, calculate normals, format, and save
    final_forecast_df = pd.concat(all_forecast_data, ignore_index=True)
    
    # FIX: Standardize dtypes immediately after combining.
    final_forecast_df['City Symbol'] = final_forecast_df['City Symbol'].astype(str)
    final_forecast_df['Date'] = pd.to_datetime(final_forecast_df['Date'])
    
    final_forecast_df = calculate_10yr_normals(final_forecast_df, historical_context_df, config)

    for col in config.final_column_order:
        if col not in final_forecast_df.columns:
            final_forecast_df[col] = np.nan
            
    final_forecast_df['Date'] = pd.to_datetime(final_forecast_df['Date']).dt.strftime('%Y-%m-%d')
    final_forecast_df = final_forecast_df[config.final_column_order]

    final_forecast_df.to_csv(config.weather_forecast_csv_path, index=False)
    print(f"\n✅ Forecast data saved to: {config.weather_forecast_csv_path}")
    print(f"   Total rows: {len(final_forecast_df)}")


def main():
    """Main function to orchestrate the entire weather data update process."""
    print("--- Initializing Weather Data Update Script ---")
    
    # Find the directory the script is in and load the .env file from there.
    # This is portable and will work on any machine.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(script_dir, '.env')
    load_dotenv(dotenv_path=dotenv_path)

    try:
        config = Config()
        client = WSIClient(config)
        
        city_symbols = get_city_symbols(config.price_admin_path)
        city_titles = fetch_city_titles(client, city_symbols)

        update_historical_data(client, config, city_symbols, city_titles)
        update_forecast_data(client, config, city_symbols, city_titles)
        
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ A critical error occurred: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
    finally:
        print("\n--- Weather Data Update Script Finished ---")

if __name__ == "__main__":
    main()