import requests
import csv
import io
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
env_path = find_dotenv(raise_error_if_not_found=False)
print(f"Path to .env file found by find_dotenv: {env_path}") # DEBUG
loaded = load_dotenv(dotenv_path=env_path, verbose=True)
print(f".env file loaded by load_dotenv: {loaded}") # DEBUG

ACCOUNT_USERNAME = os.getenv("WSI_ACCOUNT_USERNAME")
PROFILE_EMAIL = os.getenv("WSI_PROFILE_EMAIL")
PASSWORD = os.getenv("WSI_PASSWORD")

print(f"Loaded USERNAME from os.getenv: '{ACCOUNT_USERNAME}' (Type: {type(ACCOUNT_USERNAME)})") # DEBUG
print(f"Loaded EMAIL from os.getenv: '{PROFILE_EMAIL}' (Type: {type(PROFILE_EMAIL)})") # DEBUG
print(f"Loaded PASSWORD from os.getenv: '{PASSWORD}' (Type: {type(PASSWORD)})") # DEBUG


BASE_URL = "https://www.wsitrader.com/Services/CSVDownloadService.svc/GetCityTableForecast"
PARAMS = {
    "Account": ACCOUNT_USERNAME,
    "Profile": PROFILE_EMAIL,
    "Password": PASSWORD,
    "IsCustom": "false",
    "CurrentTabName": "AverageTemp",
    "TempUnits": "F",
    "SiteId": "allcities", # As per documentation for "all available cities"
    "Region": "NA"
}

# No longer using CITY_COLUMN_NAME as we'll parse based on column index after skipping header lines

def get_unique_north_american_cities():
    unique_cities = set()

    if not all([ACCOUNT_USERNAME, PROFILE_EMAIL, PASSWORD]):
        print("ERROR: Missing one or more API credentials from the .env file.")
        print("Please ensure WSI_ACCOUNT_USERNAME, WSI_PROFILE_EMAIL, and WSI_PASSWORD are set in your .env file.")
        return None

    print(f"Requesting data from: {BASE_URL}")

    try:
        response = requests.get(BASE_URL, params=PARAMS)
        response.raise_for_status()

        csv_data = response.text
        
        # --- MODIFIED CSV PARSING LOGIC ---
        csvfile = io.StringIO(csv_data)
        
        # Skip the first two lines which are general headers/dates
        try:
            next(csvfile) # Skip "ALLCITIES - Average Temp,Forecast Made..."
            next(csvfile) # Skip the primary date row ",5/21/2025,5/22/2025..."
        except StopIteration:
            print("\nWarning: CSV file has less than 2 lines. Cannot skip headers.")
            print(f"Full CSV data:\n{csv_data}")
            return None

        # Now use csv.reader because the next line is the actual header for the data rows
        # or it's the first data row if we consider the third line as the "effective header"
        reader = csv.reader(csvfile) # Use standard csv.reader

        header_row_for_data = None
        try:
            header_row_for_data = next(reader) # This should be "City:,Average:,Average:..."
        except StopIteration:
            print("\nWarning: CSV file has less than 3 lines. No data rows found after skipping headers.")
            print(f"Full CSV data after trying to skip two lines:\n{csv_data}") # This will show data starting from line 3
            return None

        print(f"Effective header row for data: {header_row_for_data}")

        # Check if the first column of this "header" is indeed 'City:' or similar
        if header_row_for_data and (header_row_for_data[0].strip().lower() == "city:" or header_row_for_data[0].strip().lower() == "city"):
            # Now iterate through the actual data rows
            for row in reader:
                if row: # Ensure row is not empty
                    city_name = row[0].strip() # City name is in the first column
                    if city_name: # Ensure city_name is not empty after stripping
                        unique_cities.add(city_name)
        else:
            print(f"\nERROR: Did not find 'City:' as the first column in the effective header row.")
            print(f"Effective header was: {header_row_for_data}")
            print("The CSV structure might have changed or is not as expected.")
            # Fallback: Let's try to see if rows from here directly start with city names
            # This part is for robustness if the "City:" header line itself is missing but data follows
            print("Attempting to parse rows directly assuming first column is city name...")
            # We need to re-evaluate the 'reader' or the lines we already consumed
            # For simplicity, let's re-parse from the point after skipping the first two lines
            csvfile.seek(0) # Reset stream
            next(csvfile); next(csvfile) # Skip first two again
            # If header_row_for_data was actually the first data row:
            if header_row_for_data and header_row_for_data[0].strip(): # If it wasn't the "City:" line but a real city
                unique_cities.add(header_row_for_data[0].strip())

            # Continue with the rest of the reader
            temp_reader = csv.reader(csvfile)
            if header_row_for_data and (header_row_for_data[0].strip().lower() == "city:" or header_row_for_data[0].strip().lower() == "city"):
                 # We already processed the "City:" line, so the temp_reader starts at data
                 pass
            else: # header_row_for_data might have been the first data row, or an unexpected header
                  # if it was an unexpected header, we need to skip it in temp_reader.
                  # This logic becomes complex, for now, we'll rely on the "City:" check primarily.
                  # If the first data row was consumed as `header_row_for_data`, the next loop starts from 2nd data row.
                  pass


            for row_num, row in enumerate(temp_reader):
                 # If the "City:" header was NOT found, and `header_row_for_data` was the first actual data row,
                 # this loop starts from the second data row.
                 # If the "City:" header WAS found, `header_row_for_data` consumed it, and this loop starts from the first data row.
                if row and row[0].strip():
                    # Avoid re-adding the first data row if it was already added via header_row_for_data
                    if not (row_num == 0 and header_row_for_data and row[0].strip() == header_row_for_data[0].strip() and not (header_row_for_data[0].strip().lower() == "city:" or header_row_for_data[0].strip().lower() == "city") ):
                        unique_cities.add(row[0].strip())


        if not unique_cities:
            print("\nWarning: No city names were extracted. The CSV structure might be different than expected after skipping headers.")
            print(f"Original CSV data (first 500 chars):\n{csv_data[:500]}")


    except requests.exceptions.HTTPError as http_err:
        print(f"\nHTTP error occurred: {http_err}")
        if response is not None and response.text:
            print(f"Response content from API: {response.text[:500]}...")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"\nRequest error occurred: {req_err}")
        return None
    except csv.Error as csv_err: # csv.Error might not be hit with manual line skipping
        print(f"\nCSV processing error occurred: {csv_err}")
        print(f"First 500 characters of response text: {csv_data[:500] if 'csv_data' in locals() else 'N/A'}")
        return None
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return None

    return sorted(list(unique_cities))

if __name__ == "__main__":
    print("Attempting to fetch a unique list of North American cities from WSI Trader API using .env...")
    
    cities = get_unique_north_american_cities()

    if cities is not None:
        if cities:
            print(f"\nFound {len(cities)} unique North American city/station names:")
            for city in cities:
                print(city)
        else:
            print("\nNo unique city names were found. Please check the script's CSV parsing logic against the API output.")
    else:
        print("\nCould not retrieve the list of cities due to errors mentioned above.")

    print("\n--- Reminders ---")
    print("1. The script now attempts to parse a CSV structure with 2 initial header lines followed by a 'City:' column.")
    print("2. The list comprises North American cities; for strictly U.S. cities, further filtering might be needed.")
    print("3. Your credentials are now loaded from the `.env` file. Keep this file secure.")