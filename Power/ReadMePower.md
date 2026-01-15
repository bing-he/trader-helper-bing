# Power Folder

## Overview

This folder contains Python scripts designed to interact with the [GridStatus.io API](https://www.gridstatus.io/). The scripts are used to fetch, process, analyze, visualize, and save various types of electricity market data from Independent System Operators (ISOs) across North America. This includes Locational Marginal Prices (LMPs), load data, fuel mix information, and other market-specific reports.

The primary goal of these tools is to gather fundamental data relevant to power markets, which can then be used for trading analysis, strategy development, and supporting the broader "trader-helper" agent.

## General Setup & Dependencies

Before running these scripts, ensure you have the following:

* **Python Environment**: A working Python 3.x environment.
* **Required Libraries**:
    * `gridstatusio`: The official Python client for the GridStatus.io API. Install via `pip install gridstatusio`.
    * `pandas`: For data manipulation and analysis. Install via `pip install pandas`.
    * `python-dotenv`: For managing environment variables (like API keys). Install via `pip install python-dotenv`.
    * `matplotlib`: For generating plots and charts. Install via `pip install matplotlib`.
* **API Key**:
    * A valid API key for GridStatus.io.
    * This key should be stored in a `.env` file located either in the `Power/` directory or the root `trader-helper/` directory.
    * The `.env` file should contain a line like: `GRIDSTATUS_API_KEY='your_actual_api_key_here'`
    * **Important**: Ensure your `.env` file (containing the API key) is listed in your main `.gitignore` file to prevent accidental commits of sensitive credentials.

## Scripts

Below is a description of each script within this folder:

### 1. `GSPowerLMPChartCSV.py` (Interactive ISO Price Explorer)

* **Purpose**: This script provides an interactive command-line interface for users to select an ISO, specific locations within that ISO (like hubs, zones, or DLAPs), and a date range to fetch and visualize Locational Marginal Price (LMP) data.
* **Functionality**:
    * Lists available ISOs based on pre-defined configurations.
    * Dynamically fetches and lists available trading locations for the chosen ISO and location types (e.g., Hubs, Zones) using a lookback period.
    * Prompts the user for a start and end date for data retrieval.
    * Allows the user to choose between displaying a chart of daily maximum prices, saving the raw hourly price data to a CSV file, or both.
    * Fetches data using the `gridstatusio` client.
    * Processes data into a pandas DataFrame.
    * Generates a plot of daily maximum LMPs using Matplotlib if selected.
    * Saves detailed price data to a CSV file in the current working directory if selected.
* **Inputs**:
    * User selections for ISO, location(s), start date, end date, and output type (chart, CSV, or both).
    * `GRIDSTATUS_API_KEY` from the `.env` file.
* **Outputs**:
    * Console output detailing fetched data and selected options.
    * A Matplotlib chart (displayed in a new window) of daily maximum LMPs for the selected location and period.
    * A CSV file named in the format: `{ISO}_{LocationName}_{PriceColumn}_{StartDate}_to_{EndDate}.csv` (e.g., `CAISO_TH_NP15_GEN-APND_lmp_2023-01-01_to_2023-01-07.csv`) saved in the script's current working directory.

### 2. `GridstatusMultiISOMktCompCHARTandHIST.py` (Multi-ISO & Location Price Explorer)

* **Purpose**: An advanced script that allows users to select multiple ISOs and multiple locations within each of those ISOs to fetch, compare, and visualize daily maximum market prices. It also saves the combined data to a CSV.
* **Functionality**:
    * Allows users to select one or more ISOs from a predefined list.
    * For each selected ISO, it fetches and lists available trading locations.
    * Allows users to select one or more locations for each chosen ISO.
    * Prompts for an overall start and end date for data retrieval.
    * Fetches price data for all selected ISO-location pairs.
    * Combines data from all selections into a single pandas DataFrame.
    * Standardizes price columns and creates a unique `series_id` (ISO_Location) for each time series.
    * Resamples the data to find the daily maximum price for each series.
    * Pivots the data to create a wide DataFrame with dates as the index and each `series_id` as a column.
    * Saves this wide DataFrame of daily maximum prices to a CSV file.
    * Generates and displays a Matplotlib chart comparing the daily maximum prices of all selected series.
* **Inputs**:
    * User selections for multiple ISOs, multiple locations per ISO, start date, and end date.
    * `GRIDSTATUS_API_KEY` from the `.env` file.
* **Outputs**:
    * Console output guiding the user and showing fetch progress.
    * A CSV file named like `combined_daily_max_prices_{StartDate}_to_{EndDate}.csv` containing the daily maximum prices for all selected series, saved in the script's current working directory.
    * A Matplotlib chart (displayed) comparing all selected price series.

### 3. `GSLoadFuelHist.py` (Peak Load & Fuel Mix Analyzer)

* **Purpose**: This script fetches and analyzes hourly load data for a selected ISO to determine daily peak load hours. It then fetches the corresponding fuel mix data at those peak hours.
* **Functionality**:
    * Prompts the user to select an ISO from a configured list (e.g., CAISO, ISONE).
    * Prompts the user for a start and end date for the analysis period.
    * Fetches raw hourly load data for the ISO.
    * Processes the load data to calculate total hourly ISO load (either by summing TAC areas/locations or using a direct ISO total series, depending on ISO configuration).
    * Identifies the timestamp and magnitude of the daily peak load for each day in the period.
    * For each daily peak load hour, it fetches the detailed fuel mix data (e.g., solar, wind, natural_gas, nuclear contributions).
    * Combines the daily peak load information with its corresponding fuel mix into a single DataFrame.
    * Saves the resulting DataFrame to a CSV file.
* **Inputs**:
    * User selections for ISO, start date, and end date.
    * `GRIDSTATUS_API_KEY` from the `.env` file.
    * Pre-defined configurations (`ANALYSIS_ISO_CONFIGS`) within the script for dataset IDs and column names for load and fuel mix for each supported ISO.
* **Outputs**:
    * Console output detailing the analysis steps and progress.
    * A CSV file named in the format: `{ISO}_peak_load_fuel_mix_{StartDate}_to_{EndDate}.csv` (e.g., `CAISO_peak_load_fuel_mix_2023-01-01_to_2023-01-03.csv`) saved in the script's directory. This CSV contains the date, peak load hour (UTC), peak load in MW, and the contribution of each fuel type (in MW) at that peak hour.

### 4. `GSCustAPIExplorer.py` (Daily Peak Report Explorer)

* **Purpose**: This script is designed to fetch and display data from the GridStatus.io `get_daily_peak_report` endpoint. This report typically includes details about daily peak Day-Ahead Market (DAM) LMPs, peak load (with fuel mix), and peak net load (with fuel mix) for a given ISO and market date.
* **Functionality**:
    * Prompts the user to enter an ISO (e.g., CAISO, ERCOT) and a market date.
    * Calls the `GS_CLIENT.get_daily_peak_report()` method.
    * Parses the returned data, which is often a dictionary containing various sections.
    * Prints formatted details for sections like `peak_dam_lmp`, `peak_load`, and `peak_net_load`.
    * Converts structured data sections (like lists of LMPs or dictionaries of load details) into pandas DataFrames for better display.
    * Prompts the user if they want to save these extracted DataFrames to separate CSV files.
* **Inputs**:
    * User input for ISO and market date.
    * `GRIDSTATUS_API_KEY` from the `.env` file.
* **Outputs**:
    * Detailed console output of the fetched report sections.
    * Optionally, saves multiple CSV files if the user chooses, named like: `{ISO}_daily_report_{MarketDate}_{SectionName}.csv` (e.g., `caiso_daily_report_20230101_peak_dam_lmp.csv`), saved in the script's directory.

## Data Output

* Many scripts in this folder output CSV files. By default, these are often saved in the same directory as the script itself (i.e., within the `Power/` folder) or the current working directory from which the script is run.
* Consider standardizing output paths, perhaps to a subfolder within `INFO/` (e.g., `INFO/PowerData/`) if these generated files are meant to be persistent inputs for other analytical processes. This can be modified within each script.

## For AI Agent Robustness

To enhance the usability of these scripts for a future AI agent:

* **Standardize Inputs/Outputs**: Ensure all scripts have clearly defined ways to accept parameters (e.g., command-line arguments in addition to interactive prompts) and predictable output file naming and locations.
* **Structured Logging**: Implement more formal logging (using Python's `logging` module) instead of just `print()` statements for better traceability by an agent.
* **Error Handling**: Continue to build robust error handling for API request failures, data parsing issues, missing configuration, etc.
* **Configuration Files**: For parameters like dataset IDs, column names per ISO, etc., consider moving these from being hardcoded in `ANALYSIS_ISO_CONFIGS` or `ISO_CONFIGS` dicts within scripts to external configuration files (e.g., JSON or YAML) that are easier for an agent to read and potentially modify.
* **Docstrings & Comments**: Ensure comprehensive docstrings for functions and classes, and clear comments for complex code sections.

This README should provide a good starting point for understanding and using the scripts in your `Power` folder.